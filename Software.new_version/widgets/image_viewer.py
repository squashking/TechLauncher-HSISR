import logging

from PySide6.QtCore import Qt, QSize, QRect, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget, QLabel, QScrollArea, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog


class SelectableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Image to be displayed
        self.pixmap = None

        self.scale_factor = 1.0

        # selection status
        self.image_loaded = False
        self.selection_enabled = False

        self.selection_rect = QRect()
        self.selecting = False
        self.press_position = None

        # coordinates_label
        self.coordinates_label = QLabel(self)  # Create the label to display coordinates
        self.coordinates_label.setStyleSheet("background-color: rgba(255, 255, 255, 180); padding: 2px;")
        self.coordinates_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coordinates_label.setVisible(False)  # Initially hidden

        self.coordinates_label_offset_x = 0
        self.coordinates_label_offset_y = 0
        self.coordinates_label.move(0, 0)

        # Set mouse tracking enabled to display coordinates even when mouse button is not pressed
        self.setMouseTracking(False)  # Disable by default for efficiency

    def mousePressEvent(self, event):
        if not self.selection_enabled or not self.image_loaded:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            self.selection_rect.setTopLeft(pos)
            self.selection_rect.setBottomRight(pos)
            self.selecting = True
            self.press_position = pos
            self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.selection_enabled or not self.image_loaded:
            super().mouseMoveEvent(event)
            return

        if self.selecting:
            # Update the coordinates display
            mouse_coordinate = self.map_from_self_to_pixmap(event.pos())
            self.coordinates_label.setText(f"{mouse_coordinate.x():d}, {mouse_coordinate.y():d}")
            self.coordinates_label.adjustSize()
            self.coordinates_label.move(
                self.coordinates_label_offset_x - self.coordinates_label.size().width(),
                self.coordinates_label_offset_y - self.coordinates_label.size().height())
            self.coordinates_label.setVisible(True)

            pos = event.pos()
            top_left = QPoint(min(pos.x(), self.press_position.x()), min(pos.y(), self.press_position.y()))
            bottom_right = QPoint(max(pos.x(), self.press_position.x()), max(pos.y(), self.press_position.y()))
            self.selection_rect.setTopLeft(top_left)
            self.selection_rect.setBottomRight(bottom_right)
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self.selection_enabled or not self.image_loaded:
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.press_position.x() == event.pos().x() and self.press_position.y() == event.pos().y():
                self.selection_rect = QRect()  # Cancel selection if no movement after pressing

            self.selecting = False
            self.press_position = None
            self.update()

            self.coordinates_label.setVisible(False)

        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.selection_enabled or not self.image_loaded:
            return

        if not self.selection_rect.isEmpty():
            painter = QPainter(self)
            pen = QPen(QColor(0, 0, 255))  # Blue
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 0, 255, 50))  # Semi-transparent blue
            painter.drawRect(self.selection_rect)

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)

        self.image_loaded = True

    def move_coordinates_label(self, offset_x : int, offset_y : int):
        self.coordinates_label_offset_x = offset_x
        self.coordinates_label_offset_y = offset_y

    def map_from_self_to_pixmap(self, point: QPoint, /) -> QPoint:
        original_point = QPoint(
            point.x() - (self.size().width() - self.pixmap.width() * self.scale_factor) // 2,
            point.y() - (self.size().height() - self.pixmap.height() * self.scale_factor) // 2)
        return QPoint(
            int(original_point.x() / self.scale_factor),
            int(original_point.y() / self.scale_factor))

    def zoom_in(self):
        self.scale_image(1.25)

    def zoom_out(self):
        self.scale_image(0.8)

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.size())

    def scale_image(self, factor):
        self.scale_factor *= factor
        new_size = QSize(
            int(self.scale_factor * self.pixmap.size().width()),
            int(self.scale_factor * self.pixmap.size().height()))
        scaled_pixmap = self.pixmap.scaled(new_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self.resize(scaled_pixmap.size())

    def clear_selection(self):
        self.selection_rect = QRect()
        self.update()

    def enable_selection(self, enabled : bool = True):
        self.selection_enabled = enabled


class ScrollableImageView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._widget : SelectableImageLabel | None = None

        self.horizontalScrollBar().valueChanged.connect(self.on_scroll)
        self.verticalScrollBar().valueChanged.connect(self.on_scroll)

    def setWidget(self, widget):
        super().setWidget(widget)

        self._widget = widget

    def resizeEvent(self, event):
        self._widget.move_coordinates_label(
            self.horizontalScrollBar().value() + self.viewport().size().width(),
            self.verticalScrollBar().value() + self.viewport().size().height())

        super().resizeEvent(event)

    def on_scroll(self):
        self._widget.move_coordinates_label(
            self.horizontalScrollBar().value() + self.viewport().size().width(),
            self.verticalScrollBar().value() + self.viewport().size().height())

    def clear_selection(self):
        self._widget.clear_selection()


class ImageViewer(QWidget):
    def __init__(self, logger : logging.Logger, parent=None):
        super().__init__(parent)

        self.logger = logger

        # Crate QLabel to display the image
        # Put the label in a scrollable area
        self.scroll_area = ScrollableImageView(self)
        self.image_label = SelectableImageLabel(self.scroll_area)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.enable_selection(False)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # Zoom in/out buttons
        self.zoom_in_btn = QPushButton("Zoom In", self)
        self.zoom_out_btn = QPushButton("Zoom Out", self)
        self.reset_btn = QPushButton("Reset", self)
        self.save_btn = QPushButton("Save", self)

        self.zoom_in_btn.setEnabled(False)
        self.zoom_out_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_btn.clicked.connect(self.reset_zoom)
        self.save_btn.clicked.connect(self.save_image)

        # Layout for buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.zoom_in_btn)
        btn_layout.addWidget(self.zoom_out_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.save_btn)

        # Overall layout
        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def set_image(self, image: QPixmap):
        if image is None:
            self.logger.error("image is None")
            return

        self.image_label.setPixmap(image)
        self.image_label.pixmap = image

        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def zoom_in(self):
        self.scale_image(1.25)

    def zoom_out(self):
        self.scale_image(0.8)

    def reset_zoom(self):
        self.image_label.reset_zoom()

    def scale_image(self, factor):
        self.image_label.scale_image(factor)

    def save_image(self):
        im = self.image_label.pixmap.toImage()

        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if save_path:
            im.save(save_path, "png")
            self.logger.info(f"Image saved to {save_path}")

    def clear_selection(self):
        self.image_label.clear_selection()

    def enabled_selection(self, enabled=True):
        self.image_label.enable_selection(enabled)
