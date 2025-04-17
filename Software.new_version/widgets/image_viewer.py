import logging

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QScrollArea, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog


class ImageViewer(QWidget):
    def __init__(self, logger : logging.Logger):
        super().__init__()

        self.logger = logger

        self.scale_factor = 1.0

        # QLabel to display the image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load the image
        self.pixmap = None

        # Put the label in a scrollable area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # Zoom in/out buttons
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.reset_btn = QPushButton("Reset")
        self.save_btn = QPushButton("Save")

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

        self.pixmap = image
        self.image_label.setPixmap(self.pixmap)

        self.zoom_in_btn.setEnabled(True)
        self.zoom_out_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def zoom_in(self):
        self.scale_image(1.25)

    def zoom_out(self):
        self.scale_image(0.8)

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.image_label.setPixmap(self.pixmap)
        self.image_label.resize(self.pixmap.size())

    def scale_image(self, factor):
        self.scale_factor *= factor
        new_size = QSize(
            int(self.scale_factor * self.pixmap.size().width()),
            int(self.scale_factor * self.pixmap.size().height()))
        scaled_pixmap = self.pixmap.scaled(new_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

    def save_image(self):
        im = self.pixmap.toImage()

        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if save_path:
            im.save(save_path, "png")
            self.logger.info(f"Image saved to {save_path}")
