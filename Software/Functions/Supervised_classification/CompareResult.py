import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView,
                             QGraphicsScene, QSlider, QHBoxLayout, QWidget,
                             QGraphicsOpacityEffect, QVBoxLayout)
from PyQt6.QtGui import QPixmap, QPainter, QImage
from PyQt6.QtCore import Qt, QRectF
from spectral import *


class ImageComparisonSlider(QGraphicsView):
    def __init__(self, before_array, after_array, parent=None):
        super().__init__(parent)

        # Convert arrays to QPixmap
        self.before_pixmap = self.array_to_qpixmap(before_array)
        self.after_pixmap = self.array_to_qpixmap(after_array)

        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Add background image (before)
        self.before_item = self.scene.addPixmap(self.before_pixmap)

        # Add foreground image (after) with opacity effect
        self.after_item = self.scene.addPixmap(self.after_pixmap)
        self.effect = QGraphicsOpacityEffect()
        self.effect.setOpacity(0.0)  # Start at 0% visible
        self.after_item.setGraphicsEffect(self.effect)

        # Create a clipping mask
        self.mask_item = self.scene.addRect(
            QRectF(0, 0, self.before_pixmap.width() / 2, self.before_pixmap.height()),
            Qt.GlobalColor.transparent, Qt.GlobalColor.transparent
        )
        self.after_item.setParentItem(self.mask_item)

        # Add divider line
        self.divider = self.scene.addLine(
            self.before_pixmap.width() / 2, 0,
            self.before_pixmap.width() / 2, self.before_pixmap.height(),
            Qt.GlobalColor.white
        )

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def array_to_qpixmap(self, arr):
        """Convert numpy array to QPixmap"""
        h, w = arr.shape[0], arr.shape[1]
        channels = 1 if arr.ndim == 2 else arr.shape[2]

        format_mapping = {
            1: QImage.Format.Format_Grayscale8,
            3: QImage.Format.Format_RGB888,
            4: QImage.Format.Format_RGBA8888
        }

        return QPixmap.fromImage(
            QImage(arr.data, w, h, channels * w, format_mapping[channels])
        )

    def setSliderPosition(self, value):
        """Update split position (0-100) and adjust opacity of after_item"""
        width = self.before_pixmap.width() * (value / 100)
        self.mask_item.setRect(QRectF(0, 0, width, self.before_pixmap.height()))
        self.divider.setLine(width, 0, width, self.before_pixmap.height())

        # Update opacity based on the slider value (value ranges from 0 to 100)
        opacity = value / 100  # This will map slider values to opacity range [0, 1]
        self.effect.setOpacity(opacity)


class MainWindow(QMainWindow):
    def __init__(self, before_array, after_array):
        super().__init__()
        self.setWindowTitle("Image Comparison Slider")

        # Setup comparison widget
        self.comparison = ImageComparisonSlider(before_array, after_array)

        # Setup slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)  # Set initial value to 0
        self.slider.valueChanged.connect(self.comparison.setSliderPosition)

        # Call setSliderPosition to adjust opacity at the start
        self.comparison.setSliderPosition(0)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.comparison)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Size window to fit images
        self.resize(self.comparison.before_pixmap.width(),
                    self.comparison.before_pixmap.height() + 50)


def apply_spy_colormap(class_array):
    # Map the class IDs to the corresponding colors using the spy_colors array
    num_classes = spy_colors.shape[0]
    print(num_classes)

    # Initialize a blank color array (height, width, 3 for RGB channels)
    color_array = np.zeros((class_array.shape[0], class_array.shape[1], 3), dtype=np.uint8)

    # Map each class ID to its corresponding color
    for i in range(num_classes):
        color_array[class_array == i] = spy_colors[i]

    # Return the class index array for use in the ImageLabel
    return color_array


if __name__ == "__main__":
    img = open_image('92AV3C.lan').load()
    gt = open_image('92AV3GT.GIS').read_band(0)

    original = apply_spy_colormap(gt)

    classes = create_training_classes(img, gt)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(img)
    gtresults = clmap * (gt != 0)
    to_compare = apply_spy_colormap(gtresults)

    app = QApplication(sys.argv)

    window = MainWindow(to_compare, original)
    window.show()
    sys.exit(app.exec())
