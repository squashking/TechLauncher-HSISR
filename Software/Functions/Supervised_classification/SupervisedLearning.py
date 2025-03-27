#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:45:39 2024

@author: benhe
"""

from spectral import *

import sys
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt


# Define the ImageLabel class to handle QImage conversion
class ImageLabel(QLabel):
    def __init__(self, image_array):
        super().__init__()
        self.image_array = image_array
        height, width, channels  = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(self.pixmap)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        x = int(event.position().x())
        y = int(event.position().y())
        if 0 <= x < self.image_array.shape[1] and 0 <= y < self.image_array.shape[0]:
            target = np.array(self.image_array[y, x])
            class_index = int(np.where(np.all(spy_colors == target, axis=1))[0][0])
            self.parent().color_label.setText(f'Class Index: {class_index}')
        else:
            self.parent().color_label.setText('Out of bounds')


# Define the MainWindow class to manage the UI
class MainWindow(QWidget):
    def __init__(self, image_array):
        super().__init__()
        self.setWindowTitle('Class Index Display')

        self.image_label = ImageLabel(image_array)
        self.color_label = QLabel('Move mouse over image to see class index')

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.color_label)

        self.setLayout(layout)


# Define a function to apply the spy colormap and return class indices (no QImage conversion yet)
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


def main(image_array):
    app = QApplication(sys.argv)
    window = MainWindow(image_array)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':

    img = open_image('92AV3C.lan').load()
    gt = open_image('92AV3GT.GIS').read_band(0)

    # Check the properties of the hyperspectral image
    # print(f"Image shape: {img.shape}")
    # print(f"Image data range: min={img.min()}, max={img.max()}")

    # Check the properties of the ground truth image
    # print(f"Ground Truth shape: {gt.shape}")
    # print(f"Ground Truth data range: min={gt.min()}, max={gt.max()}")
    
    classes = create_training_classes(img, gt)
    
    gmlc = GaussianClassifier(classes)
    
    clmap = gmlc.classify_image(img)
    
    gtresults = clmap * (gt != 0)
    something = apply_spy_colormap(gtresults)
    main(something)

