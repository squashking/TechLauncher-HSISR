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





class ImageLabel(QLabel):
    def __init__(self, image_array):
        super().__init__()
        self.image_array = image_array
        height, width = image_array.shape
        qimage = QImage(image_array.data, width, height, width, QImage.Format.Format_Indexed8)
        self.pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(self.pixmap)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        x = int(event.position().x())
        y = int(event.position().y())
        if 0 <= x < self.image_array.shape[1] and 0 <= y < self.image_array.shape[0]:
            class_index = self.image_array[y, x]
            self.parent().color_label.setText(f'Class Index: {class_index}')
        else:
            self.parent().color_label.setText('Out of bounds')

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

def main(image_array):
    app = QApplication(sys.argv)
    window = MainWindow(image_array)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':

    img = open_image('92AV3C.lan').load()
    gt = open_image('92AV3GT.GIS').read_band(0)
    
    classes = create_training_classes(img, gt)
    
    gmlc = GaussianClassifier(classes)
    
    clmap = gmlc.classify_image(img)
    
    gtresults = clmap * (gt != 0)
    main(gtresults)

