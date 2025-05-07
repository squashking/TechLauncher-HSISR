# unsupervised_worker.py
import logging
import os
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import numpy as np

import platform


if platform.system() == 'Windows':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Software')))
else:
    # On macOS or other Unix-like systems, keep the original path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Unsupervised_classification.unsupervised_classification import load_and_process_hsi_data
from Functions.Unsupervised_classification.unsupervised_classification import find_Red_NIR_bands
from Functions.Visualization.Visualize_HSI import calculate_ndvi


class UnsupervisedClassificationWorker(QThread):
    classification_finished = pyqtSignal(QPixmap, np.ndarray, np.ndarray, np.ndarray)
    error_occurred        = pyqtSignal(str)

    def __init__(self, hsi_data, wavelengths, k, max_iterations):
        super().__init__()
        self.hsi_data      = hsi_data
        self.wavelengths   = wavelengths
        self.k             = k
        self.max_iterations = max_iterations

    def run(self):
        try:
            # 1) Perform clustering and NDVI via the existing helper
            cluster_map, ndvi = load_and_process_hsi_data(
                self.hsi_data,
                self.wavelengths,
                k=self.k,
                max_iterations=self.max_iterations
            )

            # 2) Generate color image
            num_clusters = cluster_map.max() + 1
            normed = cluster_map / (num_clusters - 1)
            from matplotlib import pyplot as plt
            color_img = plt.cm.nipy_spectral(normed)[:, :, :3]

            # 3) Convert to QPixmap (with deep-copy)
            img8 = (color_img * 255).astype(np.uint8)
            h, w, _ = img8.shape
            qimg = QImage(img8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            # 4) Emit results
            self.classification_finished.emit(pix, cluster_map, ndvi, color_img)

        except Exception as e:
            self.error_occurred.emit(str(e))
