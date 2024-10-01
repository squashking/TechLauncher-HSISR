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



class UnsupervisedClassificationWorker(QThread):
    # Modify the signal to emit cluster_map and ndvi
    classification_finished = pyqtSignal(QPixmap, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, hsi_data, wavelengths, k, max_iterations):
        super().__init__()
        self.hsi_data = hsi_data
        self.wavelengths = wavelengths
        self.k = k
        self.max_iterations = max_iterations

    def run(self):
        try:
            # Perform classification
            cluster_map, ndvi = load_and_process_hsi_data(
                self.hsi_data, self.wavelengths, self.k, self.max_iterations
            )

            # Map clusters to colors
            from matplotlib import pyplot as plt
            num_clusters = np.max(cluster_map) + 1
            norm_cluster_map = cluster_map / (num_clusters - 1)
            cluster_image_color = plt.cm.nipy_spectral(norm_cluster_map)
            cluster_image_color = cluster_image_color[:, :, :3]  # Keep as float array in [0,1]

            # Convert to QPixmap
            image_uint8 = (cluster_image_color * 255).astype(np.uint8)
            height, width, _ = image_uint8.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                image_uint8.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)

            # Emit the pixmap, cluster_map, ndvi, and cluster_image_color
            self.classification_finished.emit(pixmap, cluster_map, ndvi, cluster_image_color)

        except Exception as e:
            error_message = str(e)
            self.error_occurred.emit(error_message)