import sys

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QLineEdit, QHBoxLayout, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from matplotlib import pyplot as plt

# Assuming ndvi_calculator.py has the necessary processing functions
from ndvi_calculator import load_and_process_hsi_data

class NDVIViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NDVI Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(200, 100, 400, 400)

        # Layout for buttons and input fields
        layout = QVBoxLayout()

        # K and Maxitr input
        self.k_input = QSpinBox()
        self.k_input.setMinimum(1)
        self.k_input.setValue(20)
        self.k_input.setPrefix("K = ")

        self.max_itr_input = QSpinBox()
        self.max_itr_input.setMinimum(1)
        self.max_itr_input.setValue(15)
        self.max_itr_input.setPrefix("Max iteration = ")

        layout.addWidget(self.k_input)
        layout.addWidget(self.max_itr_input)

        # Load Image Button
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        # Save Image Button
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        layout.addWidget(save_button)

        # Display the image
        layout.addWidget(self.image_label)

        # Container widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_image = None  # To store the current processed image
        self.cluster_map = None
        self.ndvi = None

    def load_image(self):
        hdr_file, _ = QFileDialog.getOpenFileName(self, "Select HDR File", "", "HDR Files (*.hdr)")
        bil_file, _ = QFileDialog.getOpenFileName(self, "Select BIL File", "", "BIL Files (*.bil)")

        if hdr_file and bil_file:
            k = self.k_input.value()
            max_iterations = self.max_itr_input.value()
            self.cluster_map, self.ndvi, img = load_and_process_hsi_data(hdr_file, bil_file, k, max_iterations)
            self.display_image()

    def display_image(self):
        # Apply color map to the cluster map
        cluster_image_color = plt.cm.nipy_spectral(self.cluster_map / np.max(self.cluster_map))
        cluster_image_color = (cluster_image_color[:, :, :3] * 255).astype(np.uint8)

        height, width, _ = cluster_image_color.shape
        bytes_per_line = 3 * width
        q_image = QImage(cluster_image_color.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.current_image = pixmap  # Store current image for saving

    def save_image(self):
        if self.current_image:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if save_path:
                self.current_image.save(save_path)

    def get_ndvi(self, event):
        try:
            x = event.pos().x()
            y = event.pos().y()

            if x >= self.cluster_map.shape[1] or y >= self.cluster_map.shape[0]:
                return

            cluster_id = self.cluster_map[y, x]
            ndvi_cluster = np.ma.masked_where(self.cluster_map != cluster_id, self.ndvi)
            mean_ndvi = np.mean(ndvi_cluster)

            self.setWindowTitle(f"NDVI Value for Selected Region: {mean_ndvi:.2f}")

        except Exception as e:
            print(f"Error: {e}")

def main():
    app = QApplication(sys.argv)
    viewer = NDVIViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
