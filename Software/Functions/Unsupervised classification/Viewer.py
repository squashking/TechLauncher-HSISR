import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QRectF

from ndvi_calculator import load_and_process_hsi_data

class NDVIViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NDVI Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Setup Graphics View for image display
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setGeometry(200, 100, 400, 400)

        # Setup Graphics Scene for managing items in the view
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Label to display the mean NDVI value
        self.ndvi_label = QLabel("Mean NDVI: N/A", self)
        self.ndvi_label.setGeometry(200, 520, 400, 20)

        # Layout for buttons and input fields
        layout = QVBoxLayout()

        self.k_input = QSpinBox()
        self.k_input.setMinimum(1)
        self.k_input.setValue(15)
        self.k_input.setPrefix("K = ")

        self.maxitr_input = QSpinBox()
        self.maxitr_input.setMinimum(1)
        self.maxitr_input.setValue(15)
        self.maxitr_input.setPrefix("Maxitr = ")

        layout.addWidget(self.k_input)
        layout.addWidget(self.maxitr_input)

        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        layout.addWidget(save_button)

        layout.addWidget(self.graphics_view)
        layout.addWidget(self.ndvi_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_image = None  # Store the current processed image
        self.cluster_map = None
        self.ndvi = None
        self.highlight_item = None  # Store the current highlighted item
        self.cluster_image_color = None  # Store the color-mapped cluster image

    def load_image(self):
        try:
            hdr_file, _ = QFileDialog.getOpenFileName(self, "Select HDR File", "", "HDR Files (*.hdr)")
            bil_file, _ = QFileDialog.getOpenFileName(self, "Select BIL File", "", "BIL Files (*.bil)")

            if hdr_file and bil_file:
                k = self.k_input.value()
                max_iterations = self.maxitr_input.value()

                # Safeguard the K-means processing with a try-except
                try:
                    self.cluster_map, self.ndvi, img = load_and_process_hsi_data(hdr_file, bil_file, k, max_iterations)
                    if self.cluster_map is None or self.ndvi is None:
                        print("Error: Failed to process the HSI data.")
                        return
                    self.display_image()
                except Exception as e:
                    print(f"Error during K-means processing: {e}")
                    return

        except Exception as e:
            print(f"Error loading image: {e}")

    def display_image(self):
        try:
            # Apply color mapping to the cluster map
            self.cluster_image_color = plt.cm.nipy_spectral(self.cluster_map / np.max(self.cluster_map))
            self.cluster_image_color = (self.cluster_image_color[:, :, :3] * 255).astype(np.uint8)

            height, width, _ = self.cluster_image_color.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.cluster_image_color.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Clear the scene and add the pixmap
            self.scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(pixmap_item)
            self.graphics_view.setSceneRect(QRectF(pixmap.rect()))  # Convert QRect to QRectF

            # Set current image for saving
            self.current_image = pixmap
            self.highlight_item = None

            # Connect mouse press event for clicking regions
            self.graphics_view.mousePressEvent = self.get_ndvi
        except Exception as e:
            print(f"Error displaying image: {e}")

    def get_ndvi(self, event):
        try:
            # Convert click position from view to scene coordinates
            scene_pos = self.graphics_view.mapToScene(event.position().toPoint())
            x = int(scene_pos.x())
            y = int(scene_pos.y())

            if x < 0 or y < 0 or x >= self.cluster_map.shape[1] or y >= self.cluster_map.shape[0]:
                return

            cluster_id = self.cluster_map[y, x]

            if self.ndvi.shape[-1] == 1:
                ndvi_flattened = self.ndvi[:, :, 0]
            else:
                ndvi_flattened = self.ndvi

            ndvi_cluster = np.ma.masked_where(self.cluster_map != cluster_id, ndvi_flattened)
            mean_ndvi = np.mean(ndvi_cluster)

            self.ndvi_label.setText(f"Mean NDVI: {mean_ndvi:.4f}")

            # Create a mask where the selected region is gray
            mask = np.zeros_like(self.cluster_image_color, dtype=np.uint8)
            mask[self.cluster_map == cluster_id] = [128, 128, 128]  # Apply gray to the selected region

            # Combine the mask with the original image
            overlay = np.where(mask == [128, 128, 128], mask, self.cluster_image_color)

            # Convert overlay image to QImage and display it
            height, width, _ = overlay.shape
            bytes_per_line = 3 * width
            q_image_overlay = QImage(overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap_overlay = QPixmap.fromImage(q_image_overlay)

            if self.highlight_item:
                self.scene.removeItem(self.highlight_item)

            self.highlight_item = QGraphicsPixmapItem(pixmap_overlay)
            self.scene.addItem(self.highlight_item)

        except Exception as e:
            print(f"Error during NDVI calculation or display: {e}")

    def save_image(self):
        try:
            if self.current_image:
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
                if save_path:
                    self.current_image.save(save_path)
        except Exception as e:
            print(f"Error saving image: {e}")

def main():
    try:
        app = QApplication(sys.argv)
        viewer = NDVIViewer()
        viewer.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Unhandled exception: {e}")

if __name__ == "__main__":
    main()
