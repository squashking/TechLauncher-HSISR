import sys
import numpy as np
from matplotlib import pyplot as plt
from spectral import envi, kmeans
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

# Function to calculate NDVI
def calculate_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    denominator = nir + red
    denominator[denominator == 0] = np.nan  # Avoid division by zero
    ndvi = (nir - red) / denominator
    return np.nan_to_num(ndvi, nan=0.0)  # Replace NaN with 0.0

# Function to find Red and NIR bands
def find_Red_NIR_bands(listWavelength):
    R_wavelength = 682.5  # Red (625+740)/2
    NIR_wavelength = 850  # NIR

    rFound = nirFound = False
    rPreDifference = nirPreDifference = float('inf')  # previously calculated difference
    rIndex = nirIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:
                rPreDifference = difference
            else:
                rIndex = i - 1
                rFound = True

        if not nirFound:
            difference = abs(value - NIR_wavelength)
            if difference < nirPreDifference:
                nirPreDifference = difference
            else:
                nirIndex = i - 1
                nirFound = True

    return (rIndex, nirIndex)

# Load HSI data
hdr_file = '2023-06-08--17-28-29_round-0_cam-1_tray-Tray_1_envi.hdr'
bil_file = '2023-06-08--17-28-29_round-0_cam-1_tray-Tray_1.bil'
img = envi.open(hdr_file, image=bil_file)
data = img.load()

# Perform K-Means clustering on the HSI data
k = 20  # Number of clusters
max_iterations = 15  # Maximum iterations
(m, c) = kmeans(data, k, max_iterations)
cluster_map = m.reshape(data.shape[:-1])  # Reshape to 2D map (rows, cols)

# Extract wavelength data from the metadata
wavelengths = [float(w) for w in img.metadata['wavelength']]

# Find Red and NIR bands
red_band_index, nir_band_index = find_Red_NIR_bands(wavelengths)
red_band = data[:, :, red_band_index]
nir_band = data[:, :, nir_band_index]

# Calculate NDVI for the entire image
ndvi = calculate_ndvi(nir_band, red_band)

class NDVIViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("NDVI Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 800, 600)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.display_image()

    def display_image(self):
        # Apply color map to the cluster map
        cluster_image_color = plt.cm.nipy_spectral(cluster_map / np.max(cluster_map))
        cluster_image_color = (cluster_image_color[:, :, :3] * 255).astype(np.uint8)

        height, width, _ = cluster_image_color.shape
        bytes_per_line = 3 * width
        q_image = QImage(cluster_image_color.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.image_label.setPixmap(pixmap)
        self.image_label.mousePressEvent = self.get_ndvi

    def get_ndvi(self, event):
        try:
            # Get mouse click position
            x = event.pos().x()
            y = event.pos().y()

            # Ensure the click is within image bounds
            if x >= cluster_map.shape[1] or y >= cluster_map.shape[0]:
                return

            # Get the cluster ID at the clicked position
            cluster_id = cluster_map[y, x]

            # Mask the NDVI values for the selected cluster
            ndvi_cluster = np.ma.masked_where(cluster_map != cluster_id, ndvi)

            # Calculate the mean NDVI for the selected region
            mean_ndvi = np.mean(ndvi_cluster)

            # Display the NDVI value
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
