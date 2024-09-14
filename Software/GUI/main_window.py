import sys
import numpy as np
import os
import spectral.io.envi as envi
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QStackedWidget, QRadioButton, QLabel, 
                             QLineEdit, QHBoxLayout, QProgressBar, QGroupBox, 
                             QFormLayout, QComboBox, QFrame, QSizePolicy, QFileDialog, QMenuBar)
from PyQt6.QtGui import QFont, QPixmap, QAction, QImage
from PyQt6.QtCore import Qt
from hyperspectral_classifier import HyperspectralClassifier 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Visualization.Visualize_HSI import show_rgb, show_ndvi, show_evi, show_mcari, show_mtvi, show_osavi, show_pri, load_image

def find_rgb_bands(wavelengths):
    R_wavelength = 682.5  # Red wavelength
    G_wavelength = 532.5  # Green wavelength
    B_wavelength = 472.5  # Blue wavelength
    
    r_band = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - R_wavelength))
    g_band = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - G_wavelength))
    b_band = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - B_wavelength))
    
    return r_band, g_band, b_band

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.loaded_image = None  # To store the loaded image for Vis Window
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 1024, 768)

        # Initialize the classifier
        self.classifier = HyperspectralClassifier()
        
        # Create menu bar
        self.create_menu_bar()

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layouts
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        left_frame = QFrame()
        left_frame.setFixedWidth(200)
        left_frame.setStyleSheet("""
            background-color: #f0f0f0;
            border-right: 1px solid #d0d0d0;
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_frame)
        main_layout.addLayout(right_layout)
        
        # Sidebar buttons
        sidebar_buttons = ["Visualization", "Super-resolution", "Calibration", "Classification"]
        for i, button_text in enumerate(sidebar_buttons):
            btn = QPushButton(button_text)
            btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: none;
                    border-bottom: 1px solid #d0d0d0;
                    text-align: center;
                    padding: 15px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
            """)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            left_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, text=button_text: self.change_page(text))
        
        # This will make the buttons fill the available space
        for i in range(len(sidebar_buttons)):
            left_layout.setStretch(i, 1)
        
        # Stack for right layout
        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack)
        
        # Create pages
        self.create_visualization_page()
        self.create_super_resolution_page()
        self.create_calibration_page()
        self.create_classification_page()

    def create_menu_bar(self):
        """Create a menu bar with 'File' and 'About' options."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("File")

        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        # About menu
        about_menu = menu_bar.addMenu("About")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

    def show_about_dialog(self):
        """Display an 'About' dialog."""
        about_dialog = QLabel("Hyperspectral Image Classification GUI\nVersion 1.0", alignment=Qt.AlignmentFlag.AlignCenter)
        about_dialog.show()

    def load_image(self):
        """Loads the hyperspectral image and displays its RGB composite."""
        # Open file dialog to select the hyperspectral image file
        print("Opening file dialog to select .bil file...")
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open file', None, "Hyperspectral Images (*.bil *.bip *.bsq)")
        
        if not image_path:
            print("No file selected or invalid file.")
            return

        print(f"Selected image file: {image_path}")
        
        # Get corresponding header file
        header_path = image_path.replace('.bil', '.hdr')
        
        if not os.path.exists(header_path):
            print(f"Header file not found at: {header_path}")
            return

        print(f"Header file located at: {header_path}")
        
        # Load hyperspectral image using spectral library
        self.hsi = envi.open(header_path, image_path)
        print(f"Hyperspectral image loaded successfully from {image_path}")
        
        # Extract the wavelengths
        wavelengths = [float(w) for w in self.hsi.metadata['wavelength']]
        print(f"Extracted wavelengths: {wavelengths[:10]} ...")
        
        # Find RGB bands
        r_band, g_band, b_band = find_rgb_bands(wavelengths)
        print(f"RGB bands located at: R={r_band}, G={g_band}, B={b_band}")
        
        # Extract the RGB bands
        img_r = self.hsi.read_band(r_band)
        img_g = self.hsi.read_band(g_band)
        img_b = self.hsi.read_band(b_band)
        
        # Stack into a single RGB image and normalize it
        img_rgb = np.dstack((img_r, img_g, img_b))
        img_rgb = (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb))  # Normalize to [0, 1]
        img_rgb = (img_rgb * 255).astype(np.uint8)  # Scale to 8-bit [0, 255]
        
        print(f"Image stacked and normalized: shape={img_rgb.shape}")
        
        # Convert to QImage and store the result in self.loaded_image for both visualization and classification
        height, width, _ = img_rgb.shape
        bytes_per_line = 3 * width
        qimage = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qimage)
        
        # Store the loaded RGB image for later use in both classification and visualization
        self.loaded_image = pixmap

        # Display the image in classification tab (if you're in the classification tab)
        self.image_label.setPixmap(self.loaded_image)
        self.image_label.setScaledContents(True)
        print("RGB Image loaded and displayed in classification tab.")
        
        # Set the file path in the visualization input for further use
        self.file_input.setText(image_path)


    def save_image(self):
        """Save the result image."""
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
        if save_path:
            result_image_path = "result/gtresults.png"  # Assuming the result image is saved with this name
            pixmap = QPixmap(result_image_path)
            pixmap.save(save_path)


    def update_visualization(self, file_path):
        """Update the visualization page with the loaded image."""
        # Load the image
        img_data = self.classifier.hsi.load()
        if img_data is not None:
            img_rgb = img_data[:, :, :3]  # Assuming 3 bands for RGB
            img_rgb = (img_rgb * 255 / img_rgb.max()).astype(np.uint8)  # Scale image to 8-bit format
            
            # Convert the numpy image to QPixmap
            height, width, channels = img_rgb.shape
            bytes_per_line = channels * width
            qimage = QImage(img_rgb.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            # Update the label in the visualization area
            self.visualization_label.setPixmap(pixmap)
        else:
            self.visualization_label.setText("Error: Unable to load the image")
            
    def visualize_selected_mode(self):
        image_path = self.file_input.text()
        header_path = image_path.replace(".bil", ".hdr")

        print(f"Image Path: {image_path}")
        print(f"Header Path: {header_path}")

        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            self.visualization_label.setText("Error: Image file not found")
            return

        if not os.path.exists(header_path):
            print(f"Error: Header file not found at {header_path}")
            self.visualization_label.setText("Error: Header file not found")
            return

        try:
            # Load image using the more comprehensive Visualize_HSI load_image function
            hsi = load_image(image_path, header_path)

            # Based on the selected mode, call the appropriate function
            if self.radio_rgb.isChecked():
                show_rgb(hsi, "visualization_rgb.png")
                pixmap = QPixmap("visualization_rgb.png")
            elif self.radio_ndvi.isChecked():
                show_ndvi(hsi, "visualization_ndvi.png")
                pixmap = QPixmap("visualization_ndvi.png")
            elif self.radio_evi.isChecked():
                show_evi(hsi, "visualization_evi.png")
                pixmap = QPixmap("visualization_evi.png")
            elif self.radio_mcari.isChecked():
                show_mcari(hsi, "visualization_mcari.png")
                pixmap = QPixmap("visualization_mcari.png")
            elif self.radio_mtvi.isChecked():
                show_mtvi(hsi, "visualization_mtvi.png")
                pixmap = QPixmap("visualization_mtvi.png")
            elif self.radio_osavi.isChecked():
                show_osavi(hsi, "visualization_osavi.png")
                pixmap = QPixmap("visualization_osavi.png")
            elif self.radio_pri.isChecked():
                show_pri(hsi, "visualization_pri.png")
                pixmap = QPixmap("visualization_pri.png")
            else:
                self.visualization_label.setText("Error: No mode selected")
                return

            # Display the image in the visualization label
            self.visualization_label.setPixmap(pixmap)
            self.visualization_label.setScaledContents(True)

        except Exception as e:
            self.visualization_label.setText(f"Error: {str(e)}")


    def create_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Visualization image label (takes up most of the space)
        self.visualization_label = QLabel("Visualization Content", alignment=Qt.AlignmentFlag.AlignCenter)
        self.visualization_label.setFixedHeight(400)  # Set an appropriate height or let it scale with content
        layout.addWidget(self.visualization_label)

        # Add a spacer to push the banner to the bottom
        layout.addStretch(1)
        
        # Mode selection banner
        file_path_label = QLabel("File path:")
        self.file_input = QLineEdit("Path/to/actual/image")

        # Visualization Mode Options
        mode_group = QGroupBox("Mode:")
        mode_layout = QHBoxLayout()

        # Add radio buttons for each visualization mode
        self.radio_rgb = QRadioButton("RGB")
        self.radio_ndvi = QRadioButton("NDVI")
        self.radio_evi = QRadioButton("EVI")
        self.radio_mcari = QRadioButton("MCARI")
        self.radio_mtvi = QRadioButton("MTVI")
        self.radio_osavi = QRadioButton("OSAVI")
        self.radio_pri = QRadioButton("PRI")

        mode_layout.addWidget(self.radio_rgb)
        mode_layout.addWidget(self.radio_ndvi)
        mode_layout.addWidget(self.radio_evi)
        mode_layout.addWidget(self.radio_mcari)
        mode_layout.addWidget(self.radio_mtvi)
        mode_layout.addWidget(self.radio_osavi)
        mode_layout.addWidget(self.radio_pri)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Button to trigger visualization
        visualize_button = QPushButton("Visualize")
        visualize_button.clicked.connect(self.visualize_selected_mode)
        layout.addWidget(visualize_button)

        # Add the page to the stack
        self.stack.addWidget(page)



    def create_super_resolution_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # File path input
        file_layout = QHBoxLayout()
        file_label = QLabel("File path:")
        self.file_input = QLineEdit("Path/to/actual/image")
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input)
        layout.addLayout(file_layout)
        
        # Super resolution options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.radio_super_res = QRadioButton("Super Resolution")
        self.radio_low_res = QRadioButton("Low Res")
        self.radio_high_res = QRadioButton("High Res")
        options_layout.addWidget(self.radio_super_res)
        options_layout.addWidget(self.radio_low_res)
        options_layout.addWidget(self.radio_high_res)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(24)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch(1)
        self.stack.addWidget(page)

    def create_calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        form = QFormLayout()
        dark_file_input = QLineEdit()
        ref_file_input = QLineEdit()
        form.addRow("Dark File:", dark_file_input)
        form.addRow("Reference File:", ref_file_input)
        calibrate_btn = QPushButton("Calibrate")
        form.addRow(calibrate_btn)
        
        layout.addLayout(form)
        layout.addStretch(1)
        self.stack.addWidget(page)

    def create_classification_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # File path input
        file_layout = QHBoxLayout()
        file_label = QLabel("File path:")
        self.file_input_class = QLineEdit("One_sample/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.bil")
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_input_class)
        layout.addLayout(file_layout)
        
        # Groundtruth input
        groundtruth_layout = QHBoxLayout()
        groundtruth_label = QLabel("Groundtruth:")
        self.groundtruth_input = QLineEdit("One_sample/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1_mask.jpg")
        groundtruth_layout.addWidget(groundtruth_label)
        groundtruth_layout.addWidget(self.groundtruth_input)
        layout.addLayout(groundtruth_layout)
        
        # Classifier selection
        classifier_layout = QHBoxLayout()
        classifier_label = QLabel("Classifier:")
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItem("GaussianClassifier")
        classifier_layout.addWidget(classifier_label)
        classifier_layout.addWidget(self.classifier_combo)
        layout.addLayout(classifier_layout)
        
        # Classify button
        classify_button = QPushButton("Classify")
        classify_button.clicked.connect(self.run_classification)
        layout.addWidget(classify_button)

        # Label to display classified image
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        
        layout.addStretch(1)
        self.stack.addWidget(page)

    def run_classification(self):
        """Load and classify the image, display in classification tab."""
        image_path = self.file_input_class.text()
        groundtruth_path = self.groundtruth_input.text()

        header_path = image_path.replace('.bil', '.hdr')

        # Load and classify the image
        self.classifier.load_image(image_path, header_path)
        result_image_path = self.classifier.classify(groundtruth_path)

        # Load the classified image as QPixmap
        pixmap = QPixmap(result_image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        # Store the loaded image in self.loaded_image for visualization tab
        self.loaded_image = pixmap

    def update_visualization_tab(self):
        """Update the visualization tab with the loaded RGB image from the .bil file."""
        if self.loaded_image:  # If an image was loaded
            self.visualization_label.setPixmap(self.loaded_image)
            self.visualization_label.setScaledContents(True)
        else:
            self.visualization_label.setText("No image loaded")


    def change_page(self, button_text):
        """Switch between pages and update the visualization tab if necessary."""
        index = ["Visualization", "Super-resolution", "Calibration", "Classification"].index(button_text)
        self.stack.setCurrentIndex(index)

        if button_text == "Visualization":
            self.update_visualization_tab()  # Update the visualization tab with the loaded image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
