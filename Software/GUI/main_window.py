import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QPushButton, QStackedWidget, QRadioButton, QLabel,
                             QLineEdit, QHBoxLayout, QProgressBar, QGroupBox,
                             QFormLayout, QComboBox, QFrame, QSizePolicy, QFileDialog, QMenuBar, QSpinBox)
from PyQt6.QtGui import QFont, QPixmap, QAction, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from hyperspectral_classifier import HyperspectralClassifier 
import matplotlib.pyplot as plt
import shutil
import time
import threading
import spectral.io.envi as envi
from spectral import get_rgb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Software.Functions.Basic_Functions.Load_HSI import load_hsi
from Software.Functions.Visualization.Visualize_HSI import find_RGB_bands, show_rgb, show_ndvi, show_evi, show_mcari, show_mtvi, show_osavi, show_pri
from Software.Functions.Super_resolution.Run_Super_Resolution import run_super_resolution
from Software.Functions.Calibration.calibrate import calibration
from Software.Functions.Hypercube_Spectrum.Hypercube import show_cube
from unsupervised_worker import UnsupervisedClassificationWorker
from Software.Functions.Unsupervised_classification.unsupervised_classification import load_and_process_hsi_data


class ClickableImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loaded_image = None
        self.mode = "RGB"  # Default mode

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Trigger plot when the image is clicked
            self.show_plot()

    def show_plot(self):
        # This function will trigger the plt.show()
        if self.loaded_image is not None:
            if self.mode == "RGB":
                plt.imshow(self.loaded_image)
                plt.title("RGB Image")
            elif self.mode == "NDVI":
                plt.imshow(self.ndvi_image)
                plt.title("NDVI Image")
            plt.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.loaded_image = None  # To store the loaded image for Vis Window
        self.image_path = ""
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 1024, 768)

        # Initialize the classifier
        self.classifier = HyperspectralClassifier()
        
        # Create menu bar
        self.create_menu_bar()

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts (create these first!)
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

        # Clickable image label (after defining main_layout)
        self.image_label = ClickableImage(self)  # Using the custom ClickableImage class
        right_layout.addWidget(self.image_label)  # Add it to right_layout instead of main_layout

        # Mode selection and visualization controls
        self.mode = "RGB"
        
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
        self.image_path = image_path
        print(f"Selected image file: {self.image_path}")

        # Get corresponding header file
        header_path = image_path.replace('.bil', '.hdr')
        if not os.path.exists(header_path):
            print(f"Header file not found at: {header_path}")
            return
        print(f"Header file located at: {header_path}")
        # Load hyperspectral image using spectral library
        self.hsi = load_hsi(image_path,header_path)
        print(f"Hyperspectral image loaded successfully from {image_path}")
        
        # Convert to QImage and store the result in self.loaded_image for both visualization and classification
        height, width, _ = self.hsi.shape
        empty_image = QImage(width, height, QImage.Format.Format_RGB888)  # 交换 width 和 height 的位置
        empty_image.fill(Qt.GlobalColor.white)  # 填充白色
        pixmap = QPixmap.fromImage(empty_image)
        
        # # # Store the loaded RGB image for later use in both classification and visualization
        self.loaded_image = pixmap

    def save_image(self):
        """Save the currently visualized image."""
        if self.loaded_image is not None:
            # Open a file dialog to choose the save location
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
            if save_path:
                self.loaded_image.save(save_path)  # Save the image
                print(f"Image saved to {save_path}")
        else:
            print("No image to save.")
            
    def visualize_selected_mode(self):
        # Create a dictionary mapping modes to their corresponding functions and output file names
        mode_mapping = {
            "RGB": ("img/visualization_rgb.png", show_rgb),
            "NDVI": ("img/visualization_ndvi.png", show_ndvi),
            "EVI": ("img/visualization_evi.png", show_evi),
            "MCARI": ("img/visualization_mcari.png", show_mcari),
            "MTVI": ("img/visualization_mtvi.png", show_mtvi),
            "OSAVI": ("img/visualization_osavi.png", show_osavi),
            "PRI": ("img/visualization_pri.png", show_pri),
            "hypercube": ("img/visualization_cube.png", show_cube)
        }

        # Determine which radio button is checked and select the appropriate function and file name
        selected_mode = None
        if self.radio_rgb.isChecked():
            selected_mode = "RGB"
        elif self.radio_ndvi.isChecked():
            selected_mode = "NDVI"
        elif self.radio_evi.isChecked():
            selected_mode = "EVI"
        elif self.radio_mcari.isChecked():
            selected_mode = "MCARI"
        elif self.radio_mtvi.isChecked():
            selected_mode = "MTVI"
        elif self.radio_osavi.isChecked():
            selected_mode = "OSAVI"
        elif self.radio_pri.isChecked():
            selected_mode = "PRI"
        elif self.radio_cube.isChecked():
            selected_mode = "hypercube"
        
        if selected_mode is None:
            self.visualization_label.setText("Error: No mode selected")
            return

        # Get the file path and function for the selected mode
        save_path, visualization_function = mode_mapping[selected_mode]

        # Call the corresponding visualization function
        try:
            visualization_function(self.hsi, save_path)
            pixmap = QPixmap(save_path)
            self.visualization_label.setPixmap(pixmap)
            self.visualization_label.setScaledContents(True)
        except Exception as e:
            self.visualization_label.setText(f"Error visualizing {selected_mode}: {str(e)}")

    def create_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Visualization image label (takes up most of the space)
        self.visualization_label = QLabel("Visualization Content", alignment=Qt.AlignmentFlag.AlignCenter)
        self.visualization_label.setFixedHeight(525)  # Set an appropriate height or let it scale with content
        self.visualization_label.setFixedWidth(700)
        layout.addWidget(self.visualization_label)

        # Add a spacer to push the banner to the bottom
        layout.addStretch(1)
        
        # Mode selection banner
        file_path_label = QLabel("File path:")
        self.file_input = QLineEdit("Path/to/actual/image")

        # Visualization Mode Options
        mode_group = QGroupBox("Mode:")
        mode_layout_top = QHBoxLayout()
        mode_layout_bottom = QHBoxLayout()

        # Add radio buttons for each visualization mode
        self.radio_rgb = QRadioButton("RGB")
        self.radio_ndvi = QRadioButton("NDVI")
        self.radio_evi = QRadioButton("EVI")
        self.radio_mcari = QRadioButton("MCARI")
        self.radio_mtvi = QRadioButton("MTVI")
        self.radio_osavi = QRadioButton("OSAVI")
        self.radio_pri = QRadioButton("PRI")
        self.radio_cube = QRadioButton("HyperCube")

        mode_layout_top.addWidget(self.radio_rgb)
        mode_layout_top.addWidget(self.radio_ndvi)
        mode_layout_top.addWidget(self.radio_evi)
        mode_layout_top.addWidget(self.radio_mcari)
        mode_layout_bottom.addWidget(self.radio_mtvi)
        mode_layout_bottom.addWidget(self.radio_osavi)
        mode_layout_bottom.addWidget(self.radio_pri)
        mode_layout_bottom.addWidget(self.radio_cube)

        mode_layout_vertical = QVBoxLayout()
        mode_layout_vertical.addLayout(mode_layout_top)
        mode_layout_vertical.addLayout(mode_layout_bottom)
        mode_group.setLayout(mode_layout_vertical)

        layout.addWidget(mode_group)

        # Button to trigger visualization
        visualize_button = QPushButton("Visualize")
        visualize_button.clicked.connect(self.visualize_selected_mode)
        layout.addWidget(visualize_button)

        # Add the page to the stack
        self.stack.addWidget(page)

    def show_resolution_image(self, resolution):
        if resolution == "low":
            if self.hsi is not None:
                print(f"显示{resolution}分辨率图像")
                save_path = f"img/visualization_{resolution}_res.png"
                try:
                    show_rgb(self.hsi, save_path)
                    pixmap = QPixmap(save_path)
                    self.visualization_label_sr.setPixmap(pixmap)
                    self.visualization_label_sr.setScaledContents(True)
                except Exception as e:
                    self.visualization_label_sr.setText(f"显示{resolution}分辨率图像时出错：{str(e)}")
            else:
                self.visualization_label_sr.setText("请先加载图像")
        if resolution == "high":
            image_path = 'temp_sr/result_hsidata/result.bil'
            header_path = 'temp_sr/result_hsidata/result.hdr'
            high_hsi = load_hsi(image_path,header_path)
            if high_hsi is not None:
                print(f"显示{resolution}分辨率图像")
                save_path = f"img/visualization_{resolution}_res.png"
                try:
                    show_rgb(high_hsi, save_path)
                    pixmap = QPixmap(save_path)
                    self.visualization_label_sr.setPixmap(pixmap)
                    self.visualization_label_sr.setScaledContents(True)
                except Exception as e:
                    self.visualization_label_sr.setText(f"显示{resolution}分辨率图像时出错：{str(e)}")

    def handle_super_resolution(self):
        if self.radio_super_res.isChecked():
            print("开始超分辨率处理")
            desired_path = os.path.dirname(self.image_path) + os.path.sep
            temp_folders = ['temp_sr/ori_matdata/', 'temp_sr/mid_matdata/', 'temp_sr/result_matdata/',
                            'temp_sr/result_hsidata/']
            for folder in temp_folders:
                full_path = os.path.join(os.getcwd(), folder)
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                os.makedirs(full_path)

            self.progress_bar.setValue(0)
            self.current_progress = 0
            self.target_progress = 0

            def update_progress(speed=0.05):
                while self.current_progress < self.target_progress:
                    self.current_progress += speed  # 每次增加0.5%
                    self.progress_bar.setValue(int(self.current_progress))
                    time.sleep(0.05)

            def progress_callback(message):
                print(f"进度更新: {message}")
                if message == "完成":
                    self.target_progress = 100
                    threading.Thread(target=update_progress, args=(1,), daemon=True).start()
                else:
                    self.target_progress = min(self.target_progress + 25, 99)
                    threading.Thread(target=update_progress, daemon=True).start()
                QApplication.processEvents()

            run_super_resolution(desired_path, temp_folders[0], temp_folders[1], temp_folders[2], temp_folders[3],
                                 callback=progress_callback)
        else:
            print("取消超分辨率处理")

    def create_super_resolution_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.visualization_label_sr = QLabel("Visualization Content")
        self.visualization_label_sr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label_sr.setFixedHeight(525)  # Set an appropriate height or let it scale with content
        self.visualization_label_sr.setFixedWidth(700)
        layout.addWidget(self.visualization_label_sr)

        # 文件路径输入
        file_layout = QHBoxLayout()
        self.super_resolution_file_label = QLabel("File path: No image loaded")
        file_layout.addWidget(self.super_resolution_file_label)
        layout.addLayout(file_layout)

        # 超分辨率选项
        options_group = QGroupBox()
        options_layout = QHBoxLayout()

        self.radio_super_res = QPushButton("Super Resolution")
        self.radio_super_res.setCheckable(True)
        self.radio_super_res.setChecked(False)
        self.radio_low_res = QRadioButton("Low Res")
        self.radio_high_res = QRadioButton("High Res")
        self.radio_super_res.clicked.connect(self.handle_super_resolution)
        self.radio_low_res.clicked.connect(lambda: self.show_resolution_image("low"))
        self.radio_high_res.clicked.connect(lambda: self.show_resolution_image("high"))

        options_layout.addWidget(self.radio_super_res)
        options_layout.addWidget(self.radio_low_res)
        options_layout.addWidget(self.radio_high_res)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        self.stack.addWidget(page)
        
    def run_calibration(self):
        """Run the calibration process based on user input and display the result."""
        dark_hdr = self.dark_file_hdr_input.text()
        dark_bil = self.dark_file_bil_input.text()
        ref_hdr = self.ref_file_hdr_input.text()
        ref_bil = self.ref_file_bil_input.text()
        threshold = int(self.threshold_input.text())

        if not all([dark_hdr, dark_bil, ref_hdr, ref_bil]):
            print("Error: Please provide all file paths.")
            self.calibration_image_label.setText("Error: Missing file paths")
            return

        try:
            # Use the imported calibration function from Functions.Calibration.demo
            calibration(dark_hdr, dark_bil, ref_hdr, ref_bil, ["average", "demo"], threshold)

            # Load the resulting image
            result_image_path = "result.bil"
            result_header_path = "result.hdr"

            result_hsi = envi.open(result_header_path, result_image_path)
            tuple_rgb_bands = find_RGB_bands([float(i) for i in result_hsi.metadata['wavelength']])
            rgb_image = get_rgb(result_hsi, tuple_rgb_bands)
            rgb_image = (rgb_image * 255).astype(np.uint8)

            # Convert to QImage for display
            height, width, _ = rgb_image.shape
            bytes_per_line = 3 * width
            qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap(qimage)

            # Update the label to display the calibration result
            self.calibration_image_label.setPixmap(pixmap)
            self.calibration_image_label.setScaledContents(True)

        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            self.calibration_image_label.setText(f"Error: {str(e)}")

    def create_calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Dark file input (hdr and bil)
        dark_file_hdr_layout = QHBoxLayout()
        dark_file_hdr_label = QLabel("Dark File HDR:")
        self.dark_file_hdr_input = QLineEdit()
        dark_file_hdr_button = QPushButton("Browse")
        dark_file_hdr_button.clicked.connect(lambda: self.browse_file(self.dark_file_hdr_input))
        dark_file_hdr_layout.addWidget(dark_file_hdr_label)
        dark_file_hdr_layout.addWidget(self.dark_file_hdr_input)
        dark_file_hdr_layout.addWidget(dark_file_hdr_button)
        
        dark_file_bil_layout = QHBoxLayout()
        dark_file_bil_label = QLabel("Dark File BIL:")
        self.dark_file_bil_input = QLineEdit()
        dark_file_bil_button = QPushButton("Browse")
        dark_file_bil_button.clicked.connect(lambda: self.browse_file(self.dark_file_bil_input))
        dark_file_bil_layout.addWidget(dark_file_bil_label)
        dark_file_bil_layout.addWidget(self.dark_file_bil_input)
        dark_file_bil_layout.addWidget(dark_file_bil_button)

        # Reference file input (hdr and bil)
        ref_file_hdr_layout = QHBoxLayout()
        ref_file_hdr_label = QLabel("Reference File HDR:")
        self.ref_file_hdr_input = QLineEdit()
        ref_file_hdr_button = QPushButton("Browse")
        ref_file_hdr_button.clicked.connect(lambda: self.browse_file(self.ref_file_hdr_input))
        ref_file_hdr_layout.addWidget(ref_file_hdr_label)
        ref_file_hdr_layout.addWidget(self.ref_file_hdr_input)
        ref_file_hdr_layout.addWidget(ref_file_hdr_button)

        ref_file_bil_layout = QHBoxLayout()
        ref_file_bil_label = QLabel("Reference File BIL:")
        self.ref_file_bil_input = QLineEdit()
        ref_file_bil_button = QPushButton("Browse")
        ref_file_bil_button.clicked.connect(lambda: self.browse_file(self.ref_file_bil_input))
        ref_file_bil_layout.addWidget(ref_file_bil_label)
        ref_file_bil_layout.addWidget(self.ref_file_bil_input)
        ref_file_bil_layout.addWidget(ref_file_bil_button)

        # Calibration threshold input
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_input = QLineEdit("10")  # Default threshold value
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_input)

        # Calibrate button
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.run_calibration)

        # Layout organization
        layout.addLayout(dark_file_hdr_layout)
        layout.addLayout(dark_file_bil_layout)
        layout.addLayout(ref_file_hdr_layout)
        layout.addLayout(ref_file_bil_layout)
        layout.addLayout(threshold_layout)
        layout.addWidget(calibrate_button)

        # Label to display calibration result image
        self.calibration_image_label = QLabel("Calibration Result", alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.calibration_image_label)

        layout.addStretch(1)
        self.stack.addWidget(page)

    def browse_file(self, line_edit):
        """Helper function to browse and set file paths."""
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', None, "All Files (*.*)")
        if file_path:
            line_edit.setText(file_path)

    def create_classification_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.visualization_label_class = QLabel("Visualization Content")
        self.visualization_label_class.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label_class.setFixedHeight(525)
        self.visualization_label_class.setFixedWidth(700)
        layout.addWidget(self.visualization_label_class)

        self.classification_inputfile_label = QLabel("File path: No image loaded")
        layout.addWidget(self.classification_inputfile_label)

        self.tab_widget = QTabWidget()
        unsupervised_tab = QWidget()
        supervised_tab = QWidget()

        self.tab_widget.addTab(unsupervised_tab, "Unsupervised")
        self.tab_widget.addTab(supervised_tab, "Supervised")

        # 无监督标签页的内容
        unsupervised_layout = QVBoxLayout(unsupervised_tab)

        num_classes_layout = QHBoxLayout()
        num_classes_label = QLabel("Num of Classes:")
        self.num_classes_input = QSpinBox()
        self.num_classes_input.setMinimum(1)
        self.num_classes_input.setValue(5)
        num_classes_layout.addWidget(num_classes_label)
        num_classes_layout.addWidget(self.num_classes_input)
        unsupervised_layout.addLayout(num_classes_layout)

        max_iterations_layout = QHBoxLayout()
        max_iterations_label = QLabel("Max Iterations:")
        self.max_iterations_input = QSpinBox()
        self.max_iterations_input.setMinimum(1)
        self.max_iterations_input.setValue(10)
        max_iterations_layout.addWidget(max_iterations_label)
        max_iterations_layout.addWidget(self.max_iterations_input)
        unsupervised_layout.addLayout(max_iterations_layout)

        unsupervised_classify_button = QPushButton("Classify")
        unsupervised_classify_button.setFixedWidth(100)
        unsupervised_classify_button.clicked.connect(self.run_unsupervised_classification)
        unsupervised_layout.addWidget(unsupervised_classify_button)

        # **在这里创建进度条，并赋值给 self.unsupervised_progress_bar**
        self.unsupervised_progress_bar = QProgressBar()
        self.unsupervised_progress_bar.setValue(0)
        unsupervised_layout.addWidget(self.unsupervised_progress_bar)


        # 有监督标签页的内容
        supervised_layout = QVBoxLayout(supervised_tab)

        groundtruth_layout = QHBoxLayout()
        groundtruth_label = QLabel("Groundtruth:")
        self.groundtruth_input = QLineEdit("One_sample/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1_mask.jpg")
        groundtruth_layout.addWidget(groundtruth_label)
        groundtruth_layout.addWidget(self.groundtruth_input)
        supervised_layout.addLayout(groundtruth_layout)

        classifier_layout = QHBoxLayout()
        classifier_label = QLabel("Classifier:")
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItem("GaussianClassifier")
        classifier_layout.addWidget(classifier_label)
        classifier_layout.addWidget(self.classifier_combo)
        classifier_layout.addStretch()
        supervised_layout.addLayout(classifier_layout)

        supervised_classify_button = QPushButton("Classify")
        supervised_classify_button.setFixedWidth(100)
        supervised_classify_button.clicked.connect(self.run_supervised_classification)
        supervised_layout.addWidget(supervised_classify_button)

        layout.addWidget(self.tab_widget)

        layout.addStretch(1)
        self.stack.addWidget(page)


    def unsupervised_classification_finished(self, pixmap):
        # 更新显示区域，显示分类结果
        self.visualization_label_class.setPixmap(pixmap)
        self.visualization_label_class.setScaledContents(True)

        # 将进度条设置为确定状态并更新为 100%
        self.unsupervised_progress_bar.setRange(0, 100)
        self.unsupervised_progress_bar.setValue(100)


    def unsupervised_classification_error(self, error_message):
        self.visualization_label_class.setText(f"Error: {error_message}")
        # 将进度条设置为确定状态并重置为 0%
        self.unsupervised_progress_bar.setRange(0, 100)
        self.unsupervised_progress_bar.setValue(0)


    def run_unsupervised_classification(self):
        try:
            # 确保已经加载了图像
            if self.hsi is None:
                print("No image loaded.")
                self.visualization_label_class.setText("No image loaded.")
                return

            # 更新显示区域，显示 "Classification in progress..."
            self.visualization_label_class.setText("Classification in progress...")

            # 重置进度条为不确定状态
            self.unsupervised_progress_bar.setRange(0, 0)  # 设置为不确定进度
            self.unsupervised_progress_bar.setValue(0)

            # 获取用户输入的参数
            k = self.num_classes_input.value()
            max_iterations = self.max_iterations_input.value()

            # 获取高光谱数据和波长信息
            hsi_data = self.hsi.load()
            wavelengths = [float(w) for w in self.hsi.metadata['wavelength']]

            # 创建并启动工作线程
            self.unsupervised_worker = UnsupervisedClassificationWorker(
                hsi_data, wavelengths, k, max_iterations
            )

            # 连接信号和槽
            # 删除 progress_updated 信号的连接
            # self.unsupervised_worker.progress_updated.connect(self.update_unsupervised_progress)
            self.unsupervised_worker.classification_finished.connect(self.unsupervised_classification_finished)
            self.unsupervised_worker.error_occurred.connect(self.unsupervised_classification_error)

            # 启动工作线程
            self.unsupervised_worker.start()

        except Exception as e:
            print(f"Error during unsupervised classification: {e}")
            self.visualization_label_class.setText(f"Error: {str(e)}")

    def run_supervised_classification(self):
        """Load and classify the image, display in classification tab."""
        groundtruth_path = self.groundtruth_input.text()
        header_path = self.image_path.replace('.bil', '.hdr')
        # Load and classify the image
        self.classifier.load_image(self.image_path, header_path)
        result_image_path = self.classifier.classify(groundtruth_path)

        # Load the classified image as QPixmap
        pixmap = QPixmap(result_image_path)
        self.visualization_label_class.setPixmap(pixmap)
        self.visualization_label_class.setScaledContents(True)

        # Store the loaded image in self.loaded_image for visualization tab
        self.loaded_image = pixmap

    def update_visualization_tab(self):
        """Update the visualization tab with the loaded RGB image from the .bil file."""
        if self.loaded_image:  # If an image was loaded
            self.visualization_label.setPixmap(self.loaded_image)
            self.visualization_label.setScaledContents(True)
        else:
            self.visualization_label.setText("No image loaded")

    def update_super_resolution_tab(self):
        if self.loaded_image:  # If an image was loaded
            self.super_resolution_file_label.setText(f"File path: {self.image_path} ")
        else:
            self.super_resolution_file_label.setText("File path: No image loaded")

    def update_classification_tab(self):
        if self.loaded_image:  # If an image was loaded
            self.classification_inputfile_label.setText(f"File path: {self.image_path} ")
        else:
            self.classification_inputfile_label.setText("File path: No image loaded")

    def change_page(self, button_text):
        """Switch between pages and update the visualization tab if necessary."""
        index = ["Visualization", "Super-resolution", "Calibration", "Classification"].index(button_text)
        self.stack.setCurrentIndex(index)

        if button_text == "Visualization":
            self.update_visualization_tab()  # Update the visualization tab with the loaded image
        elif button_text == "Super-resolution":
            self.update_super_resolution_tab()  # 更新超分辨率页面的文件路径
        elif button_text == "Classification":
            self.update_classification_tab()

# # 定义 UnsupervisedClassificationWorker 类
# class UnsupervisedClassificationWorker(QThread):
#     progress_updated = pyqtSignal(int)
#     classification_finished = pyqtSignal(QPixmap)
#     error_occurred = pyqtSignal(str)
#
#     def __init__(self, hsi_data, wavelengths, k, max_iterations):
#         super().__init__()
#         self.hsi_data = hsi_data
#         self.wavelengths = wavelengths
#         self.k = k
#         self.max_iterations = max_iterations
#
#     def run(self):
#         try:
#             # 模拟进度更新
#             import time
#
#             total_steps = 5
#             for step in range(1, total_steps + 1):
#                 time.sleep(1)  # 模拟一些工作正在进行
#                 progress = int(step / total_steps * 100)
#                 self.progress_updated.emit(progress)
#
#             # 执行实际的分类
#             cluster_map, ndvi = load_and_process_hsi_data(
#                 self.hsi_data, self.wavelengths, self.k, self.max_iterations
#             )
#
#             # 将聚类结果映射为彩色图像
#             from matplotlib import pyplot as plt
#             cluster_image_color = plt.cm.nipy_spectral(cluster_map / np.max(cluster_map))
#             cluster_image_color = (cluster_image_color[:, :, :3] * 255).astype(np.uint8)
#
#             # 转换为 QPixmap
#             height, width, _ = cluster_image_color.shape
#             bytes_per_line = 3 * width
#             q_image = QImage(
#                 cluster_image_color.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
#             )
#             pixmap = QPixmap.fromImage(q_image)
#
#             # 发出完成信号
#             self.classification_finished.emit(pixmap)
#
#         except Exception as e:
#             error_message = str(e)
#             self.error_occurred.emit(error_message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
