import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QStackedWidget, QRadioButton, QLabel, 
                             QLineEdit, QHBoxLayout, QProgressBar, QGroupBox, 
                             QFormLayout, QComboBox, QFrame, QSizePolicy, QFileDialog)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt
from hyperspectral_classifier import HyperspectralClassifier 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 800, 600)

        # Initialize the classifier
        self.classifier = HyperspectralClassifier()
        
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

    def create_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel("Visualization Content", alignment=Qt.AlignmentFlag.AlignCenter))
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
        image_path = self.file_input_class.text()
        groundtruth_path = self.groundtruth_input.text()
        
        header_path = image_path.replace('.bil', '.hdr')  # Assuming the header is .hdr and image is .bil
        
        # Load and classify the image
        self.classifier.load_image(image_path, header_path)
        result_image_path = self.classifier.classify(groundtruth_path)

        # Display the classified result
        pixmap = QPixmap(result_image_path)
        self.image_label.setPixmap(pixmap)

    def change_page(self, button_text):
        index = ["Visualization", "Super-resolution", "Calibration", "Classification"].index(button_text)
        self.stack.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
