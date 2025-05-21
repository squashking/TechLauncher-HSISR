from PySide6.QtWidgets import QWidget, QRadioButton, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QStackedWidget, \
    QGroupBox, QFormLayout, QFrame

from widgets.image_viewer import ImageViewer


class TabCalibrationView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create stacked widget for different views
        self.stack = QStackedWidget()

        # Create pages for different views
        input_page = QWidget()
        input_layout = QVBoxLayout()
        self.input_view = ImageViewer(self.controller.logger)
        input_layout.addWidget(self.input_view)
        input_page.setLayout(input_layout)

        output_page = QWidget()
        output_layout = QVBoxLayout()
        self.output_view = ImageViewer(self.controller.logger)
        output_layout.addWidget(self.output_view)
        output_page.setLayout(output_layout)

        combined_page = QWidget()
        combined_layout = QHBoxLayout()
        self.input_view_0 = ImageViewer(self.controller.logger)
        self.output_view_0 = ImageViewer(self.controller.logger)

        combined_layout.addWidget(self.input_view_0)

        # Add separator line
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        combined_layout.addWidget(line)

        combined_layout.addWidget(self.output_view_0)
        combined_page.setLayout(combined_layout)

        self.stack.addWidget(input_page)
        self.stack.addWidget(output_page)
        self.stack.addWidget(combined_page)
        self.stack.setCurrentIndex(0)  # default

        main_layout.addWidget(self.stack)

        # View selection radio buttons
        view_selection_layout = QHBoxLayout()
        self.input_only_button = QRadioButton("Input Only")
        self.output_only_button = QRadioButton("Output Only")
        self.input_and_output_button = QRadioButton("Input and Output")

        self.input_only_button.setChecked(True)
        self.output_only_button.setDisabled(True)
        self.input_and_output_button.setDisabled(True)

        view_selection_layout.addWidget(self.input_only_button)
        view_selection_layout.addWidget(self.output_only_button)
        view_selection_layout.addWidget(self.input_and_output_button)
        view_selection_layout.addStretch()
        main_layout.addLayout(view_selection_layout)

        # Create file selection group
        file_group = QGroupBox("Input Files")
        file_layout = QFormLayout()

        # Input file row
        input_file_layout = QHBoxLayout()
        self.input_file_path = QLineEdit(str(self.controller.main_controller.hyperspectral_image_path))
        self.input_file_path.setReadOnly(True)
        self.run_calibration_button = QPushButton("Calibrate")
        self.run_calibration_button.setDisabled(True)
        input_file_layout.addWidget(self.input_file_path)
        input_file_layout.addWidget(self.run_calibration_button)
        file_layout.addRow("Hyperspectral Image:", input_file_layout)

        # Dark file row
        dark_file_layout = QHBoxLayout()
        self.dark_file_path = QLineEdit()
        self.dark_file_path.setReadOnly(True)
        self.dark_file_button = QPushButton("Browse")
        dark_file_layout.addWidget(self.dark_file_path)
        dark_file_layout.addWidget(self.dark_file_button)
        file_layout.addRow("Dark File:", dark_file_layout)

        # Reference file row
        ref_file_layout = QHBoxLayout()
        self.ref_file_path = QLineEdit()
        self.ref_file_path.setReadOnly(True)
        self.ref_file_button = QPushButton("Browse")
        ref_file_layout.addWidget(self.ref_file_path)
        ref_file_layout.addWidget(self.ref_file_button)
        file_layout.addRow("Reference File:", ref_file_layout)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)