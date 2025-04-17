from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QWidget, QRadioButton, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QPushButton, QStackedWidget

from widgets.image_viewer import ImageViewer


class TabCalibrationView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # Layout for radio buttons
        self.input_only_button = QRadioButton("Input Only")
        self.output_only_button = QRadioButton("Output Only")
        self.input_and_output_button = QRadioButton("Input and Output")
        self.input_only_button.setChecked(True)
        self.output_only_button.setDisabled(True)
        self.input_and_output_button.setDisabled(True)
        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.input_only_button)
        self.radio_layout.addWidget(self.output_only_button)
        self.radio_layout.addWidget(self.input_and_output_button)

        # Layout for files
        TITLE_WIDTH = 150

        self.input_file_title = QLabel("Hyperspectral Image:")
        self.input_file_path = QTextEdit(str(self.controller.main_controller.hyperspectral_image_path))
        self.run_calibration_button = QPushButton("Calibrate")

        self.run_calibration_button.setDisabled(True)
        self.input_file_path.setReadOnly(True)
        self.input_file_title.setFixedWidth(TITLE_WIDTH)
        font_height = QFontMetrics(self.input_file_title.font()).lineSpacing() + 10
        self.input_file_path.setFixedHeight(font_height)

        self.input_file_layout = QHBoxLayout()
        self.input_file_layout.addWidget(self.input_file_title)
        self.input_file_layout.addWidget(self.input_file_path)
        self.input_file_layout.addWidget(self.run_calibration_button)


        self.dark_file_title = QLabel("Dark File:")
        self.dark_file_path = QTextEdit()
        self.dark_file_button = QPushButton("Browse")

        self.dark_file_path.setReadOnly(True)
        self.dark_file_title.setFixedWidth(TITLE_WIDTH)
        self.dark_file_title.setFixedHeight(font_height)
        self.dark_file_path.setFixedHeight(font_height)

        self.dark_file_layout = QHBoxLayout()
        self.dark_file_layout.addWidget(self.dark_file_title)
        self.dark_file_layout.addWidget(self.dark_file_path)
        self.dark_file_layout.addWidget(self.dark_file_button)


        self.ref_file_title = QLabel("Reference File:")
        self.ref_file_path = QTextEdit()
        self.ref_file_button = QPushButton("Browse")

        self.ref_file_path.setReadOnly(True)
        self.ref_file_title.setFixedWidth(TITLE_WIDTH)
        self.ref_file_title.setFixedHeight(font_height)
        self.ref_file_path.setFixedHeight(font_height)

        self.ref_file_layout = QHBoxLayout()
        self.ref_file_layout.addWidget(self.ref_file_title)
        self.ref_file_layout.addWidget(self.ref_file_path)
        self.ref_file_layout.addWidget(self.ref_file_button)


        self.file_layout = QVBoxLayout()
        self.file_layout.addLayout(self.input_file_layout)
        self.file_layout.addLayout(self.dark_file_layout)
        self.file_layout.addLayout(self.ref_file_layout)


        # # Create different views and stack the views (only one visible at a time)
        self.input_view = ImageViewer(self.controller.logger)
        self.output_view = ImageViewer(self.controller.logger)
        self.input_view_0 = ImageViewer(self.controller.logger)
        self.output_view_0 = ImageViewer(self.controller.logger)
        self.input_and_output_layout = QHBoxLayout()
        self.input_and_output_layout.addWidget(self.input_view_0)
        self.input_and_output_layout.addWidget(self.output_view_0)
        self.input_and_output_container = QWidget()
        self.input_and_output_container.setLayout(self.input_and_output_layout)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.input_view)
        self.stack.addWidget(self.output_view)
        self.stack.addWidget(self.input_and_output_container)
        self.stack.setCurrentIndex(0)  # default

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        main_layout.addLayout(self.radio_layout)
        main_layout.addLayout(self.file_layout)
        self.setLayout(main_layout)
