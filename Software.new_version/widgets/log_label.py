import logging
import os

from PySide6.QtCore import Qt, QFileInfo, QTimer
from PySide6.QtWidgets import QLabel, QWidget, QScrollArea, QHBoxLayout


class LogScrollArea(QWidget):
    def __init__(self, logger : logging.Logger, log_file_path : str):
        super().__init__()

        self.logger = logger

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(LogLabel(self.logger, log_file_path))

        main_layout = QHBoxLayout()
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)


class LogLabel(QLabel):
    def __init__(self, logger : logging.Logger, log_file_path : str):
        super().__init__()

        self.logger = logger

        self.log_file_path = log_file_path
        self.setTextInteractionFlags(self.textInteractionFlags() | Qt.TextSelectableByMouse)
        self.setWordWrap(True)

        # Keep track of last position
        self.last_size = 0

        # Timer to check for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(1000)  # check every second

        self.update_text()  # load initial content

    def update_text(self):
        try:
            file_info = QFileInfo(self.log_file_path)
            if not file_info.exists():
                self.setText(f"File not found: {self.log_file_path}")
                return

            with open(self.log_file_path, 'r') as f:
                f.seek(self.last_size)
                new_content = f.read()
                self.last_size = f.tell()

            if new_content:
                current_text = self.text()
                updated_text = current_text + new_content
                self.setText(updated_text)

        except Exception as e:
            self.setText(f"Error: {str(e)}")
