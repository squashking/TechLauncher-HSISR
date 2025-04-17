import logging

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, logger: logging.Logger, parent=None):
        super().__init__(parent)
        self.logger = logger

        if not self.objectName():
            self.setObjectName(u"MainWindow")
        self.setWindowTitle("MainWindow")
        self.resize(QSize(1200, 800))

        # QMetaObject.connectSlotsByName(self)

    def resizeEvent(self, event):
        new_size = event.size()
        self.logger.debug(f"Window resized to {new_size.width()}x{new_size.height()}")
