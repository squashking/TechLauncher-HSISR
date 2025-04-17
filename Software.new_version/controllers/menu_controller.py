import logging
import os
import typing

from PySide6.QtCore import QRect
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMessageBox, QMenuBar, QFileDialog

from controllers.base_controller import BaseController
if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController

from Functions.Basic_Functions import Load_HSI


class MenuController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        # Create the menu bar
        self.menu_bar = QMenuBar(self.main_window)
        self.menu_bar.setObjectName(u"MenuBar")
        self.menu_bar.setGeometry(QRect(0, 0, self.main_window.size().width(), 23))  # why 23?
        self.main_window.setMenuBar(self.menu_bar)

        # Menu: File
        file_menu = self.menu_bar.addMenu("File")

        # Menu: File -> Open
        open_action = QAction("Open", self.main_window)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()  # Adds a separator line

        # Menu: File -> Exit
        exit_action = QAction("Exit", self.main_window)
        exit_action.triggered.connect(self.quit)
        file_menu.addAction(exit_action)

        # Menu: About
        about_action = QAction("About", self.main_window)
        about_action.triggered.connect(self.show_about)
        self.menu_bar.addAction(about_action)

    def open_file(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Open File",
            None,
            "Hyperspectral Images (*.bil *.bip *.bsq)")
        if not image_path:
            self.logger.error("No file selected or invalid file.")
            return
        self.main_controller.hyperspectral_image_path = image_path
        self.logger.info(f"Selected image file: {image_path}")

        # Get corresponding header file
        header_path = image_path.replace(".bil", ".hdr")
        if not os.path.exists(header_path):
            self.logger.error(f"Header file not found at: {header_path}")
            return
        self.logger.info(f"Header file located at: {header_path}")

        # Load hyperspectral image using spectral library
        self.main_controller.hyperspectral_image = Load_HSI.load_hsi(
            self.main_controller.hyperspectral_image_path,
            header_path)
        self.logger.info(f"Hyperspectral image loaded successfully from {image_path}")

        # Update views - tab_visualisation, tab_calibration
        self.main_controller.tab_widget_controller.tab_visualisation_controller.on_load_file()
        self.main_controller.tab_widget_controller.tab_calibration_controller.on_load_file()

    def quit(self):
        self.logger.info("Quitting by menu action")
        self.main_controller.app.quit()

    def show_about(self):
        QMessageBox.about(self.main_window, "About", "Hyperspectral Image Classification GUI\nVersion 2.0")
