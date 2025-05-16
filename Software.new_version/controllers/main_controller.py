import logging
from typing import List

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from controllers.menu_controller import MenuController
from controllers.status_bar_controller import StatusBarController
from controllers.tab_widget_controller import TabWidgetController
from widgets.main_window import MainWindow


class MainController:
    _controller = None

    def __new__(cls, *args, **kwargs):
        if not cls._controller:
            cls._controller = super(MainController, cls).__new__(cls)
        return cls._controller

    def __init__(self, logger: logging.Logger, argv: List):
        self.logger = logger

        self.hyperspectral_image = None
        self.hyperspectral_image_path = None

        self.app = QApplication(argv)
        self.main_window = MainWindow(logger)
        self.menu_controller = MenuController(logger, self)
        self.status_bar_controller = StatusBarController(logger, self)
        self.tab_widget_controller = TabWidgetController(logger, self)

        icon = QIcon("logo/LEAF-icon.png")
        self.main_window.setWindowIcon(icon)

    @staticmethod
    def get():
        if MainController._controller:
            return MainController._controller
        raise Exception("MainController not initialized")

    def update_hyperspectral_image(self, hyperspectral_image, hyperspectral_image_path=None):
        self.hyperspectral_image = hyperspectral_image
        if hyperspectral_image_path is not None:
            self.hyperspectral_image_path = hyperspectral_image_path

        # Update views - tab_visualisation, tab_calibration
        self.tab_widget_controller.tab_visualisation_controller.on_load_file()
        self.tab_widget_controller.tab_cropping_controller.on_load_file()
        self.tab_widget_controller.tab_super_resolution_controller.on_load_file()
        self.tab_widget_controller.tab_calibration_controller.on_load_file()
        self.tab_widget_controller.tab_unsupervised_segmentation_controller.on_load_file()
