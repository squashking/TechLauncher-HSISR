import logging
import os
import typing

import numpy as np
import spectral
from PySide6.QtWidgets import QFileDialog

import Functions.Calibration.calibrate
from Functions.Basic_Functions import Load_HSI
from Functions.Visualization import Visualize_HSI
from controllers.base_controller import BaseController
from controllers.tab_visualisation_controller import TabVisualisationController
from widgets.tab_calibration_view import TabCalibrationView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabCalibrationController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.dark_image = None
        self.ref_image = None

        self.tab_view = TabCalibrationView(self)

        self.tab_view.dark_file_button.clicked.connect(lambda: self.on_click_load_file_button("dark"))
        self.tab_view.ref_file_button.clicked.connect(lambda: self.on_click_load_file_button("ref"))
        self.tab_view.run_calibration_button.clicked.connect(self.run_calibration)

        self.tab_view.input_only_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(0))
        self.tab_view.output_only_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(1))
        self.tab_view.input_and_output_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(2))

    def run_calibration(self):
        calibrated_image = Functions.Calibration.calibrate.calibration(
            self.dark_image, self.ref_image, self.main_controller.hyperspectral_image, "data/calibration")

        # From Functions.Visualization.Visualize_HSI.py - show_rgb
        tuple_rgb_bands = Visualize_HSI.find_RGB_bands(
            [float(i) for i in calibrated_image.metadata["wavelength"]])
        rgb_image = spectral.get_rgb(calibrated_image, tuple_rgb_bands)  # (100, 54, 31)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_image = rgb_image.copy()  # Spy don't load it to memory automatically, so must be copie
        self.logger.info(f"RGB Image Shape: {rgb_image.shape}")

        self.tab_view.output_view.set_image(TabVisualisationController.get_QPixmap(rgb_image))
        self.tab_view.output_view_0.set_image(TabVisualisationController.get_QPixmap(rgb_image))

        self.tab_view.output_only_button.setEnabled(True)
        self.tab_view.input_and_output_button.setEnabled(True)

    def on_load_file(self):
        self.tab_view.input_file_path.setText(self.main_controller.hyperspectral_image_path)

        self.tab_view.input_view.set_image(
            self.main_controller.tab_widget_controller.tab_visualisation_controller.mode_output["RGB"])
        self.tab_view.input_view_0.set_image(
            self.main_controller.tab_widget_controller.tab_visualisation_controller.mode_output["RGB"])

    def on_click_load_file_button(self, mode):
        assert mode in ["dark", "ref"]

        image_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Open File",
            None,
            "BIL Files (*.bil)")
        if not image_path:
            self.logger.error(f"No {mode} file selected or invalid file.")
            return
        self.main_controller.hyperspectral_image_path = image_path
        self.logger.info(f"Selected {mode} image file: {image_path}")

        # Get corresponding header file
        header_path = image_path.replace(".bil", ".hdr")
        if not os.path.exists(header_path):
            self.logger.error(f"Header {mode} file not found at: {header_path}")
            return
        self.logger.info(f"Header {mode} file located at: {header_path}")

        # Load hyperspectral image using spectral library
        loaded_image = Load_HSI.load_hsi(image_path, header_path)
        self.logger.info(f"{mode} image loaded successfully from {image_path}")

        if mode == "dark":
            self.dark_image = loaded_image
            self.tab_view.dark_file_path.setText(image_path)
        else:
            self.ref_image = loaded_image
            self.tab_view.ref_file_path.setText(image_path)

        if self.dark_image is not None and self.ref_image is not None:
            self.tab_view.run_calibration_button.setEnabled(True)
