import logging
import os
import typing
import glob
import re
from datetime import datetime

import numpy as np
import spectral
from PySide6.QtWidgets import QFileDialog, QMessageBox
from matplotlib import pyplot as plt

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
        self.model_path = None

        self.tab_view = TabCalibrationView(self)

        self.tab_view.dark_file_button.clicked.connect(lambda: self.on_click_load_file_button("dark"))
        self.tab_view.ref_file_button.clicked.connect(lambda: self.on_click_load_file_button("ref"))
        self.tab_view.model_file_button.clicked.connect(self.on_click_load_model_button)
        self.tab_view.run_calibration_button.clicked.connect(self.run_calibration)
        self.tab_view.auto_search_button.clicked.connect(self.auto_search_files)

        self.tab_view.input_only_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(0))
        self.tab_view.output_only_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(1))
        self.tab_view.input_and_output_button.toggled.connect(lambda: self.tab_view.stack.setCurrentIndex(2))

    def run_calibration(self):
        calibrated_image = Functions.Calibration.calibrate.calibration(
            self.dark_image, self.ref_image, self.main_controller.hyperspectral_image, "data/calibration")

        # Store the model path for future reference
        self.model_path = "data/calibration.bil"
        self.tab_view.model_file_path.setText(self.model_path)

        # From Functions.Visualization.Visualize_HSI.py - show_rgb
        tuple_rgb_bands = Visualize_HSI.find_RGB_bands(
            [float(i) for i in calibrated_image.metadata["wavelength"]])
        rgb_image = spectral.get_rgb(calibrated_image, tuple_rgb_bands)  # (100, 54, 31)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_image = rgb_image.copy()  # Spy don't load it to memory automatically, so must be copied
        self.logger.info(f"RGB Image Shape: {rgb_image.shape}")

        self.tab_view.output_view.set_image(TabVisualisationController.get_QPixmap(lambda buf: plt.imsave(buf, rgb_image)))
        self.tab_view.output_view_0.set_image(TabVisualisationController.get_QPixmap(lambda buf: plt.imsave(buf, rgb_image)))

        self.tab_view.output_only_button.setEnabled(True)
        self.tab_view.input_and_output_button.setEnabled(True)

    def on_load_file(self):
        self.tab_view.input_file_path.setText(self.main_controller.hyperspectral_image_path)

        self.tab_view.input_view.set_image(
            self.main_controller.tab_widget_controller.tab_visualisation_controller.get_RGB())
        self.tab_view.input_view_0.set_image(
            self.main_controller.tab_widget_controller.tab_visualisation_controller.get_RGB())
        
        # Auto search for related files if enabled
        if self.tab_view.auto_search_checkbox.isChecked():
            self.auto_search_files()

    def on_click_load_file_button(self, mode):
        assert mode in ["dark", "ref"]

        image_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            f"Open {mode.capitalize()} File",
            os.path.dirname(self.main_controller.hyperspectral_image_path) if self.main_controller.hyperspectral_image_path else None,
            "BIL Files (*.bil)")
        if not image_path:
            self.logger.error(f"No {mode} file selected or invalid file.")
            return
        self.logger.info(f"Selected {mode} image file: {image_path}")

        self._load_file(mode, image_path)

    def _load_file(self, mode, image_path):
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

    def on_click_load_model_button(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Open Calibration Model File",
            os.path.dirname(self.main_controller.hyperspectral_image_path) if self.main_controller.hyperspectral_image_path else None,
            "BIL Files (*.bil)")
        
        if not model_path:
            self.logger.error("No model file selected or invalid file.")
            return
        
        self.logger.info(f"Selected model file: {model_path}")
        
        # Load the model
        header_path = model_path.replace(".bil", ".hdr")
        if not os.path.exists(header_path):
            self.logger.error(f"Header file not found at: {header_path}")
            return
            
        try:
            model = spectral.io.envi.open(header_path, model_path)
            # You might want to verify the model here
            
            self.model_path = model_path
            self.tab_view.model_file_path.setText(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
            
            # Show confirmation to user
            QMessageBox.information(self.main_window, "Model Loaded", 
                                   f"Calibration model loaded successfully from:\n{os.path.basename(model_path)}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            QMessageBox.warning(self.main_window, "Model Load Error", 
                               f"Failed to load calibration model:\n{str(e)}")

    def auto_search_files(self):
        """
        Auto-search for dark and reference files based on the input file path.
        Finds calibration files with '_calibFrame' suffix that were created 
        before the currently loaded file. The earliest timestamp is considered
        the dark file and the second earliest is the reference file.
        """
        if not self.main_controller.hyperspectral_image_path:
            self.logger.warning("No input file loaded, cannot auto-search for related files")
            return
            
        input_dir = os.path.dirname(self.main_controller.hyperspectral_image_path)
        current_file = os.path.basename(self.main_controller.hyperspectral_image_path)
        self.logger.info(f"Searching for calibration files in directory: {input_dir}")
        
        # Extract timestamp from current file
        current_timestamp = None
        timestamp_pattern = r"(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})"
        match = re.search(timestamp_pattern, current_file)
        if match:
            timestamp_str = match.group(1)
            try:
                current_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d--%H-%M-%S")
                self.logger.info(f"Current file timestamp: {current_timestamp}")
            except ValueError:
                self.logger.warning(f"Could not parse timestamp from current file: {current_file}")
                current_timestamp = None
        
        if not current_timestamp:
            self.logger.warning("Could not extract timestamp from current file")
            QMessageBox.warning(self.main_window, "Auto-Search Error", 
                            "Could not extract timestamp from current file. Manual selection required.")
            return
        
        # Look for files with _calibFrame suffix in the BIL format
        calib_files = []
        for file in os.listdir(input_dir):
            if file.endswith(".bil") and "_calibFrame" in file:
                calib_files.append(os.path.join(input_dir, file))
        
        if len(calib_files) < 2:
            self.logger.warning(f"Found only {len(calib_files)} calibration files, need at least 2")
            QMessageBox.information(self.main_window, "Auto-Search Results", 
                                f"Found only {len(calib_files)} calibration files with '_calibFrame' suffix. Need both dark and reference files.")
            return
        
        # Extract timestamps from filenames and keep only those before current timestamp
        calib_files_with_time = []
        
        for file_path in calib_files:
            filename = os.path.basename(file_path)
            match = re.search(timestamp_pattern, filename)
            if match:
                timestamp_str = match.group(1)
                try:
                    # Format: YYYY-MM-DD--HH-MM-SS
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d--%H-%M-%S")
                    # Only include files with timestamps before the current file
                    if timestamp < current_timestamp:
                        calib_files_with_time.append((file_path, timestamp))
                except ValueError:
                    self.logger.warning(f"Could not parse timestamp from {filename}")
        
        # Sort by timestamp (descending order to get the most recent first)
        calib_files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        if len(calib_files_with_time) < 2:
            self.logger.warning("Could not find enough calibration files with valid timestamps before current file")
            QMessageBox.information(self.main_window, "Auto-Search Results", 
                                "Could not find enough calibration files with valid timestamps before current file.")
            return
        
        # The two most recent calibration files before the current file
        ref_file = calib_files_with_time[0][0]  # Most recent = reference file
        dark_file = calib_files_with_time[1][0]  # Second most recent = dark file
        
        self.logger.info(f"Auto-detected reference file (nearest timestamp): {ref_file}")
        self.logger.info(f"Auto-detected dark file (second nearest timestamp): {dark_file}")
        
        # Load the files
        if not self.dark_image:
            self._load_file("dark", dark_file)
            
        if not self.ref_image:
            self._load_file("ref", ref_file)
        
        # Notify user
        QMessageBox.information(self.main_window, "Auto-Search Results", 
                            f"Found calibration files:\nReference (most recent): {os.path.basename(ref_file)}\nDark (second most recent): {os.path.basename(dark_file)}")