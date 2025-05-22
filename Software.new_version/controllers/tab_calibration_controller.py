from datetime import datetime
import logging
import os
import re
import typing

from PySide6.QtWidgets import QFileDialog

from controllers.base_controller import BaseController
from utils.leaf_utils.basic import load_hsi, calibrate, convert_hsi_to_rgb_qpixmap
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
        calibrated_image = calibrate(
            self.dark_image, self.ref_image, self.main_controller.hyperspectral_image, "data/calibration")

        rgb_qpixmap = convert_hsi_to_rgb_qpixmap(calibrated_image)
        self.logger.info(f"RGB Image Shape: {(rgb_qpixmap.height(), rgb_qpixmap.width())}")

        self.tab_view.output_view.set_image(rgb_qpixmap)
        self.tab_view.output_view_0.set_image(rgb_qpixmap)

        self.tab_view.output_only_button.setEnabled(True)
        self.tab_view.input_and_output_button.setEnabled(True)

    def on_load_file(self):
        self.tab_view.input_file_path.setText(self.main_controller.hyperspectral_image_path)

        rgb_qpixmap = convert_hsi_to_rgb_qpixmap(self.main_controller.hyperspectral_image)
        self.tab_view.input_view.set_image(rgb_qpixmap)
        self.tab_view.input_view_0.set_image(rgb_qpixmap)

        # Auto search for related files by default
        self.auto_search_files()

        self.tab_view.input_only_button.setEnabled(True)
        self.tab_view.output_only_button.setEnabled(False)
        self.tab_view.input_and_output_button.setEnabled(False)
        self.tab_view.input_only_button.setChecked(True)

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
        loaded_image = load_hsi(image_path, header_path)
        self.logger.info(f"{mode} image loaded successfully from {image_path}")

        if mode == "dark":
            self.dark_image = loaded_image
            self.tab_view.dark_file_path.setText(image_path)
        else:
            self.ref_image = loaded_image
            self.tab_view.ref_file_path.setText(image_path)

        if self.dark_image is not None and self.ref_image is not None:
            self.tab_view.run_calibration_button.setEnabled(True)

    def auto_search_files(self):
        """
        Auto-search for dark and reference files based on the input file path.
        Finds calibration files with '_calibFrame' suffix that were created 
        before the currently loaded file. The earliest timestamp is considered
        the dark file and the second earliest is the reference file.
        
        If files are not found, the input fields remain empty.
        """
        if not self.main_controller.hyperspectral_image_path:
            self.logger.warning("No input file loaded, cannot auto-search for related files")
            return
            
        input_dir = os.path.dirname(self.main_controller.hyperspectral_image_path)
        current_file = os.path.basename(self.main_controller.hyperspectral_image_path)
        self.logger.info(f"Searching for calibration files in directory: {input_dir}")
        
        # Reset previous dark and reference files
        self.dark_image = None
        self.ref_image = None
        self.tab_view.dark_file_path.setText("")
        self.tab_view.ref_file_path.setText("")
        self.tab_view.run_calibration_button.setEnabled(False)
        
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
            return
        
        # Look for files with _calibFrame suffix in the BIL format
        calib_files = []
        for file in os.listdir(input_dir):
            if file.endswith(".bil") and "_calibFrame" in file:
                calib_files.append(os.path.join(input_dir, file))
        
        if len(calib_files) < 2:
            self.logger.warning(f"Found only {len(calib_files)} calibration files, need at least 2")
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
            return
        
        # The two most recent calibration files before the current file
        ref_file = calib_files_with_time[0][0]  # Most recent = reference file
        dark_file = calib_files_with_time[1][0]  # Second most recent = dark file
        
        self.logger.info(f"Auto-detected reference file (nearest timestamp): {ref_file}")
        self.logger.info(f"Auto-detected dark file (second nearest timestamp): {dark_file}")
        
        # Load the files
        self._load_file("dark", dark_file)
        self._load_file("ref", ref_file)