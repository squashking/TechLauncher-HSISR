import logging
import os
import types
import typing

from PySide6.QtWidgets import QTabWidget, QLabel, QScrollArea

from controllers.base_controller import BaseController
from controllers.tab_cropping_controller import TabCroppingController
from controllers.tab_visualisation_controller import TabVisualisationController
from controllers.tab_calibration_controller import TabCalibrationController
from controllers.tab_unsupervied_segmentation_controller import TabUnsupervisedSegmentationController
from widgets.log_label import LogScrollArea

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabWidgetController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.tab_widget = QTabWidget(self.main_window)
        self.tab_widget.setTabPosition(QTabWidget.West)
        self.main_window.setCentralWidget(self.tab_widget)

        self.tab_visualisation_controller = TabVisualisationController(logger, main_controller)
        self.tab_widget.addTab(self.tab_visualisation_controller.tab_view, "Visualisation")

        self.tab_cropping_controller = TabCroppingController(logger, main_controller)
        self.tab_widget.addTab(self.tab_cropping_controller.tab_view, "Cropping")

        self.tab_calibration_controller = TabCalibrationController(logger, main_controller)
        self.tab_widget.addTab(self.tab_calibration_controller.tab_view, "Calibration")

        self.tab_unsupervised_segmentation_controller = TabUnsupervisedSegmentationController(logger, main_controller)
        self.tab_widget.addTab(self.tab_unsupervised_segmentation_controller.tab_view, "Segmentation")

        self.tab_log_label = LogScrollArea(self.logger, "logs/" + sorted(os.listdir("logs"))[-1])
        self.tab_widget.addTab(self.tab_log_label, "Logs")

        self.tab_visualisation_controller.tab_view.resizeEvent = types.MethodType(
            lambda _, event:
                self.logger.debug(f"tab_view_1 resized to {event.size().width()}x{event.size().height()}"),
            self.tab_visualisation_controller.tab_view.resizeEvent)
        self.tab_cropping_controller.tab_view.resizeEvent = types.MethodType(
            lambda _, event:
                self.logger.debug(f"tab_view_2 resized to {event.size().width()}x{event.size().height()}"),
            self.tab_cropping_controller.tab_view)
        self.tab_calibration_controller.tab_view.resizeEvent = types.MethodType(
            lambda _, event:
                self.logger.debug(f"tab_view_3 resized to {event.size().width()}x{event.size().height()}"),
            self.tab_calibration_controller.tab_view)
        self.tab_log_label.resizeEvent = types.MethodType(
            lambda _, event:
                self.logger.debug(f"tab_view_4 resized to {event.size().width()}x{event.size().height()}"),
            self.tab_log_label)
