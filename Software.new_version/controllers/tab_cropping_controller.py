import logging
import typing

import spectral.image
from PySide6.QtWidgets import QMessageBox

from controllers.base_controller import BaseController
from widgets.tab_cropping_view import TabCroppingView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabCroppingController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.tab_view = TabCroppingView(self)
        self.tab_view.selectable_image_viewer.enabled_selection(True)

        self.tab_view.clear_selection_button.setDisabled(True)
        self.tab_view.crop_button.setDisabled(True)
        self.tab_view.save_as_hsi_button.setDisabled(True)

        self.tab_view.clear_selection_button.clicked.connect(self.on_click_clear_selection_button)
        self.tab_view.crop_button.clicked.connect(self.on_click_crop)
        self.tab_view.save_as_hsi_button.clicked.connect(self.on_click_save_as_hsi)

    def on_click_clear_selection_button(self):
        self.tab_view.selectable_image_viewer.clear_selection()
        self.tab_view.clear_selection_button.setDisabled(True)
        self.tab_view.crop_button.setDisabled(True)

    def on_click_crop(self):
        selectable_image_label = self.tab_view.selectable_image_viewer.image_label
        top_left = selectable_image_label.map_from_self_to_pixmap(selectable_image_label.selection_rect.topLeft())
        bottom_right = selectable_image_label.map_from_self_to_pixmap(selectable_image_label.selection_rect.bottomRight())
        self.logger.info(f"top_left: {top_left}, bottom_right: {bottom_right}")

        x_lower, x_upper = max(0, top_left.y()), min(self.main_controller.hyperspectral_image.shape[0], bottom_right.y())
        y_lower, y_upper = max(0, top_left.x()), min(self.main_controller.hyperspectral_image.shape[1], bottom_right.x())
        new_hsi_data = self.main_controller.hyperspectral_image[x_lower: x_upper, y_lower : y_upper, :]

        hsi_image = spectral.image.ImageArray(new_hsi_data, self.main_controller.hyperspectral_image)
        self.logger.info(f"old shape: {self.main_controller.hyperspectral_image.shape}")
        self.logger.info(f"new shape: {hsi_image.shape}")
        self.main_controller.update_hyperspectral_image(hsi_image)
        self.on_click_clear_selection_button()

    def on_click_save_as_hsi(self):
        QMessageBox.about(self.main_window, "Save", "Save Hyperspectral Image: to be implemented")

    def on_load_file(self):
        self.tab_view.selectable_image_viewer.set_image(
            self.main_controller.tab_widget_controller.tab_visualisation_controller.get_RGB())
        self.tab_view.save_as_hsi_button.setEnabled(True)
