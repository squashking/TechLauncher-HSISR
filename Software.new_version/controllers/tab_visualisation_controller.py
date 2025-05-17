import logging
import typing

from controllers.base_controller import BaseController
from utils.leaf_utils.basic import convert_hsi_to_rgb_qpixmap, convert_hsi_to_ndvi_qpixmap, convert_hsi_to_evi_qpixmap, \
    convert_hsi_to_mcari_qpixmap, convert_hsi_to_mtvi_qpixmap, convert_hsi_to_osavi_qpixmap, convert_hsi_to_pri_qpixmap, \
    convert_hsi_to_hypercube_qpixmap
from widgets.tab_visualisation_view import TabVisualisationView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabVisualisationController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.modes = ["RGB", "NDVI", "EVI", "MCARI", "MTVI", "OSAVI", "PRI", "HyperCube"]
        self.tab_view = TabVisualisationView(self)

        self.mode_button_mapping = dict()
        self.mode_output = dict()
        for i, mode in enumerate(self.modes):
            self.mode_button_mapping[mode] = self.tab_view.radio_buttons[i]
            self.mode_output[mode] = None
        self.func = {
            "RGB": lambda _hsi : convert_hsi_to_rgb_qpixmap(_hsi),
            "NDVI": lambda _hsi : convert_hsi_to_ndvi_qpixmap(_hsi),
            "EVI": lambda _hsi : convert_hsi_to_evi_qpixmap(_hsi),
            "MCARI": lambda _hsi : convert_hsi_to_mcari_qpixmap(_hsi),
            "MTVI": lambda _hsi : convert_hsi_to_mtvi_qpixmap(_hsi),
            "OSAVI": lambda _hsi : convert_hsi_to_osavi_qpixmap(_hsi),
            "PRI": lambda _hsi : convert_hsi_to_pri_qpixmap(_hsi),
            "HyperCube": lambda _hsi : convert_hsi_to_hypercube_qpixmap(_hsi),
        }

        # Connect radio buttons to slot
        for i in range(len(self.modes)):
            self.tab_view.radio_buttons[i].toggled.connect(self.on_click)

    def on_click(self):
        for i, mode in enumerate(self.modes):
            if self.tab_view.radio_buttons[i].isChecked():
                if self.mode_output[mode] is None:
                    self.mode_output[mode] = self.func[mode](self.main_controller.hyperspectral_image)
                    self.tab_view.mode_view_mapping[mode].set_image(self.mode_output[mode])
                self.tab_view.stack.setCurrentIndex(i)
                break

    def on_load_file(self):
        for mode in self.mode_button_mapping:
            button = self.mode_button_mapping[mode]
            button.setEnabled(True)
            self.mode_output[mode] = None
        self.tab_view.radio_buttons[0].setChecked(True)
        self.on_click()
