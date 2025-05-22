import gc
import logging
import threading
import time
import typing

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from scipy.ndimage import zoom
from spectral.io import envi
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchnet import meter

from controllers.base_controller import BaseController
from model.common import default_conv
from model.MSDformer import MSDformer
from utils.leaf_utils.basic import load_hsi, convert_hsi_to_rgb_qpixmap, create_header_by_template
from widgets.tab_super_resolution_view import TabSuperResolutionView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabSuperResolutionController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.tab_view = TabSuperResolutionView(self)

        self.model_path = "./model/trained_model/fin_msdformer.pth"
        self.result_path = "data/sr_hsi"
        self.high_hsi = None

        self.mode_output_image = {
            "low": None,
            "high": None,
        }

        self.tab_view.button_super_res.setDisabled(True)
        self.tab_view.radio_low_res.setDisabled(True)
        self.tab_view.radio_low_res.setChecked(True)
        self.tab_view.radio_high_res.setDisabled(True)
        self.tab_view.radio_high_res.setChecked(False)
        self.tab_view.stack.setCurrentIndex(0)

    def on_load_file(self):
        self.mode_output_image["low"] = convert_hsi_to_rgb_qpixmap(self.main_controller.hyperspectral_image)
        self.tab_view.vis_lr_label.set_image(self.mode_output_image["low"])
        self.mode_output_image["high"] = None

        self.tab_view.button_super_res.setEnabled(True)
        self.tab_view.radio_low_res.setDisabled(True)
        self.tab_view.radio_low_res.setChecked(True)
        self.tab_view.radio_high_res.setDisabled(True)
        self.tab_view.radio_high_res.setChecked(False)
        self.tab_view.stack.setCurrentIndex(0)

    def show_low_resolution(self):
        if self.mode_output_image["low"] is None:
            self.mode_output_image["low"] = convert_hsi_to_rgb_qpixmap(self.main_controller.hyperspectral_image)
            self.tab_view.vis_lr_label.set_image(self.mode_output_image["low"])

        self.tab_view.stack.setCurrentIndex(0)

    def show_high_resolution(self):
        if self.high_hsi is None:
            image_path = self.result_path + ".bil"
            header_path = self.result_path + ".hdr"
            self.high_hsi = load_hsi(image_path, header_path)

        self.logger.info(f"Show image of super resolution")

        if self.mode_output_image["high"] is None:
            self.mode_output_image["high"] = convert_hsi_to_rgb_qpixmap(self.high_hsi)
            self.tab_view.vis_sr_label.set_image(self.mode_output_image["high"])

        self.tab_view.stack.setCurrentIndex(1)

    def handle_super_resolution(self):
        self.logger.info("Starting Super Resolution")

        self.tab_view.button_super_res.setDisabled(True)

        self.tab_view.progress_bar.setValue(0)
        self.tab_view.current_progress = 0
        self.tab_view.target_progress = 0

        def update_progress(speed=0.05):
            while self.tab_view.current_progress < self.tab_view.target_progress:
                self.tab_view.current_progress += speed
                self.tab_view.progress_bar.setValue(int(self.tab_view.current_progress))
                time.sleep(0.05)

        def progress_callback(message):
            self.logger.info(f"Update progress: {message}")
            if message == "finish":
                self.tab_view.target_progress = 100
                threading.Thread(target=update_progress, args=(1,), daemon=True).start()
                QTimer.singleShot(1000, self.show_super_resolution_result)
            else:
                self.tab_view.target_progress = min(self.tab_view.target_progress + 25, 99)
                threading.Thread(target=update_progress, daemon=True).start()
            QApplication.processEvents()

        self.run_super_resolution(progress_callback)

    def show_super_resolution_result(self):
        self.show_high_resolution()

        self.tab_view.radio_low_res.setDisabled(False)
        self.tab_view.radio_low_res.setChecked(False)
        self.tab_view.radio_high_res.setDisabled(False)
        self.tab_view.radio_high_res.setChecked(True)

    def run_super_resolution(self, callback, scale_factor=2):

        hsi = self.main_controller.hyperspectral_image
        image = hsi.read_bands(range(hsi.nbands)).astype(np.float32)
        # wavelengths = hsi.metadata.get("wavelength", [])

        interpolated_images = []
        for i in range(image.shape[2]):
            band = image[:, :, i]
            interpolated_band = zoom(band, scale_factor, order=3)
            interpolated_images.append(interpolated_band)
            del interpolated_band
        interpolated_image = np.stack(interpolated_images, axis=2)
        image_data = {"ms_bicubic": interpolated_image, "ms": image}
        del interpolated_images, interpolated_image, image
        callback("finish interpolation")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Running Super Resolution on {device}")
        start_time = time.time()

        result_set = HSResultData(image_data)
        result_loader = DataLoader(result_set, batch_size=1, shuffle=False)
        output = []
        with torch.no_grad():
            epoch_meter = meter.AverageValueMeter()
            epoch_meter.reset()
            net = MSDformer(n_subs=8, n_ovls=2, n_colors=480, scale=2, n_feats=240, n_DCTM=4, conv=default_conv)
            net.to(device).eval()
            state_dict = torch.load(self.model_path, map_location=device, weights_only=True)
            net.load_state_dict(state_dict["model"])
            callback("finish loading model")
            for i, (ms, lms) in enumerate(result_loader):
                ms, lms = ms.to(device), lms.to(device)
                y = net(ms, lms, device)
                y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
                output.append(y)
            del epoch_meter, net, state_dict
        image = np.concat(output, 0).squeeze()
        del result_loader, result_set, image_data, output
        gc.collect()
        callback("finish model inference")

        image_new = []
        for i in range(480):
            image_bond = image[:, :, i]
            image_bond[image_bond < 0] = 0
            image_bond = image_bond / np.max(image_bond)
            image_bond = np.round(image_bond * 4095)
            image_new.append(image_bond)
        image = np.stack(image_new, axis=2)

        envi.save_image(self.result_path + ".hdr", image, dtype=np.float32, interleave="bil", ext="bil", force=True)
        create_header_by_template(
            self.main_controller.hyperspectral_image_path,
            self.result_path + ".hdr",
            image.shape[0],
            image.shape[1],
            4)

        end_time = time.time()
        self.logger.info(f"Super-Resolution run time: {end_time - start_time}s")
        callback("finish")


class HSResultData(data.Dataset):
    def __init__(self, image_data: dict, use_3d_conv : bool = False):
        self.use_3d_conv = use_3d_conv

        try:
            self.ms = np.transpose(np.array(image_data["ms"][...], dtype=np.float32),(3,2,1,0))
            self.lms = np.transpose(np.array(image_data["ms_bicubic"][...], dtype=np.float32),(3,2,1,0))
        except Exception as _:
            image_ms = image_data["ms"]
            image_ms_bicubic = image_data["ms_bicubic"]
            self.ms = image_ms[np.newaxis,:,:,:]
            self.lms = image_ms_bicubic[np.newaxis, ...]

    def __getitem__(self, index):
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]
        if self.use_3d_conv:
            ms, lms = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        return ms, lms

    def __len__(self):
        return self.lms.shape[0]

    def get_shape(self):
        return self.ms.shape
