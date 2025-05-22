import logging
import typing

import numpy as np
import spectral
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from controllers.base_controller import BaseController
from utils.leaf_utils.basic import find_rgb_bands, convert_hsi_to_rgb_qpixmap
from utils.leaf_utils.unsupervised_classification_utils import load_and_process_hsi_data, preprocess_hsi_with_apriltag_multiscale
from widgets.tab_segmentation_view import TabSegmentationView

if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class TabSegmentationController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.tab_view = TabSegmentationView(self)

        self.hsi = None
        self.unsupervised_worker = None

    def on_load_file(self):
        self.tab_view.classification_inputfile_label.setText(f"File path: {self.main_controller.hyperspectral_image_path}")

        pixmap = convert_hsi_to_rgb_qpixmap(self.main_controller.hyperspectral_image)
        self.tab_view._pixmap = pixmap

        self.tab_view.visualization_label_class.cluster_map = None
        self.tab_view.visualization_label_class.selected_clusters = set()
        self.tab_view.visualization_label_class.ndvi_display.setText("No selection")

        self.tab_view.set_resized_pixmap()

    def run_unsupervised_classification(self):
        try:
            if self.main_controller.hyperspectral_image is None:
                self.tab_view.visualization_label_class.setText("No image loaded.")
                return

            # 1) Extract wavelengths
            if hasattr(self.main_controller.hyperspectral_image, 'metadata') and 'wavelength' in self.main_controller.hyperspectral_image.metadata:
                wavelengths = [float(w) for w in self.main_controller.hyperspectral_image.metadata['wavelength']]
            else:
                QMessageBox.warning(self.tab_view, "Wavelengths Error", "Cannot retrieve wavelength metadata.")
                return

            # 2) Preprocess: crop HSI only
            try:
                hsi_cropped = preprocess_hsi_with_apriltag_multiscale(self.main_controller.hyperspectral_image)
            except ValueError as ve:
                QMessageBox.warning(self.tab_view, "Apriltag Detection Error", str(ve))
                return

            # Update HSI to cropped version
            self.hsi = hsi_cropped

            # 3) Show busy indicator
            self.tab_view.visualization_label_class.setText("Classification in progress.")
            self.tab_view.unsupervised_progress_bar.setRange(0, 0)

            # 4) Start worker with 4 parameters (no mask)
            k = self.tab_view.num_classes_input.value()
            max_iter = self.tab_view.max_iterations_input.value()
            self.unsupervised_worker = UnsupervisedClassificationWorker(
                hsi_cropped,
                wavelengths,
                k,
                max_iter
            )
            self.unsupervised_worker.classification_finished.connect(
                self.unsupervised_classification_finished
            )
            self.unsupervised_worker.error_occurred.connect(
                self.unsupervised_classification_error
            )
            self.unsupervised_worker.start()

        except Exception as e:
            self.tab_view.visualization_label_class.setText(f"Error: {e}")

    def unsupervised_classification_finished(self, pixmap, cluster_map, ndvi, color_img):
        """
        接收分类完成信号后：
        1) 停止进度条
        2) 标记已分类
        3) 注入数据到 ClassificationImageLabel 并渲染
        """
        # 停 busy bar
        self.tab_view.unsupervised_progress_bar.setRange(0, 1)
        self.tab_view.unsupervised_progress_bar.setValue(1)

        # 标记分类完成
        self.classification_done = True

        # —— 新增：先生成底层 RGB 图阵列 ——
        hsi_arr = self.unsupervised_worker.hsi_data
        wavelengths = self.unsupervised_worker.wavelengths
        # 选三波段生成 RGB
        rgb_bands = find_rgb_bands(wavelengths)
        rgb_arr = spectral.get_rgb(hsi_arr, rgb_bands)
        rgb_arr = (rgb_arr * 255).astype(np.uint8)
        self.tab_view.visualization_label_class.set_base_rgb_image(rgb_arr)

        # 注入聚类结果和 NDVI
        self.tab_view.visualization_label_class.set_cluster_map(cluster_map)
        self.tab_view.visualization_label_class.set_ndvi(ndvi)
        # 最后一次性渲染
        self.tab_view.visualization_label_class.update_display()
        self.tab_view.visualization_label_class.update_ndvi_display()

    def unsupervised_classification_error(self, error_message):
        self.tab_view.visualization_label_class.setText(f"Error: {error_message}")
        # 将进度条设置为确定状态并重置为 0%
        self.tab_view.unsupervised_progress_bar.setRange(0, 100)
        self.tab_view.unsupervised_progress_bar.setValue(0)


class UnsupervisedClassificationWorker(QThread):
    classification_finished = Signal(QPixmap, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, hsi_data, wavelengths, k, max_iterations):
        super().__init__()
        self.hsi_data      = hsi_data
        self.wavelengths   = wavelengths
        self.k             = k
        self.max_iterations = max_iterations

    def run(self):
        try:
            # 1) Perform clustering and NDVI via the existing helper
            cluster_map, ndvi = load_and_process_hsi_data(
                self.hsi_data,
                self.wavelengths,
                k=self.k,
                max_iterations=self.max_iterations
            )

            # 2) Generate color image
            num_clusters = cluster_map.max() + 1
            normed = cluster_map / (num_clusters - 1)
            from matplotlib import pyplot as plt
            color_img = plt.cm.nipy_spectral(normed)[:, :, :3]

            # 3) Convert to QPixmap (with deep-copy)
            img8 = (color_img * 255).astype(np.uint8)
            h, w, _ = img8.shape
            qimg = QImage(img8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            # 4) Emit results
            self.classification_finished.emit(pix, cluster_map, ndvi, color_img)

        except Exception as e:
            self.error_occurred.emit(str(e))
