from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QPushButton, QStackedWidget, QRadioButton, QLabel,
                             QLineEdit, QHBoxLayout, QProgressBar, QGroupBox, QMenu,
                             QFormLayout, QComboBox, QFrame, QSizePolicy, QFileDialog, QMenuBar, QSpinBox, QMessageBox)
from PyQt6.QtGui import QFont, QPixmap, QAction, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QRect, QSize

import sys
import os
import shutil
import copy
import time
import threading
import spectral.io.envi as envi
from spectral import get_rgb
import numpy as np
import platform
from matplotlib import colormaps


if platform.system() == 'Windows':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Software')))
else:
    # On macOS or other Unix-like systems, keep the original path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.Basic_Functions.Load_HSI import load_hsi
from Functions.Visualization.Visualize_HSI import find_RGB_bands, show_rgb, show_ndvi, show_evi, show_mcari, show_mtvi, show_osavi, show_pri
from Functions.Super_resolution.Run_Super_Resolution import run_super_resolution
from Functions.Calibration.calibrate import calibration
from Functions.Hypercube_Spectrum.Hypercube import show_cube
from unsupervised_worker import UnsupervisedClassificationWorker
from Functions.Hypercube_Spectrum.Spectrum_plot import plot_spectrum
from Functions.Supervised_classification.hyperspectral_classifier import HyperspectralClassifier
from Functions.Unsupervised_classification.unsupervised_classification import preprocess_hsi_with_apriltag_multiscale



class ClickableImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loaded_image = None
        self.right_click_position = None  # To store the right-click position
        self.hsi = None
        self.mode = None

    def set_hsi(self, hsi):
        self.hsi = hsi

    def set_mode(self, mode):
        self.mode = mode

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pass
        elif event.button() == Qt.MouseButton.RightButton:
            if self.mode == "hypercube":
                return
            self.right_click_position = event.pos()
            self.show_context_menu(event.pos())

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        action1 = QAction("Spectrum plot", self)
        action1.triggered.connect(self.action1_triggered)
        context_menu.addAction(action1)
        context_menu.exec(self.mapToGlobal(position))

    def action1_triggered(self):
        if self.right_click_position is not None:
            try:
                x = self.right_click_position.x() / 1.4
                y = self.right_click_position.y() / 1.4
                print(x,y)
                plot_spectrum(self.hsi, int(x), int(y))
            except Exception as e:
                error_message = f"Error plotting spectrum: {str(e)}"
                print(error_message)

class ClassificationImageLabel(QLabel):
    def __init__(self, ndvi_display, parent=None):
        super().__init__(parent)
        self.cluster_map = None
        self.ndvi = None
        self.selected_clusters = set()
        self.ndvi_display = ndvi_display

        # 不自动缩放
        self.setScaledContents(False)

        # 新版获取 colormap，避免 deprecation warning
        from matplotlib import colormaps
        self.overlay_cmap = colormaps['tab20']
        self.num_overlay_colors = self.overlay_cmap.N
        self.alpha = 0.4
        self.base_rgb_image = None  # HxWx3 uint8 原始 RGB 底图
        self.default_alpha = 0.2  # 默认透明度
        self.highlight_alpha = 0.7  # 点击后更不透明
        self.num_clusters = 0  # 聚类数

    def set_cluster_map(self, cluster_map):
        self.cluster_map = cluster_map
        self.num_clusters = int(cluster_map.max()) + 1

    def set_ndvi(self, ndvi):
        self.ndvi = ndvi

    def set_base_rgb_image(self, rgb_arr):
        """
           设置底层的 RGB 图像数组 (HxWx3 uint8)，后续在 update_display 时叠加 mask。
        """
        self.base_rgb_image = rgb_arr.copy()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            x = pos.x()
            y = pos.y()
            if self.cluster_map is not None:
                # Handle image scaling
                pixmap = self.pixmap()
                if pixmap:
                    img_width = pixmap.width()
                    img_height = pixmap.height()
                    lbl_width = self.width()
                    lbl_height = self.height()
                    scale_x = img_width / lbl_width
                    scale_y = img_height / lbl_height
                    img_x = int(x * scale_x)
                    img_y = int(y * scale_y)
                    if 0 <= img_x < self.cluster_map.shape[1] and 0 <= img_y < self.cluster_map.shape[0]:
                        cluster_label = self.cluster_map[img_y, img_x]
                        if cluster_label in self.selected_clusters:
                            self.selected_clusters.remove(cluster_label)
                        else:
                            self.selected_clusters.add(cluster_label)
                        self.update_display()
                        self.update_ndvi_display()
        elif event.button() == Qt.MouseButton.RightButton:
            # Handle right-click: show context menu
            pos = event.pos()
            self.show_context_menu(pos)
        else:
            super().mousePressEvent(event)


    def change_visualization_method(self, method_id):
        self.visualization_method = method_id
        self.update_display()

    def show_context_menu(self, position):
        """
        右键菜单：仅保留光谱绘制和保存图像功能。
        """
        context_menu = QMenu(self)

        # 光谱图
        action_spectrum = QAction("Spectrum plot", self)
        action_spectrum.triggered.connect(lambda: self.show_spectrum_plot(position))
        context_menu.addAction(action_spectrum)

        # 保存当前蒙版图
        action_save_image = QAction("Save Image", self)
        action_save_image.triggered.connect(self.save_masked_image)
        context_menu.addAction(action_save_image)

        context_menu.exec(self.mapToGlobal(position))

    # Add a method to set HSI data
    def set_hsi(self, hsi):
        self.hsi = hsi

    # Method to show spectrum plot
    def show_spectrum_plot(self, position):
        if self.hsi is None:
            print("HSI data not available")
            return
        # Convert position to image coordinates
        x, y = self.convert_position_to_image_coords(position)
        if x is not None and y is not None:
            try:
                plot_spectrum(self.hsi, x, y)
            except Exception as e:
                print(f"Error plotting spectrum: {e}")
        else:
            print("Invalid position for spectrum plot")

    def convert_position_to_image_coords(self, position):
        x = position.x()
        y = position.y()
        pixmap = self.pixmap()
        if pixmap:
            label_width = self.width()
            label_height = self.height()
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            ratio_x = pixmap_width / label_width
            ratio_y = pixmap_height / label_height

            img_x = int(x * ratio_x)
            img_y = int(y * ratio_y)

            # Ensure coordinates are within bounds
            img_x = min(max(img_x, 0), self.cluster_map.shape[1] - 1)
            img_y = min(max(img_y, 0), self.cluster_map.shape[0] - 1)

            return img_x, img_y
        else:
            return None, None


    def set_cluster_image_color(self, cluster_image_color):
        self.cluster_image_color = cluster_image_color.copy()  # Store the current color image array
        self.original_cluster_image_color = cluster_image_color.copy()  # Store the original color image array

    def save_masked_image(self):
        if self.cluster_map is None or self.cluster_image_color is None:
            print("No cluster map or image data available")
            return

        try:
            # Create a mask of the selected clusters
            if self.selected_clusters:
                mask = np.isin(self.cluster_map, list(self.selected_clusters))
            else:
                mask = np.ones_like(self.cluster_map, dtype=bool)  # If no clusters selected, include all

            # Apply the mask to the image
            masked_image = self.cluster_image_color.copy()
            masked_image[~mask, :] = [0.0, 0.0, 0.0]  # Black out unselected regions

            # Convert to uint8
            image_to_save = (masked_image[:, :, :3] * 255).astype(np.uint8)

            # Use QFileDialog to choose save location (options adjusted for PyQt6)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Masked Image",
                "",
                "PNG Files (*.png);;All Files (*)"
                # No 'options' parameter needed if not setting any options
            )
            if file_path:
                # Save the image using PIL
                from PIL import Image
                image = Image.fromarray(image_to_save)
                image.save(file_path)
                print(f"Masked image saved to {file_path}")
            else:
                print("Save operation cancelled")
        except Exception as e:
            print(f"Error in save_masked_image: {e}")

    # def update_display(self):
    #     """
    #     在底层分类色彩图上，对每个 selected_clusters 用不同彩色半透明 mask 做 alpha-blend，
    #     100% 在 NumPy 层面计算，最后一次性构造 QImage 并 copy，避免内存野指针。
    #     """
    #     if self.cluster_map is None or self.cluster_image_color is None:
    #         return
    #
    #     # 1) 准备底图：HxWx3, uint8
    #     base = (self.cluster_image_color * 255).astype(np.uint8)
    #     # 确保 C-连续
    #     base = np.ascontiguousarray(base)
    #     h, w, _ = base.shape
    #
    #     # 2) 对每个选中 segment 做 overlay
    #     for lbl in self.selected_clusters:
    #         mask = (self.cluster_map == lbl)
    #         if not mask.any():
    #             continue
    #         # 取一个对比色（tab20）
    #         rgba = self.overlay_cmap(lbl % self.num_overlay_colors)
    #         overlay_color = np.array(rgba[:3]) * 255  # float->0-255
    #         # NumPy 向量化 alpha blend
    #         for c in range(3):
    #             channel = base[..., c]
    #             # 公式：out = alpha*overlay + (1-alpha)*orig
    #             channel[mask] = (
    #                     overlay_color[c] * self.alpha +
    #                     channel[mask] * (1.0 - self.alpha)
    #             ).astype(np.uint8)
    #             base[..., c] = channel
    #
    #     # 3) 一次性构造 QImage，并深拷贝
    #     bytes_per_line = 3 * w
    #     qimg = QImage(base.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    #     pix = QPixmap.fromImage(qimg)
    #
    #     # 4) 显示
    #     self.setFixedSize(pix.size())
    #     self.setPixmap(pix)

    def update_display(self):
        """
           在原始 RGB 底图上，对所有 segment （或仅 selected_clusters）叠加半透明彩色 mask。
           selected_clusters 为空时，显示所有 segment；否则仅对 selected_clusters 用更高不透明度叠加。
        """
        if self.cluster_map is None or self.base_rgb_image is None:
            return

        base = self.base_rgb_image.copy()
        h, w, _ = base.shape

        # 对每个 segment 叠加 mask
        for lbl in range(self.num_clusters):
            mask = (self.cluster_map == lbl)
            if not mask.any():
                continue
            # 点击后用 highlight_alpha，其它用 default_alpha；若无选中，则都用 default_alpha
            if self.selected_clusters:
                alpha = self.highlight_alpha if lbl in self.selected_clusters else self.default_alpha
            else:
                alpha = self.default_alpha

            rgba = self.overlay_cmap(lbl % self.num_overlay_colors)
            color = (np.array(rgba[:3]) * 255).astype(np.uint8)
            # 叠加
            for c in range(3):
                chan = base[..., c]
                chan[mask] = (
                        color[c] * alpha +
                        chan[mask] * (1.0 - alpha)
                ).astype(np.uint8)
                base[..., c] = chan

        # 转为 QPixmap 并显示
        bytes_per_line = 3 * w
        qimg = QImage(base.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.setFixedSize(pix.size())
        self.setPixmap(pix)


    def update_ndvi_display(self):
        try:
            print("update_ndvi_display called")
            if self.ndvi is None or self.cluster_map is None:
                print("NDVI or cluster_map is None")
                return
            print(f"NDVI shape: {self.ndvi.shape}")
            print(f"Cluster map shape: {self.cluster_map.shape}")

            selected_clusters = self.selected_clusters
            print(f"Selected clusters: {selected_clusters}")
            if not selected_clusters:
                self.ndvi_display.setText("No selection")
                return
            mask = np.isin(self.cluster_map, list(selected_clusters))
            # print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            selected_ndvi_values = self.ndvi[mask]
            # print(f"Selected NDVI values count: {selected_ndvi_values.size}")
            avg_ndvi = np.mean(selected_ndvi_values)
            print(f"Average NDVI: {avg_ndvi:.4f}")
            self.ndvi_display.setText(f"Average NDVI: {avg_ndvi:.4f}")
        except Exception as e:
            print(f"Error updating NDVI display: {str(e)}")
            self.ndvi_display.setText("Error computing NDVI")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.loaded_image = None  # To store the loaded image for Vis Window
        self.classification_done = False
        self.image_path = ""
        self.setWindowTitle("MainWindow")
        self.setGeometry(100, 100, 1024, 768)

        # Initialize the classifier
        self.classifier = HyperspectralClassifier()

        # Create menu bar
        self.create_menu_bar()

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_frame = QFrame()
        left_frame.setFixedWidth(200)
        left_frame.setStyleSheet("""
            background-color: #f0f0f0;
            border-right: 1px solid #d0d0d0;
        """)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 20, 20, 20)

        main_widget.setLayout(main_layout)
        main_layout.addWidget(left_frame)
        main_layout.addLayout(right_layout)

        # Clickable image label (after defining main_layout)
        # self.image_label = ClickableImage(self)  # Using the custom ClickableImage class
        # right_layout.addWidget(self.image_label)  # Add it to right_layout instead of main_layout

        # Mode selection and visualization controls
        self.mode = "RGB"

        # Sidebar buttons
        sidebar_buttons = ["Visualization", "Super-resolution", "Calibration", "Classification"]
        self.current_tab = "Visualization"  # init tab
        self.sidebar_button_index = dict()
        for i, button_text in enumerate(sidebar_buttons):
            self.sidebar_button_index[button_text] = i
        self.sidebar_buttons = sidebar_buttons
        for i, button_text in enumerate(sidebar_buttons):
            btn = QPushButton(button_text)
            btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: none;
                    border-bottom: 1px solid #d0d0d0;
                    text-align: center;
                    padding: 15px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
            """)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            left_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, text=button_text: self.change_page(text))

        # Register Resize Handler
        self.installEventFilter(self)

        # tab states
        self.tab_state = dict()
        for button_text in self.sidebar_buttons:
            self.tab_state[button_text] = 0

        # HSI and image to save
        self.output_sr_hsi = None
        self.output_calibration_hsi = None
        self.vis_viewer_image = None
        self.sr_viewer_image = None
        self.calibration_viewer_image = None

        # This will make the buttons fill the available space
        for i in range(len(sidebar_buttons)):
            left_layout.setStretch(i, 1)

        # Stack for right layout
        self.stack = QStackedWidget()
        right_layout.addWidget(self.stack)

        # Create pages
        self.create_visualization_page()
        self.create_super_resolution_page()
        self.create_calibration_page()
        self.create_classification_page()

    def create_menu_bar(self):
        """Create a menu bar with 'File' and 'About' options."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("File")

        load_action = QAction("Load Hyperspectral Image", self)
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        load_action.setDisabled(False)

        save_hyperspectral_image_action = QAction("Save Hyperspectral Image", self)
        save_hyperspectral_image_action.triggered.connect(self.save_hyperspectral_image)
        file_menu.addAction(save_hyperspectral_image_action)
        save_hyperspectral_image_action.setDisabled(True)
        self.save_hyperspectral_image_action = save_hyperspectral_image_action

        save_viewer_image_action = QAction("Save Viewer Image", self)
        save_viewer_image_action.triggered.connect(self.save_viewer_image)
        file_menu.addAction(save_viewer_image_action)
        save_viewer_image_action.setDisabled(True)
        self.save_viewer_image_action = save_viewer_image_action

        # About menu
        about_menu = menu_bar.addMenu("About")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

    def show_about_dialog(self):
        """Display an 'About' dialog."""
        about_dialog = QLabel("Hyperspectral Image Classification GUI\nVersion 1.0", alignment=Qt.AlignmentFlag.AlignCenter)
        about_dialog.show()

    def load_image(self):
        """加载 HSI 并立即在所有页显示原始 RGB，重置 classification_done"""
        # 1) 打开文件对话
        image_path, _ = QFileDialog.getOpenFileName(
            self, 'Open file', None,
            "Hyperspectral Images (*.bil *.bip *.bsq)"
        )
        if not image_path:
            return
        self.image_path = image_path

        # 2) 读取 .hdr & HSI 数据
        header_path = image_path.replace('.bil', '.hdr')
        if not os.path.exists(header_path):
            QMessageBox.warning(self, "Load Error", "Header file not found.")
            return
        try:
            self.hsi = load_hsi(image_path, header_path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load HSI:\n{e}")
            return

        # 3) 生成 RGB 预览并保留
        save_path = "img/temp_rgb.png"
        show_rgb(self.hsi, save_path)
        pixmap = QPixmap(save_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Visualization Error", "Failed to generate RGB preview.")
            return
        self.loaded_image = pixmap
        # 重置分类状态
        self.classification_done = False

        # 4) 在 Visualization 页贴图
        self.visualization_file_label.setText(f"File path: {self.image_path}")
        self.set_label_scaled_pixmap(self.visualization_label, pixmap)
        self.visualization_label.setScaledContents(True)
        self.radio_rgb.setChecked(True)
        self.tab_state["Visualization"] = 1
        self.save_hyperspectral_image_action.setDisabled(False)
        self.save_viewer_image_action.setDisabled(False)

        # 5) 启用其它页面
        for page in ["Super-resolution", "Calibration", "Classification"]:
            self.tab_state[page] = 1

        # 6) 切到 Visualization
        self.change_page("Visualization")

    def save_hyperspectral_image(self):
        """Save the currently visualized image."""
        if self.loaded_image is not None:
            # Open a file dialog to choose the save location
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, "Save Hyperspectral Image", "", "HDR Image (*.hdr);;BIL Image (*.bil)")
            save_hdr = save_path + ".hdr"
            save_bil = save_path + ".bil"
            if save_path:
                if self.current_tab == "Visualization":
                    envi.save_image(save_hdr, self.hsi, dtype=np.uint16, interleave="bil", ext="bil", force=True, metadata=self.hsi.metadata)
                elif self.current_tab == "Super-resolution":
                    envi.save_image(save_hdr, self.output_sr_hsi, dtype=np.uint16, interleave="bil", ext="bil", force=True, metadata=self.output_sr_hsi.metadata)
                elif self.current_tab == "Calibration":
                    envi.save_image(save_hdr, self.output_calibration_hsi, dtype=np.uint16, interleave="bil", ext="bil", force=True, metadata=self.output_calibration_hsi.metadata)
                else:
                    assert False
                print(f"Image saved to {save_path}")
        else:
            print("No image to save.")

    def save_viewer_image(self):
        """Save the currently visualized image."""
        if self.loaded_image is not None:
            # Open a file dialog to choose the save location
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)")
            if save_path:
                if self.current_tab == "Visualization":
                    self.vis_viewer_image.save(save_path, "png")
                elif self.current_tab == "Super-resolution":
                    self.sr_viewer_image.save(save_path, "png")
                elif self.current_tab == "Calibration":
                    self.calibration_viewer_image.save(save_path, "png")
                else:
                    assert False
                print(f"Image saved to {save_path}")
        else:
            print("No image to save.")

    def eventFilter(self, widget, event):
        if event.type() == QEvent.Type.Resize:
            print("QEvent.Resize", self.current_tab, widget, event, self.width(), self.height())

            if self.current_tab == "Visualization":
                self.update_visualization_tab()
            elif self.current_tab == "Super-resolution":
                self.update_super_resolution_tab()
            elif self.current_tab == "Calibration":
                self.update_calibration_tab()
            elif self.current_tab == "Classification":
                self.update_classification_tab()
            else:
                assert False
            return True
        #if (event.type() == QtCore.QEvent.Resize and widget is self.label):
        #    self.label.setPixmap(self._pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio))
        #    return True
        return QMainWindow.eventFilter(self, widget, event)

    def get_scaled_pixmap(self, pixmap):
        #print("scaled", pixmap.scaled(self.width() - 300, self.height() - 310, Qt.AspectRatioMode.KeepAspectRatio))
        print("not scaled", pixmap.width(), pixmap.height(), self.width(), self.height())
        ret = pixmap.scaled(QSize(self.width() - 300, self.height() - 310), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
        return ret

    def set_label_scaled_pixmap(self, label, pixmap):
        print("set_label_scaled_pixmap", label, pixmap.width(), pixmap.height())
        #label.setFixedWidth(self.width() - 300)
        #label.setFixedHeight(self.height() - 310)
        #label.setGeometry(QRect(label.x(), label.y(), self.width() - 260, self.height() - 310))
        pixmap = self.get_scaled_pixmap(pixmap)
        print("set_label_scaled_pixmap", pixmap.width(), pixmap.height())
        label.setFixedWidth(pixmap.width())
        label.setFixedHeight(pixmap.height())
        label.setPixmap(pixmap)

    def visualize_selected_mode(self, mode):
        # Create a dictionary mapping modes to their corresponding functions and output file names
        mode_mapping = {
            "RGB": ("img/visualization_rgb.png", show_rgb),
            "NDVI": ("img/visualization_ndvi.png", show_ndvi),
            "EVI": ("img/visualization_evi.png", show_evi),
            "MCARI": ("img/visualization_mcari.png", show_mcari),
            "MTVI": ("img/visualization_mtvi.png", show_mtvi),
            "OSAVI": ("img/visualization_osavi.png", show_osavi),
            "PRI": ("img/visualization_pri.png", show_pri),
            "hypercube": ("img/visualization_cube.png", show_cube)
        }

        self.visualization_label.set_mode(mode)
        if mode is None:
            self.visualization_label.setText("Error: No mode selected")
            return

        # Get the file path and function for the selected mode
        save_path, visualization_function = mode_mapping[mode]

        # Call the corresponding visualization function
        try:
            visualization_function(self.hsi, save_path)
            pixmap = QPixmap(save_path)
            #self.visualization_label.setPixmap(pixmap)
            self.set_label_scaled_pixmap(self.visualization_label, pixmap)
            self.visualization_label.setScaledContents(True)
            self.vis_viewer_image = self.get_scaled_pixmap(pixmap)
        except Exception as e:
            self.visualization_label.setText(f"Error visualizing {mode}: {str(e)}")

        self.tab_state["Visualization"] = 2

    def create_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Visualization image label
        self.visualization_label = ClickableImage(self)
        self.visualization_label.setText("APPN-Tech")
        self.visualization_label.setFixedHeight(539)
        self.visualization_label.setFixedWidth(700)
        self.visualization_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.visualization_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label.setStyleSheet("font-size: 36px; font-weight: bold; color: grey;")
        layout.addWidget(self.visualization_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add a spacer to push the banner to the bottom
        layout.addStretch(1)

        # File path label (same layout as Super-resolution tab)
        file_layout = QHBoxLayout()
        self.visualization_file_label = QLabel("File path: No image loaded")
        file_layout.addWidget(self.visualization_file_label)
        layout.addLayout(file_layout)

        # Visualization Mode Options
        mode_group = QGroupBox("Mode:")
        mode_layout_top = QHBoxLayout()
        mode_layout_bottom = QHBoxLayout()

        # Add radio buttons for each visualization mode
        self.radio_rgb = QRadioButton("RGB")
        self.radio_ndvi = QRadioButton("NDVI")
        self.radio_evi = QRadioButton("EVI")
        self.radio_mcari = QRadioButton("MCARI")
        self.radio_mtvi = QRadioButton("MTVI")
        self.radio_osavi = QRadioButton("OSAVI")
        self.radio_pri = QRadioButton("PRI")
        self.radio_cube = QRadioButton("HyperCube")

        # Connect buttons to visualization function
        self.radio_rgb.clicked.connect(lambda: self.visualize_selected_mode("RGB"))
        self.radio_ndvi.clicked.connect(lambda: self.visualize_selected_mode("NDVI"))
        self.radio_evi.clicked.connect(lambda: self.visualize_selected_mode("EVI"))
        self.radio_mcari.clicked.connect(lambda: self.visualize_selected_mode("MCARI"))
        self.radio_mtvi.clicked.connect(lambda: self.visualize_selected_mode("MTVI"))
        self.radio_osavi.clicked.connect(lambda: self.visualize_selected_mode("OSAVI"))
        self.radio_pri.clicked.connect(lambda: self.visualize_selected_mode("PRI"))
        self.radio_cube.clicked.connect(lambda: self.visualize_selected_mode("hypercube"))

        mode_layout_top.addWidget(self.radio_rgb)
        mode_layout_top.addWidget(self.radio_ndvi)
        mode_layout_top.addWidget(self.radio_evi)
        mode_layout_top.addWidget(self.radio_mcari)
        mode_layout_bottom.addWidget(self.radio_mtvi)
        mode_layout_bottom.addWidget(self.radio_osavi)
        mode_layout_bottom.addWidget(self.radio_pri)
        mode_layout_bottom.addWidget(self.radio_cube)

        mode_layout_vertical = QVBoxLayout()
        mode_layout_vertical.addLayout(mode_layout_top)
        mode_layout_vertical.addLayout(mode_layout_bottom)
        mode_group.setLayout(mode_layout_vertical)

        layout.addWidget(mode_group)

        # Add the page to the stack
        self.stack.addWidget(page)

    def show_resolution_image(self, resolution):
        if resolution == "low":
            if self.hsi is not None:
                print(f"显示{resolution}分辨率图像")
                save_path = f"img/visualization_{resolution}_res.png"
                try:
                    show_rgb(self.hsi, save_path)
                    pixmap = QPixmap(save_path)
                    #self.visualization_label_sr.setPixmap(pixmap)
                    #self.visualization_label_sr.setPixmap(self.get_scaled_pixmap(pixmap))
                    self.set_label_scaled_pixmap(self.visualization_label_sr, pixmap)
                    self.visualization_label_sr.setScaledContents(True)
                    self.sr_viewer_image = self.get_scaled_pixmap(pixmap)
                except Exception as e:
                    self.visualization_label_sr.setText(f"显示{resolution}分辨率图像时出错：{str(e)}")
            else:
                self.visualization_label_sr.setText("请先加载图像")
        if resolution == "high":
            image_path = 'temp_sr/result_hsidata/result.bil'
            header_path = 'temp_sr/result_hsidata/result.hdr'
            high_hsi = load_hsi(image_path,header_path)
            if high_hsi is not None:
                print(f"显示{resolution}分辨率图像")
                save_path = f"img/visualization_{resolution}_res.png"
                try:
                    show_rgb(high_hsi, save_path)
                    pixmap = QPixmap(save_path)
                    #self.visualization_label_sr.setPixmap(pixmap)
                    #self.visualization_label_sr.setPixmap(self.get_scaled_pixmap(pixmap))
                    self.set_label_scaled_pixmap(self.visualization_label_sr, pixmap)
                    self.visualization_label_sr.setScaledContents(True)
                    self.sr_viewer_image = self.get_scaled_pixmap(pixmap)
                except Exception as e:
                    self.visualization_label_sr.setText(f"显示{resolution}分辨率图像时出错：{str(e)}")

    def handle_super_resolution(self):
        if self.radio_super_res.isChecked():
            print("开始超分辨率处理")
            desired_path = os.path.dirname(self.image_path) + os.path.sep
            temp_folders = ['temp_sr/ori_matdata/', 'temp_sr/mid_matdata/', 'temp_sr/result_matdata/',
                            'temp_sr/result_hsidata/']
            for folder in temp_folders:
                full_path = os.path.join(os.getcwd(), folder)
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                os.makedirs(full_path)

            self.progress_bar.setValue(0)
            self.current_progress = 0
            self.target_progress = 0

            def update_progress(speed=0.05):
                while self.current_progress < self.target_progress:
                    self.current_progress += speed  # 每次增加0.5%
                    self.progress_bar.setValue(int(self.current_progress))
                    time.sleep(0.05)

            def progress_callback(message):
                print(f"进度更新: {message}")
                if message == "完成":
                    self.target_progress = 100
                    threading.Thread(target=update_progress, args=(1,), daemon=True).start()
                    QTimer.singleShot(1000, self.show_super_resolution_result)
                else:
                    self.target_progress = min(self.target_progress + 25, 99)
                    threading.Thread(target=update_progress, daemon=True).start()
                QApplication.processEvents()

            run_super_resolution(desired_path, temp_folders[0], temp_folders[1], temp_folders[2], temp_folders[3],
                                 callback=progress_callback)
        else:
            print("取消超分辨率处理")

    def show_super_resolution_result(self):
        # 显示超分辨率处理的结果
        self.show_resolution_image("high")
        self.radio_high_res.setChecked(True)
        self.radio_low_res.setChecked(False)

    def create_super_resolution_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.visualization_label_sr = QLabel("Visualization Content")
        self.visualization_label_sr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label_sr.setFixedHeight(539)  # Set an appropriate height or let it scale with content
        self.visualization_label_sr.setFixedWidth(700)
        self.visualization_label_sr.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.visualization_label_sr, alignment=Qt.AlignmentFlag.AlignCenter)  # Center the image

        layout.addStretch(1)

        # 文件路径输入
        file_layout = QHBoxLayout()
        self.super_resolution_file_label = QLabel("File path: No image loaded")
        file_layout.addWidget(self.super_resolution_file_label)
        layout.addLayout(file_layout)

        # 超分辨率选项
        options_group = QGroupBox()
        options_layout = QHBoxLayout()

        self.radio_super_res = QPushButton("Super Resolution")
        self.radio_super_res.setCheckable(True)
        self.radio_super_res.setChecked(False)
        self.radio_low_res = QRadioButton("Low Res")
        self.radio_high_res = QRadioButton("High Res")

        self.radio_super_res.clicked.connect(self.handle_super_resolution)
        self.radio_low_res.clicked.connect(lambda: self.show_resolution_image("low"))
        self.radio_high_res.clicked.connect(lambda: self.show_resolution_image("high"))

        options_layout.addWidget(self.radio_super_res)
        options_layout.addWidget(self.radio_low_res)
        options_layout.addWidget(self.radio_high_res)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stack.addWidget(page)

    def run_calibration(self):
        # Load and set file path for the calibration
        if self.image_path:
            self.calibration_file_label.setText(f"File path: {self.image_path}")
        else:
            self.calibration_file_label.setText("File path: No image loaded")
            self.calibration_image_label.setText("File path: No image loaded")
            return

        dark_filename = self.calibration_input_mapping["dark_file"].text()
        ref_filename = self.calibration_input_mapping["ref_file"].text()
        output_filename = None

        for filename in [dark_filename, ref_filename]:
            if filename is None:
                print("Error: Missing file paths")
                self.calibration_image_label.setText("Error: Missing file paths")
                return

        input_bil = self.image_path
        input_hdr = ".".join(self.image_path.split(".")[: -1] + ["hdr"])

        dark_hdr = f"{dark_filename}.hdr"
        dark_bil = f"{dark_filename}.bil"
        ref_hdr = f"{ref_filename}.hdr"
        ref_bil = f"{ref_filename}.bil"
        for fname in [dark_hdr, dark_bil, ref_hdr, ref_bil, input_hdr, input_bil]:
            if not os.path.exists(fname):
                print(f"Error: {fname} not found")
                self.calibration_image_label.setText("Error: {fname} not found")

        try:
            dark_hsi = load_hsi(dark_bil, dark_hdr)
            ref_hsi = load_hsi(ref_bil, ref_hdr)
            input_hsi = load_hsi(input_bil, input_hdr)
        except Exception as _:
            print("Load file error")
            self.calibration_image_label.setText(f"Load file error")

        try:
            # Call calibration function in Functions module
            result_hsi = calibration(dark_hsi, ref_hsi, input_hsi, output_filename)

            # Load the result
            result_image_path = "calibration.png" if output_filename is None else (output_filename + ".png")
            tuple_rgb_bands = find_RGB_bands([float(i) for i in result_hsi.metadata['wavelength']])
            rgb_image = get_rgb(result_hsi, tuple_rgb_bands)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            show_rgb(result_hsi, result_image_path)

            # Convert to QImage for display
            height, width, _ = rgb_image.shape
            bytes_per_line = 3 * width
            pixmap = QPixmap(result_image_path)

            # Update the label to display the calibration result
            #self.calibration_image_label.setPixmap(pixmap)
            #self.calibration_image_label.setPixmap(self.get_scaled_pixmap(pixmap))
            self.set_label_scaled_pixmap(self.calibration_image_label, pixmap)
            self.calibration_image_label.setScaledContents(True)
            self.calibration_viewer_image = self.get_scaled_pixmap(pixmap)

            # Remove temporary files
            if output_filename is None:
                os.remove(result_image_path)

        except Exception as _:
            print(f"Calibration error")
            self.calibration_image_label.setText(f"Calibration error")

        self.tab_state["Calibration"] = 2

    def create_calibration_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Label to display calibration result image
        self.calibration_image_label = QLabel("Calibration Result", alignment=Qt.AlignmentFlag.AlignCenter)
        self.calibration_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_image_label.setFixedHeight(539)  # Set an appropriate height or let it scale with content
        self.calibration_image_label.setFixedWidth(700)
        self.calibration_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.calibration_image_label, alignment=Qt.AlignmentFlag.AlignCenter)  # Center the image

        layout.addStretch(1)

        # File path label (same layout as Super-resolution tab)
        file_layout = QHBoxLayout()
        self.calibration_file_label = QLabel("File path: No image loaded")

        file_layout.addWidget(self.calibration_file_label)
        layout.addLayout(file_layout)

        calibration_display_text_mapping = {
            "dark_file": "Dark File",
            "ref_file": "Reference File",
        }
        calibration_file_types = calibration_display_text_mapping.keys()
        calibration_default_input = {
            "dark_file": "dark",
            "ref_file": "ref",
        }
        self.calibration_input_mapping = dict()

        def calibration_browse_file(line_edit):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open File", None, "Header Files (*.hdr);;BIL Files (*.bil)")
            if file_path:
                file_path = ".".join(file_path.split(".")[: -1])
                line_edit.setText(file_path)

        def connect_fn_generator(file_type):
            ft = copy.copy(file_type)
            return lambda: calibration_browse_file(self.calibration_input_mapping[ft])

        for file_type in calibration_file_types:
            current_layout = QHBoxLayout()
            current_label = QLabel(calibration_display_text_mapping[file_type])
            self.calibration_input_mapping[file_type] = QLineEdit(calibration_default_input[file_type]) if file_type in calibration_default_input else QLineEdit()

            current_button = QPushButton("Browse")
            current_button.clicked.connect(connect_fn_generator(file_type))

            current_layout.addWidget(current_label)
            current_layout.addWidget(self.calibration_input_mapping[file_type])
            current_layout.addWidget(current_button)

            layout.addLayout(current_layout)

        # Calibrate button
        calibrate_button = QPushButton("Calibrate")
        calibrate_button.clicked.connect(self.run_calibration)

        # Layout organization
        layout.addWidget(calibrate_button)

        self.stack.addWidget(page)


    def browse_file(self, line_edit):
        """Helper function to browse and set file paths."""
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', None, "All Files (*.*)")
        if file_path:
            line_edit.setText(file_path)

    def create_classification_page(self):
        page = QWidget()
        page.setObjectName("Classification")

        # Create main layout
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        # Create NDVI display label
        self.ndvi_display = QLabel("No selection")
        self.ndvi_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ndvi_display.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Use ClassificationImageLabel instead of QLabel
        self.visualization_label_class = ClassificationImageLabel(self.ndvi_display, self)
        self.visualization_label_class.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label_class.setFixedHeight(539)
        self.visualization_label_class.setFixedWidth(700)
        self.visualization_label_class.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.visualization_label_class.setText("No image loaded")
        main_layout.addWidget(self.visualization_label_class, alignment=Qt.AlignmentFlag.AlignCenter)  # Center the image

        # --------- 下部操作面板与 NDVI 显示窗口 ---------
        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(20)  # 设置左右间距

        # --------- 操作面板左侧（占约2/3空间） ---------
        left_op_layout = QVBoxLayout()
        left_op_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 文件路径显示
        self.classification_inputfile_label = QLabel("File path: No image loaded")
        left_op_layout.addWidget(self.classification_inputfile_label)

        # Tab Widget
        self.tab_widget = QTabWidget()
        unsupervised_tab = QWidget()
        supervised_tab = QWidget()

        self.tab_widget.addTab(unsupervised_tab, "Unsupervised")
        self.tab_widget.addTab(supervised_tab, "Supervised")

        # --------- 无监督标签页内容 ---------
        unsupervised_layout = QVBoxLayout(unsupervised_tab)

        # Num of Classes 输入
        num_classes_layout = QHBoxLayout()
        num_classes_label = QLabel("Num of Classes:")
        self.num_classes_input = QSpinBox()
        self.num_classes_input.setMinimum(1)
        self.num_classes_input.setValue(5)
        num_classes_layout.addWidget(num_classes_label)
        num_classes_layout.addWidget(self.num_classes_input)
        unsupervised_layout.addLayout(num_classes_layout)

        # Max Iterations 输入
        max_iterations_layout = QHBoxLayout()
        max_iterations_label = QLabel("Max Iterations:")
        self.max_iterations_input = QSpinBox()
        self.max_iterations_input.setMinimum(1)
        self.max_iterations_input.setValue(10)
        max_iterations_layout.addWidget(max_iterations_label)
        max_iterations_layout.addWidget(self.max_iterations_input)
        unsupervised_layout.addLayout(max_iterations_layout)

        # Classify 按钮
        unsupervised_classify_button = QPushButton("Classify")
        unsupervised_classify_button.setFixedWidth(100)
        unsupervised_classify_button.clicked.connect(self.run_unsupervised_classification)
        unsupervised_layout.addWidget(unsupervised_classify_button)

        # 进度条
        self.unsupervised_progress_bar = QProgressBar()
        self.unsupervised_progress_bar.setValue(0)
        unsupervised_layout.addWidget(self.unsupervised_progress_bar)

        # --------- 有监督标签页内容 ---------
        supervised_layout = QVBoxLayout(supervised_tab)

        # Groundtruth 输入
        groundtruth_layout = QHBoxLayout()
        groundtruth_label = QLabel("Groundtruth:")
        self.groundtruth_input = QLineEdit("One_sample/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1_mask.jpg")
        groundtruth_layout.addWidget(groundtruth_label)
        groundtruth_layout.addWidget(self.groundtruth_input)
        supervised_layout.addLayout(groundtruth_layout)

        # Classifier 选择
        classifier_layout = QHBoxLayout()
        classifier_label = QLabel("Classifier:")
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItem("Gaussian")
        self.classifier_combo.addItem("Mahalanobis")
        self.classifier_combo.addItem("Perceptron")

        classifier_layout.addWidget(classifier_label)
        classifier_layout.addWidget(self.classifier_combo)
        classifier_layout.addStretch()
        supervised_layout.addLayout(classifier_layout)

        # Supervised Classify 按钮
        supervised_classify_button = QPushButton("Classify")
        supervised_classify_button.setFixedWidth(100)
        supervised_classify_button.clicked.connect(self.run_supervised_classification)
        supervised_layout.addWidget(supervised_classify_button)

        # 将 Tab Widget 添加到操作面板左侧布局
        left_op_layout.addWidget(self.tab_widget)

        # 将操作面板左侧布局添加到下部布局中，并设置伸缩因子为2
        lower_layout.addLayout(left_op_layout, 2)

        # --------- NDVI 显示窗口右侧（占约1/3空间） ---------
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # NDVI display group
        self.ndvi_display_group = QGroupBox("NDVI Value")
        ndvi_layout = QVBoxLayout()
        ndvi_layout.addWidget(self.ndvi_display)
        self.ndvi_display_group.setLayout(ndvi_layout)

        # 设置 GroupBox 的样式
        self.ndvi_display_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
            }
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 10px;
            }
        """)

        # 将 NDVI 显示组添加到右侧布局
        right_layout.addWidget(self.ndvi_display_group)
        right_layout.addStretch(1)  # 添加伸缩以填充剩余空间

        # 将右侧布局添加到下部布局中，并设置伸缩因子为1
        lower_layout.addLayout(right_layout, 1)

        # 将下部布局添加到主布局中
        main_layout.addLayout(lower_layout)

        # 添加伸缩以填充剩余空间（可选）
        main_layout.addStretch(1)

        # 将 Classification 页面添加到 QStackedWidget
        self.stack.addWidget(page)
        # print("Classification page created and added to stack.")  # 调试信息

    def unsupervised_classification_finished(self, pixmap, cluster_map, ndvi, color_img):
        """
        接收分类完成信号后：
        1) 停止进度条
        2) 标记已分类
        3) 注入数据到 ClassificationImageLabel 并渲染
        """
        # 停 busy bar
        self.unsupervised_progress_bar.setRange(0, 1)
        self.unsupervised_progress_bar.setValue(1)

        # 标记分类完成
        self.classification_done = True

        # —— 新增：先生成底层 RGB 图阵列 ——
        hsi_arr = self.unsupervised_worker.hsi_data
        wavelengths = self.unsupervised_worker.wavelengths
        # 选三波段生成 RGB
        rgb_bands = find_RGB_bands(wavelengths)
        rgb_arr = get_rgb(hsi_arr, rgb_bands)
        rgb_arr = (rgb_arr * 255).astype(np.uint8)
        self.visualization_label_class.set_base_rgb_image(rgb_arr)

        # 注入聚类结果和 NDVI
        self.visualization_label_class.set_cluster_map(cluster_map)
        self.visualization_label_class.set_ndvi(ndvi)
        # 最后一次性渲染
        self.visualization_label_class.update_display()
        self.visualization_label_class.update_ndvi_display()



    def unsupervised_classification_error(self, error_message):
        self.visualization_label_class.setText(f"Error: {error_message}")
        # 将进度条设置为确定状态并重置为 0%
        self.unsupervised_progress_bar.setRange(0, 100)
        self.unsupervised_progress_bar.setValue(0)

    def run_unsupervised_classification(self):
        try:
            if self.hsi is None:
                self.visualization_label_class.setText("No image loaded.")
                return

            # 1) Extract wavelengths
            if hasattr(self.hsi, 'metadata') and 'wavelength' in self.hsi.metadata:
                wavelengths = [float(w) for w in self.hsi.metadata['wavelength']]
            else:
                QMessageBox.warning(self, "Wavelengths Error",
                                    "Cannot retrieve wavelength metadata.")
                return

            # 2) Preprocess: crop HSI only
            try:
                hsi_cropped = preprocess_hsi_with_apriltag_multiscale(
                    self.hsi, self.image_path
                )
            except ValueError as ve:
                QMessageBox.warning(self, "Apriltag Detection Error", str(ve))
                return

            # Update HSI to cropped version
            self.hsi = hsi_cropped

            # 3) Show busy indicator
            self.visualization_label_class.setText("Classification in progress.")
            self.unsupervised_progress_bar.setRange(0, 0)

            # 4) Start worker with 4 parameters (no mask)
            k = self.num_classes_input.value()
            max_iter = self.max_iterations_input.value()
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
            self.visualization_label_class.setText(f"Error: {e}")

    def run_supervised_classification(self):
        """Load and classify the image, display in classification tab."""
        groundtruth_path = self.groundtruth_input.text()
        header_path = self.image_path.replace('.bil', '.hdr')

        # Get the selected classifier type from the combo box
        selected_classifier = self.classifier_combo.currentText()

        # Load and classify the image based on the selected classifier
        self.classifier.load_image(self.image_path, header_path)
        result_image_path = self.classifier.classify(groundtruth_path, classifier_type=selected_classifier)

        # Load the classified image as QPixmap
        pixmap = QPixmap(result_image_path)
        #self.visualization_label_class.setPixmap(pixmap)
        #self.visualization_label_class.setPixmap(self.get_scaled_pixmap(pixmap))
        self.set_label_scaled_pixmap(self.visualization_label_class, pixmap)
        self.visualization_label_class.setScaledContents(True)

        # Store the loaded image in self.loaded_image for visualization tab
        self.loaded_image = pixmap

    def update_visualization_tab(self):
        """Update the visualization tab with the loaded RGB image from the .bil file."""
        state = self.tab_state["Visualization"]
        if state == 0:
            self.visualization_file_label.setText("File path: No image loaded")
            self.visualization_label.setText("No image loaded")
            self.save_hyperspectral_image_action.setDisabled(True)
            self.save_viewer_image_action.setDisabled(True)
        elif state == 1:
            self.visualization_file_label.setText(f"File path: {self.image_path}")
            pixmap = self.loaded_image
            #self.visualization_label.setPixmap(pixmap)
            #self.visualization_label.setPixmap(self.get_scaled_pixmap(pixmap))
            self.set_label_scaled_pixmap(self.visualization_label, pixmap)
            self.visualization_label.setScaledContents(True)
            self.radio_rgb.setChecked(True)
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        elif state == 2:
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        else:
            assert False

    def update_super_resolution_tab(self):
        state = self.tab_state["Super-resolution"]
        if state == 0:
            self.super_resolution_file_label.setText("File path: No image loaded")
            self.visualization_label_sr.setText("No image loaded")
            self.save_hyperspectral_image_action.setDisabled(True)
            self.save_viewer_image_action.setDisabled(True)
        elif state == 1:
            self.super_resolution_file_label.setText(f"File path: {self.image_path}")
            self.show_resolution_image("low")
            self.radio_low_res.setChecked(True)
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        elif state == 2:
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        else:
            assert False

    def update_calibration_tab(self):
        """Update the calibration tab with the loaded image path."""
        state = self.tab_state["Calibration"]
        if state == 0:
            self.calibration_file_label.setText("File path: No image loaded")
            self.calibration_image_label.setText("No image loaded")
            self.save_hyperspectral_image_action.setDisabled(True)
            self.save_viewer_image_action.setDisabled(True)
        elif state == 1:
            self.calibration_file_label.setText(f"File path: {self.image_path}")
            #self.calibration_image_label.setPixmap(self.loaded_image)
            #self.calibration_image_label.setPixmap(self.get_scaled_pixmap(self.loaded_image))
            pixmap = self.calibration_viewer_image
            self.set_label_scaled_pixmap(self.calibration_image_label, pixmap)
            self.calibration_image_label.setScaledContents(True)
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        elif state == 2:
            pixmap = self.calibration_viewer_image
            self.set_label_scaled_pixmap(self.calibration_image_label, pixmap)
            self.calibration_image_label.setScaledContents(True)
            self.save_hyperspectral_image_action.setDisabled(False)
            self.save_viewer_image_action.setDisabled(False)
        else:
            assert False

    def update_classification_tab(self):
        """
        切到 Classification 页时：
        - 如果未分类，显示原始 RGB；
        - 如果已分类，保持现有显示（分类结果）。
        """
        if not self.loaded_image:
            # 无图
            self.classification_inputfile_label.setText("File path: No image loaded")
            self.visualization_label_class.setText("No image loaded")
        else:
            # 有图：更新路径
            self.classification_inputfile_label.setText(f"File path: {self.image_path} ")
            if not self.classification_done:
                # 尚未分类，显示原始 RGB
                self.set_label_scaled_pixmap(self.visualization_label_class, self.loaded_image)
                self.visualization_label_class.setScaledContents(True)
            # 已分类时不覆盖，让 unsupervised_classification_finished 保持的结果留在界面上
        # 分类页按钮状态
        self.save_hyperspectral_image_action.setDisabled(True)
        self.save_viewer_image_action.setDisabled(True)

    def change_page(self, button_text):
        """Switch between pages and update the visualization tab if necessary."""
        index = ["Visualization", "Super-resolution", "Calibration", "Classification"].index(button_text)
        self.stack.setCurrentIndex(index)
        self.current_tab = button_text

        if button_text == "Visualization":
            self.update_visualization_tab()
        elif button_text == "Super-resolution":
            self.update_super_resolution_tab()
        elif button_text == "Calibration":
            self.update_calibration_tab()
        elif button_text == "Classification":
            self.update_classification_tab()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
