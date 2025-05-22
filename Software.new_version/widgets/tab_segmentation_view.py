import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QTabWidget, QSpinBox, \
    QProgressBar, QLineEdit, QComboBox, QGroupBox, QMenu, QFileDialog, QFrame

from utils.leaf_utils.basic import plot_spectrum


class TabSegmentationView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        self._pixmap = None

        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(20)

        # Create NDVI display label
        self.ndvi_display = QLabel("No selection")
        self.ndvi_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ndvi_display.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Use ClassificationImageLabel instead of QLabel
        self.visualization_label_class = ClassificationImageLabel(self.ndvi_display, self)
        self.visualization_label_class.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label_class.setMinimumHeight(540)
        self.visualization_label_class.setMinimumWidth(700)
        self.visualization_label_class.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.visualization_label_class.setText("No image loaded")

        self.visualization_frame = QFrame(self)
        self.visualization_frame.setMinimumHeight(self.visualization_label_class.height() + 10)
        self.visualization_frame.setMinimumWidth(self.visualization_label_class.width() + 10)
        self.visualization_frame.setLineWidth(2)
        self.visualization_frame.setStyleSheet("border: 1px solid #ccc;")
        self.visualization_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.frame_layout = QVBoxLayout(self)
        self.frame_layout.addWidget(self.visualization_label_class, alignment=Qt.AlignmentFlag.AlignCenter)
        self.frame_layout.setSpacing(5)
        self.frame_layout.setStretch(0, 1)
        self.visualization_frame.setLayout(self.frame_layout)

        self.main_layout.addWidget(self.visualization_frame)

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
        self.tab_widget.setTabEnabled(1, False)  # disable supervised tab before it's done

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
        unsupervised_classify_button.clicked.connect(self.controller.run_unsupervised_classification)
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
        # supervised_classify_button.clicked.connect(self.run_supervised_classification)
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

        right_layout.addWidget(self.ndvi_display_group)
        right_layout.addStretch(1)
        lower_layout.addLayout(right_layout, 1)

        self.main_layout.setStretch(0, 2)
        self.main_layout.setStretch(1, 1)
        self.main_layout.addLayout(lower_layout)

        self.setLayout(self.main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.visualization_label_class.cluster_map is None:
            self.set_resized_pixmap()

    def set_resized_pixmap(self):
        if self._pixmap is not None:
            self.visualization_label_class.setPixmap(self._pixmap.scaled(
                self.visualization_frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

class ClassificationImageLabel(QLabel):
    def __init__(self, ndvi_display, parent=None):
        super().__init__(parent)

        self.tab_view = parent

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
        if self.cluster_map is None:
            super().mousePressEvent(event)
            return

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

    # Method to show spectrum plot
    def show_spectrum_plot(self, position):
        hsi = self.tab_view.controller.hsi
        if self.tab_view.controller.hsi is None:
            print("HSI data not available")
            return
        # Convert position to image coordinates
        x, y = self.convert_position_to_image_coords(position)
        if x is not None and y is not None:
            try:
                plot_spectrum(hsi, self.tab_view.controller.main_controller.hyperspectral_image.metadata, x, y)
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
