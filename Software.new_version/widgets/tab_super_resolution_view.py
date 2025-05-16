from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QProgressBar, QGroupBox, \
    QRadioButton, QStackedWidget

from widgets.image_viewer import ImageViewer


class TabSuperResolutionView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # stack
        self.stack = QStackedWidget()
        self.vis_lr_label = ImageViewer(self.controller.logger)
        self.vis_lr_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack.addWidget(self.vis_lr_label)
        self.vis_sr_label = ImageViewer(self.controller.logger)
        self.vis_sr_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stack.addWidget(self.vis_sr_label)
        self.stack.setCurrentIndex(0)  # Low Resolution

        # options
        self.button_super_res = QPushButton("Run Super Resolution")
        self.radio_low_res = QRadioButton("Low Resolution")
        self.radio_high_res = QRadioButton("High Resolution")
        self.button_super_res.clicked.connect(self.controller.handle_super_resolution)
        self.radio_low_res.clicked.connect(self.controller.show_low_resolution)
        self.radio_high_res.clicked.connect(self.controller.show_high_resolution)

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.button_super_res)
        options_layout.addWidget(self.radio_low_res)
        options_layout.addWidget(self.radio_high_res)

        options_group = QGroupBox()
        options_group.setLayout(options_layout)

        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # main_layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        main_layout.addWidget(options_group)
        main_layout.addWidget(self.progress_bar)
        self.setLayout(main_layout)
