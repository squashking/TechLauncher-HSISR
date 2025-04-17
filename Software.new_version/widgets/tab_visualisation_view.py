from PySide6.QtWidgets import QWidget, QRadioButton, QVBoxLayout, QStackedWidget, QHBoxLayout

from widgets.image_viewer import ImageViewer


class TabVisualisationView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # Create radio buttons
        self.radio_buttons = [QRadioButton(x) for x in self.controller.modes]
        for btn in self.radio_buttons:
            btn.setDisabled(True)

        # Layout for radio buttons
        radio_layout = QVBoxLayout()
        for btn in self.radio_buttons:
            radio_layout.addWidget(btn)

        # Create different views and stack the views (only one visible at a time)
        self.stack = QStackedWidget()
        self.mode_view_mapping = dict()
        for mode in self.controller.modes:
            self.mode_view_mapping[mode] = ImageViewer(self.controller.logger)
            self.stack.addWidget(self.mode_view_mapping[mode])
        self.stack.setCurrentIndex(0)  # default

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.stack)
        main_layout.addLayout(radio_layout)
        self.setLayout(main_layout)
