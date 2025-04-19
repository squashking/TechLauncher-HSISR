from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from widgets.image_viewer import ImageViewer


class TabCroppingView(QWidget):
    def __init__(self, controller):
        super().__init__()

        self.controller = controller

        # Selectable Image View
        self.selectable_image_viewer = ImageViewer(controller.logger, self)

        # Buttons
        self.clear_selection_button = QPushButton("Clear Selection")
        self.crop_button = QPushButton("Crop")
        self.save_as_hsi_button = QPushButton("Save as HSI")

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_selection_button)
        button_layout.addWidget(self.crop_button)
        button_layout.addWidget(self.save_as_hsi_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.selectable_image_viewer)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def mouseReleaseEvent(self, event):
        if self.selectable_image_viewer.image_label.selection_rect.isEmpty():
            self.clear_selection_button.setDisabled(True)
            self.crop_button.setDisabled(True)
        else:
            self.clear_selection_button.setEnabled(True)
            self.crop_button.setEnabled(True)

        super().mouseReleaseEvent(event)
