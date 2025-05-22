import logging
import typing

from PySide6.QtWidgets import QStatusBar

from controllers.base_controller import BaseController
if typing.TYPE_CHECKING:
    from controllers.main_controller import MainController


class StatusBarController(BaseController):
    def __init__(self, logger: logging.Logger, main_controller: "MainController"):
        super().__init__(logger, main_controller)
        self.main_window = main_controller.main_window

        self.status_bar = QStatusBar(self.main_window)
        self.status_bar.setObjectName(u"StatusBar")
        self.main_window.setStatusBar(self.status_bar)
