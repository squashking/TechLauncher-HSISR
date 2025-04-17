import typing

if typing.TYPE_CHECKING:
    import logging
    from controllers.main_controller import MainController


class BaseController:
    def __init__(self, logger: "logging.Logger", main_controller: "MainController"):
        self.logger = logger
        self.main_controller = main_controller
