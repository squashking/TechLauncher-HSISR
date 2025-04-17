import logging
import os
import sys
import time

from controllers.main_controller import MainController


def main():
    logger = init_logger()
    main_controller = MainController(logger, sys.argv)

    main_controller.main_window.show()
    sys.exit(main_controller.app.exec())


def init_logger() -> logging.Logger:
    log_dir = "logs"
    start_time = int(time.time() * 1000)

    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        pass
    else:
        try:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            else:
                os.remove(log_dir)
                os.mkdir(log_dir)
        except OSError:
            print("Creation of the directory logs failed")
            raise OSError

    logging.basicConfig(
        filename=f"{log_dir}/log.{start_time}",
        filemode="w",
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
        level=logging.DEBUG,
    )
    logger = logging.getLogger("main_logger")

    return logger


if __name__ == "__main__":
    main()
