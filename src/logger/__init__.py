import logging
import os
import sys

from datetime import datetime

LOG_FILE_NAME = (
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
)

logs_dir_path = os.path.join(
    os.getcwd(),
    "logs"
)

os.makedirs(
    logs_dir_path,
    exist_ok=True
)

LOG_FILE_PATH = os.path.join(
    logs_dir_path,
    LOG_FILE_NAME
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "[ %(asctime)s ] "
        "%(name)s - %(levelname)s - %(message)s"
    ),
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)