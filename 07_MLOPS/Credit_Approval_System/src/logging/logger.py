import logging
import src.constants as const

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(const.LOGGER_FILE_PATH), logging.StreamHandler()],
    format="[%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s]",
)
