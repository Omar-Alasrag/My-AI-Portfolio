import pandas as pd
from pathlib import Path
from src.logging.logger import logging
import joblib
import numpy as np

logger = logging.getLogger(__name__)


def create_dirs(path: str | Path):
    path = Path(path)
    logger.info(f"create dir:{path}")
    # check if it is a file through the suffix
    if path.suffix:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def is_path_exist(path: Path):
    if not path.exists():
        logger.error(f"the dir:{path} is not exist")
        raise Exception("dir is not exist")

    return True


def read_data(path: Path) -> pd.DataFrame:
    try:
        logger.info(f"reading the dataframe path:{path}")

        is_path_exist(path)
        if path.suffix == ".parquet":
            data = pd.read_parquet(path)
        elif path.suffix == ".pkl":
            data = joblib.load(path)
        else:
            data = pd.read_csv(path)
        logger.info("dataframe was read successfully")
        return data

    except Exception as ex:
        logger.exception("failed to read the dataframe")
        raise ex


def save_object(path: Path, object):
    try:
        joblib.dump(object, path)
        logger.info(f"object was written successfully path:{path}")

    except Exception as ex:
        logger.exception("failed to save the object")
        raise ex


def load_object(path: Path):
    try:
        logger.info(f"object was loaded successfully path:{path}")
        return joblib.load(path)

    except Exception as ex:
        logger.exception("failed to load the object")
        raise ex


def save_numpy_array(self, path: str, array: np.ndarray):
    try:
        np.save(path, array)
        logger.info(f"saved numpy array at:{path}")
    except Exception as ex:
        logger.exception("failed to write the numpy")
        raise ex
