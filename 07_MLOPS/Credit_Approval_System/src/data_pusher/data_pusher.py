import pymongo
from src.entity.config_entity import PusherConfig, MainConfig
import requests
from src.utils.main_utils import read_data, create_dirs

from src.logging.logger import logging

logger = logging.getLogger(__name__)


class DataPusher:
    def __init__(self, pusher_config: PusherConfig):
        self.main_config = pusher_config.main_config
        self.data_pusher_download_url = pusher_config.data_pusher_download_url
        self.data_pusher_download_dir = pusher_config.data_pusher_download_dir
        self.data_pusher_download_file = pusher_config.data_pusher_download_file

    def download_data(
        self,
    ):

        try:
            logger.info("downloading data....")

            if self.data_pusher_download_file.exists():
                logger.info("data already downloaded....")
                return
            create_dirs(self.data_pusher_download_dir)

            with requests.get(
                url=self.data_pusher_download_url, stream=True, timeout=10
            ) as r:
                r.raise_for_status()

                create_dirs(self.data_pusher_download_file)
                with open(self.data_pusher_download_file, "wb") as f:
                    for chunk in r.iter_content(1024 * 100):
                        f.write(chunk)
            logger.info("data downloaded successfully")
        except requests.HTTPError as ex:
            logger.exception("data url is wrong")
            raise ex

        except Exception as ex:
            logger.exception("failed to download the data")
            raise ex

    def push_data_to_mongo(self):
        try:

            logger.info("pushing data to the database....")
            df = read_data(path=self.data_pusher_download_file)

            client = pymongo.MongoClient(host=self.main_config.mongo_conn_url)
            db = client[self.main_config.mongo_db_name]
            collection = db[self.main_config.mongo_collection_name]
            if collection.find_one():
                logger.info("data already pushed....")
                return

            collection.insert_many(df.to_dict(orient="records"))
            logger.info("data pushed successfully")
        except Exception as ex:
            logger.exception("Failed to push the data to mongo DB")
            raise ex


if __name__ == "__main__":
    try:
        logger.info("start the push data process")
        main_config = MainConfig(ts="my_dataset")
        pusher_config = PusherConfig(main_config)
        data_pusher = DataPusher(pusher_config)
        data_pusher.download_data()
        data_pusher.push_data_to_mongo()
        logger.info("end the push data process")
    except Exception as ex:
        logger.exception("error in pushing the data to mongo DB")
        raise ex
