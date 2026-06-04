from src.entity.artifacts_entity import DataIngestionArtifact
from src.entity.config_entity import IngestionConfig, MainConfig
from src.utils.main_utils import create_dirs
import pymongo
import pandas as pd
from src.logging.logger import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, ingestion_config: IngestionConfig):
        self.ingested_main_dir = ingestion_config.ingested_main_dir
        self.ingested_data_file = ingestion_config.ingested_data_file
        self.client = pymongo.MongoClient(
            ingestion_config.main_config.mongo_conn_url, socketTimeoutMS=8000
        )
        self.db_name = ingestion_config.main_config.mongo_db_name
        self.collection_name = ingestion_config.main_config.mongo_collection_name

    def check_db_connction(self):
        logger.info("checking the db connection")

        db_names = self.client.list_database_names()
        if not db_names:
            logger.info("there are no databases!!!")
        if self.db_name not in db_names:
            logger.error(f"databases:{self.db_name} doesn't exist")
            raise Exception(f"databases:{self.db_name} doesn't exist")

        db = self.client[self.db_name]
        collection_names = db.list_collection_names()
        if self.collection_name not in collection_names:
            logger.warning(
                f"collection:{self.collection_name}' does not exist inside {self.db_name}"
            )
            raise Exception(
                f"collection:{self.collection_name}' does not exist inside {self.db_name}"
            )
        logger.info("the db connection is successful")

    def read_data_from_mongo(self) -> pd.DataFrame:

        logger.info("reading mongo DB data")

        db = self.client[self.db_name]
        collection_name = db[self.collection_name]
        df = pd.DataFrame(list(collection_name.find({}, {"_id": 0}).limit(10000)))

        logger.info("mongo DB data was read successfully")
        return df

    def initiate_data_ingestion(self):

        self.check_db_connction()
        df = self.read_data_from_mongo()
        create_dirs(self.ingested_main_dir)
        logger.debug(self.ingested_data_file)
        df.to_csv(self.ingested_data_file, index=None)
        return DataIngestionArtifact(self.ingested_data_file)


if __name__ == "__main__":
    try:
        logger.info("start the data ingestion process")
        main_config = MainConfig()
        ingestion_config = IngestionConfig(main_config)
        data_ingestion = DataIngestion(ingestion_config)
        data_ingestion.initiate_data_ingestion()
        logger.info("end the data ingestion process")

    except:
        logger.exception("error in data ingestions")
        raise
