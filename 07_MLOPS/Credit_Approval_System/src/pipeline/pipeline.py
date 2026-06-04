from src.components.ingestion import DataIngestion
from src.components.validation import DataValidation
from src.components.transformation import DataTransformation
from src.components.train import ModelTraining
from src.entity.config_entity import (
    MainConfig,
    IngestionConfig,
    ValidationConfig,
    TransformationConfig,
    ModelTrainConfig,
)

from src.logging.logger import logging
from datetime import datetime
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        pass

    def start_pipeline(self, ts=datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")):
        # ingestion
        try:
            logger.info("start the data ingestion process")
            main_config = MainConfig(ts=ts)
            ingestion_config = IngestionConfig(main_config)
            data_ingestion = DataIngestion(ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("end the data ingestion process")

        except:
            logger.exception("error in data ingestions")
            raise

        # validation
        try:
            logger.info("start the data validation process")
            validation_config = ValidationConfig(main_config)
            data_validation = DataValidation(
                validation_config,
                data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("end the data validation process")

        except:
            logger.exception("error in data validation")
            raise

        # transformation
        try:
            logger.info("start the data transformation process")
            transformation_config = TransformationConfig(main_config)
            data_transformation = DataTransformation(
                transformation_config, data_validation_artifact
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logger.info("end the data transformation process")

        except:
            logger.exception("error in data transformation")
            raise

        # training
        try:
            logger.info("start the model_train process")
            model_train_config = ModelTrainConfig(main_config)
            model_train = ModelTraining(
                model_train_config, data_transformation_artifact
            )
            model_train.initiate_model_training()
            logger.info("end the model training process")

        except:
            logger.exception("error in model training")
            raise
