from pandera.errors import SchemaError


from src.entity.config_entity import ValidationConfig, MainConfig
from src.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact
from src.utils.main_utils import read_data, create_dirs

from src.logging.logger import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(
        self,
        validation_config: ValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.ingested_data_path = Path(ingestion_artifact.ingested_data_file)
        self.data_schema_validator = validation_config.data_schema_validator
        self.data_val_invalid_data_file = validation_config.data_val_invalid_data_file
        self.data_val_valid_data_file = validation_config.data_val_valid_data_file

    def initiate_data_validation(self):
        try:
            df = read_data(self.ingested_data_path)
            valid_df = self.data_schema_validator.validate(df, inplace=True)
            create_dirs(self.data_val_valid_data_file)
            valid_df.to_parquet(self.data_val_valid_data_file)
            return DataValidationArtifact(self.data_val_valid_data_file)

        except SchemaError:
            logger.exception("data schema is not valid")
            create_dirs(self.data_val_invalid_data_file)
            df.to_parquet(self.data_val_invalid_data_file)
            raise
        except:
            logger.exception("error happen in initiating data validation class")
            raise


if __name__ == "__main__":
    try:
        logger.info("start the data validation process")
        validation_config = ValidationConfig(main_config=MainConfig())
        data_validation = DataValidation(
            validation_config,
            DataIngestionArtifact(r"artifacts\ingestion\credit_approval.csv"),
        )
        data_validation.initiate_data_validation()
        logger.info("end the data validation process")

    except:
        logger.exception("error in data validation")
        raise
