import src.constants as const
from pathlib import Path

from src.entity.data_schema import data_schema_validator


class MainConfig:
    def __init__(self, ts):
        self.artifacts_dir = Path(const.PIPELINE_ARTIFACT_DIR) / ts
        self.mongo_conn_url = const.MONGO_CONNECTION_URL
        self.mongo_db_name = const.MONGO_DB_NAME
        self.mongo_collection_name = const.MONGO_COLLECTION_NAME


class PusherConfig:
    def __init__(self, main_config: MainConfig):
        self.main_config = main_config
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.data_pusher_download_url = const.DATA_PUSHER_DOWNLOAD_LINK
        self.data_pusher_download_dir = self.artifact_dir / const.DATA_PUSHER_MAIN_DIR
        self.data_pusher_download_file = (
            self.data_pusher_download_dir / const.DATA_PUSHER_DOWNLOAD_FILE
        )


class IngestionConfig:
    def __init__(self, main_config: MainConfig):
        self.main_config = main_config
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.ingested_main_dir = self.artifact_dir / const.DATA_INGESTION_ARTIFACT_DIR
        self.ingested_data_file = (
            self.ingested_main_dir / const.DATA_INGESTION_INGESTED_DATA_FILE
        )


class ValidationConfig:
    def __init__(self, main_config: MainConfig):
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.data_val_artifact_main_dir = (
            self.artifact_dir / const.DATA_VALIDATION_ARTIFACT_MAIN_DIR
        )
        self.data_val_valid_data_dir = (
            self.data_val_artifact_main_dir / const.DATA_VALIDATION_VALID_DATA_DIR
        )
        self.data_val_valid_data_file = (
            self.data_val_valid_data_dir / const.DATA_VALIDATION_VALID_DATA_FILE
        )
        self.data_val_invalid_data_dir = (
            self.data_val_artifact_main_dir / const.DATA_VALIDATION_INVALID_DATA_DIR
        )
        self.data_val_invalid_data_file = (
            self.data_val_invalid_data_dir / const.DATA_VALIDATION_INVALID_DATA_FILE
        )
        self.data_schema_validator = data_schema_validator


class TransformationConfig:
    def __init__(self, main_config: MainConfig):
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.data_transformation_artifact_main_dir = (
            self.artifact_dir / const.DATA_TRANSFORMATION_ARTIFACT_MAIN_DIR
        )
        self.data_transformation_file = (
            self.data_transformation_artifact_main_dir
            / const.DATA_TRANSFORMATION_DATA_FILE
        )
        self.data_transformation_cat_cols = const.DATA_TRANSFORMATION_CAT_COLS
        self.data_transformation_target_col = const.DATA_TRANSFORMATION_TARGET_COL
        self.data_transformation_unneeded_cols = const.DATA_TRANSFORMATION_UNNEEDED_COLS
        self.data_transformation_preprocessor_file = (
            self.data_transformation_artifact_main_dir
            / const.DATA_TRANSFORMATION_PREPROCESSOR_FILE
        )


class ModelTrainConfig:
    def __init__(self, main_config: MainConfig):
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.model_training_main_dir = self.artifact_dir / const.MODEL_TRAINING_MAIN_DIR
        self.model_training_model_path = (
            self.model_training_main_dir / const.MODEL_TRAINING_MODEL_FILE
        )


class ModelPredictionConfig:
    def __init__(self, main_config: MainConfig):
        self.artifact_dir = Path(main_config.artifacts_dir)
        self.model_training_main_dir = self.artifact_dir / const.MODEL_TRAINING_MAIN_DIR
        self.model_training_model_path = (
            self.model_training_main_dir / const.MODEL_TRAINING_MODEL_FILE
        )
        self.data_transformation_artifact_main_dir = (
            self.artifact_dir / const.DATA_TRANSFORMATION_ARTIFACT_MAIN_DIR
        )

        self.data_transformation_preprocessor_file = (
            self.data_transformation_artifact_main_dir
            / const.DATA_TRANSFORMATION_PREPROCESSOR_FILE
        )
