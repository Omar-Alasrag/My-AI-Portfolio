from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    ingested_data_file: str


@dataclass
class DataValidationArtifact:
    data_val_valid_data_file: str


@dataclass
class DataTransformationArtifact:
    data_transformation_file: str
    data_transformation_preprocessor_file: str


@dataclass
class ModelTrainArtifacts:
    trasnformed_data_file: str
