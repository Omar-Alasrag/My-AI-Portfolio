from src.entity.config_entity import MainConfig, TransformationConfig
from src.entity.artifacts_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.utils.main_utils import read_data, create_dirs, save_object
from pathlib import Path
from src.logging.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

logger = logging.getLogger(__name__)


class DataTransformation:
    def __init__(
        self,
        transformation_config: TransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):

        self.data_validation_file = Path(
            data_validation_artifact.data_val_valid_data_file
        )
        self.data_transformation_file = transformation_config.data_transformation_file
        self.data_transformation_cat_cols = (
            transformation_config.data_transformation_cat_cols
        )
        self.data_transformation_target_col = (
            transformation_config.data_transformation_target_col
        )
        self.data_transformation_unneeded_cols = (
            transformation_config.data_transformation_unneeded_cols
        )
        self.data_transformation_preprocessor_file = (
            transformation_config.data_transformation_preprocessor_file
        )

    def check_validation_status(self):

        if not self.data_validation_file.exists():
            logger.info("validation failed, data validation file is not exist")
            raise Exception("validation failed, data validation file is not exist")
        else:
            logger.info("data validation file is exist")

    def split_features(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def configure_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = self.data_transformation_cat_cols
        num_cols = X.columns.drop(cat_cols).tolist()
        num_imputer = SimpleImputer(strategy="mean")
        num_scaler = StandardScaler()
        num_tr = Pipeline([("impute", num_imputer), ("scale", num_scaler)])

        cat_imputer = SimpleImputer(strategy="most_frequent")
        ohe = OneHotEncoder(drop="first")
        cat_tr = Pipeline(
            [
                ("impute", cat_imputer),
                ("encode", ohe),
            ]
        )

        self.preprocessor = ColumnTransformer(
            [("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)],
            remainder="passthrough",
        )
        return self.preprocessor

    def transform(self, df: pd.DataFrame):
        df.drop(columns=self.data_transformation_unneeded_cols, inplace=True)
        y = df[self.data_transformation_target_col]
        X = df.drop(columns=[self.data_transformation_target_col])

        X_train, X_test, y_train, y_test = self.split_features(X, y.to_numpy())

        self.preprocessor = self.configure_preprocessor(X)

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        return X_train_processed, X_test_processed, y_train, y_test

    def initiate_data_transformation(self):
        try:
            logger.info("starting data transformation steps...")
            self.check_validation_status()
            df = read_data(self.data_validation_file)
            X_train_processed, X_test_processed, y_train, y_test = self.transform(df)

            create_dirs(self.data_transformation_file)
            save_object(
                self.data_transformation_file,
                (X_train_processed, X_test_processed, y_train, y_test),
            )

            create_dirs(self.data_transformation_preprocessor_file)
            save_object(self.data_transformation_preprocessor_file, self.preprocessor)
            logger.info("data transformation completed successfully.")

            return DataTransformationArtifact(
                self.data_transformation_file,
                self.data_transformation_preprocessor_file,
            )
        except Exception as ex:
            logger.exception("error caught during pipeline execution")
            raise ex


if __name__ == "__main__":
    try:
        logger.info("start the data transformation process")
        transformation_config = TransformationConfig(main_config=MainConfig())
        data_transformation = DataTransformation(
            transformation_config,
            DataValidationArtifact(
                r"artifacts\validation\valid\credit_approval_valid.parquet"
            ),
        )
        data_transformation.initiate_data_transformation()
        logger.info("end the data transformation process")

    except:
        logger.exception("error in data transformation")
        raise
