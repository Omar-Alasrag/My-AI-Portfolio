from src.entity.config_entity import MainConfig, ModelTrainConfig
from src.entity.artifacts_entity import DataTransformationArtifact
from src.utils.main_utils import read_data, create_dirs, save_object
from pathlib import Path
from src.logging.logger import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import optuna as opt
import mlflow
from urllib.parse import urlparse
import dagshub

logger = logging.getLogger(__name__)

dagshub.init(repo_owner="alasrag013", repo_name="Credit-Approval", mlflow=True)


class ModelTraining:
    def __init__(
        self,
        model_train_config: ModelTrainConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_training_model_path = model_train_config.model_training_model_path

        self.data_transformation_file = Path(
            data_transformation_artifact.data_transformation_file
        )
        self.data_transformation_preprocessor_file = Path(
            data_transformation_artifact.data_transformation_preprocessor_file
        )
        self.algorithms = {
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
            "SVC": SVC,
            "MLPClassifier": MLPClassifier,
        }

    def choose_algorithms_with_params(self, trial: opt.Trial):

        algorithm_name = trial.suggest_categorical(
            "algorithm_name", list(self.algorithms.keys())
        )

        if algorithm_name == "DecisionTreeClassifier":
            model_params = {
                "max_depth": trial.suggest_int("dt_max_depth", low=3, high=10),
                "min_samples_split": trial.suggest_int(
                    "dt_min_samples_split", low=2, high=15
                ),
                "min_samples_leaf": trial.suggest_int(
                    "dt_min_samples_leaf", low=1, high=10
                ),
                "class_weight": "balanced",
            }
        elif algorithm_name == "RandomForestClassifier":
            model_params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", low=100, high=300),
                "max_depth": trial.suggest_int("rf_max_depth", low=3, high=10),
                "min_samples_split": trial.suggest_int(
                    "rf_min_samples_split", low=2, high=15
                ),
                "min_samples_leaf": trial.suggest_int(
                    "rf_min_samples_leaf", low=1, high=10
                ),
                "class_weight": "balanced",
            }
        elif algorithm_name == "XGBClassifier":
            model_params = {
                "n_estimators": trial.suggest_int("xg_n_estimators", low=100, high=300),
                "scale_pos_weight": trial.suggest_float(
                    "xg_scale_pos_weight", low=1, high=10
                ),
            }
        elif algorithm_name == "HistGradientBoostingClassifier":
            model_params = {
                "learning_rate": trial.suggest_float(
                    "gb_learning_rate", low=0.01, high=0.3
                ),
                "l2_regularization": trial.suggest_float(
                    "gb_l2_regularization", low=0.01, high=0.3
                ),
                "class_weight": "balanced",
            }
        elif algorithm_name == "SVC":
            model_params = {
                "C": trial.suggest_float("SVC_C", low=0.1, high=10.0),
                "class_weight": "balanced",
            }
        elif algorithm_name == "MLPClassifier":
            hidden_layer_string = trial.suggest_categorical(
                "MLP_hidden_layer_sizes", ["(20, 10)", "(10, 5)", "(20,)"]
            )
            model_params = {
                "hidden_layer_sizes": eval(hidden_layer_string),
            }

        mlflow.log_param("algorithm_name", algorithm_name)
        logger.info(f"best model is {algorithm_name}")
        mlflow.log_params(model_params)
        trial.set_user_attr("model_params", model_params)
        return self.algorithms[algorithm_name](**model_params)

    def select_best_params(self, X_train, X_test, y_train, y_test):

        study = opt.create_study(direction="maximize")

        def objective(trial: opt.Trial):
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                algorithm = self.choose_algorithms_with_params(trial)
                algorithm.fit(X_train, y_train)
                y_pred = algorithm.predict(X_test)
                f1 = f1_score(
                    y_test,
                    y_pred,
                )

                mlflow.log_metric("f1_score", f1)

                return f1

        study.optimize(objective, n_trials=10)

        best_algorithm = study.best_params["algorithm_name"]
        best_params = study.best_trial.user_attrs["model_params"]

        return best_algorithm, best_params

    def save_and_register_the_model(self, model):

        uri = mlflow.get_tracking_uri()
        url_schema = urlparse(uri).scheme
        if url_schema.lower().strip() != "file":
            mlflow.sklearn.log_model(
                model,
                name="best_model",
                registered_model_name="credit_approval",
            )
        else:
            mlflow.sklearn.log_model(model, "best_model")
        create_dirs(self.model_training_model_path)
        save_object(self.model_training_model_path, model)

    def initiate_model_training(self):

        logger.info("initiate model training")
        mlflow.get_registry_uri()
        mlflow.set_experiment("Credit-Approval")

        with mlflow.start_run(run_name="credit_approval_parent"):

            X_train, X_test, y_train, y_test = read_data(self.data_transformation_file)

            best_algorithm, best_params = self.select_best_params(
                X_train, X_test, y_train, y_test
            )

            model = self.algorithms[best_algorithm](**best_params)
            model.fit(X_train, y_train)
            self.save_and_register_the_model(model)
        logger.info("model training has ended")


if __name__ == "__main__":
    try:
        logger.info("start the model_train process")
        model_train_config = ModelTrainConfig(main_config=MainConfig())
        model_train = ModelTraining(
            model_train_config,
            DataTransformationArtifact(
                r"artifacts\tranformation\credit_approval_transformed.pkl",
                r"artifacts\tranformation\preprocessor.pkl",
            ),
        )
        model_train.initiate_model_training()
        logger.info("end the model_train process")

    except:
        logger.exception("error in model training")
        raise
