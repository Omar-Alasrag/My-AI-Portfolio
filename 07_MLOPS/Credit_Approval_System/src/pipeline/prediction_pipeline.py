from sklearn.compose import ColumnTransformer
import numpy as np
from src.logging.logger import logging

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self, preprocessor: ColumnTransformer, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, data):
        try:
            logger.info("start prediction process")
            tr_data = self.preprocessor.transform(data)

            if hasattr(self.model, "c"):
                probabilities = self.model.predict_proba(tr_data)
                prediction = np.argmax(probabilities, axis=1)
                prediction_probability = np.max(probabilities, axis=1)
                logger.info("predict_proba is not available")
                logger.info(f"prediction is {prediction}")
                logger.info(f"probability is {prediction_probability}")

            else:
                prediction = self.model.predict(tr_data)
                prediction_probability = np.array(["unknown"] * len(prediction))
                logger.info("predict_proba is not available, using direct prediction")
            logger.info("end prediction process successfully")
            return prediction, prediction_probability
        except:
            logger.exception("prediction process failed")
            raise
