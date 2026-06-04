from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from src.pipeline.pipeline import Pipeline
from fastapi.templating import Jinja2Templates
from src.entity.data_schema import DataSchema
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelPredictionConfig, MainConfig
from sklearn.compose import ColumnTransformer
from pandas import DataFrame
from src.pipeline.prediction_pipeline import PredictionPipeline
from pathlib import Path
import src.constants as const
from src.logging.logger import logging

logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates("templates")


@app.get("/")
async def welcome(request: Request):

    return RedirectResponse("/predict")


@app.get("/train")  # it should have been "POST" but i added this for simplicity
@app.post("/train")
async def train():
    try:
        pipeline = Pipeline()
        pipeline.start_pipeline()
        return {"message": "****** The Model Trained Successfully ******"}

    except Exception as ex:
        return {"message": f"Exception Happened{ex}"}


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):

    return templates.TemplateResponse(
        request, "index.html", context={"prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, data: DataSchema = Form()):
    try:

        data_dict = data.model_dump()
        df = DataFrame([data_dict])
        main_artifacts_path = Path(const.PIPELINE_ARTIFACT_DIR)
        all_dirs = [
            d
            for d in main_artifacts_path.iterdir()
            if d.is_dir() and d.name != "my_dataset"
        ]
        if len(all_dirs) == 0:
            warning = "train the model first"
            return templates.TemplateResponse(
                request, "index.html", context={"warning": warning, "prediction": None}
            )
        last_dir = sorted(all_dirs, key=lambda x: x.name)[-1].name
        main_config = MainConfig(last_dir)
        model_prediction_config = ModelPredictionConfig(main_config)

        preprocessor: ColumnTransformer = load_object(
            model_prediction_config.data_transformation_preprocessor_file
        )
        model = load_object(model_prediction_config.model_training_model_path)

        prediction_pipeline = PredictionPipeline(preprocessor, model)
        prediction, prediction_probability = prediction_pipeline.predict(df)

        prediction, prediction_probability = prediction[0], prediction_probability[0]
        logger.info(df.to_string())

        logger.info(
            f"result is: {prediction}, prediction_probability: {prediction_probability}"
        )
        return templates.TemplateResponse(
            request, "index.html", context={"prediction": prediction}
        )
    except Exception as ex:
        logger.exception("prediction failed")
        return templates.TemplateResponse(
            request, "index.html", context={"warring": str(ex)}
        )
