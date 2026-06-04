from dotenv import load_dotenv
import os

load_dotenv()

LOGGER_FILE_PATH = "logs.txt"
MONGO_CONNECTION_URL = f"mongodb+srv://{os.getenv("MONGO_USERNAME")}:{os.getenv("MONGO_PASSWORD")}@cluster0.5lddb3w.mongodb.net/?appName=Cluster0"
MONGO_DB_NAME = "credit_approval"
MONGO_COLLECTION_NAME = "records"

PIPELINE_ARTIFACT_DIR = "artifacts"


DATA_PUSHER_MAIN_DIR = "pusher"
DATA_PUSHER_DOWNLOAD_FILE = "credit_approval.csv"
# main link -> https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data - you should download clean_dataset.csv only
DATA_PUSHER_DOWNLOAD_LINK = r"https://storage.googleapis.com/kagglesdsdata/datasets/2121492/3526626/clean_dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com@kaggle-161607.iam.gserviceaccount.com/20260602/auto/storage/goog4_request&X-Goog-Date=20260602T095532Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3b489916544713c47d8520f8178ce6edb7e745ef54d46a0f935dd1fff68a25760fd11c0da87b5d4d459a60b352d84cd99c6823985b6a25503accaa5a3a9814b55c2eebeba3441d38a2f15e1effcd5acf1046fbb4b92f6fb4c8a2cd2134d14e5be537fa256d32c54c534288c1ab2fbea8fbdd2442756fede43db72b8345595af9ed6e605567534dfdf9396b4e683112dbce4598dc2a0bdad1200d296c76ea7c20b190c9bfb91f3b494760d7f2853b6d0fa25eac39b2467361bf0718e7cdade69d7116be792c30a5ba3631d7739f147996f6310bfc0aa21312ae68f2d818e395fb87bb0745de0af488503a48d7fcfe9ad227b6f55eb731e3a3e32aa7ca7f19fb0b"


DATA_INGESTION_ARTIFACT_DIR = "ingestion"
DATA_INGESTION_INGESTED_DATA_FILE = "credit_approval.csv"


DATA_VALIDATION_ARTIFACT_MAIN_DIR = "validation"
DATA_VALIDATION_VALID_DATA_DIR = "valid"
DATA_VALIDATION_VALID_DATA_FILE = "credit_approval_valid.parquet"
DATA_VALIDATION_INVALID_DATA_DIR = "invalid"
DATA_VALIDATION_INVALID_DATA_FILE = "credit_approval_invalid.parquet"


DATA_TRANSFORMATION_ARTIFACT_MAIN_DIR = "tranformation"
DATA_TRANSFORMATION_DATA_FILE = "credit_approval_transformed.pkl"
DATA_TRANSFORMATION_PREPROCESSOR_FILE = "preprocessor.pkl"
DATA_TRANSFORMATION_TARGET_COL = "Approved"
DATA_TRANSFORMATION_CAT_COLS = [
    "Married",
    "BankCustomer",
    "Industry",
    "PriorDefault",
    "Citizen",
]
DATA_TRANSFORMATION_UNNEEDED_COLS = ["ZipCode", "DriversLicense", "Gender", "Ethnicity"]


MODEL_TRAINING_MAIN_DIR = "train"
MODEL_TRAINING_MODEL_FILE = "best_model.pkl"
