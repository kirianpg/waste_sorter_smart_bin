import os
import numpy as np

##################  VARIABLES  ##################
DATA_SOURCE="kaggle"
SPLIT_RATIO_1=float(os.environ.get("SPLIT_RATIO_1"))
SPLIT_RATIO_2=float(os.environ.get("SPLIT_RATIO_2"))
DATA_SIZE = "all" # ["1k", "200k", "all"]
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "waste_sorter_smart_bin", "training_outputs")
COLUMN_NAMES_RAW = ['image_base64','y']

# TO MODIFY
DTYPES_RAW = {
    "image_base64": "float32",
    "y": "datetime64[ns, UTC]",
}

CATEGORIES_MAP = {
    'trash': 0,
    'glass': 1,
    'paper': 2,
    'cardboard': 3,
    'plastic': 4,
    'metal': 5
    }

DTYPES_PROCESSED = np.float32
