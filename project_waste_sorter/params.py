import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "all" # ["1k", "200k", "all"]
CHUNK_SIZE = 100000
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "waste_sorter_smart_bin", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "waste_sorter_smart_bin", "training_outputs")
COLUMN_NAMES_RAW = ['fare_amount','pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'] # TO MODIFY

# TO MODIFY
DTYPES_RAW = {
    "fare_amount": "float32",
    "pickup_datetime": "datetime64[ns, UTC]",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16"
}

CATEGORIES_MAP = {
    'glass': 1,
    'paper': 2,
    'cardboard': 3,
    'plastic': 4,
    'metal': 5,
    'trash': 0
    }

DTYPES_PROCESSED = np.float32
