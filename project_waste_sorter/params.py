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
LOCAL_REGISTRY_PATH =  os.path.join("models", ".lewagon", "waste_sorter_smart_bin", "training_outputs")
#LOCAL_REGISTRY_PATH =  os.path.join(os.path.sep,"models")
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

###################### RECYCLING POLICIES FRONTEND #################################
# Recycling policies by city
# Lyon
lyon_RP_message = "\nBrown bin : organic \nGreen bin : glass \nBlue bin : cardboard, paper, plastic, metal \nGrey bin : trash"
lyon_bins = {
    'trash': 'grey',
    'glass': 'green',
    'paper': 'blue',
    'cardboard': 'blue',
    'plastic': 'blue',
    'metal': 'blue',
    'organic' : 'brown'
    }
lyon_bins_images = {
    'trash': 'lyon_grey_bin.png',
    'glass': 'lyon_green_bin.png',
    'paper': 'lyon_blue_bin.png',
    'cardboard': 'lyon_blue_bin.png',
    'plastic': 'lyon_blue_bin.png',
    'metal': 'lyon_blue_bin.png',
    'organic' : 'lyon_brown_bin.png'
    }

# Other city
other_city_RP_message = "\nBrown bin : organic \nGreen bin : glass \nBlue bin : cardboard, paper, plastic, metal \nGrey bin : trash"
other_city_bins = {
    'trash': 'grey',
    'glass': 'green',
    'paper': 'blue',
    'cardboard': 'blue',
    'plastic': 'blue',
    'metal': 'blue',
    'organic' : 'brown'
    }
other_city_bins_images = {
    'trash': 'lyon_grey_bin.png',
    'glass': 'lyon_green_bin.png',
    'paper': 'lyon_blue_bin.png',
    'cardboard': 'lyon_blue_bin.png',
    'plastic': 'lyon_blue_bin.png',
    'metal': 'lyon_blue_bin.png',
    'organic' : 'lyon_brown_bin.png'
    }

# Recycling points by city (paths to databases)
# Lyon
lyon_recycling_points_files = {
    'trash': 'lyon_grey.csv',
    'glass': 'lyon_green.csv',
    'paper': 'lyon_blue.csv',
    'cardboard': 'lyon_blue.csv',
    'plastic': 'lyon_blue.csv',
    'metal': 'lyon_blue.csv',
    'organic' : 'lyon_brown.csv'
    }

# Custom policies [policy text, bin-class correspondance, images of the bins, data files for recycling points]
custom_policies = {
    'Lyon' : [lyon_RP_message, lyon_bins, lyon_bins_images, lyon_recycling_points_files],
    'Other city': [other_city_RP_message, other_city_bins, other_city_bins_images, None]
    }
