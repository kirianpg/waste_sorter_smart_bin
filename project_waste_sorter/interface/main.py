'''
Main Python entry point containing all "routes" for GCP running
'''
import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from project_waste_sorter.params import *
from project_waste_sorter.ml_logic.data import *
from project_waste_sorter.ml_logic.preprocessing import *
from project_waste_sorter.ml_logic.encoders import *
from project_waste_sorter.ml_logic.model import *
from project_waste_sorter.ml_logic.registry import load_model, save_model, save_results
from project_waste_sorter.ml_logic.registry import mlflow_run, mlflow_transition_model


#🚨 Needs to be fixed => it seems we need to install/import tf.keras??
from tensorflow import keras
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input


# PREPROCESSING FOR VGG16 BASED MODEL
def preprocess_vgg16():
    '''
    Preprocessing for VGG16-based model
    1. Load images from dataset (or datasets) from local dir in the (224,224) required size as arrays
    2. VGG16 specific preprocessing
    3. Store processed data on your project BQ (truncate existing table if it exists)
    '''

    print("\n⭐️ Preprocessing for VGG16 model...")

    # Define root folder TBD!!!! => 🚨 We need to make sure everone has the same local structure and has the data in the same dir
    root_folder = os.path.join(LOCAL_DATA_PATH,'raw','Garbage classification')

    # Load images with labels (from ml_logic.data) => target_size by default is (224,224) in the function
    images_with_labels = load_images_with_labels(root_folder)

    # Get images and labels lists
    labels_list = images_with_labels[1]
    images_list = images_with_labels[0]

    # Transform categories to numeric
    categories_series = preprocess_labels(labels_list)

    # Convert lists to NumPy arrays
    images_array = np.array(images_list)
    labels_array = np.array(categories_series)

    # VGG16 specific preprocessing
    X_processed = preprocess_input(images_array)

    # Encode the X_processed into base64
    X_processed_base64 = tensor_to_series(X_processed)

    # Target encoding
    y_processed = to_categorical(labels_array, num_classes=6)

    # TO BE DONE
    # Load a DataFrame onto BigQuery containing [X_processed, y_preprocessed] using data.load_data_to_bq()
    # 🚨 Data needs to be flattened if we want to transform it into a DataFrame !!!
    data_processed = pd.DataFrame(np.concatenate((
        X_processed_base64,
        y_processed), columns=['image_base64', 'y'], axis=1))

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_data_batch_{BATCH_NUMBER}',
        truncate=True
    )

    print("✅ preprocess_vgg16() done \n")



# TRAINING

def train(
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)



    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_data_batch_{BATCH_NUMBER}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_data_batch_{BATCH_NUMBER}.csv")
    data_processed_base64 = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed_base64.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    first_split = int(len(data_processed_base64) /SPLIT_RATIO_1)
    second_split = first_split + int(len(data_processed_base64) * SPLIT_RATIO_2)

    data_processed_val = data_processed_base64.iloc[first_split:second_split, :].sample(frac=1).to_numpy()
    data_processed_train = data_processed_base64.iloc[second_split:, :].sample(frac=1).to_numpy()

    X_train_processed = series_to_tensor(data_processed_train[:, 0])
    y_train = data_processed_train[:, -1]

    X_val_processed = series_to_tensor(data_processed_val[:, 0])
    y_val = data_processed_val[:, -1]

    # Train model using `model.py`
    model = load_model_VGG16()

    if model is None:
        model = initialize_model_VGG16(input_shape=X_train_processed.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_accuracy = np.min(history.history['val_accuracy'])

    params = dict(
        context="train",
        #training_set_size=DATA_SIZE,
        batch_number=1,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_accuracy))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    mlflow_transition_model(current_stage='None', new_stage='Staging')

    print("✅ train() done \n")

    return val_accuracy

# EVALUATING

def evaluate(
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_data_batch_{BATCH_NUMBER}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_data_batch_{BATCH_NUMBER}.csv")
    data_processed_base64 = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed_base64.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None


    # Create (X_test_processed, y_test)
    first_split = int(len(data_processed_base64) /SPLIT_RATIO_1)

    data_processed_test = data_processed_base64.iloc[:first_split, :].sample(frac=1).to_numpy()

    X_test_processed = series_to_tensor(data_processed_test[:, 0])
    y_test = data_processed_test[:, -1]


    metrics_dict = evaluate_model(model=model, X=X_test_processed, y=y_test)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        #training_set_size=DATA_SIZE,
        batch_number=1,
        row_count=len(X_test_processed)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


# PREDICTING



# TO BE MODIFIED
if __name__ == '__main__':
    try:
        preprocess_vgg16()
        # preprocess()
        # train()
        # evaluate()
        # predict()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
