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
from keras.utils import load_img, img_to_array


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
    root_folder = os.path.join(LOCAL_DATA_PATH,'raw_data', 'kaggle_data', 'Garbage classification', 'Garbage classification')

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

    # Encode the X_processed into msgpack
    X_processed_msgpack = msgpack_tensor_to_series(X_processed)

    # Target encoding
    y_processed = to_categorical(labels_array, num_classes=6)

    # Finally we shuffle:
    p = np.random.permutation(len(X_processed_msgpack))
    X_processed_msgpack, y_processed = X_processed_msgpack[p], y_processed[p]

    # TO BE DONE
    # Load a DataFrame onto BigQuery containing [X_processed, y_preprocessed] using data.load_data_to_bq()
    # 🚨 Data needs to be flattened if we want to transform it into a DataFrame !!!
    data_processed = pd.DataFrame(np.column_stack((
        X_processed_msgpack,
        y_processed)))

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_data_{DATA_SOURCE}',
        truncate=True
    )

    print("✅ preprocess_vgg16() done \n")



# TRAINING
#@mlflow_run
def train(
        learning_rate=0.0001,
        batch_size = 64,
        patience = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_accuracy as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)



    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_data_{DATA_SOURCE}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_data_{DATA_SOURCE}.csv")
    data_processed_msgpack = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed_msgpack.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    first_split = int(data_processed_msgpack.shape[0] /SPLIT_RATIO_1)
    second_split = first_split + int(data_processed_msgpack.shape[0] * SPLIT_RATIO_2)

    #print(f"first split : {first_split} \n second split : {second_split}")

    data_processed_val = data_processed_msgpack.iloc[first_split:second_split, :]
    data_processed_train = data_processed_msgpack.iloc[second_split:, :]

    X_train_processed = msgpack_series_to_tensor(data_processed_train.iloc[:, 0])
    y_train = data_processed_train.iloc[:, 1:].to_numpy(dtype=np.float32)

    X_val_processed = msgpack_series_to_tensor(data_processed_val.iloc[:, 0])
    y_val = data_processed_val.iloc[:, 1:].to_numpy(dtype=np.float32)

    #print(X_train_processed.shape)
    #print(y_val)

    # Train model using `model.py`
    model = load_model_VGG16(input_shape=X_train_processed.shape[1:])

    model = initialize_model_VGG16(input_shape=X_train_processed.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )

    val_accuracy = np.max(history.history['val_accuracy'])

    params = dict(
        context="train",
        #training_set_size=DATA_SIZE,
        data_source=DATA_SOURCE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy=val_accuracy))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # The latest model should be moved to staging
    #mlflow_transition_model(current_stage='None', new_stage='Staging')

    print("✅ train() done \n")

    return val_accuracy

# EVALUATING
#@mlflow_run
def evaluate(
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return ACCURACY as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_data_{DATA_SOURCE}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_data_{DATA_SOURCE}.csv")
    data_processed_msgpack = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed_msgpack.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None


    # Create (X_test_processed, y_test)
    first_split = int(data_processed_msgpack.shape[0] /SPLIT_RATIO_1)

    data_processed_test = data_processed_msgpack.iloc[:first_split, :]

    X_test_processed = msgpack_series_to_tensor(data_processed_test.iloc[:, 0])
    y_test = data_processed_test.iloc[:, 1:].to_numpy(dtype=np.float32)


    metrics_dict = evaluate_model(model=model, X=X_test_processed, y=y_test)
    accuracy = metrics_dict["accuracy"]

    params = dict(
        context="evaluate", # Package behavior
        #training_set_size=DATA_SIZE,
        data_source=DATA_SOURCE,
        row_count=len(X_test_processed)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return accuracy


# PREDICTING

def pred(image_path: str = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    INDEX_TO_CATEGORIES = {v: k for k, v in CATEGORIES_MAP.items()}

    if image_path is None:
        image_path = Path(LOCAL_DATA_PATH).joinpath("test", "14290_une.jpg")

    model = load_model()
    assert model is not None

    img = load_img(image_path, target_size=(224, 224))

    img_array = img_to_array(img)

    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    img_preprocessed = preprocess_input(img_array_expanded_dims)

    predictions = model.predict(img_preprocessed)

    predictions_with_categories = [(INDEX_TO_CATEGORIES[i], prob) for i, prob in enumerate(predictions[0])]

    predictions_with_categories.sort(key=lambda x: x[1], reverse=True)

    best_prediction = predictions_with_categories[0]

    #print("\n✅ prediction done: ", f"Number of classes : {len(predictions_with_categories)}\n", predictions_with_categories,  "\n")
    print(dict(result = best_prediction))

    return best_prediction

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
