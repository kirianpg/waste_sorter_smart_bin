'''
Main Python entry point containing all "routes" for GCP running
'''

from project_waste_sorter.params import *
from project_waste_sorter.ml_logic.data import *
from project_waste_sorter.ml_logic.preprocessing import *
#from project_waste_sorter.ml_logic.model import *

#🚨 Needs to be fixed => it seems we need to install/import tf.keras??
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input


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

    # Target encoding
    y_processed = to_categorical(labels_array, num_classes=6)

    # TO BE DONE
    # Load a DataFrame onto BigQuery containing [X_processed, y_preprocessed] using data.load_data_to_bq()
    # 🚨 Data needs to be flattened if we want to transform it into a DataFrame !!!
    data_processed = pd.DataFrame(np.concatenate((
        X_processed,
        y_processed), axis=1))

    load_data_to_bq(
        data_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_data',
        truncate=True
    )

    print("✅ preprocess_vgg16() done \n")



# TRAINING


# EVALUATING



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
