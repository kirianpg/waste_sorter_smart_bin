'''
Main Python entry point containing all "routes"
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
    1. Load images from dataset (or datasets)
    2. Basic preprocessing
    3. VGG16 specific preprocessing
    '''

    # Define root folder TBD!!!! => 🚨 We need to make sure everone has the same local structure and has the data in the same dir
    root_folder = os.path.join(LOCAL_DATA_PATH,'raw','Garbage classification')

    # Load images with labels (from ml_logic.data)
    images_with_labels = load_images_with_labels(root_folder)

    # Define target size for resizing
    target_size = (224, 224)

    # Basic preprocessing : Resize and normalize images and convert categories to numbers
    # Preprocess each image in the dataset
    preprocessed_images = []
    labels_list = []
    for img, label in images_with_labels:
        labels_list.append(label)
        preprocessed_img = preprocess_image(img, target_size)
        preprocessed_images.append(preprocessed_img)
    preprocessed_labels = preprocess_labels(labels_list)

    # Convert basic preprocessed_data to NumPy arrays
    images_array = np.array(preprocessed_images)
    labels_array = np.array(preprocessed_labels)

    # VGG16 specific preprocessing
    X_preprocessed = preprocess_input(images_array)

    # Target encoding
    y_preprocessed = to_categorical(labels_array, num_classes=6)

    return X_preprocessed, y_preprocessed



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
