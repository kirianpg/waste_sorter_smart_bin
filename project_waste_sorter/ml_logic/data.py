'''
Package that contains functions to load images (data)
'''

#Imports
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from project_waste_sorter.params import *

#🚨 Needs to be fixed => it seems we need to install/import tf.keras??
from tensorflow.keras.preprocessing.image import load_img, img_to_array


'''
Data for training
'''
# Load Kaggle dataset
def load_images_with_labels(root_folder, target_size=(244,244)):
    '''
    Load images from local data folder
    Convert them to arrays
    Create a list with all the images and the equivalent with all the labels
    '''
    images_list = []
    labels_list = []
    for object_folder in os.listdir(root_folder):
        object_path = os.path.join(root_folder, object_folder)
        if os.path.isdir(object_path):
            for filename in os.listdir(object_path):
                img_path = os.path.join(object_path, filename)
                if os.path.isfile(img_path):
                    img = tensorflow.keras.utils.load_img(img_path, target_size=target_size)
                    img_array = tensorflow.keras.utils.img_to_array(img)
                    if img_array is not None:
                        label = object_folder
                        labels_list.append(label)
                        images_list.append(img_array)
    return images_list, labels_list


def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(f"\nSave data to BigQuery @ {full_table_name}...:")

    # Load data onto full_table_name
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")



# Load TACO dataset
# TO BE DONE

'''
Data for predictions
'''
# Load user image
# TO BE DONE => not sure that we need it here
