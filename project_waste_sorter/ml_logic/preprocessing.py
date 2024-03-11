import os
import cv2
import numpy as np
import pandas as pd
from project_waste_sorter.params import *


# 🚨 SUSI : I would put this function in ml_logic.data
# def load_images_with_labels(root_folder):
#     images_with_labels = []
#     for object_folder in os.listdir(root_folder):
#         object_path = os.path.join(root_folder, object_folder)
#         if os.path.isdir(object_path):
#             for filename in os.listdir(object_path):
#                 img_path = os.path.join(object_path, filename)
#                 if os.path.isfile(img_path):
#                     img = cv2.imread(img_path)
#                     if img is not None:
#                         label = object_folder
#                         images_with_labels.append((img, label))
#     return images_with_labels

def preprocess_image(img, target_size):
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    return img_normalized

def preprocess_labels(labels_list):
    labels_series = pd.Series(labels_list)
    categories_series = labels_series.map(CATEGORIES_MAP)
    return categories_series


# 🚨 SUSI : I would put this function in interface.main
# def main():
#     # Define root folder TBD!!!!
#     root_folder = os.path.dirname(os.getcwd())

#     # Load images with labels
#     images_with_labels = load_images_with_labels(root_folder)

#     # Define target size for resizing
#     target_size = (224, 224)

#     # Preprocess each image in the dataset
#     preprocessed_data = []
#     for img, label in images_with_labels:
#         preprocessed_img = preprocess_image(img, target_size)
#         preprocessed_data.append((preprocessed_img, label))

#     # Convert preprocessed_data to NumPy arrays
#     X_preprocessed = np.array([data[0] for data in preprocessed_data])
#     y_preprocessed = np.array([data[1] for data in preprocessed_data])
#     return X_preprocessed, y_preprocessed
