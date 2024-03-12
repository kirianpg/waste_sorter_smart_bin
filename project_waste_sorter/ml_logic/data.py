'''
Package that contains functions to load images (data)
'''

#Imports
import os
import cv2
import numpy as np



'''
Data for training
'''
# Load Kaggle dataset
def load_images_with_labels(root_folder):

    images_with_labels = []
    for object_folder in os.listdir(root_folder):
        object_path = os.path.join(root_folder, object_folder)
        if os.path.isdir(object_path):
            for filename in os.listdir(object_path):
                img_path = os.path.join(object_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        label = object_folder
                        images_with_labels.append((img, label))
    return images_with_labels



# Load TACO dataset
# TO BE DONE

'''
Data for predictions
'''
# Load user image
# TO BE DONE => not sure that we need it here
