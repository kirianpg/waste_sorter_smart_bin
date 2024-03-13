import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

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
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        img = np.expand_dims(img, axis=0)
                        img = preprocess_input(img)
                        label = object_folder
                        images_with_labels.append((img, label))
    return images_with_labels
