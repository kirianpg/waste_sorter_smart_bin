import base64
from PIL import Image
import io
import numpy as np
import pandas as pd
import tensorflow as tf

def image_to_base64(image_array):
    """
    Encodes an image array into a base64 string.

    :param image_array: A Numpy array representing an image.
    :return: A string representing the image encoded in base64.
    """
    image_pil = Image.fromarray(np.uint8(image_array))
    buff = io.BytesIO()
    image_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def tensor_to_series(tensor):
    """
    Converts a tensor of images into a Pandas Series with the images encoded in base64.

    :param tensor: A TensorFlow tensor containing images.
    :return: A Pandas Series with each item being an image encoded in base64.
    """
    images_base64 = pd.Series([image_to_base64(image.numpy()) for image in tensor])
    return images_base64


def base64_to_image(base64_str):
    """
    Decodes a base64 string into an image array.

    :param base64_str: The base64 string to decode.
    :return: A Numpy array of the image.
    """
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

def series_to_tensor(series):
    """
    Converts a Pandas Series containing base64 encoded images into a tensor.

    :param series: Series containing the base64 encoded images.
    :return: A TensorFlow tensor of the images.
    """
    images_array = [base64_to_image(item) for item in series]
    return tf.convert_to_tensor(images_array, dtype=tf.float32)
