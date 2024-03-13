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
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(np.uint8(image_array))
    # Create a bytes buffer for the image
    buffer = io.BytesIO()
    # Save the image to the buffer in PNG format
    image.save(buffer, format="PNG")
    # Get the buffer's contents as a byte string
    byte_data = buffer.getvalue()
    # Encode the byte string in base64 and return it
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def tensor_to_series(tensor):
    """
    Converts a tensor of images into a Pandas Series with the images encoded in base64.

    :param tensor: A TensorFlow tensor containing images.
    :return: A Pandas Series with each item being an image encoded in base64.
    """
    images_base64 = pd.Series([image_to_base64(image) for image in tensor])
    return images_base64


def base64_to_image(base64_string):
    """
    Decodes a base64 string into a PIL Image. Assumes the image is already
    preprocessed (in terms of resizing and pixel values) to match expectations.

    :param base64_string: The base64 encoded string of an image.
    :return: A NumPy array representing the decoded image.
    """
    # Decode the base64 string to bytes, then open it as an image
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    # Convert the image to a NumPy array
    image_array = np.array(image, dtype=np.float32)
    return image_array

def series_to_tensor(series):
    """
    Converts a Pandas Series containing base64 encoded images into a NumPy array.

    :param series: Series containing the base64 encoded images.
    :return: A NumPy array of the images.
    """
    # Use np.stack to combine the arrays into a single numpy.ndarray
    images_array = np.stack([base64_to_image(item) for item in series], axis=0)
    return images_array
