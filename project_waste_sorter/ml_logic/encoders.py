import base64
import msgpack
from PIL import Image
import io
import numpy as np
import pandas as pd
import tensorflow as tf

def image_to_base64(image_array):
    """
    Encodes a Numpy array (with float values) into a base64 string.
    """
    buffer = io.BytesIO()
    # Save array to buffer using numpy's save function
    np.save(buffer, image_array, allow_pickle=True, fix_imports=True)
    # Encode the buffer's contents
    byte_data = buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('ascii')
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
    Decodes a base64 string into a Numpy array with float values.
    """
    image_data = base64.b64decode(base64_string)
    # Load array from buffer using numpy's load function
    image_array = np.load(io.BytesIO(image_data), allow_pickle=True, fix_imports=True)
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

### MSGPACK ###

def image_to_msgpack(image_array):
    """
    Encodes a Numpy array into a msgpack byte string.
    """
    buffer = io.BytesIO()
    # Use numpy to save the array in the buffer
    np.save(buffer, image_array, allow_pickle=True, fix_imports=True)
    # Use msgpack to encode the content of the buffer
    byte_data = buffer.getvalue()
    packed_data = msgpack.packb(byte_data)
    return packed_data

def msgpack_tensor_to_series(tensor):
    """
    Converts a tensor of images into a Pandas Series with the images encoded in msgpack.
    """
    images_packed = pd.Series([image_to_msgpack(image) for image in tensor])
    return images_packed

def msgpack_to_image(packed_data):
    """
    Decodes a msgpack byte string into a Numpy array.
    """
    # Decode the msgpack content to obtain bytes
    byte_data = msgpack.unpackb(packed_data)
    # Load the array from the bytes
    image_array = np.load(io.BytesIO(byte_data), allow_pickle=True, fix_imports=True)
    return image_array

def msgpack_series_to_tensor(series):
    """
    Converts a Pandas Series containing msgpack encoded images into a NumPy array.
    """
    # Use np.stack to combine the arrays in an unique numpy.ndarray
    images_array = np.stack([msgpack_to_image(item) for item in series], axis=0)
    return images_array
