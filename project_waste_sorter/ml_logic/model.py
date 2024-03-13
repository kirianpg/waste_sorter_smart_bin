import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.applications import VGG16


end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def load_model_VGG16(input_shape: tuple):

    model = VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
    )

    return model

def set_nontrainable_layers(model):

    model.trainable = False

    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top, + regularizers'''
    reg = regularizers.l1_l2(l2=0.005)
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(512, activation='relu', kernel_regularizer=reg)
    dropout_layer = layers.Dropout(0.5)
    prediction_layer = layers.Dense(6, activation='softmax')

    final_model = Sequential([
        model,
        flattening_layer,
        dense_layer,
        dropout_layer,
        prediction_layer
    ])

    return final_model



def initialize_model_VGG16(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model_VGG16 = load_model_VGG16(input_shape)
    model_VGG16_snl = set_nontrainable_layers(model_VGG16)
    customized_VGG16 = add_last_layers(model_VGG16_snl)

    print("✅ Model initialized")

    return customized_VGG16


def compile_model(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the Neural Network
    """
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
    )
    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=5,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with last val accuracy: {round(history.history['val_accuracy'][-1], 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, ACCURACY: {round(accuracy, 2)}")

    return metrics
