# %%
# dataset link
# https://www.kaggle.com/competitions/cifar-10/data



import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# 1. Load data with the split defined HERE
train_dset = tf.keras.utils.image_dataset_from_directory(
    r"cifar10\train",
    validation_split=0.2,
    subset="training",
    seed=1,
    image_size=(32, 32),
    batch_size=32,
)

val_dset = tf.keras.utils.image_dataset_from_directory(
    r"cifar10\test",
    validation_split=0.2,
    subset="validation",
    seed=1,
    image_size=(32, 32),
    batch_size=32,
)


cnn = tf.keras.Sequential([
    # Data augmentation
    tf.keras.layers.RandomFlip(input_shape=(32, 32, 3)),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomRotation(0.1),

    # Normalize
    tf.keras.layers.Rescaling(1.0 / 255),

    # Block 1
    tf.keras.layers.Conv2D(32, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),

    # Block 2
    tf.keras.layers.Conv2D(64, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),

    # Block 3
    tf.keras.layers.Conv2D(128, 3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPool2D(),

    # Head
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax"),
])


cnn.compile(
    "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
r = cnn.fit(train_dset, validation_data=val_dset, epochs=10)


plt.plot(r.history["loss"], label="loss", c="green")
plt.plot(r.history["val_loss"], label="val_loss", c="blue")
plt.legend()
plt.show()

# %%
