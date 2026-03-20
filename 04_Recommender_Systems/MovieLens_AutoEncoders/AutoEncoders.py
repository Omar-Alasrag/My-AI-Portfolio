# %%
# Dataset: The MovieLens 1M dataset can be downloaded from
# https://grouplens.org/datasets/movielens/1m/.
# Place the files in the `data/` folder before running the scripts.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.models import Model

# Load data
data_path = r"ml-1m\ratings.dat"
df = pd.read_csv(data_path, sep="::")

num_users = df["user_id"].max() + 1
num_movies = df["movie_id"].max() + 1

x = np.zeros((num_users, num_movies), dtype="float32")


for user in df["user_id"].unique():
    idxs = df.index[df["user_id"] == user]
    movies = df["movie_id"].iloc[idxs]
    ratings = df["rating"].iloc[idxs]

    x[user, movies] = ratings

x = np.where(x >= 3, 1, 0)


autoencoder = tf.keras.Sequential(
    [
        # Encoder
        layers.Dense(512, activation="relu", shape=(num_movies,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),  # bottleneck
        # Decoder
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_movies, activation="sigmoid"),
    ]
)


autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


def masked_BCELoss(y_true, y_pred):
    mask = tf.cast(y_true > 0, tf.float32)
    # mask = tf.where(y_true > 0, 1.0, 0.0)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-8)
    
autoencoder.compile(
    optimizer="adam",
    loss=masked_BCELoss
)

autoencoder.fit(
    x, x,  epochs=40, batch_size=64, validation_split=0.2
)   


# %%
