# %%
# dataset link
# https://www.kaggle.com/code/mervetorkan/churn-prediction

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

dset = pd.read_csv("ANN\Churn_Modelling.csv")
X = dset.iloc[:, 2:-1].to_numpy()
y = dset.iloc[:, -1].to_numpy()


# num_idxs = np.where(dset.dtypes != "object")[0] - 2
# print("Numerical Indices:", num_idxs)
# num_idxs = np.where(dset.dtypes == "object")[0] - 2
# print("Numerical Indices:", num_idxs)

ct = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), [0, 2, 3]),
        ("num", StandardScaler(), [1, 4, 5, 6, 7, 10]),
    ],
    remainder="passthrough",
)


X = ct.fit_transform(X).toarray()

ann = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.1), 
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


ann.compile("adam", loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
r = ann.fit(X, y, batch_size=32, epochs=4, validation_split=0.2)

plt.plot(r.history["loss"], label="loss",  c="green")
plt.plot(r.history["val_loss"], label="val_loss", c="blue")
plt.legend()
plt.show()


print("roc_auc_score", roc_auc_score(y, ann.predict(X)))