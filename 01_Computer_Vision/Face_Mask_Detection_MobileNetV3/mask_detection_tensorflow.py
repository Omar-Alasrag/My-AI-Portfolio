import keras as keras
import keras.layers
import keras.losses
import keras.optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input
from keras.models import Model
from keras.utils import image_dataset_from_directory
from sklearn.metrics import f1_score, roc_auc_score



tr_dset, val_dset = image_dataset_from_directory(
    "data\mask_dataset", subset="both", validation_split=0.2, seed=123
)

tr_dset = tr_dset.map(lambda x, y: (preprocess_input(x), y)).prefetch(
    buffer_size=tf.data.AUTOTUNE
)

val_dset = val_dset.map(lambda x, y: (preprocess_input(x), y)).prefetch(
    buffer_size=tf.data.AUTOTUNE
)




base_model: Model = MobileNetV3Small((256, 256, 3), include_top=False)


model: Model = keras.Sequential(
    [
        keras.layers.Input(shape=(256, 256, 3)),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomFlip("HORIZONTAL"),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomContrast(0.2),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, "relu"),
        keras.layers.Dense(2, "softmax"),
    ]
)


for layer in base_model.layers:
    layer.trainable = False

model.summary()


model.compile(
    keras.optimizers.Adam(),
    keras.losses.SparseCategoricalCrossentropy(),
    metrics=["acc"],
)

n_epochs = 20
r = model.fit(tr_dset, validation_data=val_dset, epochs=n_epochs)

val_dset = image_dataset_from_directory(
    "data\mask_dataset",
    subset="validation",
    validation_split=0.2,
    seed=123,
    shuffle=True,
).cache().prefetch(tf.data.AUTOTUNE)

val_dset = val_dset.map(lambda x, y: (preprocess_input(x), y))

labels = tf.concat([y for x, y in val_dset], axis=0)

predictions = model.predict(val_dset)
best = tf.argmax(predictions, axis=-1)

f1 = f1_score(labels, best)
auc = roc_auc_score(labels, predictions[:, 1])


plt.plot(r.history["loss"], label="tr_loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.title(f"Loss Curves\nF1 Score: {f1:.2f} | AUC: {auc:.2f}")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()


