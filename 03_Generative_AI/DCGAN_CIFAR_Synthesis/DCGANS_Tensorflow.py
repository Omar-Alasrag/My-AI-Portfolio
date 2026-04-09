import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential, layers
from keras.utils import image_dataset_from_directory
from tqdm import tqdm

scaler = layers.Rescaling(scale=1.0 / 127.5, offset=-1)

dset = (
    image_dataset_from_directory(
        r"cifar10", # suppose that the img size is  128 * 128
        image_size=(128, 128),
        batch_size=32,
    )
    .map(lambda x, y: (scaler(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

gen = Sequential(
    [
        layers.Input((1, 1, 100)),
        layers.Conv2DTranspose(512, (4, 4), 2, "valid", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(265, (4, 4), 2, "same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(128, (4, 4), 2, "same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (4, 4), 2, "same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(32, (4, 4), 2, "same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (4, 4), 2, "same", activation="tanh", use_bias=False),
    ]
)

gen.summary()

dis = Sequential(
    [
        layers.Input((128, 128, 3)),
        layers.Conv2D(32, (4, 4), 2, "same"),
        layers.LeakyReLU(),
        layers.Conv2D(64, (4, 4), 2, "same"),
        layers.LeakyReLU(),
        layers.Conv2D(128, (4, 4), 2, "same"),
        layers.LeakyReLU(),
        layers.Conv2D(256, (4, 4), 2, "same"),
        layers.LeakyReLU(),
        layers.Conv2D(512, (4, 4), 2, "same"),
        layers.LeakyReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

dis.summary()


def save_images(epoch, generator, num_examples=16):
    fixed_noise = tf.random.normal([num_examples, 1, 1, 100])
    predictions = generator(fixed_noise, training=False)

    predictions = (predictions + 1) / 2.0

    plt.figure(figsize=(14, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis("off")

    if not os.path.exists("gan_results"):
        os.makedirs("gan_results")

    plt.tight_layout() 
    plt.savefig(f"gan_results/image_at_epoch_{epoch:04d}.png")
    plt.show()


loss_fn = tf.keras.losses.BinaryCrossentropy()
G_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
D_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)


@tf.function
def train_step(real):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        batch_size = real.shape[0]
        noise = tf.random.normal((batch_size, 1, 1, 100))

        fake = gen(noise, training=True)

        D_fake_output = dis(fake, training=True)
        D_real_output = dis(real, training=True)

        D_loss_fake = loss_fn(tf.zeros_like(D_fake_output), D_fake_output)
        D_loss_real = loss_fn(tf.ones_like(D_real_output), D_real_output)

        D_total_loss = D_loss_fake + D_loss_real

        G_loss = loss_fn(tf.ones_like(D_fake_output), D_fake_output)

    D_grad = disc_tape.gradient(D_total_loss, dis.trainable_variables)
    G_grad = gen_tape.gradient(G_loss, gen.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, dis.trainable_variables))
    G_optimizer.apply_gradients(zip(G_grad, gen.trainable_variables))
    return D_total_loss, G_loss


n_epochs = 10
for e in range(n_epochs):
    for step, (real, _) in enumerate(tqdm(dset, leave=False)):
        D_total_loss, G_loss = train_step(real)
        if step % 100 == 0:
            print(
                f"Epoch {e}, Step {step}, D: {D_total_loss.numpy():.4f}, G: {G_loss.numpy():.4f}"
            )
    save_images(e, gen, 8)
