
import os
from argparse import ArgumentParser
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import attention as att
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import (
    Dense,
    Dropout
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    # parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    args = parser.parse_args()

    ds = tfds.load("imagenet_r", as_supervised=True)
    # ds = tfds.load("mnist", as_supervised=True)
    ds_train = (
        ds["train"]
        .cache()
        .shuffle(5 * args.batch_size)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = (
        ds["test"]
        .cache()
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = tf.keras.models.Sequential()

        model.add(Rescaling(1.0 / 255))

        model.add(att.VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            num_classes=10,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=1,
            dropout=0.1,
        ))

        model.add(Dense(args.mlp_dim, activation=tfa.activations.gelu))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='softmax'))

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
            metrics=["accuracy"],
        )

    history = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0), ],

    )

    test = model.predict(ds_test)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    model.save_weights(os.path.join(args.logdir, "vit"))
