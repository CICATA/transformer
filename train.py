import pathlib
import os
import tensorflow as tf
import tensorflow_addons as tfa
from argparse import ArgumentParser
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import imtools as imt
import attention as att
import math


from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--patch-size", default=128, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    args = parser.parse_args()

    # Custom dataset

    # Fiting dataset
    x = imt.load_dataset('/data/gee/SAR/transformer/data/train/c2', args.image_size, args.image_size)
    y = imt.load_dataset('/data/gee/SAR/transformer/data/train/c2', args.patch_size, args.patch_size)

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.shuffle(10).batch(args.batch_size).prefetch(AUTOTUNE).cache()

    # End Custom dataset

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
            channels=3,
            dropout=0.1,
        ))

        model.add(Dense(args.mlp_dim, activation=tfa.activations.gelu))
        model.add(Dropout(0.1))
        model.add(Dense((args.patch_size**2)*3))
        model.add(tf.keras.layers.Reshape((args.patch_size, args.patch_size, 3)))
        # model.add(Dense(10, activation='softmax'))

        model.compile(
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(
                #from_logits=True
            #),
            loss=tf.keras.losses.MeanAbsoluteError(),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
            #metrics=["accuracy"],
        )


    history = model.fit(
        train_dataset,
        #validation_data=ds_val,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0), ],
    )

    test = model.predict(x)


    loss = history.history['loss']
    plt.plot(loss, label='Training loss')
    plt.show()

    plt.imshow(test[0, :, :, 0])
    plt.show()
    plt.imshow(y[0, :, :, 0])
    plt.show()

    model.save_weights(os.path.join(args.logdir, "vit"))
