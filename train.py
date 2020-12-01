import pathlib
import os
import tensorflow as tf
import tensorflow_addons as tfa
from argparse import ArgumentParser
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from attention import VisionTransformer

from tensorflow.keras.layers import (
    Dense,
    Dropout
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=28, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    args = parser.parse_args()

    # Custom dataset

    data_dir = pathlib.Path('data')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 28
    img_width = 28

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        seed=128,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True)

    class_names = ds_train.class_names
    print(class_names)

    ds_train = ds_train.prefetch(AUTOTUNE).cache()

    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        seed=128,
        image_size=(img_height, img_width),
        batch_size=batch_size).prefetch(AUTOTUNE).cache()

    # End Custom dataset

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

        model = tf.keras.models.Sequential()

        model.add(VisionTransformer(
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
        ds_val,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=[TensorBoard(log_dir=args.logdir, profile_batch=0), ],
    )

    test = model.predict(ds_val)

    # Plots
    plt.figure(figsize=(10, 10))
    for images, labels in ds_val.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

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
