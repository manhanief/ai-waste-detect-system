"""
model.py
--------
Defines the CNN architectures used in the project.

Two builders:
  - build_custom_cnn():     a CNN built from scratch (your baseline).
  - build_transfer_model(): MobileNetV2 pre-trained on ImageNet (the
                            "improvement" experiment).

Both reuse the augmentation/normalization layers from data_loader so the
exact same preprocessing is applied consistently.
"""

import tensorflow as tf
from data_loader import data_augmentation, normalization_layer


def build_custom_cnn(num_classes, img_size=224):
    """A simple CNN built from scratch.

    Pattern: 4 convolution blocks (each = Conv -> BatchNorm -> MaxPool) that
    progressively extract more complex features, then a dense classifier head.
    Filters grow 32 -> 64 -> 128 -> 128 as spatial size shrinks.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),

        data_augmentation,        # random flip/rotate/zoom/contrast (train only)
        normalization_layer,      # scale pixels 0-255 -> 0-1

        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 4
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Classifier head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),                  # reduces overfitting
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="custom_cnn")

    return model


def build_transfer_model(num_classes, img_size=224):
    """MobileNetV2 pre-trained on ImageNet with a new classifier head.

    The convolutional base is frozen so we only train the new top layers.
    This usually reaches much higher accuracy than the custom CNN because the
    base already knows generic visual features (edges, textures, shapes).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,          # drop ImageNet's 1000-class head
        weights="imagenet",
    )
    base.trainable = False          # freeze the pre-trained weights

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    # MobileNetV2 has its OWN preprocessing (scales to [-1, 1]); use it instead
    # of our Rescaling layer.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
