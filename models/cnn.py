"""
CNN model architectures for image classification.
"""

import tensorflow as tf
from utils.preprocessing import build_augmentation_layer


def build_cnn(input_shape=(224, 224, 3), num_classes=10, use_augmentation=True):
    """
    Build a custom CNN model for image classification.

    Architecture:
        - Data augmentation (optional)
        - Rescaling (0-255 -> 0-1)
        - 3 Conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout)
        - Global Average Pooling
        - Dense classifier head

    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        use_augmentation: Whether to include augmentation layers.

    Returns:
        Compiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if use_augmentation:
        x = build_augmentation_layer()(x)

    # Rescale pixel values to [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="image_classifier_cnn")
    return model


def build_transfer_model(input_shape=(224, 224, 3), num_classes=10, use_augmentation=True):
    """
    Build a transfer learning model using MobileNetV2 as the backbone.
    Best for small datasets where a custom CNN may overfit.

    Args:
        input_shape: Input image shape (H, W, C).
        num_classes: Number of output classes.
        use_augmentation: Whether to include augmentation layers.

    Returns:
        Compiled Keras model.
    """
    # Upscale small images to minimum 96x96 for MobileNetV2
    target_size = max(input_shape[0], 96)
    effective_shape = (target_size, target_size, 3)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    if input_shape[0] < 96:
        x = tf.keras.layers.Resizing(target_size, target_size)(x)

    if use_augmentation:
        x = build_augmentation_layer()(x)

    # MobileNetV2 expects inputs in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=effective_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze backbone initially

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_mobilenetv2")
    return model
