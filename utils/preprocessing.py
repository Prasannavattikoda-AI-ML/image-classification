"""
Data preprocessing and augmentation pipeline for image classification.
"""

import os
import tensorflow as tf


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Create training and validation data generators with augmentation.

    Args:
        data_dir: Path to dataset directory (expects subdirectories per class).
        img_size: Target image size (height, width).
        batch_size: Batch size for training.
        val_split: Fraction of data to use for validation.

    Returns:
        train_ds, val_ds, class_names
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    # Performance optimization
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def build_augmentation_layer():
    """
    Build a data augmentation layer to improve generalization on small datasets.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def download_sample_dataset(dest_dir="data"):
    """
    Download a small sample dataset (CIFAR-10) for quick experimentation.

    Args:
        dest_dir: Directory to save extracted images.

    Returns:
        Path to the created dataset directory.
    """
    import numpy as np
    from PIL import Image

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    dataset_dir = os.path.join(dest_dir, "cifar10_sample")

    # Save a subset (500 images per class) as individual files
    samples_per_class = 500
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        indices = np.where(y_train.flatten() == class_idx)[0][:samples_per_class]
        for i, idx in enumerate(indices):
            img = Image.fromarray(x_train[idx])
            img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.png"))

    print(f"Sample dataset saved to: {dataset_dir}")
    print(f"Classes: {class_names}")
    print(f"Images per class: {samples_per_class}")

    return dataset_dir
