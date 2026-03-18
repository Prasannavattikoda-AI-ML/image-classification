"""
Training script for image classification model.

Usage:
    # Train with sample CIFAR-10 dataset
    python train.py --download-data

    # Train with your own dataset
    python train.py --data-dir data/my_dataset

    # Use transfer learning for small datasets
    python train.py --data-dir data/my_dataset --model transfer
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn import build_cnn, build_transfer_model
from utils.preprocessing import create_data_generators, download_sample_dataset


def plot_training_history(history, save_path="results/training_history.png"):
    """Plot and save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def evaluate_model(model, val_ds, class_names):
    """Run full evaluation and print classification report."""
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred, class_names)

    return y_true, y_pred


def train(args):
    """Main training function."""
    # Prepare data
    if args.download_data:
        data_dir = download_sample_dataset("data")
        img_size = (32, 32)  # CIFAR-10 native size
    else:
        data_dir = args.data_dir
        img_size = (args.img_size, args.img_size)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Use --download-data to get a sample dataset, or provide --data-dir."
        )

    train_ds, val_ds, class_names = create_data_generators(
        data_dir, img_size=img_size, batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print(f"\nDataset: {num_classes} classes - {class_names}")

    # Build model
    input_shape = (*img_size, 3)
    if args.model == "cnn":
        model = build_cnn(input_shape, num_classes)
    else:
        model = build_transfer_model(input_shape, num_classes)

    model.summary()

    # Use lower learning rate for transfer learning
    lr = args.lr if args.model == "cnn" else min(args.lr, 1e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.keras", monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks,
    )

    # Evaluate
    plot_training_history(history)
    evaluate_model(model, val_ds, class_names)

    # Save class names for inference
    meta = {"class_names": class_names, "img_size": list(img_size)}
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete!")
    print(f"Best model saved to: models/best_model.keras")
    print(f"Model metadata saved to: models/model_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--data-dir", type=str, default="data/cifar10_sample",
                        help="Path to dataset directory with class subdirectories")
    parser.add_argument("--download-data", action="store_true",
                        help="Download sample CIFAR-10 dataset")
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn",
                        help="Model type: custom cnn or transfer learning")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size (ignored when using --download-data)")
    args = parser.parse_args()
    train(args)
