"""
Inference script for classifying images using a trained model.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image-dir path/to/images/
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def load_model_and_meta(model_path="models/best_model.keras", meta_path="models/model_meta.json"):
    """Load trained model and metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    model = tf.keras.models.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta["class_names"], tuple(meta["img_size"])


def predict_image(model, image_path, class_names, img_size, top_k=3):
    """
    Predict the class of a single image.

    Returns:
        List of (class_name, confidence) tuples for top-k predictions.
    """
    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_array = np.expand_dims(np.array(img), axis=0).astype("float32")

    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]

    results = [(class_names[i], float(predictions[i])) for i in top_indices]
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify images using trained model")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Path to directory of images")
    parser.add_argument("--model", type=str, default="models/best_model.keras")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model, class_names, img_size = load_model_and_meta(args.model)

    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
    else:
        parser.error("Provide --image or --image-dir")

    for img_path in images:
        results = predict_image(model, img_path, class_names, img_size, args.top_k)
        print(f"\n{img_path}:")
        for class_name, confidence in results:
            print(f"  {class_name:20s} {confidence:.2%}")


if __name__ == "__main__":
    main()
