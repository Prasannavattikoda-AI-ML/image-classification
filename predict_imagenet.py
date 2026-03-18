"""
Classify any image using MobileNetV2 pretrained on ImageNet (1000 classes).

No training required — uses pretrained weights directly.

Usage:
    python3 predict_imagenet.py --image path/to/image.jpg
    python3 predict_imagenet.py --image-dir path/to/images/
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)


def load_model():
    """Load MobileNetV2 with full ImageNet weights (1000 classes)."""
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet",
    )
    return model


def predict_image(model, image_path, top_k=5):
    """Classify a single image and return top-k predictions."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top_k)[0]

    return [(label, class_name, float(score)) for label, class_name, score in decoded]


def main():
    parser = argparse.ArgumentParser(description="Classify images using ImageNet (1000 classes)")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, help="Path to directory of images")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()

    print("Loading MobileNetV2 (ImageNet 1000 classes)...")
    model = load_model()

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
        results = predict_image(model, img_path, args.top_k)
        print(f"\n{img_path}:")
        for label_id, class_name, score in results:
            print(f"  {class_name:30s} {score:.2%}")


if __name__ == "__main__":
    main()
