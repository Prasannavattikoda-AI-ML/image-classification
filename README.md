# Image Classification with CNN & Transfer Learning

A complete image classification pipeline built with TensorFlow/Keras, featuring a custom CNN and MobileNetV2 transfer learning. Includes automated preprocessing, data augmentation, training with early stopping, evaluation with confusion matrices, and inference scripts for both custom-trained models and pretrained ImageNet (1000 classes).

## Results

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Custom CNN (3-block) | CIFAR-10 (500/class) | 48.1% |
| MobileNetV2 Transfer | CIFAR-10 (500/class) | 79.4% |
| MobileNetV2 ImageNet | ImageNet (pretrained) | 1000 classes |

## Project Structure

```
image-classification/
├── models/
│   ├── cnn.py                 # Custom CNN & MobileNetV2 architectures
│   ├── best_model.keras       # Saved best model weights
│   └── model_meta.json        # Class names & image size metadata
├── utils/
│   └── preprocessing.py       # Data loading, augmentation, CIFAR-10 downloader
├── results/
│   ├── training_history.png   # Accuracy/loss curves
│   └── confusion_matrix.png   # Per-class evaluation
├── train.py                   # Training script with CLI arguments
├── predict.py                 # Inference using trained model
├── predict_imagenet.py        # Inference using pretrained ImageNet (1000 classes)
└── requirements.txt           # Python dependencies
```

## Setup

```bash
git clone https://github.com/Prasannavattikoda-AI-ML/image-classification.git
cd image-classification
pip3 install -r requirements.txt
```

**Requirements:** Python 3.9+, TensorFlow 2.12+, NumPy, Matplotlib, scikit-learn, Pillow

## Usage

### Quick Start - Train on CIFAR-10

```bash
python3 train.py --download-data
```

This downloads 5,000 CIFAR-10 images (500 per class), trains the custom CNN, saves the best model, and generates evaluation plots.

### Train with Your Own Dataset

Organize your images into class subdirectories:

```
data/your_dataset/
├── cats/
│   ├── img001.jpg
│   └── img002.jpg
├── dogs/
│   ├── img001.jpg
│   └── ...
```

Then train:

```bash
# Custom CNN
python3 train.py --data-dir data/your_dataset --img-size 224

# Transfer learning (recommended for small datasets)
python3 train.py --data-dir data/your_dataset --model transfer --img-size 224
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/cifar10_sample` | Path to dataset directory |
| `--download-data` | off | Download CIFAR-10 sample dataset |
| `--model` | `cnn` | `cnn` or `transfer` (MobileNetV2) |
| `--epochs` | 30 | Maximum training epochs |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 0.001 | Learning rate (auto-reduced for transfer) |
| `--img-size` | 224 | Image resize dimension |

### Predict with Trained Model

```bash
# Single image
python3 predict.py --image path/to/image.jpg

# Folder of images
python3 predict.py --image-dir path/to/images/
```

### Predict with ImageNet (1000 Classes)

No training needed - uses pretrained MobileNetV2 weights:

```bash
# Single image
python3 predict_imagenet.py --image photo.jpg

# Folder of images
python3 predict_imagenet.py --image-dir photos/ --top-k 5
```

## Architecture

### Custom CNN

```
Input (H x W x 3)
  -> Data Augmentation (flip, rotation, zoom, contrast)
  -> Rescaling (0-1)
  -> Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
  -> Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
  -> Conv2D(128) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
  -> GlobalAveragePooling2D
  -> Dense(256, ReLU) -> Dropout(0.5)
  -> Dense(num_classes, Softmax)
```

### Transfer Learning (MobileNetV2)

```
Input (H x W x 3)
  -> Data Augmentation
  -> MobileNetV2 Backbone (frozen, ImageNet weights)
  -> GlobalAveragePooling2D
  -> Dense(128, ReLU) -> Dropout(0.3)
  -> Dense(num_classes, Softmax)
```

## Training Features

- **Early Stopping** - Stops training when validation accuracy plateaus (patience=5)
- **Learning Rate Scheduling** - Reduces LR by 50% when validation loss stalls (patience=3)
- **Model Checkpointing** - Saves only the best model based on validation accuracy
- **Data Augmentation** - Random flip, rotation (15%), zoom (10%), contrast (10%)
- **Evaluation** - Classification report + confusion matrix generated automatically

## Live Demo

Try the browser-based classifier on the [portfolio site](https://Prasannavattikoda-AI-ML.github.io) - uses TensorFlow.js with MobileNetV2 for client-side inference (no server needed).

## Technologies

- **TensorFlow / Keras** - Model building, training, and inference
- **MobileNetV2** - Pretrained backbone for transfer learning and ImageNet classification
- **scikit-learn** - Classification report and confusion matrix
- **Matplotlib** - Training curves and evaluation plots
- **Pillow** - Image loading and preprocessing
- **TensorFlow.js** - Browser-based inference in portfolio demo
