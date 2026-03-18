# Image Classification with CNN

CNN-based image classifier built with TensorFlow. Includes automated preprocessing, data augmentation, and support for both custom CNN and transfer learning (MobileNetV2).

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
image-classification/
├── train.py                 # Training script
├── predict.py               # Inference script
├── models/
│   └── cnn.py               # CNN & transfer learning architectures
├── utils/
│   └── preprocessing.py     # Data loading & augmentation pipeline
├── data/                    # Dataset directory
└── results/                 # Training plots & metrics
```

## Quick Start

### 1. Train with sample dataset (CIFAR-10)

```bash
python train.py --download-data
```

### 2. Train with your own dataset

Organize images into class subdirectories:

```
data/my_dataset/
├── class_a/
│   ├── img1.jpg
│   └── img2.jpg
├── class_b/
│   └── ...
```

Then train:

```bash
python train.py --data-dir data/my_dataset --epochs 30
```

### 3. Use transfer learning (recommended for small datasets)

```bash
python train.py --data-dir data/my_dataset --model transfer
```

### 4. Run predictions

```bash
python predict.py --image path/to/image.jpg
python predict.py --image-dir path/to/images/
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/cifar10_sample` | Dataset path |
| `--download-data` | - | Download CIFAR-10 sample |
| `--model` | `cnn` | `cnn` or `transfer` |
| `--epochs` | `30` | Max training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--img-size` | `224` | Image resize dimension |

## Features

- **Data Augmentation**: Random flip, rotation, zoom, and contrast
- **Early Stopping**: Stops training when validation accuracy plateaus
- **Learning Rate Scheduling**: Reduces LR on validation loss plateau
- **Evaluation**: Classification report + confusion matrix
- **Transfer Learning**: MobileNetV2 backbone for small datasets
