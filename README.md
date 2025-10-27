# Diabetic Retinopathy Classification with PyTorch

A comprehensive deep learning pipeline for classifying diabetic retinopathy (DR) stages from retinal images using **transfer learning** with **ResNet18**. This project addresses class imbalance, implements domain-specific preprocessing (CLAHE), and provides detailed evaluation metrics including confusion matrices.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training Process](#training-process)
- [Outputs](#outputs)
- [Performance Improvements](#performance-improvements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Transfer Learning:** Pretrained ResNet18 backbone with custom classifier head
- **Class Imbalance Handling:** Focal Loss with class weights
- **Domain-Specific Preprocessing:** CLAHE for retinal image enhancement
- **Data Augmentation:** Random flips, rotations, and color jitter
- **Fine-Tuning:** Optional unfreezing of ResNet's layer4
- **Adaptive Learning Rate:** ReduceLROnPlateau scheduler
- **Evaluation:** Precision, recall, F1-score, confusion matrix
- **Training Visualization:** Loss and accuracy curves saved as PNG
- **Modular Architecture:** Clean separation of data loading, model building, training, and evaluation

---

## Project Structure
Diabetic-Retinopathy/
├── README.md
├── config.yaml
├── main.py
├── requirements.txt
├── outputs/
│ ├── retina_classifier.pth
│ ├── metrics.json
│ ├── training_curves.png
│ └── confusion_matrix.png
├── data/
│ └── retina_dataset/
│ ├── Healthy/
│ ├── Mild DR/
│ ├── Moderate DR/
│ ├── Proliferate DR/
│ └── Severe DR/
└── src/
├── init.py
├── data_loader.py
├── evaluate.py
├── model.py
├── train.py
└── utils.py

yaml
Copy code

---

## Dataset
- **Format:** ZIP archive containing class folders
- **Classes:** 5 - Healthy, Mild DR, Moderate DR, Proliferate DR, Severe DR
- **Recommended Size:** 500+ images (100+ per class)
- **Resolution:** Resized to 224x224 during preprocessing
- **Sources:** [Kaggle APTOS 2019](https://www.kaggle.com/c/diabetic-retinopathy-detection), Messidor, or custom datasets

---

## Installation
### Prerequisites
- Python 3.8+
- Windows/Linux/macOS
- At least 8GB RAM (16GB recommended)
- Optional NVIDIA GPU with CUDA for faster training

### Setup
```bash
git clone https://github.com/westobaba/Diabetic-Retinopathy
cd Diabetic-Retinopathy
python -m venv venv
# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
Place your dataset ZIP file in a convenient location and update zip_path in config.yaml.

Verify installation:

bash
Copy code
python -c "import torch, torchvision, cv2; print('All dependencies installed!')"
Configuration
All settings are in config.yaml:

yaml
Copy code
data:
  zip_path: "C:\\Users\\SKRIMGADGETS\\Downloads\\archive.zip"
  extract_dir: "data/retina_dataset"
  img_size: 224
  batch_size: 16
  train_split: 0.8
  augment: True

training:
  epochs: 20
  lr: 0.0001
  num_classes: 5
  model_name: "resnet18"
  save_path: "outputs/retina_classifier.pth"
  fine_tune: True
Tips:

Small datasets → reduce batch_size, increase epochs

GPU → increase batch_size for faster training

Class imbalance → adjust class_counts in main.py

Freeze layers for faster training → fine_tune: False

Usage
Training
bash
Copy code
python main.py
This will:

Extract dataset

Apply CLAHE & data augmentation

Train ResNet18 with focal loss

Save model & training metrics

Inference
python
Copy code
import torch
from src.model import get_model
from src.utils import load_config
from PIL import Image
import torchvision.transforms as transforms

config = load_config()
model = get_model(config["training"]["num_classes"])
model.load_state_dict(torch.load("outputs/retina_classifier.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

image = Image.open("path/to/retinal_image.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    class_names = ["Healthy","Mild DR","Moderate DR","Proliferate DR","Severe DR"]
    print(f"Prediction: {class_names[predicted.item()]}")
Training Process
Data Preparation: Extract, preprocess, augment, split

Model Building: Load pretrained ResNet18, replace classifier

Training Loop: Focal Loss + AdamW + ReduceLROnPlateau

Evaluation: Confusion matrix, precision, recall, F1

Model Saving: Outputs saved in outputs/

Outputs
File	Description
retina_classifier.pth	Trained model weights
metrics.json	Accuracy, F1-scores
training_curves.png	Loss & accuracy plots
confusion_matrix.png	Confusion matrix heatmap

Performance Improvements
Class Imbalance: Focal Loss with class weights

Feature Adaptation: Fine-tuning ResNet layer4

Image Quality: CLAHE enhances retinal vessels & lesions

Training Stability: ReduceLROnPlateau scheduler

Troubleshooting
FileNotFoundError: Ensure ZIP file exists & correct zip_path

Class Count Mismatch: Update class_counts in main.py

CUDA OOM: Reduce batch_size

OpenCV Error: pip install opencv-python

Contributing
Fork repository

Create a feature branch

Commit your changes

Push and create a Pull Request


Acknowledgments
PyTorch, OpenCV, Matplotlib, Seaborn

Kaggle datasets

Built with ❤️ to advance diabetic retinopathy diagnosis using deep learning.
