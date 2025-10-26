Diabetic Retinopathy Classification with PyTorch
Image

Image

A comprehensive deep learning pipeline for classifying diabetic retinopathy stages from retinal images using transfer learning with ResNet18. The project addresses class imbalance, implements domain-specific preprocessing (CLAHE), and provides detailed evaluation metrics including confusion matrices.

Table of Contents
Features
Project Structure
Dataset
Installation
Configuration
Usage
Training Process
Outputs
Performance Improvements
Troubleshooting
Contributing
License
Features
Transfer Learning: Pretrained ResNet18 backbone with custom classifier head
Class Imbalance Handling: Focal Loss with class weights for minority classes
Domain-Specific Preprocessing: CLAHE (Contrast Limited Adaptive Histogram Equalization) for retinal image enhancement
Data Augmentation: Random flips, rotations, and color jitter for robust training
Fine-Tuning: Optional unfreezing of ResNet's layer4 for task-specific adaptation
Adaptive Learning Rate: ReduceLROnPlateau scheduler for optimal convergence
Comprehensive Evaluation: Precision, recall, F1-score, and confusion matrix visualization
Training Visualization: Loss and accuracy curves saved as PNG
Modular Architecture: Clean separation of data loading, model building, training, and evaluation
Project Structure
text
Diabetic-Retinopathy/
├── README.md                    # Project documentation
├── config.yaml                  # Configuration file
├── main.py                      # Main training and evaluation script
├── requirements.txt             # Python dependencies
├── outputs/                     # Generated outputs
│   ├── retina_classifier.pth    # Trained model weights
│   ├── metrics.json             # Performance metrics
│   ├── training_curves.png      # Training progress visualization
│   └── confusion_matrix.png     # Classification confusion matrix
├── data/                        # Dataset directory
│   └── retina_dataset/          # Extracted dataset
│       ├── Healthy/             # Class 0: Healthy retinal images
│       ├── Mild DR/             # Class 1: Mild Diabetic Retinopathy
│       ├── Moderate DR/         # Class 2: Moderate Diabetic Retinopathy
│       ├── Proliferate DR/      # Class 3: Proliferative Diabetic Retinopathy
│       └── Severe DR/           # Class 4: Severe Diabetic Retinopathy
└── src/                         # Source code modules
    ├── __init__.py              # Python package initializer
    ├── data_loader.py           # Dataset loading and preprocessing
    ├── evaluate.py              # Model evaluation and metrics
    ├── model.py                 # Model architecture definition
    ├── train.py                 # Training loop and loss functions
    └── utils.py                 # Configuration and file utilities
Dataset
Requirements
Format: ZIP archive containing class folders with retinal images (JPEG/PNG)
Classes: 5 classes - Healthy, Mild DR, Moderate DR, Proliferate DR, Severe DR
Structure: Each class folder should contain corresponding retinal images
Size: Recommended minimum 500+ images total (100+ per class for best results)
Resolution: Images will be resized to 224x224 during preprocessing
Expected Dataset Structure (after extraction)
text
data/retina_dataset/
├── Healthy/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── Mild DR/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── Moderate DR/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── Proliferate DR/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── Severe DR/
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
Sample Datasets
Kaggle APTOS 2019: Diabetic Retinopathy Detection
Messidor Dataset: Standard benchmark for diabetic retinopathy
Custom Dataset: Any ZIP file following the above structure
Installation
Prerequisites
Python 3.8 or higher
Windows/Linux/macOS
At least 8GB RAM (16GB recommended for faster training)
Optional: NVIDIA GPU with CUDA support for faster training
Step-by-Step Setup
Clone or Download the Repository

bash
git clone <your-repo-url>
cd Diabetic-Retinopathy
Create Virtual Environment

bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
Install Dependencies

bash
pip install -r requirements.txt
Or install manually:

bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn numpy pyyaml opencv-python
Download Dataset

Place your dataset ZIP file at: C:\Users\SKRIMGADGETS\Downloads\archive.zip
Update the zip_path in config.yaml if using a different location
Verify Installation

bash
python -c "import torch, torchvision, cv2; print('All dependencies installed successfully!')"
Configuration
The project uses a YAML configuration file (config.yaml) for easy customization:

Key Parameters
Data Configuration
yaml
data:
  zip_path: "C:\\Users\\SKRIMGADGETS\\Downloads\\archive.zip"  # Path to dataset ZIP
  extract_dir: "data/retina_dataset"                           # Extraction destination
  img_size: 224                                                # Input image size
  batch_size: 16                                               # Batch size for training
  train_split: 0.8                                             # Train/validation split ratio
  augment: True                                                # Enable data augmentation
Training Configuration
yaml
training:
  epochs: 20                                                   # Number of training epochs
  lr: 0.0001                                                   # Learning rate
  num_classes: 5                                               # Number of classes
  model_name: "resnet18"                                       # Backbone architecture
  save_path: "outputs/retina_classifier.pth"                   # Model save path
  fine_tune: True                                              # Enable fine-tuning of layer4
Customization Tips
Small datasets: Reduce batch_size to 8 and increase epochs to 30
GPU training: Increase batch_size to 32 and lr to 0.001
Class imbalance: Adjust class_counts in main.py to match your dataset
Faster training: Set fine_tune: false to freeze all layers
Usage
Basic Training
Run the complete pipeline with default settings:

bash
python main.py
Custom Configuration
Modify config.yaml parameters as needed
Update class_counts in main.py to match your dataset distribution
Run training:
bash
python main.py
Model Inference
To use the trained model for predictions:

python
import torch
from src.model import get_model
from src.utils import load_config
from PIL import Image
import torchvision.transforms as transforms

# Load model
config = load_config()
model = get_model(config["training"]["num_classes"])
model.load_state_dict(torch.load("outputs/retina_classifier.pth"))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict on single image
image = Image.open("path/to/retinal_image.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    class_names = ["Healthy", "Mild DR", "Moderate DR", "Proliferate DR", "Severe DR"]
    print(f"Prediction: {class_names[predicted.item()]}")
Training Process
The training pipeline follows these steps:

Data Preparation (data_loader.py)
Extracts dataset from ZIP to data/retina_dataset/
Applies CLAHE preprocessing for retinal image enhancement
Implements data augmentation (flips, rotations, color jitter)
Splits data into 80% training, 20% validation
Creates batched DataLoaders
Model Building (model.py)
Loads pretrained ResNet18 from ImageNet
Freezes early layers for feature extraction
Replaces classifier with custom 512→256→5 architecture
Enables fine-tuning of layer4 if configured
Training Loop (train.py)
Uses Focal Loss with class weights for imbalance handling
Implements AdamW optimizer with ReduceLROnPlateau scheduler
Tracks training loss and validation accuracy
Saves training curves to outputs/training_curves.png
Evaluation (evaluate.py)
Computes overall accuracy and class-wise metrics
Generates confusion matrix visualization
Saves detailed metrics to outputs/metrics.json
Model Saving (utils.py)
Saves trained model weights to outputs/retina_classifier.pth
Creates output directories automatically
Outputs
Generated Files
File	Description	Location
retina_classifier.pth	Trained model weights	outputs/
metrics.json	Performance metrics (accuracy, F1-scores)	outputs/
training_curves.png	Training loss vs validation accuracy plot	outputs/
confusion_matrix.png	Classification confusion matrix heatmap	outputs/
Sample Output
text
📜 Loaded config from config.yaml
📥 Loading data...
📦 Extracted dataset to data/retina_dataset
📊 Dataset loaded: 640 train, 160 test images
🧠 Building model...
🚀 Training model...
Epoch 1/20 | Training Loss: 1.4567 | Validation Accuracy: 62.34%
...
Epoch 20/20 | Training Loss: 0.8234 | Validation Accuracy: 78.56%
🔍 Evaluating model...
✅ Test Accuracy: 76.88%
                precision    recall  f1-score   support
       Healthy       0.85      0.92      0.88        32
       Mild DR       0.74      0.68      0.71        28
   Moderate DR       0.67      0.64      0.65        31
Proliferate DR       0.69      0.73      0.71        30
     Severe DR       0.82      0.80      0.81        39
      accuracy                           0.77       160
     macro avg       0.75      0.75      0.75       160
  weighted avg       0.77      0.77      0.77       160
💾 Model saved to outputs/retina_classifier.pth
📊 Metrics saved to outputs/metrics.json
Performance Improvements
This implementation addresses common challenges in diabetic retinopathy classification:

1. Class Imbalance
Problem: Severe DR and Proliferate DR have fewer samples
Solution: Focal Loss with normalized class weights prioritizes minority classes
Expected Impact: Improved recall for Severe DR (from 0.11 to ~0.70)
2. Feature Adaptation
Problem: ImageNet-pretrained features may not capture retinopathy patterns
Solution: Fine-tuning of ResNet's layer4 with reduced learning rate
Expected Impact: Better feature extraction for medical images
3. Image Quality
Problem: Retinal images have low contrast and noise
Solution: CLAHE preprocessing enhances vessel and lesion visibility
Expected Impact: Improved model sensitivity to pathological features
4. Training Stability
Problem: Fixed learning rate may cause suboptimal convergence
Solution: ReduceLROnPlateau scheduler adapts LR based on validation loss
Expected Impact: More stable training and higher final accuracy
Performance Comparison
Metric	Original	Improved	Improvement
Test Accuracy	55.53%	75-85%	+20-30%
Severe DR Recall	0.11	0.70-0.80	+60%
Macro F1-Score	0.48	0.70-0.75	+22%
Troubleshooting
Common Issues
Dataset Structure Error
text
FileNotFoundError: No such file or directory: 'data/retina_dataset'
Solution: Ensure archive.zip exists and contains class folders
Class Count Mismatch
text
ValueError: Expected 5 classes but found X
Solution: Update class_counts in main.py to match your dataset
Memory Issues
text
RuntimeError: CUDA out of memory
Solution: Reduce batch_size to 8 or 4 in config.yaml
Slow Training on CPU Solution:
Set num_workers: 0 in data_loader.py
Reduce batch_size to 8
Consider using Google Colab with GPU
OpenCV Import Error
bash
ModuleNotFoundError: No module named 'cv2'
Solution:
bash
pip install opencv-python
Pin Memory Warning
text
UserWarning: 'pin_memory' argument is set as true but no accelerator is found
Solution: This is harmless for CPU training; ignore or set pin_memory=False
Debug Steps
Verify Dataset Extraction:
bash
ls data/retina_dataset/
# Should show: Healthy, Mild DR, Moderate DR, Proliferate DR, Severe DR
Check Class Distribution: Add to data_loader.py after line 70:
python
from collections import Counter
class_counts = Counter([dataset.targets[i] for i in range(len(dataset))])
print(f"Class distribution: {dict(class_counts)}")
Test DataLoader:
python
from src.data_loader import prepare_data
from src.utils import load_config
config = load_config()
train_loader, test_loader, class_names = prepare_data(config)
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}, Labels: {labels}")
    break
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
Development Setup
bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/

# Run tests
pytest tests/
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
PyTorch Team: For the excellent deep learning framework
Kaggle Community: For diabetic retinopathy datasets
OpenCV Team: For robust image processing capabilities
Matplotlib & Seaborn: For beautiful visualizations
Contact
For questions or feedback, please open an issue on the GitHub repository or contact the maintainer.

Built with ❤️ for advancing diabetic retinopathy diagnosis through deep learning

Additional Notes for Developers
Extending the Model: Add new architectures in model.py by modifying get_model()
Custom Preprocessing: Extend custom_preprocess() in data_loader.py for additional retinal image enhancements
Advanced Metrics: Add ROC-AUC, sensitivity/specificity in evaluate.py
Model Ensemble: Implement multiple model fusion in main.py
Hyperparameter Search: Use Optuna or Ray Tune for automated tuning
Happy training! 🎯