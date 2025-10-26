import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """Pretrained ResNet18 with frozen layers and custom classifier."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers for feature extraction
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),  # Increased dropout for better regularization
        nn.Linear(256, num_classes)
    )
    return model