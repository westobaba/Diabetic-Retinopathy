import os
import zipfile
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from PIL import Image
import random

def custom_preprocess(image):
    """Apply CLAHE preprocessing for retinal images."""
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

def prepare_data(config):
    """
    Prepare DataLoaders for training and testing.

    Args:
        config: Dictionary from load_config
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        class_names: List of class names
    """
    # Extract dataset if not already extracted
    zip_path = config["data"]["zip_path"]
    extract_dir = config["data"]["extract_dir"]
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"📦 Extracted dataset to {extract_dir}")

    # Define transforms
    img_size = config["data"]["img_size"]
    augment = config["data"].get("augment", False)
    
    if augment:
        transform_train = transforms.Compose([
            transforms.Lambda(custom_preprocess),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Lambda(custom_preprocess),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_test = transforms.Compose([
        transforms.Lambda(custom_preprocess),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset (assuming folder structure: extract_dir/class_name/images)
    dataset = datasets.ImageFolder(root=extract_dir, transform=None)
    class_names = dataset.classes

    # Split into train and test
    train_split = config["data"]["train_split"]
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    train_size = int(train_split * total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Apply transforms to subsets
    train_dataset.dataset.transform = transform_train
    test_dataset.dataset.transform = transform_test

    # Create DataLoaders
    pin_memory = torch.cuda.is_available()  # Only use pin_memory if CUDA is available
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory
    )

    print(f"📊 Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test images")
    return train_loader, test_loader, class_names