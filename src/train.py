import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.softmax(inputs, dim=1)[range(inputs.size(0)), targets]
        loss = ce_loss * ((1 - pt) ** self.gamma)
        return loss.mean()

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, scheduler=None):
    """
    Train the model with optional scheduler and validation during training.

    Args:
        model: torch model
        train_loader: training DataLoader
        test_loader: validation/test DataLoader
        criterion: loss function
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        num_epochs: int, number of epochs
        scheduler: optional learning rate scheduler
    """
    model.to(device)
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {avg_loss:.4f}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # Validation accuracy per epoch
        if test_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch+1}/{num_epochs} | Validation Accuracy: {val_acc:.2f}%")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='#1f77b4')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('outputs/training_curves.png')
    plt.close()

    return model