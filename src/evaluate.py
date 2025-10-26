import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set and compute metrics.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'
        class_names: List of class names
    Returns:
        accuracy: Overall test accuracy
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = 100 * sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    print(f"✅ Test Accuracy: {accuracy:.2f}%")

    # Compute precision, recall, f1-score
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print("\n                precision    recall  f1-score   support\n")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>20}    {precision[i]:.2f}      {recall[i]:.2f}      {f1[i]:.2f}       {support[i]}")

    print(f"\n      accuracy                          {accuracy/100:.2f}       {len(all_labels)}")
    print(f"     macro avg       {np.mean(precision):.2f}      {np.mean(recall):.2f}      {np.mean(f1):.2f}       {len(all_labels)}")
    print(f"  weighted avg       {np.average(precision, weights=support):.2f}      {np.average(recall, weights=support):.2f}      {np.average(f1, weights=support):.2f}       {len(all_labels)}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    return accuracy