import torch
import torch.nn as nn
from src.data_loader import prepare_data
from src.model import get_model
from src.train import train_model, FocalLoss
from src.evaluate import evaluate_model
from src.utils import load_config, save_model, save_metrics

def main():
    # Load configuration
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("📥 Loading data...")
    train_loader, test_loader, class_names = prepare_data(config)

    # Build model
    print("🧠 Building model...")
    model = get_model(config["training"]["num_classes"])
    model.to(device)

    # ----- Weighted Loss for class imbalance -----
    class_counts = [102, 80, 119, 47, 50]  # Healthy, Mild DR, Moderate DR, Proliferate DR, Severe DR
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)  # Use FocalLoss

    # ----- Optimizer + Scheduler -----
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=config["training"]["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Fine-tuning: Unfreeze layer4 after initial training
    if config.get("training", {}).get("fine_tune", False):
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["lr"] / 10,
            weight_decay=1e-5
        )

    # Train
    print("🚀 Training model...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device,
                        num_epochs=config["training"]["epochs"], scheduler=scheduler)

    # Evaluate
    print("🔍 Evaluating model...")
    accuracy = evaluate_model(model, test_loader, device, class_names)

    # Save model and metrics
    save_model(model, config["training"]["save_path"])
    save_metrics({"accuracy": accuracy}, "outputs/metrics.json")

if __name__ == "__main__":
    main()