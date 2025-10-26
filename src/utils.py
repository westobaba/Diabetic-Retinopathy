import yaml
import torch
import json
import os

def load_config(path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"📜 Loaded config from {path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file {path} not found")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file {path}: {e}")
        raise

def save_model(model, path):
    """Save trained model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def save_metrics(metrics, path="outputs/metrics.json"):
    """Save performance metrics to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 Metrics saved to {path}")