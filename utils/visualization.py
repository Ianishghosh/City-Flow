import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, val_losses, save_path="logs/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("City-Flow — Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Training curve saved → {save_path}")


def plot_predictions(y_true, y_pred, sensor_idx=0, n_samples=100,
                     save_path="logs/predictions.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    true_vals = y_true[:n_samples, 0, sensor_idx]
    pred_vals = y_pred[:n_samples, 0, sensor_idx]

    plt.figure(figsize=(12, 4))
    plt.plot(true_vals, label="Ground Truth", color="blue",  alpha=0.8)
    plt.plot(pred_vals, label="Predicted",    color="red",   alpha=0.8)
    plt.xlabel("Time Steps")
    plt.ylabel("Speed (mph)")
    plt.title(f"City-Flow — Sensor {sensor_idx} Predicted vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Prediction plot saved → {save_path}")


def plot_error_distribution(y_true, y_pred, save_path="logs/error_dist.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    errors = (y_true - y_pred).flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Prediction Error (mph)")
    plt.ylabel("Frequency")
    plt.title("City-Flow — Error Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Error distribution saved → {save_path}")