import numpy as np


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_all_metrics(y_true, y_pred):
    """Return dict of all metrics."""
    return {
        "MAE" : round(mae(y_true,  y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
    }


def print_metrics(metrics: dict):
    print("\n📊 Evaluation Metrics")
    print("─" * 35)
    print(f"   MAE  : {metrics['MAE']:.4f}  mph")
    print(f"   RMSE : {metrics['RMSE']:.4f}  mph")
    print(f"   MAPE : {metrics['MAPE']:.4f}  %")
    print("─" * 35)