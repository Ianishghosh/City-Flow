import numpy as np
import pickle
import pandas as pd


class StandardScaler:
    """Z-score normalizer — fit on train, apply to all splits."""

    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data):
        self.mean = np.mean(data)
        self.std  = np.std(data)

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)
        print(f"💾 Scaler saved → {path}")

    def load(self, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.mean = obj["mean"]
        self.std  = obj["std"]
        print(f"✅ Scaler loaded ← {path}")


def load_cleaned_data(cleaned_file):
    """Load the cleaned pickle produced by the preprocessing notebook."""
    with open(cleaned_file, "rb") as f:
        data = pickle.load(f)

    speed_data = data["speed_data"]   # (T, N)
    adj_mx     = data["adj_mx"]       # (N, N)

    # Fix any residual NaNs with forward/backward fill
    df         = pd.DataFrame(speed_data)
    df         = df.fillna(method="ffill").fillna(method="bfill")
    speed_data = df.values

    print(f"✅ Data loaded  | shape: {speed_data.shape}")
    print(f"✅ Adj loaded   | shape: {adj_mx.shape}")
    return speed_data, adj_mx


def create_sequences(data, input_steps=12, output_steps=12):
    """Sliding-window sequence creation."""
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i            : i + input_steps])
        y.append(data[i+input_steps: i + input_steps + output_steps])
    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.70, val_ratio=0.90):
    """Chronological train / val / test split."""
    n          = len(X)
    train_end  = int(n * train_ratio)
    val_end    = int(n * val_ratio)

    return (
        X[:train_end],  y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:],    y[val_end:]
    )