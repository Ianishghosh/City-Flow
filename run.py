"""
run.py — City-Flow inference runner
Loads trained model and runs predictions on sample data.
"""

import numpy as np
import pickle

from config    import Config
from inference import Predictor


def main():
    cfg = Config()
    predictor = Predictor(cfg)

    # Load adjacency matrix
    with open(cfg.adj_file, "rb") as f:
        adj_raw = pickle.load(f, encoding='latin1')
        adj_mx = adj_raw[2] if isinstance(adj_raw, (tuple, list)) and len(adj_raw) == 3 else adj_raw  # (207, 207)

    # Simulate a speed window (replace with real sensor data in production)
    # Shape must be (12, 207) — last 60 minutes of readings
    dummy_window = np.random.uniform(30, 70, size=(12, 207)).astype(np.float32)

    predictions = predictor.predict(dummy_window, adj_mx)

    print(f"\n✅ Prediction shape  : {predictions.shape}  → (12 steps, 207 sensors)")
    print(f"✅ Speed range (mph) : {predictions.min():.2f} – {predictions.max():.2f}")
    print(f"\n📍 Sample — Sensor 0, next 12 timesteps (5-min intervals):")
    for t, speed in enumerate(predictions[:, 0]):
        print(f"   t+{(t+1)*5:2d} min → {speed:.2f} mph")


if __name__ == "__main__":
    main()