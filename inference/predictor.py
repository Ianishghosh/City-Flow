import torch
import numpy as np
import pickle

from models.stgnn import STGNN
from data.dataset import StandardScaler


class Predictor:
    """
    Load trained model + scaler and run inference.

    Usage:
        predictor = Predictor(cfg)
        predictions = predictor.predict(speed_window, adj_mx)
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = cfg.device

        # Load scaler
        self.scaler = StandardScaler()
        self.scaler.load(cfg.scaler_path)

        # Build and load model
        self.model = STGNN(
            num_sensors  = cfg.num_sensors,
            input_steps  = cfg.input_steps,
            output_steps = cfg.output_steps,
            hidden       = cfg.hidden_dim
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(cfg.model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"✅ Model loaded from {cfg.model_path}")

    def predict(self, speed_window: np.ndarray, adj_mx: np.ndarray) -> np.ndarray:
        """
        Args:
            speed_window : (12, 207) — last 60 minutes of raw speed data
            adj_mx       : (207, 207) — adjacency matrix

        Returns:
            predictions  : (12, 207) — next 60 minutes in real mph
        """
        # Normalize input
        x_norm = self.scaler.transform(speed_window)

        # To tensor
        x_tensor   = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)   # (1,12,207)
        adj_tensor = torch.FloatTensor(adj_mx).to(self.device)                 # (207,207)

        # Inference
        with torch.no_grad():
            output = self.model(x_tensor, adj_tensor)                          # (1,12,207)

        # Inverse transform → real mph
        pred_np = output.squeeze(0).cpu().numpy()
        pred_real = self.scaler.inverse_transform(pred_np)

        return pred_real   # (12, 207)