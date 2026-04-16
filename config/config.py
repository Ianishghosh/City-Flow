import yaml
import os
import torch


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            # Resolve relative to this file so it works from any CWD
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        # Project root = parent of the config/ directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(config_path, "r", encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f)

        # ── Data ──────────────────────────────────────────────────────
        self.raw_dir         = self._abs(cfg["data"]["raw_dir"])
        self.processed_dir   = self._abs(cfg["data"]["processed_dir"])
        self.h5_file         = self._abs(cfg["data"]["h5_file"])
        self.adj_file        = self._abs(cfg["data"]["adj_file"])
        self.cleaned_file    = self._abs(cfg["data"]["cleaned_file"])
        self.scaler_file     = self._abs(cfg["data"]["scaler_file"])

        # ── Model ─────────────────────────────────────────────────────
        self.num_sensors     = cfg["model"]["num_sensors"]
        self.input_steps     = cfg["model"]["input_steps"]
        self.output_steps    = cfg["model"]["output_steps"]
        self.hidden_dim      = cfg["model"]["hidden_dim"]

        # ── Training ──────────────────────────────────────────────────
        self.epochs          = cfg["training"]["epochs"]
        self.batch_size      = cfg["training"]["batch_size"]
        self.lr              = cfg["training"]["learning_rate"]
        self.weight_decay    = cfg["training"]["weight_decay"]
        self.patience        = cfg["training"]["patience"]
        self.min_delta       = cfg["training"]["min_delta"]
        self.train_ratio     = cfg["training"]["train_ratio"]
        self.val_ratio       = cfg["training"]["val_ratio"]
        self.checkpoint_dir  = cfg["training"]["checkpoint_dir"]
        self.best_model_path = cfg["training"]["best_model_path"]
        self.log_dir         = cfg["training"]["log_dir"]

        # ── Inference ─────────────────────────────────────────────────
        self.model_path      = self._abs(cfg["inference"]["model_path"])
        self.scaler_path     = self._abs(cfg["inference"]["scaler_path"])
        self.device          = self._resolve_device(cfg["inference"]["device"])

        # ── API ───────────────────────────────────────────────────────
        self.api_host        = cfg["api"]["host"]
        self.api_port        = cfg["api"]["port"]
        self.api_title       = cfg["api"]["title"]
        self.api_version     = cfg["api"]["version"]

    def _abs(self, path):
        """If path is relative, resolve it against the project root."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.project_root, path))

    def _resolve_device(self, device_str):
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

    def __repr__(self):
        return (
            f"<Config | sensors={self.num_sensors} | "
            f"hidden={self.hidden_dim} | epochs={self.epochs} | "
            f"device={self.device}>"
        )