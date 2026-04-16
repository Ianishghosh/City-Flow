import torch
import torch.nn as nn
import os

from training.callbacks import EarlyStopping, ModelCheckpoint

# Optional MLflow — only used if installed
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Trainer:
    """Handles the full training loop for the ST-GNN model."""

    def __init__(self, model, cfg, adj_tensor, use_mlflow=True):
        self.model     = model
        self.cfg       = cfg
        self.device    = cfg.device
        self.adj       = adj_tensor.to(self.device)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.early_stopping = EarlyStopping(cfg.patience, cfg.min_delta)
        self.checkpoint     = ModelCheckpoint(cfg.best_model_path)

        self.train_losses = []
        self.val_losses   = []

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(X_batch, self.adj)
                loss   = self.criterion(output, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader):
        print(f"🚀 Training started | epochs={self.cfg.epochs} | device={self.device}")
        print("─" * 60)

        # ── MLflow run ────────────────────────────────────────────────
        mlflow_run = None
        if self.use_mlflow:
            mlflow.set_experiment("city-flow-stgnn")
            mlflow_run = mlflow.start_run(run_name="stgnn-training")
            mlflow.log_params({
                "epochs":       self.cfg.epochs,
                "batch_size":   self.cfg.batch_size,
                "lr":           self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
                "hidden_dim":   self.cfg.hidden_dim,
                "num_sensors":  self.cfg.num_sensors,
                "input_steps":  self.cfg.input_steps,
                "output_steps": self.cfg.output_steps,
                "device":       self.cfg.device,
            })
            print("📊 MLflow tracking enabled")

        try:
            for epoch in range(1, self.cfg.epochs + 1):
                train_loss = self._run_epoch(train_loader, train=True)
                val_loss   = self._run_epoch(val_loader,   train=False)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                saved = self.checkpoint(self.model, val_loss)
                self.scheduler.step(val_loss)
                self.early_stopping(val_loss)

                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss":   val_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }, step=epoch)

                if epoch % 10 == 0 or epoch == 1:
                    marker = " ← 💾 saved" if saved else ""
                    print(
                        f"Epoch [{epoch:3d}/{self.cfg.epochs}] | "
                        f"Train: {train_loss:.4f} | Val: {val_loss:.4f}{marker}"
                    )

                if self.early_stopping.early_stop:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break

        finally:
            if self.use_mlflow and mlflow_run:
                mlflow.log_metric("best_val_loss", self.checkpoint.best_loss)
                mlflow.log_artifact(self.cfg.best_model_path)
                mlflow.end_run()
                print("📊 MLflow run logged")

        print("─" * 60)
        print(f"✅ Training complete | Best Val Loss: {self.checkpoint.best_loss:.4f}")
        return self.train_losses, self.val_losses