import torch
import os


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=15, min_delta=0.0001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpoint:
    """Save the best model weights automatically."""

    def __init__(self, save_path="models/best_model.pth"):
        self.save_path  = save_path
        self.best_loss  = float("inf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            return True   # saved
        return False       # not saved