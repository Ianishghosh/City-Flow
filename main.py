"""
main.py — City-Flow full pipeline
Runs: data loading → preprocessing → training → evaluation
"""

import torch
import numpy as np

from config       import Config
from data         import load_cleaned_data, create_sequences, split_data, StandardScaler, make_loaders
from models       import build_model
from training     import Trainer
from utils        import compute_all_metrics, print_metrics, plot_training_curves, plot_predictions, plot_error_distribution


def main():
    # ── 1. Config ─────────────────────────────────────────────────────
    cfg = Config()
    print(cfg)

    # ── 2. Load data ──────────────────────────────────────────────────
    speed_data, adj_mx = load_cleaned_data(cfg.cleaned_file)

    # ── 3. Normalize ──────────────────────────────────────────────────
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(speed_data)

    # ── 4. Sequences ──────────────────────────────────────────────────
    X, y = create_sequences(data_norm, cfg.input_steps, cfg.output_steps)
    print(f"✅ Sequences | X={X.shape} | y={y.shape}")

    # ── 5. Split ──────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, cfg.train_ratio, cfg.val_ratio
    )

    # Re-fit scaler on train only
    scaler_final = StandardScaler()
    scaler_final.fit(X_train)
    X_train = scaler_final.transform(X_train)
    X_val   = scaler_final.transform(X_val)
    X_test  = scaler_final.transform(X_test)
    y_train = scaler_final.transform(y_train)
    y_val   = scaler_final.transform(y_val)
    y_test  = scaler_final.transform(y_test)
    scaler_final.save(cfg.scaler_file)

    # ── 6. DataLoaders ────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg.batch_size
    )

    # ── 7. Model ──────────────────────────────────────────────────────
    model      = build_model(cfg).to(cfg.device)
    adj_tensor = torch.FloatTensor(adj_mx)
    print(f"✅ Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 8. Train ──────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, adj_tensor)
    train_losses, val_losses = trainer.fit(train_loader, val_loader)

    # ── 9. Evaluate ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(cfg.best_model_path))
    model.eval()

    all_preds, all_truths = [], []
    adj_dev = adj_tensor.to(cfg.device)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out = model(X_batch.to(cfg.device), adj_dev)
            all_preds.append(out.cpu().numpy())
            all_truths.append(y_batch.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_truths = np.concatenate(all_truths, axis=0)

    preds_real  = scaler_final.inverse_transform(all_preds)
    truths_real = scaler_final.inverse_transform(all_truths)

    metrics = compute_all_metrics(truths_real, preds_real)
    print_metrics(metrics)

    # ── 10. Visualize ─────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses)
    plot_predictions(truths_real, preds_real)
    plot_error_distribution(truths_real, preds_real)

    print("\n🏁 City-Flow pipeline complete!")


if __name__ == "__main__":
    main()