"""
export_onnx.py — City-Flow ONNX Model Export
Exports the trained ST-GNN to ONNX format and verifies it with ONNXRuntime.

Usage:
    python export_onnx.py
    python export_onnx.py --output models/city_flow_custom.onnx
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config    import Config
from models    import build_model


def export(output_path: str = None):
    cfg = Config()

    if output_path is None:
        output_path = os.path.join(cfg.project_root, "models", "city_flow.onnx")

    # ── Load model ────────────────────────────────────────────────────
    print(f"📦 Loading model from: {cfg.model_path}")
    model = build_model(cfg)
    state = torch.load(cfg.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Model loaded  | params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Dummy inputs (required by ONNX tracer) ────────────────────────
    # x   : (batch=1, input_steps=12, num_sensors=207)
    # adj : (num_sensors=207, num_sensors=207)
    dummy_x   = torch.randn(1, cfg.input_steps, cfg.num_sensors)
    dummy_adj = torch.randn(cfg.num_sensors, cfg.num_sensors)

    # ── Export ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n🔄 Exporting to ONNX → {output_path}")

    torch.onnx.export(
        model,
        (dummy_x, dummy_adj),
        output_path,
        export_params        = True,
        opset_version        = 17,
        do_constant_folding  = True,
        input_names          = ["speed_window", "adj_matrix"],
        output_names         = ["predictions"],
        dynamic_axes         = {
            "speed_window": {0: "batch_size"},
            "predictions":  {0: "batch_size"},
        },
    )
    print(f"✅ ONNX model saved → {output_path}")

    # ── Verify with ONNX ──────────────────────────────────────────────
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✅ ONNX graph check passed")

        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📏 Model size: {model_size_mb:.2f} MB")
    except ImportError:
        print("⚠️  onnx not installed — skipping graph check")

    # ── Verify with ONNXRuntime ────────────────────────────────────────
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        ort_inputs = {
            "speed_window": dummy_x.numpy(),
            "adj_matrix":   dummy_adj.numpy(),
        }
        ort_out = sess.run(["predictions"], ort_inputs)[0]

        # Compare against PyTorch output
        with torch.no_grad():
            torch_out = model(dummy_x, dummy_adj).numpy()

        max_diff = np.abs(ort_out - torch_out).max()
        print(f"✅ ONNXRuntime inference OK | max diff vs PyTorch: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("✅ Numerical match confirmed (diff < 1e-4)")
        else:
            print(f"⚠️  Numerical diff is {max_diff:.6f} — consider checking opset")

    except ImportError:
        print("⚠️  onnxruntime not installed — skipping runtime verification")

    print("\n🏁 ONNX export complete!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export City-Flow model to ONNX")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output ONNX file path (default: models/city_flow.onnx)"
    )
    args = parser.parse_args()
    export(args.output)
