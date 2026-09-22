#!/usr/bin/env python3
"""Convert the trained Bi-LSTM (models/bilstm_exercise.pt) to a Core ML package for iOS.

Steps
  1. rebuild the PyTorch model from bilstm_exercise.json and load the weights
  2. trace it with a fixed (1, steps, inputs) window and convert with coremltools
     (fp16 weights, ML Program, iOS 16+ so it runs on the Neural Engine)
  3. bake the class names and preprocessing constants (rate, window, mean, std) into
     the model's metadata so the Swift side has everything it needs
  4. verify: run real windows from the CSVs through both PyTorch and Core ML and
     compare probabilities and predicted labels

    python export_coreml.py                      # writes models/ExerciseClassifier.mlpackage
    python export_coreml.py --no-verify

Swift usage sketch (see the printed summary for exact names):
    let model = try ExerciseClassifier()
    let input = try MLMultiArray(shape: [1, 30, 46], dataType: .float32)   // resampled window
    let out = try model.prediction(features: input)
    // out.probabilities  -> [pullup, pushup, squat]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

from train_bilstm import DEFAULT_FOLDERS, BiLSTMClassifier, Normalizer, build_windows, load_clips

ROOT = Path(__file__).resolve().parent


def load_model(model_dir: Path):
    with (model_dir / "bilstm_exercise.json").open() as handle:
        config = json.load(handle)
    model = BiLSTMClassifier(config["inputs"], len(config["classes"]), config["hidden"])
    model.load_state_dict(torch.load(model_dir / "bilstm_exercise.pt", map_location="cpu"))
    model.eval()
    return model, config


class WithSoftmax(torch.nn.Module):
    """Wrap the classifier so the Core ML output is probabilities, not logits."""

    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, features):
        return torch.softmax(self.classifier(features), dim=1)


def convert(model, config, output: Path) -> ct.models.MLModel:
    steps, inputs = config["steps"], config["inputs"]
    example = torch.zeros(1, steps, inputs)
    traced = torch.jit.trace(WithSoftmax(model).eval(), example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=(1, steps, inputs), dtype=np.float32)],
        outputs=[ct.TensorType(name="probabilities", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )
    classes = config["classes"]
    mlmodel.short_description = "Exercise type classifier (Bi-LSTM) on Apple Vision body-pose features"
    mlmodel.input_description["features"] = (
        f"(1, {steps}, {inputs}) float32: {steps} steps at {config['rate_hz']:g} Hz covering {config['window_s']:g} s; "
        f"first {inputs // 2} channels are standardised features, last {inputs // 2} are validity masks (1 = valid, 0 = missing)"
    )
    mlmodel.output_description["probabilities"] = f"softmax over classes {classes}"
    # everything Swift needs to reproduce the preprocessing lives in the metadata
    mlmodel.user_defined_metadata["classes"] = json.dumps(classes)
    mlmodel.user_defined_metadata["feature_columns"] = json.dumps(config["feature_columns"])
    mlmodel.user_defined_metadata["feature_mean"] = json.dumps(config["mean"])
    mlmodel.user_defined_metadata["feature_std"] = json.dumps(config["std"])
    mlmodel.user_defined_metadata["rate_hz"] = str(config["rate_hz"])
    mlmodel.user_defined_metadata["window_s"] = str(config["window_s"])
    mlmodel.user_defined_metadata["mask_tolerance_s"] = str(config["mask_tolerance_s"])
    mlmodel.user_defined_metadata["preprocessing"] = (
        "per feature: value = (raw - mean) / std * mask; input = concat(values, masks) along the channel axis; "
        "raw features are the 7 angles (degrees) and 16 distances from build_features.py"
    )
    mlmodel.save(str(output))
    return mlmodel


def verify(mlmodel: ct.models.MLModel, model, config, n_windows: int) -> None:
    clips, _ = load_clips(DEFAULT_FOLDERS)
    values, mask, labels, _ = build_windows(clips)
    normalizer = Normalizer()
    normalizer.mean, normalizer.std = np.array(config["mean"]), np.array(config["std"])
    x = normalizer(values, mask)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(x), size=min(n_windows, len(x)), replace=False)

    with torch.no_grad():
        torch_probs = torch.softmax(model(torch.from_numpy(x[pick])), 1).numpy()
    coreml_probs = np.stack([
        mlmodel.predict({"features": x[i][None].astype(np.float32)})["probabilities"].reshape(-1) for i in pick
    ])
    classes = config["classes"]
    truth = np.array([classes.index(l) for l in labels[pick]])
    max_abs_diff = np.abs(torch_probs - coreml_probs).max()
    agreement = (torch_probs.argmax(1) == coreml_probs.argmax(1)).mean()
    coreml_acc = (coreml_probs.argmax(1) == truth).mean()
    print(f"verification on {len(pick)} real windows:")
    print(f"  max |torch - coreml| probability difference: {max_abs_diff:.4f}")
    print(f"  argmax agreement torch vs coreml:            {agreement:.3f}")
    print(f"  coreml accuracy on these (training) windows: {coreml_acc:.3f}")
    if max_abs_diff > 0.02 or agreement < 0.99:
        raise SystemExit("Core ML output diverges from PyTorch - do not ship this model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ExerciseClassifier.mlpackage")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verify-windows", type=int, default=500)
    args = parser.parse_args()

    model, config = load_model(args.model_dir)
    mlmodel = convert(model, config, args.output)
    size_kb = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file()) / 1024
    print(f"wrote {args.output} ({size_kb:.0f} KB on disk)")
    print(f"  input  'features'      shape (1, {config['steps']}, {config['inputs']}) float32")
    print(f"  output 'probabilities' shape (1, {len(config['classes'])}) -> {config['classes']}")
    if not args.no_verify:
        verify(mlmodel, model, config, args.verify_windows)


if __name__ == "__main__":
    main()
