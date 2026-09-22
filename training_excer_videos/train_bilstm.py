#!/usr/bin/env python3
"""Train and evaluate a small Bi-LSTM exercise classifier on the per-exercise CSVs.

Input: <folder>/outputs/{landmarks,angles,distances}.csv for each exercise folder.
Features per frame: 7 angles + 16 distances (23), each with a validity mask (46 inputs).

Frame-rate handling: windows are defined in seconds and resampled onto a fixed
15 Hz grid using the frame timestamps (linear interpolation over valid samples),
so 24/25/30 fps clips and the phone's live feed all look the same to the model.
Time-scale augmentation (+-25%) covers rep-speed variation on top of that.

Evaluation: 5-fold cross-validation grouped by clip (no frame leakage), reporting
per-window and per-clip (majority vote) accuracy. Then a final model is trained
on all clips and saved with its normalisation statistics.

    python train_bilstm.py                       # defaults: "pull Up" push-up squat
    python train_bilstm.py --epochs 40 --hidden 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDERS = {"pull Up": "pullup", "push-up": "pushup", "squat": "squat"}

RATE = 15.0          # Hz of the resampled grid
WINDOW_S = 2.0       # seconds per window
STRIDE_S = 0.25      # seconds between window starts (eval / final training)
STEPS = int(RATE * WINDOW_S)
MASK_TOLERANCE_S = 0.15   # a grid point is valid if a real sample lies within this distance
MIN_VALID_FRACTION = 0.5  # drop windows with less coverage than this


# ---------------------------------------------------------------- data loading
def load_clips(folders: dict[str, str]) -> list[dict]:
    clips = []
    for folder, label in folders.items():
        out = ROOT / folder / "outputs"
        landmarks = pd.read_csv(out / "landmarks.csv", usecols=["vid_id", "frame_order", "timestamp"])
        angles = pd.read_csv(out / "angles.csv")
        distances = pd.read_csv(out / "distances.csv")
        table = landmarks.merge(angles, on=["vid_id", "frame_order"]).merge(distances, on=["vid_id", "frame_order"])
        feature_columns = [c for c in table.columns if c not in ("vid_id", "frame_order", "timestamp")]
        for vid_id, clip in table.groupby("vid_id"):
            clip = clip.sort_values("frame_order")
            clips.append({
                "group": f"{label}_{vid_id}",
                "label": label,
                "t": clip["timestamp"].to_numpy(dtype=np.float64),
                "X": clip[feature_columns].to_numpy(dtype=np.float32),
            })
    print(f"loaded {len(clips)} clips, {sum(len(c['t']) for c in clips)} frames, {len(feature_columns)} features")
    return clips, feature_columns


def resample_window(t: np.ndarray, X: np.ndarray, end: float, scale: float = 1.0):
    """Resample the window ending at `end` onto the fixed grid. scale>1 stretches time (slower rep)."""
    grid = end - (STEPS - 1 - np.arange(STEPS)) / RATE * scale
    values = np.zeros((STEPS, X.shape[1]), dtype=np.float32)
    mask = np.zeros((STEPS, X.shape[1]), dtype=np.float32)
    for j in range(X.shape[1]):
        ok = ~np.isnan(X[:, j])
        if ok.sum() < 2:
            continue
        tv = t[ok]
        values[:, j] = np.interp(grid, tv, X[ok, j])
        idx = np.clip(np.searchsorted(tv, grid), 1, len(tv) - 1)
        nearest = np.minimum(np.abs(grid - tv[idx - 1]), np.abs(grid - tv[idx]))
        mask[:, j] = nearest <= MASK_TOLERANCE_S * scale
    values[mask == 0] = 0.0
    return values, mask


def window_ends(t: np.ndarray, stride_s: float) -> np.ndarray:
    if t[-1] - t[0] < WINDOW_S * 0.75:      # very short clip: use it once, padded by the mask
        return np.array([t[-1]])
    return np.arange(t[0] + WINDOW_S, t[-1] + 1e-6, stride_s)


def build_windows(clips: list[dict], stride_s: float = STRIDE_S):
    xs, ms, ys, groups = [], [], [], []
    for clip in clips:
        for end in window_ends(clip["t"], stride_s):
            values, mask = resample_window(clip["t"], clip["X"], end)
            if mask.mean() < MIN_VALID_FRACTION:
                continue
            xs.append(values); ms.append(mask); ys.append(clip["label"]); groups.append(clip["group"])
    return np.stack(xs), np.stack(ms), np.array(ys), np.array(groups)


class Normalizer:
    """Per-feature mean/std computed over valid entries only."""

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "Normalizer":
        flat_v, flat_m = values.reshape(-1, values.shape[-1]), mask.reshape(-1, mask.shape[-1])
        self.mean = (flat_v * flat_m).sum(0) / np.maximum(flat_m.sum(0), 1)
        var = (((flat_v - self.mean) ** 2) * flat_m).sum(0) / np.maximum(flat_m.sum(0), 1)
        self.std = np.sqrt(var) + 1e-6
        return self

    def __call__(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        normalized = (values - self.mean) / self.std * mask
        return np.concatenate([normalized, mask], axis=-1).astype(np.float32)


# ---------------------------------------------------------------- augmentation on the fly
class AugmentedWindows(torch.utils.data.Dataset):
    def __init__(self, clips, classes, normalizer, windows_per_epoch, rng):
        self.clips, self.classes, self.norm, self.n, self.rng = clips, classes, normalizer, windows_per_epoch, rng
        weights = np.array([max(c["t"][-1] - c["t"][0], WINDOW_S) for c in clips])
        self.clip_p = weights / weights.sum()  # sample proportional to clip length

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        for _attempt in range(10):
            clip = self.clips[self.rng.choice(len(self.clips), p=self.clip_p)]
            scale = self.rng.uniform(0.75, 1.25)
            span = WINDOW_S * scale
            lo = clip["t"][0] + span
            end = self.rng.uniform(lo, clip["t"][-1]) if clip["t"][-1] > lo else clip["t"][-1]
            values, mask = resample_window(clip["t"], clip["X"], end, scale)
            if mask.mean() >= MIN_VALID_FRACTION:
                break
        values = values + self.rng.normal(0, 0.02, values.shape).astype(np.float32) * values.std()
        x = self.norm(values, mask)
        return torch.from_numpy(x), self.classes.index(clip["label"])


# ---------------------------------------------------------------- model
class BiLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):                      # x: (batch, steps, features)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1))      # mean over time: robust to where the rep sits in the window


def train_model(train_clips, classes, normalizer, args, rng, log_prefix=""):
    torch.manual_seed(args.seed)
    n_windows_per_epoch = max(len(train_clips) * 40, 1000)
    dataset = AugmentedWindows(train_clips, classes, normalizer, n_windows_per_epoch, rng)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = BiLSTMClassifier(2 * len(train_clips[0]["X"][0]), len(classes), args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y); total += len(y); correct += (logits.argmax(1) == y).sum().item()
        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"{log_prefix}epoch {epoch:3d}  loss {running / total:.3f}  train acc {correct / total:.3f}")
    return model


@torch.no_grad()
def predict(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    probs = [torch.softmax(model(torch.from_numpy(x[i:i + batch_size])), 1).numpy() for i in range(0, len(x), batch_size)]
    return np.concatenate(probs)


def evaluate(model, normalizer, clips, classes):
    values, mask, labels, groups = build_windows(clips)
    probs = predict(model, normalizer(values, mask))
    pred = probs.argmax(1)
    truth = np.array([classes.index(l) for l in labels])
    window_acc = (pred == truth).mean()
    clip_pred, clip_truth, clip_report = [], [], []
    for group in np.unique(groups):
        sel = groups == group
        votes = np.bincount(pred[sel], minlength=len(classes))
        clip_pred.append(votes.argmax())
        clip_truth.append(truth[sel][0])
        coverage = mask[sel].mean()
        clip_report.append((group, classes[truth[sel][0]], classes[votes.argmax()], votes.max() / votes.sum(), coverage))
    clip_pred, clip_truth = np.array(clip_pred), np.array(clip_truth)
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for t_, p_ in zip(truth, pred):
        confusion[t_, p_] += 1
    return window_acc, (clip_pred == clip_truth).mean(), confusion, len(labels), len(clip_pred), clip_report


def print_confusion(confusion, classes):
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c:>{width}}" for c in classes) + "   (rows = true, cols = predicted)")
    for name, row in zip(classes, confusion):
        print(f"{name:>{width}}" + "".join(f"{v:>{width}}" for v in row))


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)

    clips, feature_columns = load_clips(DEFAULT_FOLDERS)
    classes = sorted(set(c["label"] for c in clips))
    labels = np.array([c["label"] for c in clips])
    groups = np.array([c["group"] for c in clips])
    print(f"classes: {classes}   clips per class: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- cross-validation grouped by clip
    started = time.time()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    window_accs, clip_accs, confusion_total = [], [], np.zeros((len(classes), len(classes)), dtype=int)
    all_reports = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(clips)), labels, groups), 1):
        train_clips = [clips[i] for i in train_idx]
        test_clips = [clips[i] for i in test_idx]
        v, m, _, _ = build_windows(train_clips)
        normalizer = Normalizer().fit(v, m)
        fold_start = time.time()
        model = train_model(train_clips, classes, normalizer, args, rng, log_prefix=f"  fold {fold} ")
        w_acc, c_acc, confusion, n_windows, n_clips, report = evaluate(model, normalizer, test_clips, classes)
        window_accs.append(w_acc); clip_accs.append(c_acc); confusion_total += confusion
        all_reports.extend(report)
        print(f"fold {fold}: window acc {w_acc:.3f} ({n_windows} windows)   clip acc {c_acc:.3f} ({n_clips} clips)   "
              f"[{time.time() - fold_start:.0f} s]")
    print(f"\n{args.folds}-fold CV: window acc {np.mean(window_accs):.3f} +- {np.std(window_accs):.3f}   "
          f"clip acc {np.mean(clip_accs):.3f} +- {np.std(clip_accs):.3f}   [{time.time() - started:.0f} s total]")
    print_confusion(confusion_total, classes)
    print("\nclips with < 90% of windows voting for the true class (feature coverage = fraction of valid inputs):")
    for group, true, predicted, agreement, coverage in sorted(all_reports, key=lambda r: r[3]):
        if predicted != true or agreement < 0.9:
            flag = "WRONG" if predicted != true else "weak "
            print(f"  {flag}  {group:12s} true={true:7s} pred={predicted:7s} vote={agreement:.2f} coverage={coverage:.2f}")

    # ---- final model on all clips
    print("\ntraining final model on all clips")
    v, m, _, _ = build_windows(clips)
    normalizer = Normalizer().fit(v, m)
    final_start = time.time()
    model = train_model(clips, classes, normalizer, args, rng, log_prefix="  final ")
    print(f"final training took {time.time() - final_start:.0f} s")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "bilstm_exercise.pt")
    with (args.output / "bilstm_exercise.json").open("w") as handle:
        json.dump({
            "classes": classes, "feature_columns": feature_columns,
            "mean": normalizer.mean.tolist(), "std": normalizer.std.tolist(),
            "rate_hz": RATE, "window_s": WINDOW_S, "steps": STEPS, "mask_tolerance_s": MASK_TOLERANCE_S,
            "hidden": args.hidden, "inputs": 2 * len(feature_columns),
        }, handle, indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {args.output / 'bilstm_exercise.pt'} ({n_params} parameters, {n_params * 4 / 1024:.0f} KB fp32) "
          f"and bilstm_exercise.json")


if __name__ == "__main__":
    main()
