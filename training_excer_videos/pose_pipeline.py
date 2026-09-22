#!/usr/bin/env python3
"""Extract Apple Vision poses, annotate video, build features, and train small models."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

JOINTS = [
    "nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee",
    "rightKnee", "leftAnkle", "rightAnkle",
]
SKELETON = [
    ("nose", "leftShoulder"), ("nose", "rightShoulder"),
    ("leftShoulder", "rightShoulder"), ("leftShoulder", "leftElbow"),
    ("leftElbow", "leftWrist"), ("rightShoulder", "rightElbow"),
    ("rightElbow", "rightWrist"), ("leftShoulder", "leftHip"),
    ("rightShoulder", "rightHip"), ("leftHip", "rightHip"),
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
]


def run_extractor(extractor: Path, video: Path, pose_file: Path) -> None:
    pose_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(extractor), str(video), str(pose_file)], check=True)


def read_poses(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def annotate(video: Path, pose_file: Path, output: Path, min_confidence: float) -> None:
    poses = read_poses(pose_file)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for pose in poses:
            ok, frame = capture.read()
            if not ok:
                break
            points = pose["joints"]
            for first, second in SKELETON:
                if first in points and second in points and min(points[first]["confidence"], points[second]["confidence"]) >= min_confidence:
                    a = points[first]
                    b = points[second]
                    cv2.line(frame, (int(a["x"] * width), int(a["y"] * height)), (int(b["x"] * width), int(b["y"] * height)), (40, 220, 255), 3)
            for point in points.values():
                if point["confidence"] >= min_confidence:
                    cv2.circle(frame, (int(point["x"] * width), int(point["y"] * height)), 5, (40, 255, 120), -1)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def pose_features(pose: dict) -> dict[str, float]:
    points = pose["joints"]
    features: dict[str, float] = {"timestamp": pose["timestamp"], "frame": pose["frame"]}
    for joint in JOINTS:
        point = points.get(joint)
        features[f"{joint}_x"] = point["x"] if point else np.nan
        features[f"{joint}_y"] = point["y"] if point else np.nan
        features[f"{joint}_c"] = point["confidence"] if point else 0.0
    return features


def build_features(pose_dir: Path, output: Path) -> None:
    rows = []
    for pose_file in sorted(pose_dir.glob("*.jsonl")):
        label = pose_file.stem.split("__", 1)[0]
        for row in read_poses(pose_file):
            features = pose_features(row)
            features["video"] = pose_file.stem
            features["label"] = label
            rows.append(features)
    if not rows:
        raise RuntimeError(f"No JSONL pose files found in {pose_dir}")
    pd.DataFrame(rows).to_csv(output, index=False)


def split_by_video(data: pd.DataFrame, seed: int = 7):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, groups=data["video"]))
    return data.iloc[train_idx], data.iloc[test_idx]


def train_type(features: Path, output: Path) -> None:
    data = pd.read_csv(features).dropna(subset=["label"])
    train, test = split_by_video(data)
    ignored = {"label", "video", "timestamp", "frame"}
    columns = [column for column in data.columns if column not in ignored]
    model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, random_state=7))
    model.fit(train[columns].fillna(0), train["label"])
    print(f"exercise type accuracy: {model.score(test[columns].fillna(0), test['label']):.3f}")
    joblib.dump({"model": model, "columns": columns}, output)


def angle(data: pd.DataFrame, a: str, b: str, c: str) -> pd.Series:
    first = data[[f"{a}_x", f"{a}_y"]].to_numpy()
    middle = data[[f"{b}_x", f"{b}_y"]].to_numpy()
    third = data[[f"{c}_x", f"{c}_y"]].to_numpy()
    ba = first - middle
    bc = third - middle
    cosine = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    return pd.Series(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def train_steps(features: Path, output: Path) -> None:
    data = pd.read_csv(features).sort_values(["video", "frame"])
    data["left_knee_angle"] = angle(data, "leftHip", "leftKnee", "leftAnkle")
    data["right_knee_angle"] = angle(data, "rightHip", "rightKnee", "rightAnkle")
    data["hip_height"] = (data["leftHip_y"] + data["rightHip_y"]) / 2
    data["shoulder_height"] = (data["leftShoulder_y"] + data["rightShoulder_y"]) / 2
    columns = ["left_knee_angle", "right_knee_angle", "hip_height", "shoulder_height"]
    # Label a frame as the bottom phase. Add phase labels to training CSVs for supervised step models.
    if "phase" not in data:
        raise ValueError("Step training needs a phase column with labels such as up, down, or bottom")
    train, test = split_by_video(data)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=8, random_state=7, class_weight="balanced"))
    model.fit(train[columns].fillna(0), train["phase"])
    print(f"movement phase accuracy: {model.score(test[columns].fillna(0), test['phase']):.3f}")
    joblib.dump({"model": model, "columns": columns}, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("video", type=Path)
    extract.add_argument("--extractor", type=Path, default=Path("./VisionPoseExtractor"))
    extract.add_argument("--poses", type=Path, default=Path("poses"))
    extract.add_argument("--annotated", type=Path)
    extract.add_argument("--min-confidence", type=float, default=0.35)
    features = subparsers.add_parser("features")
    features.add_argument("--poses", type=Path, default=Path("poses"))
    features.add_argument("--output", type=Path, default=Path("features.csv"))
    type_parser = subparsers.add_parser("train-type")
    type_parser.add_argument("--features", type=Path, default=Path("features.csv"))
    type_parser.add_argument("--output", type=Path, default=Path("exercise_type.joblib"))
    step_parser = subparsers.add_parser("train-steps")
    step_parser.add_argument("--features", type=Path, default=Path("features.csv"))
    step_parser.add_argument("--output", type=Path, default=Path("exercise_phase.joblib"))
    args = parser.parse_args()
    if args.command == "extract":
        pose_file = args.poses / f"{args.video.stem}.jsonl"
        run_extractor(args.extractor, args.video, pose_file)
        if args.annotated:
            annotate(args.video, pose_file, args.annotated, args.min_confidence)
    elif args.command == "features":
        build_features(args.poses, args.output)
    elif args.command == "train-type":
        train_type(args.features, args.output)
    elif args.command == "train-steps":
        train_steps(args.features, args.output)


if __name__ == "__main__":
    main()
