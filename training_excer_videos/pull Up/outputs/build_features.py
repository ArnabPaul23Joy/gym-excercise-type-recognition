#!/usr/bin/env python3
"""Convert VisionPoseExtractor JSONL files into dataset-style CSVs.

Produces, in --output:
  landmarks.csv            raw Vision joints: x/y normalized 0..1 (top-left origin) + confidence
  landmarks_normalized.csv hip-centered, torso-scaled joints (same normalizer as excercise_dataset)
  angles.csv               the 7 dataset angles, computed in 2D from the normalized joints
  distances.csv            the 16 dataset distances, 2D Euclidean from the normalized joints
  xy_distances.csv         the same 16 pairs split per axis (second - first, as in the dataset)

Vision gives 2D joints only, so these are the 2D analogues of the dataset's 3D columns.
Frames where a joint is missing get NaN in every column that depends on it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Vision joint name -> dataset joint name
JOINT_MAP = {
    "head_joint": "nose", "left_eye_joint": "left_eye", "right_eye_joint": "right_eye",
    "left_ear_joint": "left_ear", "right_ear_joint": "right_ear",
    "left_shoulder_1_joint": "left_shoulder", "right_shoulder_1_joint": "right_shoulder",
    "left_forearm_joint": "left_elbow", "right_forearm_joint": "right_elbow",
    "left_hand_joint": "left_wrist", "right_hand_joint": "right_wrist",
    "left_upLeg_joint": "left_hip", "right_upLeg_joint": "right_hip",
    "left_leg_joint": "left_knee", "right_leg_joint": "right_knee",
    "left_foot_joint": "left_ankle", "right_foot_joint": "right_ankle",
}
JOINTS = list(JOINT_MAP.values())

ANGLES = [
    ("right_elbow", "right_shoulder", "right_hip"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_knee", "mid_hip", "left_knee"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_wrist", "right_elbow", "right_shoulder"),
    ("left_wrist", "left_elbow", "left_shoulder"),
]
DISTANCES = [
    ("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist"),
    ("left_hip", "left_ankle"), ("right_hip", "right_ankle"),
    ("left_hip", "left_wrist"), ("right_hip", "right_wrist"),
    ("left_shoulder", "left_ankle"), ("right_shoulder", "right_ankle"),
    ("left_hip", "right_wrist"), ("right_hip", "left_wrist"),
    ("left_elbow", "right_elbow"), ("left_knee", "right_knee"),
    ("left_wrist", "right_wrist"), ("left_ankle", "right_ankle"),
    ("left_hip", "avg_left_wrist_left_ankle"), ("right_hip", "avg_right_wrist_right_ankle"),
]
TORSO_MULTIPLIER = 2.5  # same as the Google pose embedder used for the dataset
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi"}


def find_video(video_dir: Path, stem: str) -> Path | None:
    for candidate in video_dir.iterdir():
        if candidate.stem == stem and candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate
    return None


def video_size(video: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video))
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return size


def load_raw(pose_file: Path, vid_id: str) -> pd.DataFrame:
    rows = []
    with pose_file.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            pose = json.loads(line)
            row = {"vid_id": vid_id, "frame_order": pose["frame"], "timestamp": pose["timestamp"]}
            for vision_name, name in JOINT_MAP.items():
                point = pose["joints"].get(vision_name)
                row[f"x_{name}"] = point["x"] if point else np.nan
                row[f"y_{name}"] = point["y"] if point else np.nan
                row[f"c_{name}"] = point["confidence"] if point else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def normalize(raw: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Hip-center and scale by max(2.5 * torso, max distance from hips), * 100 (dataset convention)."""
    xs = raw[[f"x_{j}" for j in JOINTS]].to_numpy() * width
    ys = raw[[f"y_{j}" for j in JOINTS]].to_numpy() * height
    pts = np.stack([xs, ys], axis=-1)  # (frames, joints, 2) in pixels
    idx = {j: i for i, j in enumerate(JOINTS)}
    hips = (pts[:, idx["left_hip"]] + pts[:, idx["right_hip"]]) / 2
    shoulders = (pts[:, idx["left_shoulder"]] + pts[:, idx["right_shoulder"]]) / 2
    pts = pts - hips[:, None, :]
    torso = np.linalg.norm(shoulders - hips, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no joints are all-NaN
        max_dist = np.nanmax(np.linalg.norm(pts, axis=2), axis=1)
    size = np.fmax(torso * TORSO_MULTIPLIER, max_dist)
    pts = pts / size[:, None, None] * 100
    out = raw[["vid_id", "frame_order"]].copy()
    for j, i in idx.items():
        out[f"x_{j}"] = pts[:, i, 0]
        out[f"y_{j}"] = pts[:, i, 1]
    return out


def point(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "mid_hip":
        return (point(df, "left_hip") + point(df, "right_hip")) / 2
    if name.startswith("avg_"):
        parts = name[4:].split("_")  # avg_left_wrist_left_ankle -> left wrist / left ankle
        a, b = "_".join(parts[0:2]), "_".join(parts[2:4])
        return (point(df, a) + point(df, b)) / 2
    return df[[f"x_{name}", f"y_{name}"]].to_numpy()


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba, bc = a - b, c - b
    cosine = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def build(pose_dir: Path, video_dir: Path, output: Path) -> None:
    raw_frames, norm_frames, angle_frames, dist_frames, xy_frames = [], [], [], [], []
    for pose_file in sorted(pose_dir.glob("*.jsonl")):
        vid_id = pose_file.stem
        video = find_video(video_dir, vid_id)
        if video is None:
            raise FileNotFoundError(f"No video for {pose_file.name} in {video_dir}")
        width, height = video_size(video)
        raw = load_raw(pose_file, vid_id)
        norm = normalize(raw, width, height)

        ang = norm[["vid_id", "frame_order"]].copy()
        for a, b, c in ANGLES:
            ang[f"{a}_{b}_{c}"] = angle(point(norm, a), point(norm, b), point(norm, c))

        dist = norm[["vid_id", "frame_order"]].copy()
        xy = norm[["vid_id", "frame_order"]].copy()
        for a, b in DISTANCES:
            delta = point(norm, b) - point(norm, a)
            dist[f"{a}_{b}"] = np.linalg.norm(delta, axis=1)
            xy[f"x_{a}_{b}"] = delta[:, 0]
            xy[f"y_{a}_{b}"] = delta[:, 1]

        raw_frames.append(raw)
        norm_frames.append(norm)
        angle_frames.append(ang)
        dist_frames.append(dist)
        xy_frames.append(xy)
        print(f"{vid_id}: {len(raw)} frames, {int(raw['c_left_hip'].eq(0).sum())} without hips")

    output.mkdir(parents=True, exist_ok=True)
    pd.concat(raw_frames).to_csv(output / "landmarks.csv", index=False)
    pd.concat(norm_frames).to_csv(output / "landmarks_normalized.csv", index=False)
    pd.concat(angle_frames).to_csv(output / "angles.csv", index=False)
    pd.concat(dist_frames).to_csv(output / "distances.csv", index=False)
    pd.concat(xy_frames).to_csv(output / "xy_distances.csv", index=False)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", type=Path, default=here / "poses", help="directory of JSONL files from VisionPoseExtractor")
    parser.add_argument("--videos", type=Path, default=here.parent, help="directory holding the matching .mp4 files")
    parser.add_argument("--output", type=Path, default=here, help="where to write the CSV files")
    args = parser.parse_args()
    build(args.poses, args.videos, args.output)


if __name__ == "__main__":
    main()
