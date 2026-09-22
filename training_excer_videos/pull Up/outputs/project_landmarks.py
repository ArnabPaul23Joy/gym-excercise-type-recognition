#!/usr/bin/env python3
"""Overlay the landmarks stored in landmarks.csv onto a video.

Example (defaults reproduce "pull up_1"):
    python project_landmarks.py
    python project_landmarks.py --video "../pull up_5.mp4" --output "pull up_5_landmarks.mp4"

landmarks.csv holds x/y normalized to 0..1 with a top-left origin (as written by
build_features.py), so a joint's pixel position is simply (x * width, y * height).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

SKELETON = [
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
BONE_COLOR = (40, 220, 255)   # BGR
JOINT_COLOR = (40, 255, 120)
LOW_CONF_COLOR = (60, 60, 230)


def draw_pose(frame, row: pd.Series, width: int, height: int, min_confidence: float, joints: list[str]) -> None:
    pixels = {}
    for joint in joints:
        confidence = row[f"c_{joint}"]
        if confidence <= 0 or pd.isna(row[f"x_{joint}"]):
            continue
        pixels[joint] = (int(row[f"x_{joint}"] * width), int(row[f"y_{joint}"] * height), confidence)
    for first, second in SKELETON:
        if first in pixels and second in pixels and min(pixels[first][2], pixels[second][2]) >= min_confidence:
            cv2.line(frame, pixels[first][:2], pixels[second][:2], BONE_COLOR, 3, cv2.LINE_AA)
    for x, y, confidence in pixels.values():
        color = JOINT_COLOR if confidence >= min_confidence else LOW_CONF_COLOR
        cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)


def project(video: Path, landmarks: Path, output: Path, vid_id: str, min_confidence: float) -> None:
    table = pd.read_csv(landmarks, dtype={"vid_id": str})
    # vid_id may be the full stem ("pull up_1") or just the trailing number ("1")
    candidates = [vid_id, vid_id.rsplit("_", 1)[-1]]
    table = table[table["vid_id"].isin(candidates)].set_index("frame_order").sort_index()
    if table.empty:
        raise ValueError(f"No rows with vid_id in {candidates} in {landmarks}")
    joints = [column[2:] for column in table.columns if column.startswith("c_")]

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():  # fall back if the H.264 encoder is unavailable
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in table.index:
                draw_pose(frame, table.loc[frame_index], width, height, min_confidence, joints)
            cv2.putText(frame, f"{vid_id}  frame {frame_index}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    print(f"wrote {output} ({frame_index} frames, {width}x{height} @ {fps:.1f} fps)")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=here.parent / "pull up_1.mp4")
    parser.add_argument("--landmarks", type=Path, default=here / "landmarks.csv")
    parser.add_argument("--output", type=Path, default=None, help="default: outputs/<video stem>_landmarks.mp4")
    parser.add_argument("--vid-id", default=None, help="vid_id in landmarks.csv (default: video file stem)")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="draw bones only above this confidence")
    args = parser.parse_args()
    vid_id = args.vid_id or args.video.stem
    output = args.output or here / f"{args.video.stem}_landmarks.mp4"
    project(args.video, args.landmarks, output, vid_id, args.min_confidence)


if __name__ == "__main__":
    main()
