#!/usr/bin/env python3
"""End-to-end: videos in an exercise folder -> <folder>/outputs/{poses/*.jsonl, *.csv}.

Runs VisionPoseExtractor on every video, builds the dataset-style CSVs
(landmarks, landmarks_normalized, angles, distances, xy_distances) and strips the
common file-name prefix from vid_id so it becomes just the clip number, exactly as
was done for "pull Up/outputs".

    python make_exercise_csvs.py push-up squat
    python make_exercise_csvs.py "pull Up" --prefix "pull up_"
    python make_exercise_csvs.py squat --force      # re-extract even if JSONL exists

Reuses build_features.py and strip_prefix.py from "pull Up/outputs" so the CSV
format stays identical across exercises.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pull Up" / "outputs"))
import build_features  # noqa: E402
import strip_prefix  # noqa: E402


def list_videos(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in build_features.VIDEO_SUFFIXES and not p.name.startswith(".")
    )


def extract_poses(videos: list[Path], extractor: Path, pose_dir: Path, force: bool) -> None:
    pose_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, video in enumerate(videos, 1):
        pose_file = pose_dir / f"{video.stem}.jsonl"
        if pose_file.exists() and pose_file.stat().st_size > 0 and not force:
            print(f"  [{index}/{len(videos)}] {video.name}: JSONL exists, skipping")
            continue
        print(f"  [{index}/{len(videos)}] {video.name}", flush=True)
        subprocess.run([str(extractor), str(video), str(pose_file)], check=True)
    print(f"  extraction took {time.time() - started:.0f} s")


def process_folder(folder: Path, extractor: Path, prefix: str | None, force: bool) -> None:
    videos = list_videos(folder)
    if not videos:
        raise SystemExit(f"No videos found in {folder}")
    if prefix is None:
        prefix = os.path.commonprefix([v.stem for v in videos])
    outputs = folder / "outputs"
    print(f"== {folder.name}: {len(videos)} videos, prefix to strip: {prefix!r}")

    extract_poses(videos, extractor, outputs / "poses", force)
    build_features.build(outputs / "poses", folder, outputs)
    if prefix:
        for csv in sorted(outputs.glob("*.csv")):
            changed = strip_prefix.strip_prefix(csv, prefix)
            print(f"  {csv.name}: stripped prefix in {changed} cells")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path, help="exercise folders containing videos")
    parser.add_argument("--extractor", type=Path, default=ROOT / "VisionPoseExtractor")
    parser.add_argument("--prefix", default=None, help="text to strip from vid_id (default: common prefix of file names)")
    parser.add_argument("--force", action="store_true", help="re-run pose extraction even if JSONL files exist")
    args = parser.parse_args()
    if not args.extractor.exists():
        raise SystemExit(f"Extractor not found: {args.extractor} (see README for the swiftc build command)")
    for folder in args.folders:
        process_folder(folder.resolve(), args.extractor.resolve(), args.prefix, args.force)


if __name__ == "__main__":
    main()
