# Apple Vision exercise pose pipeline

This project uses macOS Vision's `VNDetectHumanBodyPoseRequest` locally on the Mac. The Swift helper performs pose estimation; Python handles video annotation, feature tables, and lightweight scikit-learn models. Video frames and pose data stay local.

## 1. Build the Apple Vision helper

From this directory on the M4 MacBook:

```bash
xcrun swiftc VisionPoseExtractor.swift -o VisionPoseExtractor \
  -framework AVFoundation -framework CoreMedia -framework Foundation -framework Vision
```

## 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Extract and annotate a video

```bash
python pose_pipeline.py extract videos/squat_01.mp4 \
  --poses poses --annotated annotated/squat_01.mp4
```

The pose file is JSONL. Coordinates are normalized to 0..1 with origin at the top-left, which matches OpenCV drawing coordinates.

For a dataset, name videos as `label__subject_or_trial.mp4`, for example `squat__person01__trial01.mp4`. Run `extract` for every video, then:

```bash
python pose_pipeline.py features --poses poses --output features.csv
python pose_pipeline.py train-type --features features.csv
```

The type model splits by video, so frames from one recording cannot leak into both train and test sets.

## Step-count model

For repetition counting, first add a `phase` column to `features.csv` for labeled frames. Use labels such as `up`, `down`, and `bottom`; label the same way across every exercise you want to count. Then run:

```bash
python pose_pipeline.py train-steps --features features.csv --output exercise_phase.joblib
```

At inference time, predict one phase per frame, smooth the predictions with a short majority window, and count a repetition on a stable transition such as `up -> bottom -> up`. Do not train directly on frame number: split and label by recording, and keep separate people in the test set when measuring generalization.

## Data and modeling notes

- Record multiple camera angles, body sizes, speeds, and complete repetitions for each exercise.
- Keep the confidence values and reject or interpolate long low-confidence gaps.
- Normalize joint coordinates by torso scale and center them at the hip midpoint before adding a temporal model; this reduces sensitivity to distance from the camera.
- The included type model is a frame classifier baseline. A stronger lightweight model uses a 0.5-2 second sliding window of pose features with a small 1D temporal CNN, GRU, or gradient-boosted summary features.
- The included step model is a supervised phase classifier. Counting is a temporal state-machine problem, so smoothing plus hysteresis is usually more stable than counting every classifier change.
- Vision pose estimation is not a medical or safety system. Use adequate lighting, a mostly visible person, and do not infer identity or sensitive health attributes.
