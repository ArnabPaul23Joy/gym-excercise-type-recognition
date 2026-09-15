#!/usr/bin/env python3
"""
Converts RepNet from TensorFlow Hub to a CoreML package for iOS.

Requirements (Python 3.9–3.11 recommended):
    pip3 install tensorflow tensorflow-hub coremltools numpy

Usage:
    python3 convert_repnet.py

Output:
    RepNet.mlpackage  — drag this into your Xcode project, add to the app target.
    Xcode will compile it to RepNet.mlmodelc inside the app bundle automatically.

Model input:
    name : "frames"
    shape: [1, 64, 112, 112, 3]   (batch=1, 64 frames, 112×112 px, RGB)
    dtype: float32, values in [0.0, 1.0]

Model outputs:
    "within_period_scores"  — [1, 64, 1]    per-frame probability of being in a periodic segment
    "period_length_scores"  — [1, 64, bins]  logits for period-length prediction (bin b → period b+2 frames)
"""

import sys
import numpy as np

print("Importing TensorFlow…")
try:
    import tensorflow as tf
except Exception as e:
    sys.exit(f"ERROR importing tensorflow: {e}\nRun: pip3 install tensorflow tensorflow-hub coremltools numpy")

print("Importing TensorFlow Hub…")
try:
    import tensorflow_hub as hub
except Exception as e:
    sys.exit(f"ERROR importing tensorflow_hub: {e}\nRun: pip3 install tensorflow-hub")

print("Importing coremltools…")
try:
    import coremltools as ct
except Exception as e:
    sys.exit(f"ERROR importing coremltools: {e}\nRun: pip3 install coremltools")

NUM_FRAMES = 64
INPUT_SIZE = 112

print(f"Loading RepNet from TensorFlow Hub (num_frames={NUM_FRAMES})…")
repnet = hub.load("https://tfhub.dev/google/repnet/1")

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE, 3],
                  dtype=tf.float32, name="frames")
])
def predict(frames):
    out = repnet(frames, training=False)
    return {
        "within_period_scores": out["within_period_scores"],
        "period_length_scores": out["period_length_scores"],
    }

print("Tracing concrete function…")
concrete_fn = predict.get_concrete_function()

print("Converting to CoreML (this may take a few minutes)…")
mlmodel = ct.convert(
    concrete_fn,
    inputs=[
        ct.TensorType(
            name="frames",
            shape=[1, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE, 3],
            dtype=np.float32,
        )
    ],
    outputs=[
        ct.TensorType(name="within_period_scores"),
        ct.TensorType(name="period_length_scores"),
    ],
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
)

mlmodel.short_description = "RepNet — class-agnostic repetition counter"

output_path = "RepNet.mlpackage"
print(f"Saving {output_path}…")
mlmodel.save(output_path)

print()
print("Done!  Next steps:")
print("  1. In Xcode: File → Add Files to project, select RepNet.mlpackage")
print("     Make sure 'Add to targets: GymRepCounterWithMLX' is checked.")
print("  2. Also remove the MLX / MLXVLM Swift packages from:")
print("     Xcode → Project → Package Dependencies (they are no longer used).")
print("  3. Build and run on device.")
