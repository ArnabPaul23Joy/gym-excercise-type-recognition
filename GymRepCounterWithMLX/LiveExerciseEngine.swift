// LiveExerciseEngine.swift
// Drives the live-camera pipeline: AVCaptureSession → per-frame Vision body-pose →
// { keypoint smoothing → PoseFeatures → WindowResampler → Core ML classification,
//   geometric StreamingRepCounter } → LiveUpdate delivered to the UI.
//
// TRAIN/INFERENCE MISMATCH COMPENSATION (model was trained on video files, not live):
//   1. WindowResampler resamples the live stream onto the model's exact 15 Hz / 2 s
//      (30-step) grid with validity masks — identical temporal footing to training,
//      independent of the camera's frame rate.
//   2. Per-keypoint EMA smoothing tames live jitter so features match the cleaner
//      decoded-video distribution the model learned.
//   3. PoseFeatures is hip-centred + torso-scaled → invariant to camera distance/zoom;
//      it's also reflection-invariant, so top-left coords match training math exactly.
//   4. Confidence gating + majority-vote LabelSmoother + window-coverage gating reject
//      noisy, low-visibility, or flickering predictions.
//   5. A warm-up (isReady) period ("Calibrating…") until 2 s of history exists.

import AVFoundation
import CoreGraphics
import CoreML
import Foundation
import ImageIO
import Vision

/// A snapshot pushed to the UI on every processed frame.
struct LiveUpdate: Sendable {
    var pose: PoseFrame?
    var orientedSize: CGSize      // upright buffer size the joints are normalised against
    var rawLabel: String?
    var displayLabel: String
    var confidence: Double
    var reps: Int
    var calibrating: Bool
}

nonisolated final class LiveExerciseEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

    // Tunables
    private let predictionStride = 0.25       // seconds between classifier predictions
    private let minCoverage = 0.5             // skip low-visibility windows
    private let minConfidence = 0.55          // ignore unsure classifications
    private let keypointAlpha = 0.5           // EMA weight for keypoint smoothing
    private let minKeypointConfidence: Float = 0.1

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "live.session")
    private let sampleQueue = DispatchQueue(label: "live.samples")
    private let output = AVCaptureVideoDataOutput()

    /// Delivered on the main actor by the view model (see LiveWorkoutViewModel).
    var onUpdate: (@Sendable (LiveUpdate) -> Void)?

    // Pipeline state — only ever touched on `sampleQueue`.
    private var classifier: PoseExerciseClassifier?
    private var resampler: WindowResampler?
    private let smoother = LabelSmoother(size: 12)
    private let counter = StreamingRepCounter()
    private var emaJoints: [String: CGPoint] = [:]
    private var nextPrediction = 0.0
    private var firstTimestamp: Double?
    private var currentRawLabel: String?
    private var currentConfidence = 0.0
    private var position: AVCaptureDevice.Position = .back

    // MARK: - Session setup

    /// Configures the session for the given camera and returns whether it succeeded.
    func configure(position: AVCaptureDevice.Position) {
        sessionQueue.async { [self] in
            self.position = position
            session.beginConfiguration()
            session.sessionPreset = .high

            // Load the classifier once (compiled from ExerciseClassifier.mlpackage).
            if classifier == nil, let url = Bundle.main.url(forResource: "ExerciseClassifier", withExtension: "mlmodelc") {
                classifier = try? PoseExerciseClassifier(modelURL: url)
                resampler = classifier?.makeResampler()
            }

            // Camera input.
            session.inputs.forEach { session.removeInput($0) }
            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
               let input = try? AVCaptureDeviceInput(device: device),
               session.canAddInput(input) {
                session.addInput(input)
            }

            // Frame output.
            if !session.outputs.contains(output) {
                output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                output.alwaysDiscardsLateVideoFrames = true
                output.setSampleBufferDelegate(self, queue: sampleQueue)
                if session.canAddOutput(output) { session.addOutput(output) }
            }

            // NB: we deliberately do NOT rotate/mirror the connection. Buffers stay in the
            // camera's native landscape orientation; instead we hand Vision the correct
            // CGImagePropertyOrientation per frame so its coordinates come out upright.
            session.commitConfiguration()
        }
    }

    func start() { sessionQueue.async { [self] in if !session.isRunning { session.startRunning() } } }
    func stop()  { sessionQueue.async { [self] in if session.isRunning { session.stopRunning() } } }

    func resetCounter() {
        sampleQueue.async { [self] in
            counter.reset()
            currentRawLabel = nil
        }
    }

    // MARK: - Per-frame pipeline (runs on sampleQueue)

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer), let resampler else { return }
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

        // The buffer is in the camera's native landscape orientation. For a device held in
        // portrait, .right (back) / .leftMirrored (front, matches its mirrored preview) makes
        // Vision return coordinates in upright space. A 90° orientation swaps width/height.
        let orientation: CGImagePropertyOrientation = (position == .front) ? .leftMirrored : .right
        let width = Double(CVPixelBufferGetHeight(pixelBuffer))
        let height = Double(CVPixelBufferGetWidth(pixelBuffer))
        let orientedSize = CGSize(width: width, height: height)

        let request = VNDetectHumanBodyPoseRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation).perform([request])
        let observation = request.results?.first

        // (1) Raw keypoints → (2) EMA smoothing.
        let raw = normalisedJoints(from: observation)
        let smoothedJoints = smoothKeypoints(raw)
        let pose = PoseFrame(time: timestamp, joints: smoothedJoints)

        // Build the classifier's 23 features from the smoothed joints (pixel space).
        var pixelJoints: [String: SIMD2<Double>] = [:]
        for (name, point) in smoothedJoints {
            pixelJoints[name] = SIMD2(Double(point.x) * width, Double(point.y) * height)
        }
        let values = PoseFeatures.compute(PoseFeatures.normalise(pixelJoints))
        resampler.push(FrameFeatures(timestamp: timestamp, values: values))

        // (5) Warm-up until a full window exists.
        if firstTimestamp == nil {
            firstTimestamp = timestamp
            nextPrediction = timestamp + resampler.windowSeconds
        }
        let calibrating = !resampler.isReady()

        // Classification on a fixed stride, with coverage + confidence + vote smoothing.
        if timestamp >= nextPrediction, resampler.isReady(), let classifier {
            nextPrediction += predictionStride
            if let window = resampler.window(), window.coverage >= minCoverage,
               let prediction = try? classifier.predict(window), prediction.confidence >= minConfidence {
                let stable = smoother.push(prediction.label)
                currentRawLabel = stable
                currentConfidence = prediction.confidence
                counter.setExercise(stable)
            }
        }

        // Geometric rep counting every frame once we know the exercise.
        if let exercise = currentRawLabel {
            counter.update(angle: GeometricRepCounter.repAngle(for: exercise, frame: pose, orientedSize: orientedSize))
        }

        let update = LiveUpdate(
            pose: pose,
            orientedSize: orientedSize,
            rawLabel: currentRawLabel,
            displayLabel: currentRawLabel.map { ExerciseTypeAnalyzer.displayName(for: $0) } ?? "",
            confidence: currentConfidence,
            reps: counter.reps,
            calibrating: calibrating
        )
        onUpdate?(update)
    }

    // MARK: - Helpers

    /// Normalised joint locations in upright space, top-left origin (y down).
    private func normalisedJoints(from observation: VNHumanBodyPoseObservation?) -> [String: CGPoint] {
        guard let observation, let points = try? observation.recognizedPoints(.all) else { return [:] }
        var result: [String: CGPoint] = [:]
        for joint in PoseFeatures.joints {
            guard let point = points[joint.vision], point.confidence > minKeypointConfidence else { continue }
            result[joint.name] = CGPoint(x: point.location.x, y: 1 - point.location.y)
        }
        return result
    }

    /// Exponential moving average per joint; joints missing this frame are dropped (so the
    /// resampler masks them, matching training), but their last value is retained for reuse.
    private func smoothKeypoints(_ current: [String: CGPoint]) -> [String: CGPoint] {
        var out: [String: CGPoint] = [:]
        for (name, point) in current {
            if let prev = emaJoints[name] {
                let blended = CGPoint(x: prev.x * (1 - keypointAlpha) + point.x * keypointAlpha,
                                      y: prev.y * (1 - keypointAlpha) + point.y * keypointAlpha)
                out[name] = blended
                emaJoints[name] = blended
            } else {
                out[name] = point
                emaJoints[name] = point
            }
        }
        return out
    }
}
