// ExerciseTypeAnalyzer.swift
// Runs the ExerciseClassifierKit pipeline over a whole video to recognise the
// exercise type (pull-up, push-up, squat).
//
// Per frame: AVAssetReader → VNDetectHumanBodyPoseRequest → PoseFeatures →
// WindowResampler; every `stride` seconds the last window is classified with the
// on-device Bi-LSTM Core ML model. Predictions are aggregated by majority vote,
// and the winning class's mean probability is reported as confidence.
// This mirrors the reference CLI in swift/main.swift.

import AVFoundation
import CoreML
import Foundation
import Vision

nonisolated enum ExerciseTypeAnalyzer {

    struct Result: Sendable {
        let label: String        // display name, e.g. "Squat"
        let rawLabel: String     // model class, e.g. "squat"
        let confidence: Double   // mean probability of the winning class, 0…1
    }

    enum AnalyzerError: LocalizedError {
        case modelNotFound
        case noVideoTrack
        case noPoseDetected

        var errorDescription: String? {
            switch self {
            case .modelNotFound:
                return "ExerciseClassifier.mlmodelc not found in the app bundle. Confirm ExerciseClassifier.mlpackage is added to the app target."
            case .noVideoTrack:
                return "The selected file has no video track."
            case .noPoseDetected:
                return "Couldn't see a person clearly enough to identify the exercise."
            }
        }
    }

    private static let displayNames: [String: String] = [
        "pullup": "Pull-up",
        "pushup": "Push-up",
        "squat": "Squat",
    ]

    static func displayName(for rawLabel: String) -> String {
        displayNames[rawLabel] ?? rawLabel.capitalized
    }

    /// - Parameters:
    ///   - stride: seconds between predictions once the buffer is full.
    ///   - minCoverage: skip windows whose fraction of valid (unmasked) features is below this.
    static func classify(url: URL, stride: Double = 0.25, minCoverage: Double = 0.5) async throws -> Result {
        guard let modelURL = Bundle.main.url(forResource: "ExerciseClassifier", withExtension: "mlmodelc") else {
            throw AnalyzerError.modelNotFound
        }
        let classifier = try PoseExerciseClassifier(modelURL: modelURL)
        let resampler = classifier.makeResampler()

        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw AnalyzerError.noVideoTrack
        }
        let size = try await track.load(.naturalSize)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AnalyzerError.noPoseDetected }

        var votes: [String: Int] = [:]
        var probabilitySum = Array(repeating: 0.0, count: classifier.classes.count)
        var predictions = 0
        var nextPrediction = 0.0
        var firstTimestamp: Double?

        while let sample = output.copyNextSampleBuffer(), let pixelBuffer = CMSampleBufferGetImageBuffer(sample) {
            let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            if firstTimestamp == nil {
                firstTimestamp = timestamp
                nextPrediction = timestamp + classifier.windowSeconds
            }

            // per frame: pose → features → ring buffer
            let request = VNDetectHumanBodyPoseRequest()
            try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
            let observation = request.results?.first
            let features = PoseFeatures.features(from: observation, imageWidth: Double(size.width),
                                                 imageHeight: Double(size.height), timestamp: timestamp)
            resampler.push(features)

            // every `stride` seconds: resample the last window and classify
            guard timestamp >= nextPrediction, resampler.isReady() else { continue }
            nextPrediction += stride
            guard let window = resampler.window(), window.coverage >= minCoverage else { continue }

            let prediction = try classifier.predict(window)
            predictions += 1
            votes[prediction.label, default: 0] += 1
            for (i, p) in prediction.probabilities.enumerated() { probabilitySum[i] += p }
        }

        guard predictions > 0, let winner = votes.max(by: { $0.value < $1.value })?.key else {
            throw AnalyzerError.noPoseDetected
        }
        let index = classifier.classes.firstIndex(of: winner) ?? 0
        let confidence = probabilitySum[index] / Double(predictions)
        return Result(label: displayName(for: winner), rawLabel: winner, confidence: confidence)
    }
}
