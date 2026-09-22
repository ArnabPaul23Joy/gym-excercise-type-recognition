// main.swift — macOS command-line harness for ExerciseClassifierKit.
//
// Feeds a video file through the exact pipeline an iOS app would run on camera frames:
//   AVAssetReader frames -> VNDetectHumanBodyPoseRequest -> PoseFeatures -> WindowResampler
//   -> ExerciseClassifier (Core ML) -> smoothed label
// In the app, replace the AVAssetReader loop with an AVCaptureVideoDataOutput delegate and
// call the same three kit types per frame.
//
// Usage: ExerciseClassifierCLI <model.mlmodelc> <video> [--stride 0.25] [--min-coverage 0.6] [--quiet]

import AVFoundation
import CoreML
import Foundation
import Vision

struct Options {
    var modelURL: URL
    var videoURL: URL
    var stride = 0.25          // seconds between predictions
    var minCoverage = 0.6      // skip predictions when fewer valid inputs than this
    var quiet = false
}

func parseOptions() -> Options {
    var args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else {
        fputs("Usage: ExerciseClassifierCLI <model.mlmodelc> <video> [--stride s] [--min-coverage f] [--quiet]\n", stderr)
        exit(2)
    }
    var options = Options(modelURL: URL(fileURLWithPath: args.removeFirst()), videoURL: URL(fileURLWithPath: args.removeFirst()))
    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--stride": options.stride = Double(args.removeFirst()) ?? options.stride
        case "--min-coverage": options.minCoverage = Double(args.removeFirst()) ?? options.minCoverage
        case "--quiet": options.quiet = true
        default: fputs("unknown option \(flag)\n", stderr); exit(2)
        }
    }
    return options
}

func run(_ options: Options) async throws {
    let classifier = try ExerciseClassifier(modelURL: options.modelURL)
    let resampler = classifier.makeResampler()
    let smoother = LabelSmoother()

    let asset = AVURLAsset(url: options.videoURL)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw NSError(domain: "CLI", code: 1, userInfo: [NSLocalizedDescriptionKey: "no video track"])
    }
    let size = try await track.load(.naturalSize)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
    ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else { throw reader.error ?? NSError(domain: "CLI", code: 2) }

    let started = Date()
    var frames = 0, framesWithPose = 0, predictions = 0, skipped = 0
    var votes: [String: Int] = [:]
    var probabilitySum = Array(repeating: 0.0, count: classifier.classes.count)
    var nextPrediction = 0.0
    var firstTimestamp: Double?

    while let sample = output.copyNextSampleBuffer(), let pixelBuffer = CMSampleBufferGetImageBuffer(sample) {
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        if firstTimestamp == nil { firstTimestamp = timestamp; nextPrediction = timestamp + classifier.windowSeconds }
        frames += 1

        // --- per frame: pose -> features -> ring buffer (identical to the app's camera callback)
        let request = VNDetectHumanBodyPoseRequest()
        try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([request])
        let observation = request.results?.first
        if observation != nil { framesWithPose += 1 }
        let features = PoseFeatures.features(from: observation, imageWidth: Double(size.width),
                                             imageHeight: Double(size.height), timestamp: timestamp)
        resampler.push(features)

        // --- every `stride` seconds: resample the last window and classify
        guard timestamp >= nextPrediction, resampler.isReady() else { continue }
        nextPrediction += options.stride
        guard let window = resampler.window() else { continue }
        if window.coverage < options.minCoverage {
            skipped += 1
            if !options.quiet { print(String(format: "t=%6.2fs  coverage %.2f  (skipped)", timestamp, window.coverage)) }
            continue
        }
        let prediction = try classifier.predict(window)
        predictions += 1
        votes[prediction.label, default: 0] += 1
        for (i, p) in prediction.probabilities.enumerated() { probabilitySum[i] += p }
        let smoothed = smoother.push(prediction.label)
        if !options.quiet {
            let probs = zip(classifier.classes, prediction.probabilities).map { String(format: "%@ %.2f", $0, $1) }.joined(separator: "  ")
            print(String(format: "t=%6.2fs  coverage %.2f  %@   -> %@ (smoothed: %@)", timestamp, window.coverage, probs, prediction.label, smoothed))
        }
    }

    let elapsed = Date().timeIntervalSince(started)
    let duration = (resampler.latestTimestamp ?? 0) - (firstTimestamp ?? 0)
    let finalLabel = votes.max { $0.value < $1.value }?.key ?? "none"
    let meanProbs = predictions > 0 ? probabilitySum.map { $0 / Double(predictions) } : probabilitySum
    print("== \(options.videoURL.lastPathComponent)")
    print(String(format: "   %d frames (%.1f s), pose in %d, %d predictions, %d skipped for low coverage, %.1f s wall (%.1fx realtime)",
                 frames, duration, framesWithPose, predictions, skipped, elapsed, duration / max(elapsed, 1e-9)))
    print("   votes: \(votes.sorted { $0.value > $1.value }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
    print("   mean probabilities: " + zip(classifier.classes, meanProbs).map { String(format: "%@ %.3f", $0, $1) }.joined(separator: "  "))
    print("   RESULT: \(finalLabel)")
}

let options = parseOptions()
let semaphore = DispatchSemaphore(value: 0)
var failure: Error?
Task {
    do { try await run(options) } catch { failure = error }
    semaphore.signal()
}
semaphore.wait()
if let failure {
    fputs("error: \(failure.localizedDescription)\n", stderr)
    exit(1)
}
