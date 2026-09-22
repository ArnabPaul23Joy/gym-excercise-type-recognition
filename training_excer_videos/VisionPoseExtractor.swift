import AVFoundation
import CoreMedia
import Foundation
import Vision

struct Joint: Encodable {
    let x: Double
    let y: Double
    let confidence: Double
}

struct FramePose: Encodable {
    let frame: Int
    let timestamp: Double
    let joints: [String: Joint]
}

let arguments = CommandLine.arguments
 guard arguments.count == 3 else {
    fputs("Usage: VisionPoseExtractor <input-video> <output-jsonl>\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])
let asset = AVURLAsset(url: inputURL)
let semaphore = DispatchSemaphore(value: 0)
var extractionError: Error?

Task {
    do {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "VisionPoseExtractor", code: 1, userInfo: [NSLocalizedDescriptionKey: "No video track found"])
        }

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw NSError(domain: "VisionPoseExtractor", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not configure AVAssetReader"])
        }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? NSError(domain: "VisionPoseExtractor", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not start video reader"])
        }

        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: outputURL)
        defer { try? handle.close() }

        let jointNames: [VNHumanBodyPoseObservation.JointName] = [
            .nose, .leftEye, .rightEye, .leftEar, .rightEar,
            .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
            .leftWrist, .rightWrist, .leftHip, .rightHip,
            .leftKnee, .rightKnee, .leftAnkle, .rightAnkle
        ]
        let encoder = JSONEncoder()
        var frameIndex = 0

        while let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let request = VNDetectHumanBodyPoseRequest()
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            try handler.perform([request])

            var joints: [String: Joint] = [:]
            if let observation = request.results?.first,
               let recognizedPoints = try? observation.recognizedPoints(.all) {
                for name in jointNames {
                    guard let point = recognizedPoints[name], point.confidence > 0 else { continue }
                    joints[name.rawValue.rawValue] = Joint(
                        x: Double(point.location.x),
                        y: Double(1.0 - point.location.y),
                        confidence: Double(point.confidence)
                    )
                }
            }

            let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
            let record = FramePose(frame: frameIndex, timestamp: timestamp, joints: joints)
            let data = try encoder.encode(record)
            handle.write(data)
            handle.write(Data([0x0A]))
            frameIndex += 1
        }
    } catch {
        extractionError = error
    }
    semaphore.signal()
}

semaphore.wait()
if let extractionError {
    fputs("Pose extraction failed: \(extractionError.localizedDescription)\n", stderr)
    exit(1)
}
