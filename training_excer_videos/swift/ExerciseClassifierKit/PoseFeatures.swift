// PoseFeatures.swift
// Turns one VNHumanBodyPoseObservation into the 23 features the model was trained on.
// Mirrors build_features.py exactly: hip-centred, torso-scaled joints -> 7 angles + 16 distances.

import Foundation
import Vision

/// One frame of features. `values[i]` is NaN when the joints behind feature i were not detected.
struct FrameFeatures {
    let timestamp: Double
    let values: [Double]
}

enum PoseFeatures {
    /// Same joint order and naming as build_features.JOINTS.
    static let joints: [(name: String, vision: VNHumanBodyPoseObservation.JointName)] = [
        ("nose", .nose), ("left_eye", .leftEye), ("right_eye", .rightEye),
        ("left_ear", .leftEar), ("right_ear", .rightEar),
        ("left_shoulder", .leftShoulder), ("right_shoulder", .rightShoulder),
        ("left_elbow", .leftElbow), ("right_elbow", .rightElbow),
        ("left_wrist", .leftWrist), ("right_wrist", .rightWrist),
        ("left_hip", .leftHip), ("right_hip", .rightHip),
        ("left_knee", .leftKnee), ("right_knee", .rightKnee),
        ("left_ankle", .leftAnkle), ("right_ankle", .rightAnkle),
    ]

    static let angles: [(String, String, String)] = [
        ("right_elbow", "right_shoulder", "right_hip"),
        ("left_elbow", "left_shoulder", "left_hip"),
        ("right_knee", "mid_hip", "left_knee"),
        ("right_hip", "right_knee", "right_ankle"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_wrist", "right_elbow", "right_shoulder"),
        ("left_wrist", "left_elbow", "left_shoulder"),
    ]

    static let distances: [(String, String)] = [
        ("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist"),
        ("left_hip", "left_ankle"), ("right_hip", "right_ankle"),
        ("left_hip", "left_wrist"), ("right_hip", "right_wrist"),
        ("left_shoulder", "left_ankle"), ("right_shoulder", "right_ankle"),
        ("left_hip", "right_wrist"), ("right_hip", "left_wrist"),
        ("left_elbow", "right_elbow"), ("left_knee", "right_knee"),
        ("left_wrist", "right_wrist"), ("left_ankle", "right_ankle"),
        ("left_hip", "avg_left_wrist_left_ankle"), ("right_hip", "avg_right_wrist_right_ankle"),
    ]

    static let torsoMultiplier = 2.5
    static var featureCount: Int { angles.count + distances.count }

    /// Feature names in model order, for cross-checking against the model metadata.
    static var featureNames: [String] {
        angles.map { "\($0.0)_\($0.1)_\($0.2)" } + distances.map { "\($0.0)_\($0.1)" }
    }

    /// Extract pixel-space joints (top-left origin, like VisionPoseExtractor.swift) from an observation.
    static func pixelJoints(from observation: VNHumanBodyPoseObservation,
                            imageWidth: Double, imageHeight: Double) -> [String: SIMD2<Double>] {
        guard let points = try? observation.recognizedPoints(.all) else { return [:] }
        var result: [String: SIMD2<Double>] = [:]
        for joint in joints {
            guard let point = points[joint.vision], point.confidence > 0 else { continue }
            result[joint.name] = SIMD2(Double(point.location.x) * imageWidth,
                                       (1.0 - Double(point.location.y)) * imageHeight)
        }
        return result
    }

    /// Full pipeline for one frame: joints -> normalised joints -> 23 features (NaN where missing).
    static func features(from observation: VNHumanBodyPoseObservation?,
                         imageWidth: Double, imageHeight: Double, timestamp: Double) -> FrameFeatures {
        guard let observation else {
            return FrameFeatures(timestamp: timestamp, values: Array(repeating: .nan, count: featureCount))
        }
        let raw = pixelJoints(from: observation, imageWidth: imageWidth, imageHeight: imageHeight)
        let normalised = normalise(raw)
        return FrameFeatures(timestamp: timestamp, values: compute(normalised))
    }

    /// Hip-centre, then scale by max(2.5 * torso, max distance from hips) * 100 — the dataset convention.
    static func normalise(_ joints: [String: SIMD2<Double>]) -> [String: SIMD2<Double>] {
        guard let leftHip = joints["left_hip"], let rightHip = joints["right_hip"] else {
            return [:]  // without hips nothing can be normalised (matches the NaN rows in the CSV)
        }
        let hips = (leftHip + rightHip) / 2
        let maxDist = joints.values.map { length($0 - hips) }.max() ?? 0
        var size = maxDist
        // np.fmax in build_features.py ignores a NaN torso, so missing shoulders fall back to maxDist
        if let leftShoulder = joints["left_shoulder"], let rightShoulder = joints["right_shoulder"] {
            let torso = length((leftShoulder + rightShoulder) / 2 - hips)
            size = max(torso * torsoMultiplier, maxDist)
        }
        guard size > 0 else { return [:] }
        return joints.mapValues { ($0 - hips) / size * 100 }
    }

    static func compute(_ joints: [String: SIMD2<Double>]) -> [Double] {
        func point(_ name: String) -> SIMD2<Double>? {
            if name == "mid_hip" {
                guard let l = joints["left_hip"], let r = joints["right_hip"] else { return nil }
                return (l + r) / 2
            }
            if name.hasPrefix("avg_") {   // avg_left_wrist_left_ankle -> mean of left_wrist and left_ankle
                let parts = name.dropFirst(4).split(separator: "_")
                guard parts.count == 4,
                      let a = joints["\(parts[0])_\(parts[1])"], let b = joints["\(parts[2])_\(parts[3])"] else { return nil }
                return (a + b) / 2
            }
            return joints[name]
        }
        var values: [Double] = []
        values.reserveCapacity(featureCount)
        for (a, b, c) in angles {
            guard let pa = point(a), let pb = point(b), let pc = point(c) else { values.append(.nan); continue }
            let ba = pa - pb, bc = pc - pb
            let cosine = dot(ba, bc) / (length(ba) * length(bc))
            values.append(acos(min(max(cosine, -1), 1)) * 180 / .pi)
        }
        for (a, b) in distances {
            guard let pa = point(a), let pb = point(b) else { values.append(.nan); continue }
            values.append(length(pb - pa))
        }
        return values
    }

    private static func length(_ v: SIMD2<Double>) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }
    private static func dot(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.x + a.y * b.y }
}
