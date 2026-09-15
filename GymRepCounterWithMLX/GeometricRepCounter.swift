// GeometricRepCounter.swift
// Counts repetitions purely from body-point geometry — no ML inference.
//
// For each exercise we track one joint angle that swings through a large arc once per rep:
//   • squat   → knee angle   (hip → knee → ankle)
//   • push-up → elbow angle  (shoulder → elbow → wrist)
//   • pull-up → elbow angle  (shoulder → elbow → wrist)
//
// The per-frame angle forms a 1-D signal. We smooth it, then count full
// flexion→extension cycles with a Schmitt trigger whose thresholds are derived from
// the observed range of motion (hysteresis rejects jitter). Angles are computed in
// oriented pixel space so aspect ratio doesn't distort them.

import CoreGraphics
import Foundation

nonisolated enum GeometricRepCounter {

    /// Minimum peak-to-trough swing (degrees) required to accept the motion as real reps.
    private static let minRangeDegrees = 30.0
    /// Fraction of the range used as hysteresis margin on each side of the midpoint.
    private static let hysteresis = 0.30
    /// Moving-average half-window (frames) used to smooth the raw angle signal.
    private static let smoothingRadius = 2

    static func countReps(poses: [PoseFrame], orientedSize: CGSize, exercise: String) -> Int {
        guard !poses.isEmpty else { return 0 }
        let w = Double(orientedSize.width), h = Double(orientedSize.height)

        // 1. Raw per-frame angle (NaN where the needed joints weren't seen).
        let raw = poses.map { angleSignal(for: exercise, frame: $0, width: w, height: h) }

        // 2. Fill gaps by linear interpolation, then trim leading/trailing gaps.
        guard let filled = interpolateGaps(raw) else { return 0 }

        // 3. Smooth to suppress detector jitter.
        let signal = smooth(filled, radius: smoothingRadius)

        // 4. Adaptive thresholds from the range of motion.
        guard let mn = signal.min(), let mx = signal.max() else { return 0 }
        let range = mx - mn
        guard range >= minRangeDegrees else { return 0 }
        let low = mn + hysteresis * range
        let high = mx - hysteresis * range

        // 5. Schmitt trigger: one rep per extended → flexed → extended cycle.
        return countCycles(signal, low: low, high: high)
    }

    // MARK: - Per-exercise angle

    /// The rep-tracking joint angle (degrees) for one frame, or nil if the needed joints
    /// weren't seen. Shared by the batch counter and the live `StreamingRepCounter`.
    static func repAngle(for exercise: String, frame: PoseFrame, orientedSize: CGSize) -> Double? {
        let v = angleSignal(for: exercise, frame: frame,
                            width: Double(orientedSize.width), height: Double(orientedSize.height))
        return v.isNaN ? nil : v
    }

    private static func angleSignal(for exercise: String, frame: PoseFrame, width: Double, height: Double) -> Double {
        switch exercise {
        case "squat":
            return averageAngle(frame, width: width, height: height,
                                 left: ("left_hip", "left_knee", "left_ankle"),
                                 right: ("right_hip", "right_knee", "right_ankle"))
        case "pushup", "pullup":
            return averageAngle(frame, width: width, height: height,
                                 left: ("left_shoulder", "left_elbow", "left_wrist"),
                                 right: ("right_shoulder", "right_elbow", "right_wrist"))
        default:
            return .nan
        }
    }

    /// Angle at joint `b` (vertex) for the a-b-c chain, averaged over whichever sides are visible.
    private static func averageAngle(_ frame: PoseFrame, width: Double, height: Double,
                                     left: (String, String, String),
                                     right: (String, String, String)) -> Double {
        let l = jointAngle(frame, left, width: width, height: height)
        let r = jointAngle(frame, right, width: width, height: height)
        switch (l, r) {
        case let (l?, r?): return (l + r) / 2
        case let (l?, nil): return l
        case let (nil, r?): return r
        default:           return .nan
        }
    }

    private static func jointAngle(_ frame: PoseFrame, _ names: (String, String, String),
                                   width: Double, height: Double) -> Double? {
        guard let a = frame.joints[names.0], let b = frame.joints[names.1], let c = frame.joints[names.2] else {
            return nil
        }
        // Normalised (top-left) → oriented pixel space so the angle is geometrically true.
        let pa = SIMD2(a.x * width, a.y * height)
        let pb = SIMD2(b.x * width, b.y * height)
        let pc = SIMD2(c.x * width, c.y * height)
        let ba = pa - pb, bc = pc - pb
        let denom = length(ba) * length(bc)
        guard denom > 0 else { return nil }
        let cosine = dot(ba, bc) / denom
        return acos(min(max(cosine, -1), 1)) * 180 / .pi
    }

    // MARK: - Signal processing

    /// Replaces NaN runs with linear interpolation between valid samples; nil if <2 valid samples.
    private static func interpolateGaps(_ values: [Double]) -> [Double]? {
        let valid = values.enumerated().filter { !$0.element.isNaN }
        guard valid.count >= 2 else { return nil }

        var out = values
        // Clamp the ends to the nearest valid sample.
        let first = valid.first!, last = valid.last!
        for i in 0..<first.offset { out[i] = first.element }
        for i in (last.offset + 1)..<out.count { out[i] = last.element }
        // Interpolate interior gaps.
        for k in 0..<(valid.count - 1) {
            let (i0, v0) = valid[k], (i1, v1) = valid[k + 1]
            guard i1 > i0 + 1 else { continue }
            for i in (i0 + 1)..<i1 {
                let t = Double(i - i0) / Double(i1 - i0)
                out[i] = v0 + t * (v1 - v0)
            }
        }
        return out
    }

    private static func smooth(_ values: [Double], radius: Int) -> [Double] {
        guard radius > 0, values.count > 2 * radius else { return values }
        return values.indices.map { i in
            let lo = max(0, i - radius), hi = min(values.count - 1, i + radius)
            var sum = 0.0
            for j in lo...hi { sum += values[j] }
            return sum / Double(hi - lo + 1)
        }
    }

    private static func countCycles(_ signal: [Double], low: Double, high: Double) -> Int {
        var reps = 0
        // Start in whichever half the first sample sits, so a partial opening rep isn't miscounted.
        var extended = signal[0] >= (low + high) / 2
        for v in signal {
            if extended, v < low {
                extended = false            // entered the flexed/contracted phase
            } else if !extended, v > high {
                reps += 1                   // returned to extended → one full rep
                extended = true
            }
        }
        return reps
    }

    private static func length(_ v: SIMD2<Double>) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }
    private static func dot(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.x + a.y * b.y }
}
