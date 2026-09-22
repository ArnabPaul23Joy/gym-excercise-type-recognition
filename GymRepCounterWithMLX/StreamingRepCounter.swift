// StreamingRepCounter.swift
// Live (online) version of GeometricRepCounter: fed one joint-angle sample per camera
// frame, it counts reps incrementally with the same geometry as the batch counter.
//
// Same idea as the offline Schmitt trigger, adapted to a stream:
//   • EMA-smooth the incoming angle (live pose is jittier than decoded video).
//   • Track a running min/max to derive adaptive low/high thresholds (hysteresis).
//   • One rep per extended → flexed → extended cycle.
// Thresholds only "arm" once the observed range exceeds a minimum, so idle jitter and
// the warm-up period don't produce phantom reps. Resets when the exercise changes.

import Foundation

nonisolated final class StreamingRepCounter {

    private let minRangeDegrees = 30.0
    private let hysteresis = 0.30
    private let emaAlpha = 0.35              // weight of the newest sample

    private(set) var reps = 0
    private var exercise: String?
    private var smoothed: Double?
    private var minAngle = Double.greatestFiniteMagnitude
    private var maxAngle = -Double.greatestFiniteMagnitude
    private var extended = true

    /// Switches the tracked exercise; a change restarts counting from zero.
    func setExercise(_ newExercise: String?) {
        guard newExercise != exercise else { return }
        exercise = newExercise
        reset()
    }

    func reset() {
        reps = 0
        smoothed = nil
        minAngle = .greatestFiniteMagnitude
        maxAngle = -.greatestFiniteMagnitude
        extended = true
    }

    /// Feed one frame's rep-tracking angle (nil when the joints weren't seen).
    func update(angle: Double?) {
        guard let angle else { return }

        let s = smoothed.map { $0 * (1 - emaAlpha) + angle * emaAlpha } ?? angle
        smoothed = s
        minAngle = min(minAngle, s)
        maxAngle = max(maxAngle, s)

        let range = maxAngle - minAngle
        guard range >= minRangeDegrees else { return }
        let low = minAngle + hysteresis * range
        let high = maxAngle - hysteresis * range

        if extended, s < low {
            extended = false                 // entered the flexed/contracted phase
        } else if !extended, s > high {
            reps += 1                        // returned to extended → one full rep
            extended = true
        }
    }
}
