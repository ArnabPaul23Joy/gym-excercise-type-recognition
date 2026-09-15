// WindowResampler.swift
// Ring buffer of per-frame features + resampling onto the model's fixed time grid.
// Mirrors resample_window() in train_bilstm.py: linear interpolation over valid samples,
// mask = 1 where a real sample lies within `maskTolerance` seconds of the grid point.

import Foundation

nonisolated struct ResampledWindow {
    let values: [[Double]]   // [steps][features], 0 where mask is 0
    let mask: [[Double]]     // [steps][features], 1 = valid
    var coverage: Double {   // fraction of valid entries; use it to gate predictions
        let total = mask.reduce(0) { $0 + $1.reduce(0, +) }
        return total / Double(mask.count * (mask.first?.count ?? 1))
    }
}

nonisolated final class WindowResampler {
    let steps: Int
    let rateHz: Double
    let windowSeconds: Double
    let maskTolerance: Double
    let featureCount: Int
    private var frames: [FrameFeatures] = []

    init(steps: Int, rateHz: Double, windowSeconds: Double, maskTolerance: Double, featureCount: Int) {
        self.steps = steps
        self.rateHz = rateHz
        self.windowSeconds = windowSeconds
        self.maskTolerance = maskTolerance
        self.featureCount = featureCount
    }

    /// Seconds of history kept; a little more than the window so interpolation has neighbours at the edge.
    private var retention: Double { windowSeconds + 0.5 }

    func push(_ frame: FrameFeatures) {
        frames.append(frame)
        let cutoff = frame.timestamp - retention
        if let first = frames.firstIndex(where: { $0.timestamp >= cutoff }), first > 0 {
            frames.removeFirst(first)
        }
    }

    var latestTimestamp: Double? { frames.last?.timestamp }
    var earliestTimestamp: Double? { frames.first?.timestamp }

    /// True once the buffer spans a full window (or `minimumSpan` seconds, if you want earlier, weaker predictions).
    func isReady(minimumSpan: Double? = nil) -> Bool {
        guard let first = earliestTimestamp, let last = latestTimestamp else { return false }
        return last - first >= (minimumSpan ?? windowSeconds)
    }

    /// Resample the window that ends at the latest frame.
    func window() -> ResampledWindow? {
        guard let end = latestTimestamp else { return nil }
        return window(endingAt: end)
    }

    func window(endingAt end: Double) -> ResampledWindow {
        let grid = (0..<steps).map { end - Double(steps - 1 - $0) / rateHz }
        var values = Array(repeating: Array(repeating: 0.0, count: featureCount), count: steps)
        var mask = Array(repeating: Array(repeating: 0.0, count: featureCount), count: steps)

        for j in 0..<featureCount {
            // valid (time, value) samples for feature j
            let samples = frames.compactMap { f -> (Double, Double)? in
                f.values[j].isNaN ? nil : (f.timestamp, f.values[j])
            }
            guard samples.count >= 2 else { continue }
            var cursor = 0
            for (i, t) in grid.enumerated() {
                // advance so samples[cursor] is the last sample with time <= t (clamped like np.searchsorted)
                while cursor + 1 < samples.count && samples[cursor + 1].0 <= t { cursor += 1 }
                let lower = samples[cursor]
                let upper = samples[min(cursor + 1, samples.count - 1)]
                let interpolated: Double
                if t <= samples[0].0 {
                    interpolated = samples[0].1                       // np.interp clamps at the ends
                } else if t >= samples[samples.count - 1].0 {
                    interpolated = samples[samples.count - 1].1
                } else if upper.0 == lower.0 {
                    interpolated = lower.1
                } else {
                    let w = (t - lower.0) / (upper.0 - lower.0)
                    interpolated = lower.1 + w * (upper.1 - lower.1)
                }
                let nearest = min(abs(t - lower.0), abs(t - upper.0))
                if nearest <= maskTolerance {
                    values[i][j] = interpolated
                    mask[i][j] = 1
                }
            }
        }
        return ResampledWindow(values: values, mask: mask)
    }
}
