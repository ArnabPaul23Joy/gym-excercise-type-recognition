// PoseExerciseClassifier.swift
// Loads ExerciseClassifier.mlmodelc, reads the preprocessing constants baked into its metadata,
// and turns a ResampledWindow into class probabilities.
//
// In an iOS app, add ExerciseClassifier.mlpackage to the target and load the
// compiled model with Bundle.main.url(forResource: "ExerciseClassifier", withExtension: "mlmodelc").

import CoreML
import Foundation

nonisolated struct Prediction {
    let probabilities: [Double]
    let classes: [String]
    var label: String { classes[probabilities.indices.max { probabilities[$0] < probabilities[$1] }!] }
    var confidence: Double { probabilities.max() ?? 0 }
}

// NOTE: named PoseExerciseClassifier (not ExerciseClassifier) to avoid colliding
// with the Swift class Xcode auto-generates from ExerciseClassifier.mlpackage.
nonisolated final class PoseExerciseClassifier {
    let model: MLModel
    let classes: [String]
    let featureColumns: [String]
    let mean: [Double]
    let std: [Double]
    let rateHz: Double
    let windowSeconds: Double
    let maskTolerance: Double
    let steps: Int
    let inputs: Int

    init(modelURL: URL, computeUnits: MLComputeUnits = .all) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        model = try MLModel(contentsOf: modelURL, configuration: configuration)

        let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        func json<T: Decodable>(_ key: String, as type: T.Type) throws -> T {
            guard let text = meta[key], let data = text.data(using: .utf8) else {
                throw NSError(domain: "ExerciseClassifier", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "metadata key \(key) missing from model"])
            }
            return try JSONDecoder().decode(type, from: data)
        }
        classes = try json("classes", as: [String].self)
        featureColumns = try json("feature_columns", as: [String].self)
        mean = try json("feature_mean", as: [Double].self)
        std = try json("feature_std", as: [Double].self)
        rateHz = Double(meta["rate_hz"] ?? "15") ?? 15
        windowSeconds = Double(meta["window_s"] ?? "2") ?? 2
        maskTolerance = Double(meta["mask_tolerance_s"] ?? "0.15") ?? 0.15

        guard let shape = model.modelDescription.inputDescriptionsByName["features"]?.multiArrayConstraint?.shape,
              shape.count == 3 else {
            throw NSError(domain: "ExerciseClassifier", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "model input 'features' must be rank 3"])
        }
        steps = shape[1].intValue
        inputs = shape[2].intValue
        guard inputs == 2 * featureColumns.count, featureColumns == PoseFeatures.featureNames else {
            throw NSError(domain: "ExerciseClassifier", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "PoseFeatures order does not match the model's feature_columns"])
        }
    }

    func makeResampler() -> WindowResampler {
        WindowResampler(steps: steps, rateHz: rateHz, windowSeconds: windowSeconds,
                        maskTolerance: maskTolerance, featureCount: featureColumns.count)
    }

    /// (value - mean) / std * mask, concatenated with the mask -> MLMultiArray [1, steps, inputs].
    func makeInput(_ window: ResampledWindow) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: steps), NSNumber(value: inputs)], dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let features = featureColumns.count
        for i in 0..<steps {
            for j in 0..<features {
                let m = window.mask[i][j]
                pointer[i * inputs + j] = Float32((window.values[i][j] - mean[j]) / std[j] * m)
                pointer[i * inputs + features + j] = Float32(m)
            }
        }
        return array
    }

    func predict(_ window: ResampledWindow) throws -> Prediction {
        let input = try MLDictionaryFeatureProvider(dictionary: ["features": try makeInput(window)])
        let output = try model.prediction(from: input)
        guard let probabilities = output.featureValue(for: "probabilities")?.multiArrayValue else {
            throw NSError(domain: "ExerciseClassifier", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "model returned no 'probabilities'"])
        }
        return Prediction(probabilities: (0..<probabilities.count).map { probabilities[$0].doubleValue }, classes: classes)
    }
}

/// Majority vote over recent window predictions, for a stable on-screen label.
nonisolated final class LabelSmoother {
    private var recent: [String] = []
    let size: Int
    init(size: Int = 8) { self.size = size }

    func push(_ label: String) -> String {
        recent.append(label)
        if recent.count > size { recent.removeFirst() }
        let counts = Dictionary(recent.map { ($0, 1) }, uniquingKeysWith: +)
        return counts.max { $0.value < $1.value }!.key
    }
}
