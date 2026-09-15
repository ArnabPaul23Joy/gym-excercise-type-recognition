import SwiftUI
import TensorFlowLite
import TensorFlowLiteCMetal

@MainActor
@Observable
final class RepCounterViewModel {

    // MARK: - State

    enum AnalysisState {
        case idle
        case extractingFrames
        case runningModel
        case done(repCount: Int)
        case error(String)
    }

    var analysisState: AnalysisState = .idle
    var frameThumbnails: [UIImage] = []

    // MARK: - Exercise Type

    /// Recognised exercise type for the current video (nil until identified, or if pose detection fails).
    var exerciseType: ExerciseTypeAnalyzer.Result?
    /// Text shown in the exercise-type field; kept in sync with `exerciseType`.
    var exerciseTypeText: String = ""
    /// True while the Core ML classifier is running over a freshly uploaded video.
    var isClassifyingExercise = false
    /// Guards against a stale classification (older video) overwriting a newer one.
    private var classifyToken = 0

    var isAnalyzing: Bool {
        switch analysisState {
        case .extractingFrames, .runningModel: return true
        default: return false
        }
    }

    var canAnalyze: Bool { !isAnalyzing }

    // MARK: - Exercise Type Recognition

    /// Recognises the exercise type as soon as a video finishes uploading.
    /// Runs independently of rep counting; a failure just leaves the field empty.
    func classifyExercise(url: URL) async {
        classifyToken += 1
        let token = classifyToken
        exerciseType = nil
        exerciseTypeText = ""
        isClassifyingExercise = true

        let result = try? await ExerciseTypeAnalyzer.classify(url: url)

        // Ignore if a newer video started classifying while this one ran.
        guard token == classifyToken else { return }
        exerciseType = result
        exerciseTypeText = result?.label ?? ""
        isClassifyingExercise = false
    }

    // MARK: - Analysis

    func analyzeVideo(url: URL) async {
        // Re-entrancy guard: ignore taps/picks while an analysis is in flight.
        guard !isAnalyzing else { return }

        frameThumbnails = []
        analysisState = .extractingFrames

        // Hold a background assertion so a brief trip to the background mid-run
        // doesn't get the app killed for submitting Metal GPU work while suspended.
        var bgTask: UIBackgroundTaskIdentifier = .invalid
        bgTask = UIApplication.shared.beginBackgroundTask(withName: "RepNetInference") {
            if bgTask != .invalid {
                UIApplication.shared.endBackgroundTask(bgTask)
                bgTask = .invalid
            }
        }
        defer {
            if bgTask != .invalid {
                UIApplication.shared.endBackgroundTask(bgTask)
                bgTask = .invalid
            }
        }

        do {
            let frames = try await VideoFrameExtractor.extractFrames(from: url)
            frameThumbnails = VideoFrameExtractor.thumbnails(from: frames)

            analysisState = .runningModel

            // TFLite inference is CPU/GPU-heavy; run off the main actor.
            let count = try await Task.detached(priority: .userInitiated) {
                try RepCounterViewModel.runInference(frames: frames)
            }.value

            analysisState = .done(repCount: count)
        } catch {
            analysisState = .error(error.localizedDescription)
        }
    }

    // MARK: - TFLite Inference

    private nonisolated static func runInference(frames: [CGImage]) throws -> Int {
        guard let modelPath = Bundle.main.path(forResource: "repnet", ofType: "tflite") else {
            throw RepNetError.modelNotFound
        }

        var options = Interpreter.Options()
        // Threads are the fallback for any ops the GPU delegate can't run.
        options.threadCount = max(2, ProcessInfo.processInfo.activeProcessorCount)

        // Prefer the Metal GPU delegate for speed, but this RepNet graph
        // (5D tensors + transformer ops) may be rejected by the GPU delegate —
        // in that case interpreter creation throws, so fall back to CPU.
        let interpreter: Interpreter
        do {
            let gpu = try Interpreter(modelPath: modelPath, options: options, delegates: [GPUMetalDelegate()])
            try gpu.allocateTensors()
            interpreter = gpu
        } catch {
            let cpu = try Interpreter(modelPath: modelPath, options: options)
            try cpu.allocateTensors()
            interpreter = cpu
        }

        // Input: [1, 3, 64, 224, 224] float32, values in [0, 1]
        let input = try VideoFrameExtractor.framesToInputTensor(frames)
        let inputData = input.withUnsafeBufferPointer { Data(buffer: $0) }
        try interpreter.copy(inputData, toInputAt: 0)

        try interpreter.invoke()

        // Outputs are matched by shape, not index:
        //   last dim 32 → period-length logits, last dim 1 → periodicity logits.
        var periodLogits: [Float] = []
        var periodicity: [Float] = []

        for i in 0..<interpreter.outputTensorCount {
            let tensor = try interpreter.output(at: i)
            let last = tensor.shape.dimensions.last ?? 0
            let floats = tensor.data.toFloatArray()
            if last == 32 {
                periodLogits = floats
            } else if last == 1 {
                periodicity = floats
            }
        }

        guard !periodLogits.isEmpty, !periodicity.isEmpty else {
            throw RepNetError.invalidModelOutput
        }

        return computeRepCount(periodLogits: periodLogits, periodicity: periodicity)
    }

    // RepNet counting: for each of the 64 frames, if it's inside a periodic
    // segment (sigmoid(periodicity) ≥ 0.5), it contributes 1/period reps, where
    // period = argmax(period-length logits) + 1. Total reps = Σ contributions.
    private nonisolated static func computeRepCount(periodLogits: [Float], periodicity: [Float]) -> Int {
        let numFrames = periodicity.count            // 64
        let numBins = periodLogits.count / max(numFrames, 1) // 32
        guard numFrames > 0, numBins > 0 else { return 0 }

        let threshold: Float = 0.5
        var total: Float = 0

        for t in 0..<numFrames {
            let within = 1.0 / (1.0 + exp(-periodicity[t])) // sigmoid
            guard within >= threshold else { continue }

            var maxBin = 0
            var maxVal = -Float.greatestFiniteMagnitude
            for b in 0..<numBins {
                let v = periodLogits[t * numBins + b]
                if v > maxVal { maxVal = v; maxBin = b }
            }

            let period = Float(maxBin + 1)
            total += 1.0 / period
        }
        return max(0, Int(total.rounded()))
    }

    // MARK: - Errors

    enum RepNetError: LocalizedError {
        case modelNotFound
        case invalidModelOutput

        var errorDescription: String? {
            switch self {
            case .modelNotFound:
                return "repnet.tflite not found in the app bundle. Confirm the file is added to the app target in Xcode."
            case .invalidModelOutput:
                return "RepNet returned unexpected outputs. Expected tensors with last dimensions 32 (period length) and 1 (periodicity)."
            }
        }
    }
}

// MARK: - Metal GPU Delegate

// The repackaged TensorFlowLiteSwift omits the built-in MetalDelegate wrapper,
// so we build one directly on the TensorFlowLiteCMetal C API.
private nonisolated final class GPUMetalDelegate: Delegate, @unchecked Sendable {
    let cDelegate: Delegate.CDelegate

    init() {
        var options = TFLGpuDelegateOptionsDefault()
        options.allow_precision_loss = true      // fp16 on GPU — big speedup
        options.wait_type = TFLGpuDelegateWaitTypePassive
        cDelegate = TFLGpuDelegateCreate(&options)!
    }

    deinit {
        TFLGpuDelegateDelete(cDelegate)
    }
}

// MARK: - Data → [Float]

private extension Data {
    nonisolated func toFloatArray() -> [Float] {
        let count = self.count / MemoryLayout<Float>.stride
        return withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(count))
        }
    }
}
