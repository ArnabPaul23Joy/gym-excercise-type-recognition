import SwiftUI
import CoreGraphics

@MainActor
@Observable
final class RepCounterViewModel {

    // MARK: - State

    enum AnalysisState {
        case idle
        case counting
        case done(repCount: Int)
        case error(String)
    }

    var analysisState: AnalysisState = .idle

    // MARK: - Exercise Type + Pose Tracking

    /// Recognised exercise type for the current video (nil until identified, or if pose detection fails).
    var exerciseType: ExerciseTypeAnalyzer.Result?
    /// Text shown in the exercise-type field; kept in sync with `exerciseType`.
    var exerciseTypeText: String = ""
    /// True while the upload is being analysed (pose extraction + Core ML classification).
    var isClassifyingExercise = false
    /// Per-frame body-pose keypoints for the overlay + geometric rep counter.
    var poseFrames: [PoseFrame] = []
    /// Displayed (upright) pixel size of the current video.
    private(set) var orientedSize: CGSize = .zero
    /// Guards against a stale analysis (older video) overwriting a newer one.
    private var analyzeToken = 0

    var isAnalyzing: Bool {
        if case .counting = analysisState { return true }
        return false
    }

    /// Reps can be counted once a video has been analysed and its pose tracking is ready.
    var canAnalyze: Bool { !isAnalyzing && !isClassifyingExercise && !poseFrames.isEmpty }

    // MARK: - Upload Analysis (pose tracking + exercise type)

    /// Runs one pass over a freshly uploaded video: extracts body-pose keypoints and
    /// recognises the exercise type. Failure just leaves the fields empty.
    func analyzeUpload(url: URL) async {
        analyzeToken += 1
        let token = analyzeToken
        exerciseType = nil
        exerciseTypeText = ""
        poseFrames = []
        orientedSize = .zero
        analysisState = .idle
        isClassifyingExercise = true

        let analysis = try? await ExerciseTypeAnalyzer.analyze(url: url)

        // Ignore if a newer video started analysing while this one ran.
        guard token == analyzeToken else { return }
        isClassifyingExercise = false
        guard let analysis else { return }
        poseFrames = analysis.poses
        orientedSize = analysis.orientedSize
        exerciseType = analysis.type
        exerciseTypeText = analysis.type?.label ?? ""
    }

    // MARK: - Geometric Rep Counting

    /// Counts reps from the stored pose keypoints using joint-angle geometry.
    /// Instant (pure math) — the expensive pose extraction already ran on upload.
    func countReps() {
        guard !poseFrames.isEmpty else {
            analysisState = .error("Upload a video and wait for pose tracking to finish.")
            return
        }
        guard let exercise = exerciseType?.rawLabel else {
            analysisState = .error("Couldn't identify the exercise, so reps can't be counted.")
            return
        }
        analysisState = .counting
        let reps = GeometricRepCounter.countReps(poses: poseFrames, orientedSize: orientedSize, exercise: exercise)
        analysisState = .done(repCount: reps)
    }
}
