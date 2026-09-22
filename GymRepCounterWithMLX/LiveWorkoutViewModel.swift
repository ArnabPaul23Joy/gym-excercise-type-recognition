// LiveWorkoutViewModel.swift
// MainActor UI state for the live-camera screen. Owns the LiveExerciseEngine, requests
// camera permission, and marshals per-frame LiveUpdates from the capture queue onto the
// main actor for SwiftUI.

import AVFoundation
import SwiftUI

@MainActor
@Observable
final class LiveWorkoutViewModel {

    enum Permission { case unknown, authorized, denied }

    let engine = LiveExerciseEngine()

    var permission: Permission = .unknown
    var latestPose: PoseFrame?
    var orientedSize: CGSize = .zero
    var displayLabel = ""
    var confidence = 0.0
    var reps = 0
    var calibrating = true
    private(set) var cameraPosition: AVCaptureDevice.Position = .back

    var session: AVCaptureSession { engine.session }

    func start() async {
        permission = await requestCameraAccess() ? .authorized : .denied
        guard permission == .authorized else { return }

        engine.onUpdate = { [weak self] update in
            Task { @MainActor [weak self] in self?.apply(update) }
        }
        engine.configure(position: cameraPosition)
        engine.start()
    }

    func stop() {
        engine.stop()
    }

    func resetReps() {
        engine.resetCounter()
        reps = 0
    }

    func flipCamera() {
        cameraPosition = (cameraPosition == .back) ? .front : .back
        latestPose = nil
        engine.resetCounter()
        reps = 0
        engine.configure(position: cameraPosition)
    }

    private func apply(_ update: LiveUpdate) {
        latestPose = update.pose
        orientedSize = update.orientedSize
        displayLabel = update.displayLabel
        confidence = update.confidence
        reps = update.reps
        calibrating = update.calibrating
    }

    private func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }
}
