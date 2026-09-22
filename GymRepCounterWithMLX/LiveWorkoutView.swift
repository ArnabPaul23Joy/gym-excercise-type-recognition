// LiveWorkoutView.swift
// Full-screen live-camera workout screen: camera preview + pose skeleton overlay, with a
// HUD showing the live exercise type, confidence, and rep count.

import SwiftUI

struct LiveWorkoutView: View {
    @State private var viewModel = LiveWorkoutViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch viewModel.permission {
            case .denied:
                permissionDeniedView
            case .unknown:
                ProgressView().tint(.white)
            case .authorized:
                LiveCameraView(session: viewModel.session, pose: viewModel.latestPose,
                               orientedSize: viewModel.orientedSize)
                    .ignoresSafeArea()
                hud
            }
        }
        .task { await viewModel.start() }
        .onDisappear { viewModel.stop() }
        .statusBarHidden()
    }

    // MARK: - HUD

    private var hud: some View {
        VStack(spacing: 0) {
            topBar
            Spacer()
            if viewModel.calibrating {
                calibratingBanner
            }
            repCountBar
        }
        .padding()
    }

    private var topBar: some View {
        HStack {
            Button {
                viewModel.stop()
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.white)
            }

            Spacer()

            VStack(spacing: 2) {
                Text(viewModel.displayLabel.isEmpty ? "—" : viewModel.displayLabel)
                    .font(.headline)
                    .foregroundStyle(.white)
                if !viewModel.displayLabel.isEmpty {
                    Text("\(Int((viewModel.confidence * 100).rounded()))% confidence")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.black.opacity(0.4), in: Capsule())

            Spacer()

            Button {
                viewModel.flipCamera()
            } label: {
                Image(systemName: "arrow.triangle.2.circlepath.camera.fill")
                    .font(.title2)
                    .foregroundStyle(.white)
            }
        }
    }

    private var calibratingBanner: some View {
        Label("Calibrating… hold steady in frame", systemImage: "timer")
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.black.opacity(0.5), in: Capsule())
            .padding(.bottom, 12)
    }

    private var repCountBar: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 0) {
                Text("\(viewModel.reps)")
                    .font(.system(size: 64, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .contentTransition(.numericText())
                    .animation(.spring(response: 0.35), value: viewModel.reps)
                Text(viewModel.reps == 1 ? "rep" : "reps")
                    .font(.headline)
                    .foregroundStyle(.white.opacity(0.8))
            }
            Spacer()
            Button {
                viewModel.resetReps()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(.white.opacity(0.18), in: Capsule())
            }
        }
        .padding()
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 20))
    }

    // MARK: - Permission

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "video.slash.fill")
                .font(.system(size: 48))
                .foregroundStyle(.white.opacity(0.8))
            Text("Camera access is off")
                .font(.title3.weight(.semibold))
                .foregroundStyle(.white)
            Text("Enable camera access in Settings to use live rep counting.")
                .font(.callout)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white.opacity(0.75))
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
            Button("Close") { dismiss() }
                .foregroundStyle(.white)
        }
        .padding(32)
    }
}
