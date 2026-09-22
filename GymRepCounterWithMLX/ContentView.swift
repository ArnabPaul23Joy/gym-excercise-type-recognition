import SwiftUI
import PhotosUI
import AVKit

struct ContentView: View {
    @State private var viewModel = RepCounterViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var player: AVPlayer?
    @State private var showLiveCamera = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    liveCameraButton
                    videoPickerCard
                    if videoURL != nil {
                        exerciseTypeCard
                        analyzeButton
                    }
                    analysisResultCard
                }
                .padding()
            }
            .navigationTitle("Gym Rep Counter")
            .navigationBarTitleDisplayMode(.inline)
        }
        .fullScreenCover(isPresented: $showLiveCamera) {
            LiveWorkoutView()
        }
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                guard let vid = try? await newItem.loadTransferable(type: VideoFile.self) else { return }
                videoURL = vid.url
                player = AVPlayer(url: vid.url)
                viewModel.analysisState = .idle
                // Upload finished — track body points and recognise the exercise right away.
                await viewModel.analyzeUpload(url: vid.url)
            }
        }
    }

    // MARK: - Live Camera Button

    private var liveCameraButton: some View {
        Button {
            showLiveCamera = true
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "camera.viewfinder")
                    .font(.title3)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Live Camera")
                        .fontWeight(.semibold)
                    Text("Count reps in real time")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Video Picker Card

    private var videoPickerCard: some View {
        VStack(spacing: 12) {
            if let player {
                PoseVideoView(player: player, poses: viewModel.poseFrames)
                    .frame(height: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(alignment: .bottomLeading) {
                        if viewModel.isClassifyingExercise {
                            Label("Tracking body points…", systemImage: "figure.walk.motion")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .background(.black.opacity(0.55), in: Capsule())
                                .foregroundStyle(.white)
                                .padding(8)
                        } else if !viewModel.poseFrames.isEmpty {
                            Label("Body-point tracking", systemImage: "figure.walk.motion")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .background(.black.opacity(0.55), in: Capsule())
                                .foregroundStyle(.white)
                                .padding(8)
                        }
                    }
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quaternary)
                    .frame(height: 130)
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "video.circle")
                                .font(.system(size: 36))
                                .foregroundStyle(.secondary)
                            Text("Select a workout video (up to 10 s)")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
            }

            PhotosPicker(
                selection: $pickerItem,
                matching: .videos,
                photoLibrary: .shared()
            ) {
                Label(
                    videoURL == nil ? "Select Workout Video" : "Change Video",
                    systemImage: "video.badge.plus"
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Exercise Type Card

    private var exerciseTypeCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Exercise Type")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                Image(systemName: "figure.strengthtraining.traditional")
                    .foregroundStyle(.tint)

                TextField("Identifying exercise…", text: $viewModel.exerciseTypeText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(viewModel.isClassifyingExercise)

                if viewModel.isClassifyingExercise {
                    ProgressView().controlSize(.small)
                }
            }

            if let type = viewModel.exerciseType {
                Text("\(Int((type.confidence * 100).rounded()))% confidence")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            } else if !viewModel.isClassifyingExercise {
                Text("Couldn't identify the exercise from this video.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Analyze Button

    private var analyzeButton: some View {
        Button {
            viewModel.countReps()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "figure.strengthtraining.traditional")
                Text("Count Reps")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!viewModel.canAnalyze)
        .animation(.default, value: viewModel.canAnalyze)
    }

    // MARK: - Result Card

    @ViewBuilder
    private var analysisResultCard: some View {
        switch viewModel.analysisState {
        case .done(let count):
            VStack(spacing: 6) {
                Text("\(count)")
                    .font(.system(size: 96, weight: .bold, design: .rounded))
                    .foregroundStyle(.tint)
                    .contentTransition(.numericText())
                    .animation(.spring(response: 0.4), value: count)
                Text(count == 1 ? "repetition" : "repetitions")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                Text("counted from body-pose geometry")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 28)
            .padding(.horizontal)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))

        case .error(let msg):
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .padding(.top, 1)
                Text(msg)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))

        case .counting:
            progressRow(label: "Counting reps from body points…", icon: "function")

        default:
            EmptyView()
        }
    }

    private func progressRow(label: String, icon: String) -> some View {
        HStack(spacing: 10) {
            ProgressView().controlSize(.small)
            Image(systemName: icon).foregroundStyle(.secondary)
            Text(label).font(.callout).foregroundStyle(.secondary)
            Spacer()
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Video File Transferable

struct VideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { vid in
            SentTransferredFile(vid.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".mov")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoFile(url: dest)
        }
    }
}

#Preview {
    ContentView()
}
