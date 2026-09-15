import SwiftUI
import PhotosUI
import AVKit

struct ContentView: View {
    @State private var viewModel = RepCounterViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var videoURL: URL?
    @State private var player: AVPlayer?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    videoPickerCard
                    if videoURL != nil {
                        exerciseTypeCard
                    }
                    if !viewModel.frameThumbnails.isEmpty {
                        framesStripCard
                    }
                    if videoURL != nil {
                        analyzeButton
                    }
                    analysisResultCard
                }
                .padding()
            }
            .navigationTitle("Gym Rep Counter")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                guard let vid = try? await newItem.loadTransferable(type: VideoFile.self) else { return }
                videoURL = vid.url
                player = AVPlayer(url: vid.url)
                // Don't stomp state that an in-flight analysis is still updating.
                if !viewModel.isAnalyzing {
                    viewModel.analysisState = .idle
                    viewModel.frameThumbnails = []
                }
                // Upload finished — start recognising the exercise type right away.
                await viewModel.classifyExercise(url: vid.url)
            }
        }
    }

    // MARK: - Video Picker Card

    private var videoPickerCard: some View {
        VStack(spacing: 12) {
            if let player {
                VideoPlayer(player: player)
                    .frame(height: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
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

    // MARK: - Frames Strip Card

    private var framesStripCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sample Frames — \(viewModel.frameThumbnails.count) of \(VideoFrameExtractor.frameCount) shown")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(Array(viewModel.frameThumbnails.enumerated()), id: \.offset) { i, img in
                        Image(uiImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 70, height: 70)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(alignment: .topLeading) {
                                Text("\(i + 1)")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 4))
                                    .padding(4)
                            }
                    }
                }
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Analyze Button

    private var analyzeButton: some View {
        Button {
            guard let url = videoURL else { return }
            Task { await viewModel.analyzeVideo(url: url) }
        } label: {
            HStack(spacing: 8) {
                if viewModel.isAnalyzing {
                    ProgressView().controlSize(.small).tint(.white)
                } else {
                    Image(systemName: "figure.strengthtraining.traditional")
                }
                Text(analyzeTitle)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!viewModel.canAnalyze)
        .animation(.default, value: viewModel.isAnalyzing)
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
                Text("counted by RepNet")
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

        case .extractingFrames:
            progressRow(label: "Extracting 64 frames…", icon: "film")
        case .runningModel:
            progressRow(label: "RepNet is counting reps…", icon: "brain")

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

    // MARK: - Helpers

    private var analyzeTitle: String {
        switch viewModel.analysisState {
        case .extractingFrames:     return "Extracting Frames…"
        case .runningModel:         return "Counting Reps…"
        default:                    return "Count Reps"
        }
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
