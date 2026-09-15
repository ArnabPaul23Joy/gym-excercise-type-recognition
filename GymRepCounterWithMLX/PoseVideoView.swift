// PoseVideoView.swift
// AVPlayer video with a body-pose skeleton overlay drawn in sync with playback.
//
// The view's backing layer is an AVPlayerLayer; a CAShapeLayer for bones and one for
// joints are added on top. A periodic time observer looks up the nearest PoseFrame for
// the current playback time and maps its normalised keypoints into the layer's
// `videoRect` (the actual on-screen rectangle of the aspect-fit video).

import AVFoundation
import SwiftUI

struct PoseVideoView: UIViewRepresentable {
    let player: AVPlayer
    let poses: [PoseFrame]

    func makeUIView(context: Context) -> PoseVideoUIView {
        let view = PoseVideoUIView()
        view.poses = poses
        view.player = player
        return view
    }

    func updateUIView(_ uiView: PoseVideoUIView, context: Context) {
        uiView.poses = poses
        if uiView.player !== player { uiView.player = player }
    }

    static func dismantleUIView(_ uiView: PoseVideoUIView, coordinator: ()) {
        uiView.teardown()
    }
}

final class PoseVideoUIView: UIView {

    // MARK: Skeleton definition (joint names match ExerciseTypeAnalyzer's PoseFrame)

    private static let bones: [(String, String)] = [
        ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ]

    override class var layerClass: AnyClass { AVPlayerLayer.self }
    private var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }

    private let boneLayer = CAShapeLayer()
    private let jointLayer = CAShapeLayer()
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?

    var poses: [PoseFrame] = []

    var player: AVPlayer? {
        didSet {
            if let timeObserver { oldValue?.removeTimeObserver(timeObserver); self.timeObserver = nil }
            if let endObserver { NotificationCenter.default.removeObserver(endObserver); self.endObserver = nil }
            playerLayer.player = player
            playerLayer.videoGravity = .resizeAspect
            addTimeObserver()
            startLoopingPlayback()
        }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        boneLayer.strokeColor = UIColor.systemGreen.cgColor
        boneLayer.lineWidth = 3
        boneLayer.lineCap = .round
        boneLayer.fillColor = UIColor.clear.cgColor
        jointLayer.fillColor = UIColor.systemYellow.cgColor
        jointLayer.strokeColor = UIColor.clear.cgColor
        playerLayer.addSublayer(boneLayer)
        playerLayer.addSublayer(jointLayer)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(togglePlayback)))
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        boneLayer.frame = bounds
        jointLayer.frame = bounds
        render(at: player.map { CMTimeGetSeconds($0.currentTime()) } ?? 0)
    }

    // MARK: Playback-synced rendering

    private func addTimeObserver() {
        guard let player else { return }
        let interval = CMTime(seconds: 1.0 / 30.0, preferredTimescale: 600)
        timeObserver = player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [weak self] time in
            self?.render(at: CMTimeGetSeconds(time))
        }
    }

    // AVPlayerLayer has no transport controls, so we auto-play, loop, and toggle on tap.
    private func startLoopingPlayback() {
        guard let player, let item = player.currentItem else { return }
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main
        ) { [weak player] _ in
            player?.seek(to: .zero)
            player?.play()
        }
        player.play()
    }

    @objc private func togglePlayback() {
        guard let player else { return }
        if player.timeControlStatus == .playing { player.pause() } else { player.play() }
    }

    private func render(at time: Double) {
        let videoRect = playerLayer.videoRect
        guard videoRect.width > 0, videoRect.height > 0, let pose = nearestPose(to: time) else {
            clear()
            return
        }

        func point(_ name: String) -> CGPoint? {
            guard let p = pose.joints[name] else { return nil }
            return CGPoint(x: videoRect.minX + p.x * videoRect.width,
                           y: videoRect.minY + p.y * videoRect.height)
        }

        let bonePath = CGMutablePath()
        for (a, b) in Self.bones {
            guard let pa = point(a), let pb = point(b) else { continue }
            bonePath.move(to: pa)
            bonePath.addLine(to: pb)
        }

        let jointPath = CGMutablePath()
        let r: CGFloat = 4
        for name in pose.joints.keys {
            guard let p = point(name) else { continue }
            jointPath.addEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
        }

        // Overlay updates every frame; suppress implicit animations so it tracks crisply.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        boneLayer.path = bonePath
        jointLayer.path = jointPath
        CATransaction.commit()
    }

    private func clear() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        boneLayer.path = nil
        jointLayer.path = nil
        CATransaction.commit()
    }

    private func nearestPose(to time: Double) -> PoseFrame? {
        guard !poses.isEmpty else { return nil }
        var best = poses[0]
        var bestDelta = abs(poses[0].time - time)
        for pose in poses.dropFirst() {
            let delta = abs(pose.time - time)
            if delta < bestDelta { best = pose; bestDelta = delta }
        }
        // Ignore stale poses more than ~0.2 s from the current time (e.g. gaps in detection).
        return bestDelta <= 0.2 ? best : nil
    }

    func teardown() {
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        timeObserver = nil
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
        player?.pause()
        playerLayer.player = nil
    }

    deinit {
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
    }
}
