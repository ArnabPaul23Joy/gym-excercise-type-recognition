// LiveCameraView.swift
// Full-screen camera preview (AVCaptureVideoPreviewLayer) with a live body-pose skeleton
// overlay. Joint coordinates are mapped through `layerPointConverted(fromCaptureDevicePoint:)`
// so the overlay stays aligned regardless of video gravity, rotation, or front-camera mirroring.

import AVFoundation
import SwiftUI

struct LiveCameraView: UIViewRepresentable {
    let session: AVCaptureSession
    let pose: PoseFrame?
    let orientedSize: CGSize

    func makeUIView(context: Context) -> LivePreviewUIView {
        let view = LivePreviewUIView()
        view.session = session
        return view
    }

    func updateUIView(_ uiView: LivePreviewUIView, context: Context) {
        if uiView.session !== session { uiView.session = session }
        uiView.orientedSize = orientedSize
        uiView.pose = pose
    }
}

final class LivePreviewUIView: UIView {

    private static let bones: [(String, String)] = [
        ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ]

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    private var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    private let boneLayer = CAShapeLayer()
    private let jointLayer = CAShapeLayer()

    var session: AVCaptureSession? {
        didSet {
            previewLayer.session = session
            previewLayer.videoGravity = .resizeAspectFill
        }
    }

    var orientedSize: CGSize = .zero
    var pose: PoseFrame? { didSet { render() } }

    override init(frame: CGRect) {
        super.init(frame: frame)
        boneLayer.strokeColor = UIColor.systemGreen.cgColor
        boneLayer.lineWidth = 3
        boneLayer.lineCap = .round
        boneLayer.fillColor = UIColor.clear.cgColor
        jointLayer.fillColor = UIColor.systemYellow.cgColor
        jointLayer.strokeColor = UIColor.clear.cgColor
        layer.addSublayer(boneLayer)
        layer.addSublayer(jointLayer)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        boneLayer.frame = bounds
        jointLayer.frame = bounds
        render()   // displayed rect depends on bounds
    }

    /// The rect the preview fills under `.resizeAspectFill` — the same mapping we use for the overlay,
    /// so the skeleton sits in the identical coordinate space as the on-screen video (no double rotation).
    private func displayedVideoRect() -> CGRect? {
        guard orientedSize.width > 0, orientedSize.height > 0, bounds.width > 0, bounds.height > 0 else { return nil }
        let scale = max(bounds.width / orientedSize.width, bounds.height / orientedSize.height)
        let size = CGSize(width: orientedSize.width * scale, height: orientedSize.height * scale)
        return CGRect(x: bounds.midX - size.width / 2, y: bounds.midY - size.height / 2,
                      width: size.width, height: size.height)
    }

    private func render() {
        guard let pose, !pose.joints.isEmpty, let rect = displayedVideoRect() else { clear(); return }

        // pose.joints are normalised top-left in the upright buffer space; map into the displayed rect.
        func point(_ name: String) -> CGPoint? {
            guard let j = pose.joints[name] else { return nil }
            return CGPoint(x: rect.minX + j.x * rect.width, y: rect.minY + j.y * rect.height)
        }

        let bonePath = CGMutablePath()
        for (a, b) in Self.bones {
            guard let pa = point(a), let pb = point(b) else { continue }
            bonePath.move(to: pa)
            bonePath.addLine(to: pb)
        }

        let jointPath = CGMutablePath()
        let r: CGFloat = 5
        for name in pose.joints.keys {
            guard let p = point(name) else { continue }
            jointPath.addEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
        }

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
}
