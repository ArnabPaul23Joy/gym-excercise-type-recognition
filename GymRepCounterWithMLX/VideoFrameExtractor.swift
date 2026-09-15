import AVFoundation
import CoreGraphics
import UIKit

enum VideoFrameExtractor {

    nonisolated static let frameCount = 64
    nonisolated static let pixelSize = 224

    // MARK: - Frame Extraction

    static func extractFrames(from url: URL) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let duration = CMTimeGetSeconds(try await asset.load(.duration))
        guard duration > 0 else { throw ExtractionError.emptyVideo }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter  = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.maximumSize = CGSize(width: pixelSize, height: pixelSize)

        let times: [NSValue] = (0..<frameCount).map { i in
            let t = duration * Double(i) / Double(frameCount - 1)
            return NSValue(time: CMTime(seconds: t, preferredTimescale: 600))
        }

        return try await withCheckedThrowingContinuation { continuation in
            var received = 0
            var frames = [CGImage?](repeating: nil, count: frameCount)
            var firstError: Error?
            let lock = NSLock()

            // Callbacks arrive sequentially; `received` is the slot index.
            generator.generateCGImagesAsynchronously(forTimes: times) { _, cgImage, _, result, error in
                lock.lock()
                let idx = received
                received += 1
                if let error, firstError == nil { firstError = error }
                if let cgImage, result == .succeeded { frames[idx] = cgImage }
                let done = received == frameCount
                lock.unlock()

                guard done else { return }
                let compact = frames.compactMap { $0 }
                if compact.isEmpty, let err = firstError {
                    continuation.resume(throwing: err)
                } else {
                    continuation.resume(returning: compact)
                }
            }
        }
    }

    // MARK: - Input Tensor

    // Builds RepNet's input tensor as a flat Float array in [1, 3, 64, 224, 224]
    // layout (NCHW-T: channels, then frames, then H, W), normalised to [0, 1].
    nonisolated static func framesToInputTensor(_ frames: [CGImage]) throws -> [Float] {
        let channels = 3
        let t = frameCount
        let s = pixelSize
        let available = frames.count
        guard available > 0 else { throw ExtractionError.emptyVideo }

        var buffer = [Float](repeating: 0, count: channels * t * s * s)
        let bytesPerRow = s * 4
        let planeStride = t * s * s   // one full channel plane

        for f in 0..<t {
            let cgImage = frames[min(f, available - 1)] // pad with last frame if short

            guard let ctx = CGContext(
                data: nil,
                width: s, height: s,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { throw ExtractionError.imageConversionFailed }
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: s, height: s))

            guard let pixelData = ctx.data else { throw ExtractionError.imageConversionFailed }
            let raw = pixelData.assumingMemoryBound(to: UInt8.self)

            let frameBase = f * s * s
            for y in 0..<s {
                for x in 0..<s {
                    let src = y * bytesPerRow + x * 4 // RGBX
                    let spatial = frameBase + y * s + x
                    buffer[0 * planeStride + spatial] = Float(raw[src])     / 255.0 // R
                    buffer[1 * planeStride + spatial] = Float(raw[src + 1]) / 255.0 // G
                    buffer[2 * planeStride + spatial] = Float(raw[src + 2]) / 255.0 // B
                }
            }
        }
        return buffer
    }

    // MARK: - Thumbnails

    // Returns up to maxCount evenly-spaced UIImages for the preview strip.
    static func thumbnails(from frames: [CGImage], maxCount: Int = 8) -> [UIImage] {
        let step = max(1, frames.count / maxCount)
        return stride(from: 0, to: frames.count, by: step)
            .prefix(maxCount)
            .map { UIImage(cgImage: frames[$0]) }
    }

    // MARK: - Errors

    enum ExtractionError: LocalizedError {
        case emptyVideo
        case imageConversionFailed

        var errorDescription: String? {
            switch self {
            case .emptyVideo:            return "The video has no duration."
            case .imageConversionFailed: return "Failed to convert video frames to pixel data."
            }
        }
    }
}
