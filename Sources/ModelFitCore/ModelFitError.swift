import Foundation

/// Every way reading a model can fail. Each case carries enough context to tell the
/// person what went wrong and what to do next, without ever crashing on hostile input.
public enum ModelFitError: Error, Equatable, Sendable {
    case unreadable(file: String, reason: String)
    case notAModel(file: String)
    case truncated(file: String, context: String, offset: UInt64)
    case corrupt(file: String, reason: String)
    case unsupportedVersion(file: String, version: UInt32)
    case bigEndianGGUF(file: String)
    case headerTooLarge(file: String, size: UInt64)
    case emptyFolder(folder: String)

    /// Short, human headline for an error state.
    public var title: String {
        switch self {
        case .unreadable: "Couldn't open this file"
        case .notAModel: "Not a model file ModelFit understands"
        case .truncated: "This file ends too early"
        case .corrupt: "This file's header is damaged"
        case .unsupportedVersion: "Unsupported GGUF version"
        case .bigEndianGGUF: "Big-endian GGUF isn't supported"
        case .headerTooLarge: "The header is implausibly large"
        case .emptyFolder: "No model weights in this folder"
        }
    }

    /// One or two sentences with the specifics.
    public var detail: String {
        switch self {
        case let .unreadable(file, reason):
            "\(file) could not be read: \(reason)"
        case let .notAModel(file):
            "\(file) doesn't start with a GGUF magic number or a safetensors header."
        case let .truncated(file, context, offset):
            "\(file) stopped at byte \(Format.integer(Int64(offset))) while reading \(context). The download may be incomplete."
        case let .corrupt(file, reason):
            "\(file): \(reason)."
        case let .unsupportedVersion(file, version):
            "\(file) declares GGUF version \(version). ModelFit reads versions 1 to 3."
        case let .bigEndianGGUF(file):
            "\(file) was written for big-endian machines. Apple Silicon and Intel Macs need the little-endian build of the model."
        case let .headerTooLarge(file, size):
            "\(file) declares a \(Format.bytes(Int64(clamping: size))) header, which no real model needs."
        case let .emptyFolder(folder):
            "\(folder) has no .safetensors or .gguf files at its top level."
        }
    }

    public var recoverySuggestion: String {
        switch self {
        case .unreadable: "Check that the file still exists and that you have permission to read it."
        case .notAModel: "Open a .gguf file, a .safetensors file, or a Hugging Face or MLX model folder."
        case .truncated: "Re-download the model and compare its size with the one listed on the model page."
        case .corrupt, .headerTooLarge: "Re-download the model, or try a different quantization of it."
        case .unsupportedVersion: "Update to a newer copy of the model, converted with a recent llama.cpp."
        case .bigEndianGGUF: "Download the standard (little-endian) GGUF of this model."
        case .emptyFolder: "Pick the folder that contains config.json and the .safetensors shards."
        }
    }
}

extension ModelFitError: LocalizedError {
    public var errorDescription: String? { "\(title). \(detail)" }
}
