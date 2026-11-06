import Foundation

public enum ModelFormat: String, Sendable, Equatable, CaseIterable {
    case gguf, safetensors, mlxFolder, huggingFaceFolder

    public var displayName: String {
        switch self {
        case .gguf: "GGUF"
        case .safetensors: "safetensors"
        case .mlxFolder: "MLX"
        case .huggingFaceFolder: "Hugging Face"
        }
    }

    public var symbolName: String {
        switch self {
        case .gguf: "shippingbox.fill"
        case .safetensors: "square.stack.3d.up.fill"
        case .mlxFolder: "cpu.fill"
        case .huggingFaceFolder: "folder.fill"
        }
    }
}

/// Where the architecture numbers came from, shown next to them in the UI.
public enum ArchitectureSource: String, Sendable, Equatable {
    case ggufMetadata, configJSON, inferredFromTensors, unavailable

    public var label: String {
        switch self {
        case .ggufMetadata: "From GGUF metadata"
        case .configJSON: "From config.json"
        case .inferredFromTensors: "Inferred from tensor names and shapes"
        case .unavailable: "Not available in this file"
        }
    }
}

public struct ArchitectureInfo: Sendable, Equatable {
    public var architecture: String?
    public var layerCount: Int?
    public var embeddingLength: Int?
    public var headCount: Int?
    public var kvHeadCount: Int?
    /// Per-layer KV head counts when they differ between layers.
    public var kvHeadsPerLayer: [Int]?
    public var keyLength: Int?
    public var valueLength: Int?
    public var feedForwardLength: Int?
    public var contextLength: Int?
    public var vocabSize: Int?
    public var ropeFreqBase: Double?
    public var expertCount: Int?
    public var expertUsedCount: Int?
    public var slidingWindow: Int?
    /// Multi-head latent attention (DeepSeek-style) compressed KV rank.
    public var kvLoraRank: Int?
    public var ropeDimension: Int?
    /// K plus V output width per layer, when inferred from projection shapes.
    public var inferredKVWidthPerLayer: Int?
    public var source: ArchitectureSource = .unavailable

    public init() {}

    public var headDimension: Int? {
        if let k = keyLength { return k }
        if let e = embeddingLength, let h = headCount, h > 0 { return e / h }
        return nil
    }

    public var valueDimension: Int? { valueLength ?? headDimension }

    public var effectiveKVHeads: Int? { kvHeadCount ?? kvHeadsPerLayer?.max() ?? headCount }

    static let recurrent: Set<String> = ["mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"]
    static let hybrid: Set<String> = ["jamba", "falcon-h1", "granitehybrid", "lfm2", "qwen3next", "plamo2", "nemotron_h"]

    public var isRecurrent: Bool { architecture.map { Self.recurrent.contains($0.lowercased()) } ?? false }
    public var isHybrid: Bool { architecture.map { Self.hybrid.contains($0.lowercased()) } ?? false }
}

public struct TokenizerInfo: Sendable, Equatable {
    public var model: String?
    public var preTokenizer: String?
    public var tokenCount: Int?
    public var mergeCount: Int?
    public var bosTokenID: Int?
    public var eosTokenID: Int?
    public var hasChatTemplate: Bool = false
    public init() {}
}

/// Bytes and parameters grouped by storage type ("Q4_K", "BF16", "4-bit").
public struct TensorTypeShare: Sendable, Equatable, Identifiable {
    public var name: String
    public var tensorCount: Int
    public var parameters: Int64
    public var bytes: Int64
    public var id: String { name }
}

public struct ModelFact: Sendable, Equatable, Identifiable {
    public var label: String
    public var value: String
    public var id: String { label }
    public init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }
}

/// Everything ModelFit learned from a model's headers, independent of its format.
public struct ModelAnalysis: Sendable, Equatable, Identifiable {
    public var id = UUID()
    public var url: URL
    public var displayName: String
    public var format: ModelFormat
    /// "GGUF v3", "3 safetensors shards".
    public var formatDetail: String
    /// Bytes on disk across every file of the model.
    public var fileSize: Int64
    /// Bytes of tensor data: what has to be resident in memory.
    public var weightBytes: Int64
    public var parameterCount: Int64
    public var tensorCount: Int
    public var quantizationLabel: String
    public var typeMix: [TensorTypeShare]
    public var architecture: ArchitectureInfo
    public var tokenizer: TokenizerInfo?
    public var facts: [ModelFact]
    public var fileCount: Int
    public var headerBytesRead: Int64
    public var warnings: [String]
    public var isSample: Bool

    public var bitsPerWeight: Double? {
        parameterCount > 0 ? Double(weightBytes) * 8 / Double(parameterCount) : nil
    }

    /// Largest context the model declares, if any.
    public var maxContext: Int? { architecture.contextLength }
}

extension Array where Element == TensorTypeShare {
    /// Merges shares with the same name and sorts by bytes, largest first.
    static func combined(_ shares: [TensorTypeShare]) -> [TensorTypeShare] {
        var map: [String: TensorTypeShare] = [:]
        for s in shares {
            if var existing = map[s.name] {
                existing.tensorCount += s.tensorCount
                existing.parameters += s.parameters
                existing.bytes += s.bytes
                map[s.name] = existing
            } else {
                map[s.name] = s
            }
        }
        return map.values.sorted { $0.bytes != $1.bytes ? $0.bytes > $1.bytes : $0.name < $1.name }
    }
}
