import Foundation

/// KV cache element precision, matching llama.cpp's `--cache-type-k/-v` options.
public enum KVCacheType: String, Sendable, CaseIterable, Identifiable {
    case f16, q8_0, q4_0

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .f16: "F16"
        case .q8_0: "Q8_0"
        case .q4_0: "Q4_0"
        }
    }

    /// Bytes per element as an exact fraction: Q8_0 stores 32 values in 34 bytes.
    public var bytesPerElement: (numerator: Int64, denominator: Int64) {
        switch self {
        case .f16: (2, 1)
        case .q8_0: (34, 32)
        case .q4_0: (18, 32)
        }
    }

    public var bytesPerElementValue: Double {
        Double(bytesPerElement.numerator) / Double(bytesPerElement.denominator)
    }

    public var bytesPerElementLabel: String {
        switch self {
        case .f16: "2 B"
        case .q8_0: "34/32 B"
        case .q4_0: "18/32 B"
        }
    }
}

/// Tunable assumptions behind every estimate. All are shown in "How we calculate".
public struct EstimationSettings: Sendable, Equatable {
    public var kvCacheType: KVCacheType
    /// Fixed allowance for the runtime itself, Metal context, and the OS's share.
    public var runtimeOverheadBytes: Int64
    /// Tokens processed per forward pass (llama.cpp's default `-ub` is 512).
    public var microBatch: Int64
    /// At or below this fraction of the GPU limit a fit counts as comfortable.
    public var comfortableFraction: Double

    public static let defaultRuntimeOverhead: Int64 = 512 << 20

    public init(kvCacheType: KVCacheType = .f16, runtimeOverheadBytes: Int64 = defaultRuntimeOverhead,
                microBatch: Int64 = 512, comfortableFraction: Double = 0.85) {
        self.kvCacheType = kvCacheType
        self.runtimeOverheadBytes = runtimeOverheadBytes
        self.microBatch = microBatch
        self.comfortableFraction = comfortableFraction
    }
}

public enum FitVerdict: Int, Sendable, Comparable, CaseIterable {
    case comfortable, tight, wontFit

    public static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }

    public var label: String {
        switch self {
        case .comfortable: "Fits comfortably"
        case .tight: "Tight"
        case .wontFit: "Won't fit"
        }
    }

    public var shortLabel: String {
        switch self {
        case .comfortable: "Fits"
        case .tight: "Tight"
        case .wontFit: "Won't fit"
        }
    }

    public var symbolName: String {
        switch self {
        case .comfortable: "checkmark.circle.fill"
        case .tight: "exclamationmark.triangle.fill"
        case .wontFit: "xmark.octagon.fill"
        }
    }
}

/// How the per-token KV size was derived, for display.
public struct KVCacheShape: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case standard(layers: Int, kvHeads: Int, keyDim: Int, valueDim: Int)
        case perLayer(layers: Int, totalKVHeads: Int, keyDim: Int, valueDim: Int)
        case latent(layers: Int, rank: Int, ropeDim: Int)
        case inferred(layers: Int, widthPerLayer: Int)
    }

    public var kind: Kind
    /// Cached elements per token across all layers (K and V together).
    public var elementsPerToken: Int64

    public init(kind: Kind) {
        self.kind = kind
        switch kind {
        case let .standard(l, h, k, v): elementsPerToken = Int64(l) * Int64(h) * Int64(k + v)
        case let .perLayer(_, total, k, v): elementsPerToken = Int64(total) * Int64(k + v)
        case let .latent(l, r, d): elementsPerToken = Int64(l) * Int64(r + d)
        case let .inferred(l, w): elementsPerToken = Int64(l) * Int64(w)
        }
    }
}

extension ArchitectureInfo {
    /// nil when the architecture doesn't use a growing KV cache or numbers are missing.
    public var kvCacheShape: KVCacheShape? {
        guard !isRecurrent, let layers = layerCount, layers > 0 else { return nil }
        if let rank = kvLoraRank {
            return KVCacheShape(kind: .latent(layers: layers, rank: rank, ropeDim: ropeDimension ?? 0))
        }
        if let k = headDimension, let v = valueDimension {
            if let perLayer = kvHeadsPerLayer, perLayer.count == layers {
                return KVCacheShape(kind: .perLayer(layers: layers, totalKVHeads: perLayer.reduce(0, +), keyDim: k, valueDim: v))
            }
            if let heads = effectiveKVHeads {
                return KVCacheShape(kind: .standard(layers: layers, kvHeads: heads, keyDim: k, valueDim: v))
            }
        }
        if let w = inferredKVWidthPerLayer {
            return KVCacheShape(kind: .inferred(layers: layers, widthPerLayer: w))
        }
        return nil
    }
}

public struct MemoryEstimate: Sendable, Equatable, Identifiable {
    public var contextLength: Int
    public var weights: Int64
    public var kvCache: Int64
    public var compute: Int64
    public var runtime: Int64
    public var kvModelled: Bool
    public var verdict: FitVerdict
    public var id: Int { contextLength }

    /// Compute buffers and fixed runtime allowance together.
    public var overhead: Int64 { compute + runtime }
    public var total: Int64 { weights + kvCache + compute + runtime }
}

public enum MemoryEstimator {
    public static let standardContexts = [2048, 8192, 32768, 131072]
    /// Used when vocab or embedding size is unknown.
    public static let fallbackCompute: Int64 = 256 << 20

    /// 2K / 8K / 32K / 128K, capped at the model's own maximum (which is added if it
    /// falls between steps, so a 4K model shows 2K and 4K).
    public static func contextLengths(maxContext: Int?) -> [Int] {
        guard let maxContext, maxContext > 0 else { return standardContexts }
        var list = standardContexts.filter { $0 <= maxContext }
        if maxContext < standardContexts.last!, !list.contains(maxContext) { list.append(maxContext) }
        return list
    }

    public static func kvCacheBytes(shape: KVCacheShape, context: Int, type: KVCacheType) -> Int64 {
        let (num, den) = type.bytesPerElement
        return shape.elementsPerToken * Int64(context) * num / den
    }

    /// Scratch for one micro-batch: logits (vocab) plus a few embedding-wide activations,
    /// in F32. Assumes flash attention, so no context-sized attention matrix.
    public static func computeBytes(architecture a: ArchitectureInfo, settings: EstimationSettings) -> Int64 {
        guard let vocab = a.vocabSize, let embd = a.embeddingLength else { return fallbackCompute }
        return settings.microBatch * (Int64(vocab) + 4 * Int64(embd)) * 4
    }

    public static func verdict(total: Int64, profile: MacProfile, settings: EstimationSettings) -> FitVerdict {
        let limit = Double(profile.gpuWorkingSet)
        let t = Double(total)
        if t <= limit * settings.comfortableFraction { return .comfortable }
        if t <= limit { return .tight }
        return .wontFit
    }

    public static func estimate(_ analysis: ModelAnalysis, context: Int, settings: EstimationSettings, profile: MacProfile) -> MemoryEstimate {
        let shape = analysis.architecture.kvCacheShape
        let kv = shape.map { kvCacheBytes(shape: $0, context: context, type: settings.kvCacheType) } ?? 0
        let compute = computeBytes(architecture: analysis.architecture, settings: settings)
        let total = analysis.weightBytes + kv + compute + settings.runtimeOverheadBytes
        return MemoryEstimate(
            contextLength: context, weights: analysis.weightBytes, kvCache: kv, compute: compute,
            runtime: settings.runtimeOverheadBytes, kvModelled: shape != nil,
            verdict: verdict(total: total, profile: profile, settings: settings)
        )
    }

    public static func ladder(_ analysis: ModelAnalysis, settings: EstimationSettings, profile: MacProfile) -> [MemoryEstimate] {
        contextLengths(maxContext: analysis.maxContext).map {
            estimate(analysis, context: $0, settings: settings, profile: profile)
        }
    }
}
