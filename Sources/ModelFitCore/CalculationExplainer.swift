import Foundation

/// "How we calculate", with this model's real numbers plugged into each formula.
public struct CalculationStep: Sendable, Equatable, Identifiable {
    public var title: String
    public var formula: String
    public var explanation: String
    public var id: String { title }
}

public enum CalculationExplainer {
    public static func steps(for a: ModelAnalysis, at e: MemoryEstimate, settings s: EstimationSettings, profile p: MacProfile) -> [CalculationStep] {
        var steps: [CalculationStep] = []
        let ctx = Format.integer(e.contextLength)

        steps.append(.init(
            title: "Weights",
            formula: "Σ tensor bytes = \(Format.bytes(e.weights))",
            explanation: "The stored size of all \(Format.integer(a.tensorCount)) tensors, read from the header. Quantized weights stay quantized in memory, so this is what loads."
        ))

        let bpe = s.kvCacheType.bytesPerElementLabel
        let kvFormula: String
        var kvNote = "K and V are cached for every token in context. \(s.kvCacheType.label) uses \(bpe) per element; quantized V needs flash attention in llama.cpp."
        switch a.architecture.kvCacheShape?.kind {
        case let .standard(l, h, k, v)?:
            let dims = k == v ? "2 × \(l) layers × \(h) KV heads × \(k) head dim" : "\(l) layers × \(h) KV heads × (\(k) + \(v)) dims"
            kvFormula = "\(dims) × \(ctx) tokens × \(bpe) = \(Format.bytes(e.kvCache))"
        case let .perLayer(l, total, k, v)?:
            kvFormula = "\(total) KV heads summed over \(l) layers × (\(k) + \(v)) dims × \(ctx) tokens × \(bpe) = \(Format.bytes(e.kvCache))"
        case let .latent(l, r, d)?:
            kvFormula = "\(l) layers × (\(r) latent + \(d) rope) × \(ctx) tokens × \(bpe) = \(Format.bytes(e.kvCache))"
            kvNote += " Multi-head latent attention caches one compressed vector per token instead of per-head K and V."
        case let .inferred(l, w)?:
            kvFormula = "\(l) layers × \(w) (K + V projection width) × \(ctx) tokens × \(bpe) = \(Format.bytes(e.kvCache))"
            kvNote += " Layer count and projection widths were inferred from tensor names."
        case nil:
            kvFormula = "not modelled"
            kvNote = a.architecture.isRecurrent
                ? "Recurrent models keep a fixed-size state instead of a cache that grows with context."
                : "The header doesn't give layer and head counts, so only weights and overhead are counted."
        }
        if let w = a.architecture.slidingWindow, w > 0, a.architecture.kvCacheShape != nil {
            kvNote += " Layers using a \(Format.integer(w))-token sliding window may cache less, so this is an upper bound."
        }
        steps.append(.init(title: "KV cache", formula: kvFormula, explanation: kvNote))

        let computeFormula: String
        if let v = a.architecture.vocabSize, let d = a.architecture.embeddingLength {
            computeFormula = "\(s.microBatch) × (\(Format.integer(v)) vocab + 4 × \(Format.integer(d))) × 4 B = \(Format.bytes(e.compute))"
        } else {
            computeFormula = "fallback allowance = \(Format.bytes(e.compute))"
        }
        steps.append(.init(
            title: "Compute buffers",
            formula: computeFormula,
            explanation: "Scratch space for one \(s.microBatch)-token micro-batch: F32 logits plus a few activations. Assumes flash attention; without it, attention scores add more at long context."
        ))

        steps.append(.init(
            title: "Runtime",
            formula: "fixed allowance = \(Format.bytes(e.runtime))",
            explanation: "The inference runtime, Metal command buffers and the model's own process. Adjustable in Settings."
        ))

        let limitExplanation: String
        switch p.gpuLimitSource {
        case .metal:
            limitExplanation = "Metal's recommendedMaxWorkingSetSize on this Mac. macOS caps how much unified memory the GPU may wire; `sudo sysctl iogpu.wired_limit_mb=…` raises it until reboot, at your own risk."
        case .ruleOfThumb:
            limitExplanation = "Assumed: about 2/3 of RAM up to 36 GB and 3/4 above, close to macOS's default GPU cap on Apple Silicon."
        case .custom:
            limitExplanation = "Set by you in Settings, for Macs where the GPU wired limit was raised."
        case .cpuOnly:
            limitExplanation = "Intel Macs run these models on the CPU from system memory; 75% of RAM is assumed usable."
        }
        steps.append(.init(
            title: "GPU limit",
            formula: "\(Format.bytes(p.gpuWorkingSet)) of \(Format.bytes(p.physicalMemory)) (\(p.gpuLimitSource.label))",
            explanation: limitExplanation
        ))

        steps.append(.init(
            title: "Verdict",
            formula: "total ≤ \(Format.percent(s.comfortableFraction)) of limit: fits comfortably; ≤ 100%: tight; above: won't fit",
            explanation: "Above the GPU limit a runtime must offload layers to the CPU, which is several times slower. All figures are estimates: real use varies by runtime (llama.cpp, Ollama, LM Studio, MLX), batch size and settings. 1 GB = 1024³ bytes."
        ))
        return steps
    }
}
