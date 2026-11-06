import Foundation
import XCTest
@testable import ModelFitCore

final class MemoryTests: XCTestCase {
    /// Llama 3.1 8B: 2 × 32 layers × 8 KV heads × 128 dims = 65,536 elements per token.
    func testKVCacheMatchesHandComputedValues() throws {
        let shape = try XCTUnwrap(llama8BArchitecture().kvCacheShape)
        XCTAssertEqual(shape.elementsPerToken, 65_536)
        // 65,536 × 8,192 × 2 B = 1 GiB exactly.
        XCTAssertEqual(MemoryEstimator.kvCacheBytes(shape: shape, context: 8192, type: .f16), 1_073_741_824)
        // Q8_0: 34 bytes per 32 values → 1 GiB × 34/64.
        XCTAssertEqual(MemoryEstimator.kvCacheBytes(shape: shape, context: 8192, type: .q8_0), 570_425_344)
        // Q4_0: 18 bytes per 32 values → 1 GiB × 18/64.
        XCTAssertEqual(MemoryEstimator.kvCacheBytes(shape: shape, context: 8192, type: .q4_0), 301_989_888)
        // 128K at F16: 16 GiB.
        XCTAssertEqual(MemoryEstimator.kvCacheBytes(shape: shape, context: 131_072, type: .f16), 17_179_869_184)
    }

    func testComputeBuffer() {
        // 512 × (128,256 + 4 × 4,096) × 4 B = 296,222,720 B.
        XCTAssertEqual(MemoryEstimator.computeBytes(architecture: llama8BArchitecture(), settings: .init()), 296_222_720)
        XCTAssertEqual(MemoryEstimator.computeBytes(architecture: ArchitectureInfo(), settings: .init()), MemoryEstimator.fallbackCompute)
    }

    func testLatentAndInferredShapes() {
        var mla = ArchitectureInfo()
        mla.layerCount = 61
        mla.kvLoraRank = 512
        mla.ropeDimension = 64
        mla.headCount = 128
        mla.embeddingLength = 7168
        XCTAssertEqual(mla.kvCacheShape?.elementsPerToken, 61 * 576)

        var rnn = ArchitectureInfo()
        rnn.architecture = "mamba"
        rnn.layerCount = 64
        XCTAssertNil(rnn.kvCacheShape)
        XCTAssertTrue(rnn.isRecurrent)
    }

    func testContextLadder() {
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 131_072), [2048, 8192, 32768, 131_072])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 1_048_576), [2048, 8192, 32768, 131_072])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 32768), [2048, 8192, 32768])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 4096), [2048, 4096])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 40960), [2048, 8192, 32768, 40960])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: 1000), [1000])
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: nil), [2048, 8192, 32768, 131_072])
    }

    func testRuleOfThumbGPULimit() {
        XCTAssertEqual(MacProfile.ruleOfThumbLimit(physicalMemory: 16 << 30), 11_453_246_122)
        XCTAssertEqual(MacProfile.ruleOfThumbLimit(physicalMemory: 64 << 30), 48 << 30)
        XCTAssertEqual(MacProfile.sample16GB.gpuWorkingSet, 11_453_246_122)
    }

    /// Llama 3.1 8B Q4_K_M on a 16 GB Mac (GPU limit 11,453,246,122 B; comfortable ≤ 85%):
    /// fixed part = 4,912,898,304 weights + 296,222,720 compute + 536,870,912 runtime = 5,745,991,936.
    func testVerdictLadderForLlama8BOn16GB() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("llama8b.gguf")
        try SampleModels.llama31_8B(tokenizerArrays: false).write(to: url)
        var a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        a.architecture.vocabSize = 128_256

        let ladder = MemoryEstimator.ladder(a, settings: .init(), profile: .sample16GB)
        XCTAssertEqual(ladder.map(\.contextLength), [2048, 8192, 32768, 131_072])
        let fixed: Int64 = 5_745_991_936
        let kvBytes: [Int64] = [268_435_456, 1_073_741_824, 4_294_967_296, 17_179_869_184]
        let expectedTotals: [Int64] = kvBytes.map { fixed + $0 }
        XCTAssertEqual(ladder.map(\.total), expectedTotals)
        XCTAssertEqual(ladder.map(\.verdict), [.comfortable, .comfortable, .tight, .wontFit])

        let summary = FitSummary(estimates: ladder, profile: .sample16GB)
        XCTAssertEqual(summary.headline, "Runs comfortably up to 8K context")
        XCTAssertEqual(summary.detail, "Tight at 32K. Won't fit at 128K.")
        XCTAssertEqual(summary.largestFittingContext, 32768)

        // Quantizing the KV cache to Q4_0 (18/32 B per element) makes 32K comfortable
        // (5,745,991,936 + 1,207,959,552) and 128K merely tight (+ 4,831,838,208).
        let q4 = MemoryEstimator.ladder(a, settings: .init(kvCacheType: .q4_0), profile: .sample16GB)
        XCTAssertEqual(q4.map(\.verdict), [.comfortable, .comfortable, .comfortable, .tight])
    }

    func testSummaryWording() {
        func e(_ ctx: Int, _ v: FitVerdict) -> MemoryEstimate {
            MemoryEstimate(contextLength: ctx, weights: 1, kvCache: 1, compute: 1, runtime: 1, kvModelled: true, verdict: v)
        }
        let p = MacProfile.sample16GB
        XCTAssertEqual(FitSummary(estimates: [e(2048, .comfortable), e(4096, .comfortable)], profile: p).headline,
                       "Fits comfortably at every context length")
        let tight = FitSummary(estimates: [e(2048, .tight), e(8192, .wontFit), e(32768, .wontFit)], profile: p)
        XCTAssertEqual(tight.headline, "Fits, but only just, up to 2K")
        XCTAssertEqual(tight.detail, "Quit memory-hungry apps first. Won't fit at 8K and 32K.")
        XCTAssertEqual(FitSummary(estimates: [e(2048, .wontFit)], profile: p).headline, "Too big for this Mac")
    }

    func testCustomGPULimitPolicy() {
        let p = MacProfile.current(policy: .fraction(0.5))
        XCTAssertLessThanOrEqual(abs(p.gpuWorkingSet - p.physicalMemory / 2), 1)
        if case .custom = p.gpuLimitSource {} else { XCTFail("expected custom source") }
        XCTAssertGreaterThan(MacProfile.current().gpuWorkingSet, 0)
    }

    func testExplainerPlugsInRealNumbers() throws {
        var a = ModelAnalysis(
            url: URL(fileURLWithPath: "/tmp/x.gguf"), displayName: "x", format: .gguf, formatDetail: "GGUF v3",
            fileSize: 0, weightBytes: 4_912_898_304, parameterCount: 8_030_261_312, tensorCount: 292,
            quantizationLabel: "Q4_K_M", typeMix: [], architecture: llama8BArchitecture(), tokenizer: nil,
            facts: [], fileCount: 1, headerBytesRead: 0, warnings: [], isSample: false
        )
        a.architecture.keyLength = 128
        let est = MemoryEstimator.estimate(a, context: 8192, settings: .init(), profile: .sample16GB)
        let steps = CalculationExplainer.steps(for: a, at: est, settings: .init(), profile: .sample16GB)
        let kv = try XCTUnwrap(steps.first { $0.title == "KV cache" })
        XCTAssertEqual(kv.formula, "2 × 32 layers × 8 KV heads × 128 head dim × 8,192 tokens × 2 B = 1 GB")
        XCTAssertEqual(steps.first { $0.title == "Weights" }?.formula, "Σ tensor bytes = 4.58 GB")
    }

    func testFormatting() {
        XCTAssertEqual(Format.bytes(4_912_898_304), "4.58 GB")
        XCTAssertEqual(Format.bytes(16 << 30), "16 GB")
        XCTAssertEqual(Format.bytes(11_453_246_122), "10.7 GB")
        XCTAssertEqual(Format.bytes(512 << 20), "512 MB")
        XCTAssertEqual(Format.bytes(1536), "2 KB")
        XCTAssertEqual(Format.parameters(8_030_261_312), "8.03B")
        XCTAssertEqual(Format.parameters(70_553_706_560), "70.6B")
        XCTAssertEqual(Format.parameters(494_032_768), "494M")
        XCTAssertEqual(Format.parameters(405_000_000_000), "405B")
        XCTAssertEqual(Format.context(131_072), "128K")
        XCTAssertEqual(Format.context(4096), "4K")
        XCTAssertEqual(Format.context(1_048_576), "1M")
        XCTAssertEqual(Format.context(1000), "1,000")
        XCTAssertEqual(Format.integer(1_234_567), "1,234,567")
        XCTAssertEqual(Format.integer(-42), "-42")
        XCTAssertEqual(Format.percent(0.85), "85%")
    }
}
