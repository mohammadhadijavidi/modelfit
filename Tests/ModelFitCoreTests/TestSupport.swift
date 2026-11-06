import Foundation
import XCTest
@testable import ModelFitCore

/// A fresh temporary directory, removed when the test finishes.
func makeTempDirectory(_ test: XCTestCase) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("modelfit-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    test.addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url
}

/// The tiny GGUF used across tests. Hand-computed expectations:
///
/// | tensor              | dims     | type | elements | bytes               |
/// |---------------------|----------|------|----------|---------------------|
/// | token_embd.weight   | 64 × 100 | F16  | 6,400    | 6,400 × 2 = 12,800  |
/// | blk.0.attn_q.weight | 64 × 64  | Q8_0 | 4,096    | 128 blocks × 34 = 4,352 |
/// | blk.0.attn_k.weight | 64 × 32  | Q8_0 | 2,048    | 64 × 34 = 2,176     |
/// | blk.1.ffn_down      | 128 × 64 | Q4_K | 8,192    | 32 × 144 = 4,608    |
/// | output_norm.weight  | 64       | F32  | 64       | 256                 |
///
/// Totals: 20,800 parameters, 24,192 bytes.
enum Tiny {
    static let parameters: Int64 = 20_800
    static let bytes: Int64 = 24_192

    static func writer(version: UInt32 = 3, tokens: Int = 100) -> GGUFWriter {
        var w = GGUFWriter()
        w.version = version
        w.set("general.architecture", .string("llama"))
        w.set("general.name", .string("Tiny Test Model"))
        w.set("general.file_type", .uint32(7))
        w.set("llama.block_count", .uint32(2))
        w.set("llama.embedding_length", .uint32(64))
        w.set("llama.attention.head_count", .uint32(4))
        w.set("llama.attention.head_count_kv", .uint32(2))
        w.set("llama.context_length", .uint32(4096))
        w.set("llama.feed_forward_length", .uint32(128))
        w.set("llama.rope.freq_base", .float32(10_000))
        w.set("tokenizer.ggml.model", .string("llama"))
        w.set("tokenizer.ggml.tokens", .array(GGUFArray((0 ..< tokens).map { GGUFValue.string("tok\($0)") }, elementType: .string)))
        w.set("tokenizer.ggml.scores", .array(GGUFArray(Array(repeating: GGUFValue.float32(0), count: tokens), elementType: .float32)))
        w.set("tokenizer.ggml.bos_token_id", .uint32(1))
        w.set("tokenizer.chat_template", .string("{{ messages }}"))
        w.tensors = [
            .init("token_embd.weight", [64, 100], .f16),
            .init("blk.0.attn_q.weight", [64, 64], .q8_0),
            .init("blk.0.attn_k.weight", [64, 32], .q8_0),
            .init("blk.1.ffn_down.weight", [128, 64], .q4_K),
            .init("output_norm.weight", [64], .f32),
        ]
        return w
    }
}

/// Llama 3.1 8B's published dimensions.
func llama8BArchitecture() -> ArchitectureInfo {
    var a = ArchitectureInfo()
    a.architecture = "llama"
    a.layerCount = 32
    a.embeddingLength = 4096
    a.headCount = 32
    a.kvHeadCount = 8
    a.contextLength = 131_072
    a.vocabSize = 128_256
    a.source = .ggufMetadata
    return a
}
