import Foundation
import XCTest
@testable import ModelFitCore

final class SafetensorsTests: XCTestCase {
    /// a: F16 [4, 8] = 32 elements, 64 bytes; b: BF16 [8] = 16 bytes; c: I8 [3, 3] = 9 bytes.
    func testSingleFile() throws {
        let dir = try makeTempDirectory(self)
        var w = SafetensorsWriter()
        w.tensors = [.init("a", "F16", [4, 8]), .init("b", "BF16", [8]), .init("c", "I8", [3, 3])]
        w.metadata = ["format": "pt"]
        let url = dir.appendingPathComponent("small.safetensors")
        try w.write(to: url)

        let file = try SafetensorsParser.parse(url: url)
        XCTAssertEqual(file.tensors.map(\.name), ["a", "b", "c"])
        XCTAssertEqual(file.dataSize, 89)
        XCTAssertEqual(file.metadata["format"], "pt")
        XCTAssertTrue(file.warnings.isEmpty, "\(file.warnings)")

        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        XCTAssertEqual(a.format, .safetensors)
        XCTAssertEqual(a.parameterCount, 49)
        XCTAssertEqual(a.weightBytes, 89)
        XCTAssertEqual(a.typeMix.map(\.name), ["F16", "BF16", "I8"])
        XCTAssertEqual(a.quantizationLabel, "F16")
        XCTAssertEqual(a.architecture.source, .unavailable)
        XCTAssertNil(a.architecture.kvCacheShape)
    }

    func testSniffsWithoutExtension() throws {
        let dir = try makeTempDirectory(self)
        var w = SafetensorsWriter()
        w.tensors = [.init("x", "F32", [2])]
        let url = dir.appendingPathComponent("weights.bin")
        try w.write(to: url)
        XCTAssertEqual(try ModelLoader.sniff(url), .safetensors)
        XCTAssertEqual(try ModelLoader.analyze(url: url).first?.weightBytes, 8)
    }

    func testTruncatedDataWarns() throws {
        let dir = try makeTempDirectory(self)
        var w = SafetensorsWriter()
        w.tensors = [.init("x", "F32", [1000])]
        let url = dir.appendingPathComponent("cut.safetensors")
        try w.write(to: url, fullSize: false)
        let file = try SafetensorsParser.parse(url: url)
        XCTAssertTrue(file.warnings.contains { $0.contains("truncated") })
    }

    /// Two shards plus index and config. Parameters:
    /// embed 100×64 + 2 layers × (q 64×64 + k 32×64 + v 32×64) + lm_head 100×64
    /// = 6,400 + 2 × 8,192 + 6,400 = 29,184; BF16 so 58,368 bytes.
    func testShardedHuggingFaceFolder() throws {
        let dir = try makeTempDirectory(self)
        let folder = dir.appendingPathComponent("tiny-hf", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        func layer(_ i: Int) -> [SafetensorsWriter.Tensor] {
            [
                .init("model.layers.\(i).self_attn.q_proj.weight", "BF16", [64, 64]),
                .init("model.layers.\(i).self_attn.k_proj.weight", "BF16", [32, 64]),
                .init("model.layers.\(i).self_attn.v_proj.weight", "BF16", [32, 64]),
            ]
        }
        var s1 = SafetensorsWriter()
        s1.tensors = [.init("model.embed_tokens.weight", "BF16", [100, 64])] + layer(0)
        var s2 = SafetensorsWriter()
        s2.tensors = layer(1) + [.init("lm_head.weight", "BF16", [100, 64])]
        try s1.write(to: folder.appendingPathComponent("model-00001-of-00002.safetensors"))
        try s2.write(to: folder.appendingPathComponent("model-00002-of-00002.safetensors"))
        try SampleModels.writeIndex([("model-00001-of-00002.safetensors", s1), ("model-00002-of-00002.safetensors", s2)], to: folder)
        try SampleModels.writeJSON([
            "model_type": "llama", "num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 2048, "vocab_size": 100, "torch_dtype": "bfloat16",
        ], to: folder.appendingPathComponent("config.json"))
        try SampleModels.writeJSON(["tokenizer_class": "LlamaTokenizer", "chat_template": "x"], to: folder.appendingPathComponent("tokenizer_config.json"))

        let a = try XCTUnwrap(ModelLoader.analyze(url: folder).first)
        XCTAssertEqual(a.format, .huggingFaceFolder)
        XCTAssertEqual(a.fileCount, 2)
        XCTAssertEqual(a.parameterCount, 29_184)
        XCTAssertEqual(a.weightBytes, 58_368)
        XCTAssertEqual(a.quantizationLabel, "BF16")
        XCTAssertEqual(a.architecture.source, .configJSON)
        XCTAssertEqual(a.architecture.layerCount, 2)
        XCTAssertEqual(a.architecture.headDimension, 16)
        XCTAssertEqual(a.architecture.kvCacheShape?.elementsPerToken, 2 * 2 * 32)
        XCTAssertEqual(a.tokenizer?.hasChatTemplate, true)
        XCTAssertEqual(MemoryEstimator.contextLengths(maxContext: a.maxContext), [2048])

        // Dropping one shard analyzes the whole folder; so does dropping config.json.
        let viaShard = try XCTUnwrap(ModelLoader.analyze(url: folder.appendingPathComponent("model-00002-of-00002.safetensors")).first)
        XCTAssertEqual(viaShard.parameterCount, 29_184)
        let viaConfig = try XCTUnwrap(ModelLoader.analyze(url: folder.appendingPathComponent("config.json")).first)
        XCTAssertEqual(viaConfig.weightBytes, 58_368)
    }

    /// MLX 4-bit, group 32: up_proj 128×64 packs into U32 [128, 8] (64 × 4 / 32),
    /// scales and biases BF16 [128, 2]. down_proj overridden to 8-bit: 64×128 → U32 [64, 32].
    /// Logical params: 128×64 + 64×128 + norm 64 = 16,448.
    /// Bytes: 4,096 + 2×512 (up) + 8,192 + 2×512 (down, scales [64, 4]) + 128 (norm) = 14,464.
    func testMLXQuantizedFolder() throws {
        let dir = try makeTempDirectory(self)
        let folder = dir.appendingPathComponent("tiny-mlx-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var w = SafetensorsWriter()
        w.metadata = ["format": "mlx"]
        w.tensors = [
            .init("model.layers.0.mlp.up_proj.weight", "U32", [128, 8]),
            .init("model.layers.0.mlp.up_proj.scales", "BF16", [128, 2]),
            .init("model.layers.0.mlp.up_proj.biases", "BF16", [128, 2]),
            .init("model.layers.0.mlp.down_proj.weight", "U32", [64, 32]),
            .init("model.layers.0.mlp.down_proj.scales", "BF16", [64, 4]),
            .init("model.layers.0.mlp.down_proj.biases", "BF16", [64, 4]),
            .init("model.norm.weight", "BF16", [64]),
        ]
        try w.write(to: folder.appendingPathComponent("model.safetensors"))
        try SampleModels.writeJSON([
            "model_type": "llama", "num_hidden_layers": 1, "hidden_size": 64, "num_attention_heads": 4,
            "num_key_value_heads": 4, "max_position_embeddings": 8192, "vocab_size": 100,
            "quantization": ["group_size": 32, "bits": 4, "model.layers.0.mlp.down_proj": ["group_size": 32, "bits": 8]],
        ], to: folder.appendingPathComponent("config.json"))

        let a = try XCTUnwrap(ModelLoader.analyze(url: folder).first)
        XCTAssertEqual(a.format, .mlxFolder)
        XCTAssertEqual(a.parameterCount, 16_448)
        XCTAssertEqual(a.weightBytes, 14_464)
        XCTAssertEqual(a.quantizationLabel, "Mixed 4/8-bit, group 32")
        XCTAssertEqual(Set(a.typeMix.map(\.name)), ["4-bit", "8-bit", "Scales and biases", "BF16"])
    }

    func testInfersArchitectureFromTensorNames() throws {
        let dir = try makeTempDirectory(self)
        var w = SafetensorsWriter()
        w.tensors = [.init("model.embed_tokens.weight", "F16", [100, 64])]
        for i in 0 ..< 3 {
            w.tensors += [
                .init("model.layers.\(i).self_attn.k_proj.weight", "F16", [16, 64]),
                .init("model.layers.\(i).self_attn.v_proj.weight", "F16", [16, 64]),
            ]
        }
        let url = dir.appendingPathComponent("noconfig.safetensors")
        try w.write(to: url)
        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        XCTAssertEqual(a.architecture.source, .inferredFromTensors)
        XCTAssertEqual(a.architecture.layerCount, 3)
        XCTAssertEqual(a.architecture.vocabSize, 100)
        XCTAssertEqual(a.architecture.embeddingLength, 64)
        XCTAssertEqual(a.architecture.kvCacheShape?.elementsPerToken, 3 * 32)
    }

    func testSampleFoldersMatchPublishedSizes() throws {
        let dir = try makeTempDirectory(self)
        let qwen = dir.appendingPathComponent("Qwen2.5-14B-Instruct-4bit")
        try SampleModels.writeQwen14BMLX(to: qwen)
        let q = try XCTUnwrap(ModelLoader.analyze(url: qwen).first)
        XCTAssertEqual(q.format, .mlxFolder)
        XCTAssertEqual(q.parameterCount, 14_770_033_664) // published: 14.77B
        XCTAssertEqual(q.weightBytes, 8_309_352_448)
        XCTAssertEqual(q.quantizationLabel, "4-bit, group 64")
        XCTAssertTrue(q.isSample)

        let mistral = dir.appendingPathComponent("Mistral-7B-Instruct-v0.3")
        try SampleModels.writeMistral7B(to: mistral)
        let m = try XCTUnwrap(ModelLoader.analyze(url: mistral).first)
        XCTAssertEqual(m.parameterCount, 7_248_023_552) // published: 7.25B
        XCTAssertEqual(m.weightBytes, 7_248_023_552 * 2)
        XCTAssertEqual(m.fileCount, 3)
        XCTAssertEqual(m.architecture.contextLength, 32768)
    }
}
