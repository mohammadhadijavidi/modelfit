import Foundation
import XCTest
@testable import ModelFitCore

final class GGUFTests: XCTestCase {
    func testBlockLayouts() {
        XCTAssertEqual(GGMLType.q4_K.bitsPerWeight, 4.5)
        XCTAssertEqual(GGMLType.q6_K.bitsPerWeight, 6.5625)
        XCTAssertEqual(GGMLType.q8_0.bitsPerWeight, 8.5)
        XCTAssertEqual(GGMLType.q4_0.byteSize(elements: 64), 36)
        XCTAssertNil(GGMLType.q4_K.byteSize(elements: 100), "not a multiple of the 256-weight super-block")
        XCTAssertEqual(GGUFFileType.name(for: 15), "Q4_K_M")
        XCTAssertEqual(GGUFFileType.name(for: 15 | 1024), "Q4_K_M", "the 'guessed' flag is masked")
    }

    func testParsesTinyModel() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("tiny.gguf")
        try Tiny.writer().write(to: url)

        let file = try GGUFParser.parse(url: url)
        XCTAssertEqual(file.version, 3)
        XCTAssertEqual(file.tensors.count, 5)
        XCTAssertEqual(file.tensors.map(\.byteSize), [12_800, 4_352, 2_176, 4_608, 256])
        XCTAssertEqual(file.metadata["llama.block_count"], .uint32(2))
        XCTAssertEqual(file.metadata["tokenizer.ggml.tokens"]?.arrayValue?.count, 100)
        XCTAssertEqual(file.dataOffset % 32, 0)
        XCTAssertTrue(file.warnings.isEmpty, "\(file.warnings)")

        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        XCTAssertEqual(a.format, .gguf)
        XCTAssertEqual(a.displayName, "Tiny Test Model")
        XCTAssertEqual(a.parameterCount, Tiny.parameters)
        XCTAssertEqual(a.weightBytes, Tiny.bytes)
        XCTAssertEqual(a.tensorCount, 5)
        XCTAssertEqual(a.quantizationLabel, "Q8_0")
        XCTAssertEqual(a.typeMix.map(\.name), ["F16", "Q8_0", "Q4_K", "F32"])
        XCTAssertEqual(a.typeMix.map(\.bytes), [12_800, 6_528, 4_608, 256])
        XCTAssertEqual(a.bitsPerWeight!, Double(Tiny.bytes * 8) / Double(Tiny.parameters), accuracy: 1e-9)

        let arch = a.architecture
        XCTAssertEqual(arch.architecture, "llama")
        XCTAssertEqual(arch.layerCount, 2)
        XCTAssertEqual(arch.headDimension, 16)
        XCTAssertEqual(arch.kvHeadCount, 2)
        XCTAssertEqual(arch.contextLength, 4096)
        XCTAssertEqual(arch.vocabSize, 100)
        XCTAssertEqual(arch.ropeFreqBase, 10_000)
        XCTAssertEqual(arch.source, .ggufMetadata)
        XCTAssertEqual(a.tokenizer?.tokenCount, 100)
        XCTAssertEqual(a.tokenizer?.bosTokenID, 1)
        XCTAssertEqual(a.tokenizer?.hasChatTemplate, true)
        XCTAssertFalse(a.isSample)
    }

    func testVersionsOneAndTwoParseTheSame() throws {
        let dir = try makeTempDirectory(self)
        for version: UInt32 in [1, 2] {
            let url = dir.appendingPathComponent("v\(version).gguf")
            try Tiny.writer(version: version).write(to: url)
            let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
            XCTAssertEqual(a.parameterCount, Tiny.parameters, "v\(version)")
            XCTAssertEqual(a.weightBytes, Tiny.bytes, "v\(version)")
            XCTAssertEqual(a.formatDetail, "GGUF v\(version)")
        }
    }

    func testLargeArraysKeepCountButNotContents() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("vocab.gguf")
        try Tiny.writer(tokens: 50_000).write(to: url)
        let file = try GGUFParser.parse(url: url)
        let tokens = try XCTUnwrap(file.metadata["tokenizer.ggml.tokens"]?.arrayValue)
        XCTAssertEqual(tokens.count, 50_000)
        XCTAssertNil(tokens.values, "big arrays are skipped on disk, not held in memory")
        XCTAssertEqual(file.metadata["tokenizer.ggml.scores"]?.arrayValue?.count, 50_000)
        XCTAssertEqual(file.tensors.count, 5, "parsing continues correctly after skipped arrays")
    }

    func testReadsOnlyTheHeaderOfAHugeFile() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("huge.gguf")
        try Tiny.writer().write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 3 << 30) // sparse 3 GB
        try handle.close()

        let file = try GGUFParser.parse(url: url)
        XCTAssertEqual(file.fileSize, 3 << 30)
        XCTAssertLessThanOrEqual(file.bytesRead, 1 << 20, "read at most one 1 MB chunk")
        XCTAssertTrue(file.warnings.isEmpty)
    }

    func testRealisticLlama8BSampleHeader() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        try SampleModels.llama31_8B().write(to: url)

        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        // Published: 8.03B parameters, 292 tensors, ~4.92 GB for Q4_K_M.
        XCTAssertEqual(a.parameterCount, 8_030_261_312)
        XCTAssertEqual(a.tensorCount, 292)
        XCTAssertEqual(a.weightBytes, 4_912_898_304)
        XCTAssertEqual(a.quantizationLabel, "Q4_K_M")
        XCTAssertEqual(Format.parameters(a.parameterCount), "8.03B")
        XCTAssertEqual(a.tokenizer?.tokenCount, 128_256)
        XCTAssertEqual(a.tokenizer?.mergeCount, 280_147)
        XCTAssertEqual(a.architecture.contextLength, 131_072)
        XCTAssertTrue(a.isSample)
        XCTAssertEqual(Set(a.typeMix.map(\.name)), ["Q4_K", "Q6_K", "F32"])
        XCTAssertLessThan(a.headerBytesRead, 16 << 20, "only the multi-MB tokenizer header is read")
    }

    func testSplitModelIsCombined() throws {
        let dir = try makeTempDirectory(self)
        let first = try SampleModels.writeSplit(Tiny.writer(), prefix: "Tiny-Q8_0", in: dir)
        let fromFirst = try XCTUnwrap(ModelLoader.analyze(url: first).first)
        XCTAssertEqual(fromFirst.fileCount, 2)
        XCTAssertEqual(fromFirst.parameterCount, Tiny.parameters)
        XCTAssertEqual(fromFirst.weightBytes, Tiny.bytes)
        XCTAssertEqual(fromFirst.architecture.layerCount, 2)

        let second = dir.appendingPathComponent("Tiny-Q8_0-00002-of-00002.gguf")
        let fromSecond = try XCTUnwrap(ModelLoader.analyze(url: second).first)
        XCTAssertEqual(fromSecond.parameterCount, Tiny.parameters, "opening any part finds the whole model")
        XCTAssertEqual(fromSecond.url.lastPathComponent, first.lastPathComponent)

        let folder = try ModelLoader.analyze(url: dir)
        XCTAssertEqual(folder.count, 1, "a folder lists a split model once")
    }

    func testMissingSplitPartWarns() throws {
        let dir = try makeTempDirectory(self)
        let first = try SampleModels.writeSplit(Tiny.writer(), prefix: "Gone", in: dir)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("Gone-00002-of-00002.gguf"))
        let a = try XCTUnwrap(ModelLoader.analyze(url: first).first)
        XCTAssertEqual(a.fileCount, 1)
        XCTAssertTrue(a.warnings.contains { $0.contains("part 2 of 2") }, "\(a.warnings)")
    }

    func testPerLayerKVHeads() throws {
        let dir = try makeTempDirectory(self)
        var w = Tiny.writer()
        w.metadata.removeAll { $0.0 == "llama.attention.head_count_kv" }
        w.set("llama.attention.head_count_kv", .array(GGUFArray([.int32(2), .int32(1)], elementType: .int32)))
        let url = dir.appendingPathComponent("perlayer.gguf")
        try w.write(to: url)
        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        XCTAssertEqual(a.architecture.kvHeadsPerLayer, [2, 1])
        // (2 + 1) heads × (16 + 16) dims = 96 elements per token.
        XCTAssertEqual(a.architecture.kvCacheShape?.elementsPerToken, 96)
    }
}
