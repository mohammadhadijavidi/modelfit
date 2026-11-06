import Foundation
import XCTest
@testable import ModelFitCore

/// Damaged input must produce a `ModelFitError`, never a crash or a hang.
final class CorruptionTests: XCTestCase {
    func testEveryTruncationOfTheHeaderFailsGracefully() throws {
        let dir = try makeTempDirectory(self)
        let (header, _) = Tiny.writer().header()
        let url = dir.appendingPathComponent("cut.gguf")
        // Padding is at most 31 bytes, so every cut below this point lands inside metadata or tensor infos.
        let limit = header.count - 32
        for cut in stride(from: 0, to: limit, by: 5) {
            try header.prefix(cut).write(to: url)
            XCTAssertThrowsError(try GGUFParser.parse(url: url), "cut at \(cut)") { error in
                XCTAssertTrue(error is ModelFitError, "cut at \(cut): \(error)")
            }
        }
    }

    func testTruncatedDataIsReportedNotFatal() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("partial.gguf")
        try Tiny.writer().write(to: url, fullSize: false)
        let a = try XCTUnwrap(ModelLoader.analyze(url: url).first)
        XCTAssertEqual(a.parameterCount, Tiny.parameters)
        XCTAssertTrue(a.warnings.contains { $0.contains("truncated") }, "\(a.warnings)")
    }

    func testRandomByteFlipsNeverCrash() throws {
        let dir = try makeTempDirectory(self)
        let (header, _) = Tiny.writer().header()
        let url = dir.appendingPathComponent("fuzz.gguf")
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return seed >> 33
        }
        var failures = 0
        for _ in 0 ..< 400 {
            var bytes = [UInt8](header)
            for _ in 0 ..< 1 + Int(next() % 4) {
                let i = Int(next() % UInt64(bytes.count))
                bytes[i] = UInt8(truncatingIfNeeded: next())
            }
            try Data(bytes).write(to: url)
            do {
                _ = try GGUFParser.parse(url: url)
            } catch is ModelFitError {
                failures += 1
            }
        }
        XCTAssertGreaterThan(failures, 0, "some mutations must be detected")
    }

    func testWrongMagicVersionAndEndianness() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("x.gguf")

        try Data("not a model at all".utf8).write(to: url)
        XCTAssertThrowsError(try GGUFParser.parse(url: url)) { XCTAssertEqual($0 as? ModelFitError, .notAModel(file: "x.gguf")) }

        var header = [UInt8](Tiny.writer().header().data)
        header[4] = 9
        try Data(header).write(to: url)
        XCTAssertThrowsError(try GGUFParser.parse(url: url)) {
            XCTAssertEqual($0 as? ModelFitError, .unsupportedVersion(file: "x.gguf", version: 9))
        }

        header[4 ... 7] = [0, 0, 0, 3]
        try Data(header).write(to: url)
        XCTAssertThrowsError(try GGUFParser.parse(url: url)) {
            XCTAssertEqual($0 as? ModelFitError, .bigEndianGGUF(file: "x.gguf"))
        }

        try Data().write(to: url)
        XCTAssertThrowsError(try GGUFParser.parse(url: url)) { XCTAssertTrue($0 is ModelFitError) }
    }

    func testImplausibleCountsAreRejectedQuickly() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("counts.gguf")
        var bytes = [UInt8](Tiny.writer().header().data)
        // Metadata count (bytes 16..<24) set to 2^40.
        bytes[16 ..< 24] = [0, 0, 0, 0, 0, 1, 0, 0]
        try Data(bytes).write(to: url)
        XCTAssertThrowsError(try GGUFParser.parse(url: url)) { error in
            guard case .corrupt? = error as? ModelFitError else { return XCTFail("\(error)") }
        }
    }

    func testSafetensorsDamage() throws {
        let dir = try makeTempDirectory(self)
        let url = dir.appendingPathComponent("bad.safetensors")
        func write(length: UInt64, json: String) throws {
            var d = Data()
            withUnsafeBytes(of: length.littleEndian) { d.append(contentsOf: $0) }
            d.append(Data(json.utf8))
            try d.write(to: url)
        }

        try write(length: 5000, json: "{}")
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) { error in
            guard case .truncated? = error as? ModelFitError else { return XCTFail("\(error)") }
        }

        try write(length: 200_000_000, json: "{}")
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) { error in
            guard case .headerTooLarge? = error as? ModelFitError else { return XCTFail("\(error)") }
        }

        try write(length: 6, json: "{abc]}")
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) { error in
            guard case .corrupt? = error as? ModelFitError else { return XCTFail("\(error)") }
        }

        try write(length: 3, json: "[1]")
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) {
            XCTAssertEqual($0 as? ModelFitError, .notAModel(file: "bad.safetensors"))
        }

        let mismatch = #"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#
        try write(length: UInt64(mismatch.utf8.count), json: mismatch)
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) { error in
            guard case .corrupt? = error as? ModelFitError else { return XCTFail("\(error)") }
        }

        let missing = #"{"w":{"dtype":"F32","shape":[4]}}"#
        try write(length: UInt64(missing.utf8.count), json: missing)
        XCTAssertThrowsError(try SafetensorsParser.parse(url: url)) { error in
            guard case .corrupt? = error as? ModelFitError else { return XCTFail("\(error)") }
        }
    }

    func testUnknownFilesAndEmptyFolders() throws {
        let dir = try makeTempDirectory(self)
        let txt = dir.appendingPathComponent("notes.txt")
        try Data("hello world, definitely not weights".utf8).write(to: txt)
        XCTAssertThrowsError(try ModelLoader.analyze(url: txt)) {
            XCTAssertEqual($0 as? ModelFitError, .notAModel(file: "notes.txt"))
        }
        let empty = dir.appendingPathComponent("Empty", isDirectory: true)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        XCTAssertThrowsError(try ModelLoader.analyze(url: empty)) {
            XCTAssertEqual($0 as? ModelFitError, .emptyFolder(folder: "Empty"))
        }
        XCTAssertThrowsError(try ModelLoader.analyze(url: dir.appendingPathComponent("missing.gguf"))) {
            guard case .unreadable? = $0 as? ModelFitError else { return XCTFail("\($0)") }
        }
    }

    func testErrorsHaveHumanMessages() {
        let e = ModelFitError.truncated(file: "a.gguf", context: "metadata value 'x'", offset: 1234)
        XCTAssertEqual(e.title, "This file ends too early")
        XCTAssertTrue(e.detail.contains("1,234"))
        XCTAssertFalse(e.recoverySuggestion.isEmpty)
    }
}
