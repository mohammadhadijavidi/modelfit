import Foundation

/// A ggml tensor storage type, with its block layout.
///
/// Sizes come from the block structs in ggml's `ggml-common.h`: for example a `Q4_K`
/// super-block packs 256 weights into 144 bytes (4.5 bits per weight).
public struct GGMLType: Hashable, Sendable, CustomStringConvertible {
    public let rawValue: UInt32

    public init(rawValue: UInt32) { self.rawValue = rawValue }

    struct Layout: Sendable {
        let name: String
        let blockSize: Int
        let typeSize: Int
    }

    static let table: [UInt32: Layout] = [
        0: .init(name: "F32", blockSize: 1, typeSize: 4),
        1: .init(name: "F16", blockSize: 1, typeSize: 2),
        2: .init(name: "Q4_0", blockSize: 32, typeSize: 18),
        3: .init(name: "Q4_1", blockSize: 32, typeSize: 20),
        6: .init(name: "Q5_0", blockSize: 32, typeSize: 22),
        7: .init(name: "Q5_1", blockSize: 32, typeSize: 24),
        8: .init(name: "Q8_0", blockSize: 32, typeSize: 34),
        9: .init(name: "Q8_1", blockSize: 32, typeSize: 36),
        10: .init(name: "Q2_K", blockSize: 256, typeSize: 84),
        11: .init(name: "Q3_K", blockSize: 256, typeSize: 110),
        12: .init(name: "Q4_K", blockSize: 256, typeSize: 144),
        13: .init(name: "Q5_K", blockSize: 256, typeSize: 176),
        14: .init(name: "Q6_K", blockSize: 256, typeSize: 210),
        15: .init(name: "Q8_K", blockSize: 256, typeSize: 292),
        16: .init(name: "IQ2_XXS", blockSize: 256, typeSize: 66),
        17: .init(name: "IQ2_XS", blockSize: 256, typeSize: 74),
        18: .init(name: "IQ3_XXS", blockSize: 256, typeSize: 98),
        19: .init(name: "IQ1_S", blockSize: 256, typeSize: 50),
        20: .init(name: "IQ4_NL", blockSize: 32, typeSize: 18),
        21: .init(name: "IQ3_S", blockSize: 256, typeSize: 110),
        22: .init(name: "IQ2_S", blockSize: 256, typeSize: 82),
        23: .init(name: "IQ4_XS", blockSize: 256, typeSize: 136),
        24: .init(name: "I8", blockSize: 1, typeSize: 1),
        25: .init(name: "I16", blockSize: 1, typeSize: 2),
        26: .init(name: "I32", blockSize: 1, typeSize: 4),
        27: .init(name: "I64", blockSize: 1, typeSize: 8),
        28: .init(name: "F64", blockSize: 1, typeSize: 8),
        29: .init(name: "IQ1_M", blockSize: 256, typeSize: 56),
        30: .init(name: "BF16", blockSize: 1, typeSize: 2),
        31: .init(name: "Q4_0_4_4", blockSize: 32, typeSize: 18),
        32: .init(name: "Q4_0_4_8", blockSize: 32, typeSize: 18),
        33: .init(name: "Q4_0_8_8", blockSize: 32, typeSize: 18),
        34: .init(name: "TQ1_0", blockSize: 256, typeSize: 54),
        35: .init(name: "TQ2_0", blockSize: 256, typeSize: 66),
        36: .init(name: "IQ4_NL_4_4", blockSize: 32, typeSize: 18),
        37: .init(name: "IQ4_NL_4_8", blockSize: 32, typeSize: 18),
        38: .init(name: "IQ4_NL_8_8", blockSize: 32, typeSize: 18),
        39: .init(name: "MXFP4", blockSize: 32, typeSize: 17),
    ]

    public static let f32 = GGMLType(rawValue: 0)
    public static let f16 = GGMLType(rawValue: 1)
    public static let q4_0 = GGMLType(rawValue: 2)
    public static let q8_0 = GGMLType(rawValue: 8)
    public static let q4_K = GGMLType(rawValue: 12)
    public static let q5_K = GGMLType(rawValue: 13)
    public static let q6_K = GGMLType(rawValue: 14)
    public static let bf16 = GGMLType(rawValue: 30)

    public var isKnown: Bool { Self.table[rawValue] != nil }
    public var name: String { Self.table[rawValue]?.name ?? "type \(rawValue)" }
    public var blockSize: Int? { Self.table[rawValue]?.blockSize }
    public var typeSize: Int? { Self.table[rawValue]?.typeSize }
    public var description: String { name }

    /// Storage bits per weight, e.g. 4.5 for Q4_K.
    public var bitsPerWeight: Double? {
        guard let b = blockSize, let t = typeSize else { return nil }
        return Double(t * 8) / Double(b)
    }

    /// Exact bytes for `elements` weights, or nil if unknown or not block-aligned.
    public func byteSize(elements: UInt64) -> UInt64? {
        guard let b = blockSize, let t = typeSize, elements % UInt64(b) == 0 else { return nil }
        let (r, o) = (elements / UInt64(b)).multipliedReportingOverflow(by: UInt64(t))
        return o ? nil : r
    }
}

/// `general.file_type` (llama.cpp's `llama_ftype`) as the familiar preset name.
public enum GGUFFileType {
    static let names: [Int: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1 (some F16)",
        7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
        13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
        18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
        23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
        28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
        33: "Q4_0_4_4", 34: "Q4_0_4_8", 35: "Q4_0_8_8", 36: "TQ1_0", 37: "TQ2_0",
        38: "MXFP4_MOE",
    ]

    /// The preset name; the "guessed" flag (1024) is masked off.
    public static func name(for value: Int) -> String? {
        names[value & ~1024]
    }
}
