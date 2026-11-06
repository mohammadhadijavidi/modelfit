import Foundation

/// GGUF metadata value types, as numbered in the GGUF specification.
public enum GGUFValueType: UInt32, Sendable, CaseIterable {
    case uint8 = 0, int8, uint16, int16, uint32, int32, float32, bool, string, array, uint64, int64, float64

    /// Size of a fixed-width value, nil for string and array.
    var fixedSize: Int? {
        switch self {
        case .uint8, .int8, .bool: 1
        case .uint16, .int16: 2
        case .uint32, .int32, .float32: 4
        case .uint64, .int64, .float64: 8
        case .string, .array: nil
        }
    }
}

/// A decoded metadata value. Big arrays (token lists, merges, scores) keep only their
/// count: their contents are skipped on disk and never held in memory.
public enum GGUFValue: Sendable, Equatable {
    case uint8(UInt8), int8(Int8), uint16(UInt16), int16(Int16)
    case uint32(UInt32), int32(Int32), uint64(UInt64), int64(Int64)
    case float32(Float), float64(Double)
    case bool(Bool)
    case string(String)
    /// A string longer than the in-memory cap; only its byte length is kept.
    case skippedString(length: UInt64)
    case array(GGUFArray)

    public var type: GGUFValueType {
        switch self {
        case .uint8: .uint8
        case .int8: .int8
        case .uint16: .uint16
        case .int16: .int16
        case .uint32: .uint32
        case .int32: .int32
        case .uint64: .uint64
        case .int64: .int64
        case .float32: .float32
        case .float64: .float64
        case .bool: .bool
        case .string, .skippedString: .string
        case .array: .array
        }
    }

    public var intValue: Int? {
        switch self {
        case let .uint8(v): Int(v)
        case let .int8(v): Int(v)
        case let .uint16(v): Int(v)
        case let .int16(v): Int(v)
        case let .uint32(v): Int(v)
        case let .int32(v): Int(v)
        case let .uint64(v): Int(exactly: v)
        case let .int64(v): Int(exactly: v)
        case let .float32(v): v.isFinite && v == v.rounded() && abs(v) < 1e15 ? Int(v) : nil
        case let .float64(v): v.isFinite && v == v.rounded() && abs(v) < 1e15 ? Int(v) : nil
        case let .bool(v): v ? 1 : 0
        default: nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case let .float32(v): Double(v)
        case let .float64(v): v
        default: intValue.map(Double.init)
        }
    }

    public var stringValue: String? {
        if case let .string(s) = self { return s }
        return nil
    }

    public var boolValue: Bool? {
        switch self {
        case let .bool(v): v
        default: intValue.map { $0 != 0 }
        }
    }

    public var arrayValue: GGUFArray? {
        if case let .array(a) = self { return a }
        return nil
    }

    /// Integers from either a scalar or a (stored) array, e.g. per-layer head counts.
    public var intList: [Int]? {
        if let a = arrayValue { return a.values?.compactMap(\.intValue) }
        return intValue.map { [$0] }
    }

    /// Present for both stored and skipped strings: true if a string exists and is non-empty.
    public var isNonEmptyString: Bool {
        switch self {
        case let .string(s): !s.isEmpty
        case let .skippedString(n): n > 0
        default: false
        }
    }
}

public struct GGUFArray: Sendable, Equatable {
    public var elementType: GGUFValueType
    public var count: UInt64
    /// nil when the array was too large to keep (its count is still exact).
    public var values: [GGUFValue]?

    public init(elementType: GGUFValueType, count: UInt64, values: [GGUFValue]?) {
        self.elementType = elementType
        self.count = count
        self.values = values
    }

    public init(_ values: [GGUFValue], elementType: GGUFValueType) {
        self.init(elementType: elementType, count: UInt64(values.count), values: values)
    }
}
