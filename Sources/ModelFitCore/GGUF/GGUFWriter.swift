import Foundation

/// Writes GGUF headers. Used by the test suite to build tiny synthetic models and by the
/// screenshot renderer to build "sample header" files with well-known published
/// architectures. The tensor data section is left as zeros (or a sparse hole).
public struct GGUFWriter {
    public struct Tensor {
        public var name: String
        public var dimensions: [UInt64]
        public var type: GGMLType
        public init(_ name: String, _ dimensions: [UInt64], _ type: GGMLType) {
            self.name = name
            self.dimensions = dimensions
            self.type = type
        }
    }

    public var version: UInt32 = 3
    public var alignment: UInt64 = 32
    public var metadata: [(String, GGUFValue)] = []
    public var tensors: [Tensor] = []

    public init() {}

    public mutating func set(_ key: String, _ value: GGUFValue) { metadata.append((key, value)) }

    /// Bytes for the header (padded to alignment) and the total file size it describes.
    public func header() -> (data: Data, totalSize: UInt64) {
        var out = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func u64(_ v: UInt64) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func count(_ v: UInt64) { version >= 2 ? u64(v) : u32(UInt32(truncatingIfNeeded: v)) }
        func string(_ s: String) {
            let bytes = Array(s.utf8)
            count(UInt64(bytes.count))
            out.append(contentsOf: bytes)
        }
        func value(_ v: GGUFValue) {
            switch v {
            case let .uint8(x): out.append(x)
            case let .int8(x): out.append(UInt8(bitPattern: x))
            case let .uint16(x): withUnsafeBytes(of: x.littleEndian) { out.append(contentsOf: $0) }
            case let .int16(x): withUnsafeBytes(of: x.littleEndian) { out.append(contentsOf: $0) }
            case let .uint32(x): u32(x)
            case let .int32(x): u32(UInt32(bitPattern: x))
            case let .uint64(x): u64(x)
            case let .int64(x): u64(UInt64(bitPattern: x))
            case let .float32(x): u32(x.bitPattern)
            case let .float64(x): u64(x.bitPattern)
            case let .bool(x): out.append(x ? 1 : 0)
            case let .string(s): string(s)
            case let .skippedString(n): string(String(repeating: " ", count: Int(min(n, 1 << 24))))
            case let .array(a):
                u32(a.elementType.rawValue)
                let values = a.values ?? []
                count(UInt64(values.count))
                values.forEach(value)
            }
        }

        u32(GGUFParser.magic)
        u32(version)
        count(UInt64(tensors.count))
        count(UInt64(metadata.count))
        for (key, v) in metadata {
            string(key)
            u32(v.type.rawValue)
            value(v)
        }
        var offset: UInt64 = 0
        for t in tensors {
            string(t.name)
            u32(UInt32(t.dimensions.count))
            t.dimensions.forEach(count)
            u32(t.type.rawValue)
            u64(offset)
            let elements = t.dimensions.reduce(1, *)
            let size = t.type.byteSize(elements: elements) ?? 0
            offset += (size + alignment - 1) / alignment * alignment
        }
        let padded = (UInt64(out.count) + alignment - 1) / alignment * alignment
        out.append(contentsOf: [UInt8](repeating: 0, count: Int(padded) - out.count))
        return (out, padded + offset)
    }

    /// Writes the header and extends the file to its full declared size. On APFS the
    /// extension is a sparse hole, so a "5 GB" sample costs a few kilobytes of disk.
    public func write(to url: URL, fullSize: Bool = true) throws {
        let (data, total) = header()
        try data.write(to: url)
        if fullSize, total > UInt64(data.count) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: total)
        }
    }
}
