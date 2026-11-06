import Foundation

public struct SafetensorsTensor: Sendable, Equatable {
    public var name: String
    public var dtype: String
    public var shape: [Int64]
    public var dataBegin: UInt64
    public var dataEnd: UInt64
    public var elementCount: Int64
    public var byteSize: UInt64 { dataEnd - dataBegin }
}

/// The parsed JSON header of one .safetensors file. Tensor bytes are never read.
public struct SafetensorsFile: Sendable {
    public var url: URL
    public var headerLength: UInt64
    public var tensors: [SafetensorsTensor]
    public var metadata: [String: String]
    public var fileSize: UInt64
    public var dataSize: UInt64
    public var bytesRead: UInt64
    public var warnings: [String]
}

public enum SafetensorsParser {
    /// The safetensors spec caps the JSON header at 100 MB.
    static let maxHeader: UInt64 = 100_000_000

    public static let dtypeSizes: [String: Int] = [
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1,
        "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
        "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    ]

    public static func parse(url: URL) throws -> SafetensorsFile {
        let r = try ByteReader(url: url, chunkSize: 64 * 1024)
        let file = r.fileName
        let length = try r.u64("the safetensors header length")
        guard length >= 2 else { throw ModelFitError.notAModel(file: file) }
        guard length <= maxHeader else { throw ModelFitError.headerTooLarge(file: file, size: length) }
        guard length <= r.remaining else {
            throw ModelFitError.truncated(file: file, context: "the JSON header (\(Format.bytes(Int64(length))))", offset: r.position)
        }
        let bytes = try r.bytes(Int(length), "the JSON header")
        guard let firstNonSpace = bytes.first(where: { $0 != 0x20 && $0 != 0x0A && $0 != 0x0D && $0 != 0x09 }), firstNonSpace == UInt8(ascii: "{") else {
            throw ModelFitError.notAModel(file: file)
        }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: Data(bytes), options: [])
        } catch {
            throw ModelFitError.corrupt(file: file, reason: "its JSON header doesn't parse")
        }
        guard let root = object as? [String: Any] else {
            throw ModelFitError.corrupt(file: file, reason: "its header isn't a JSON object")
        }

        var metadata: [String: String] = [:]
        var tensors: [SafetensorsTensor] = []
        var dataSize: UInt64 = 0
        for (name, value) in root {
            if name == "__metadata__" {
                if let m = value as? [String: Any] {
                    for (k, v) in m { metadata[k] = (v as? String) ?? "\(v)" }
                }
                continue
            }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shapeRaw = entry["shape"] as? [Any],
                  let offsetsRaw = entry["data_offsets"] as? [Any], offsetsRaw.count == 2
            else {
                throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' is missing dtype, shape or data_offsets")
            }
            let shape = shapeRaw.compactMap { ($0 as? NSNumber)?.int64Value }
            let offsets = offsetsRaw.compactMap { ($0 as? NSNumber)?.int64Value }
            guard shape.count == shapeRaw.count, offsets.count == 2,
                  shape.allSatisfy({ $0 >= 0 }), offsets[0] >= 0, offsets[1] >= offsets[0]
            else {
                throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' has an invalid shape or byte range")
            }
            var elements: Int64 = 1
            for d in shape {
                let (p, o) = elements.multipliedReportingOverflow(by: d)
                guard !o else { throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' has an impossible shape") }
                elements = p
            }
            let begin = UInt64(offsets[0]), end = UInt64(offsets[1])
            if let size = dtypeSizes[dtype] {
                let (expected, o) = elements.multipliedReportingOverflow(by: Int64(size))
                guard !o, UInt64(expected) == end - begin else {
                    throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' is \(Format.integer(Int64(end - begin))) bytes but its \(dtype) shape needs \(o ? "more" : Format.integer(expected))")
                }
            }
            dataSize = max(dataSize, end)
            tensors.append(SafetensorsTensor(name: name, dtype: dtype, shape: shape, dataBegin: begin, dataEnd: end, elementCount: elements))
        }
        tensors.sort { $0.dataBegin != $1.dataBegin ? $0.dataBegin < $1.dataBegin : $0.name < $1.name }

        var warnings: [String] = []
        let needed = 8 + length + dataSize
        if needed > r.fileSize {
            warnings.append("\(file) is truncated: its header describes \(Format.bytes(Int64(clamping: needed))) but only \(Format.bytes(Int64(clamping: r.fileSize))) are on disk.")
        }
        let unknown = Set(tensors.map(\.dtype)).subtracting(dtypeSizes.keys)
        if !unknown.isEmpty {
            warnings.append("Unrecognised dtype \(unknown.sorted().joined(separator: ", ")); sizes taken from byte ranges.")
        }
        return SafetensorsFile(url: url, headerLength: length, tensors: tensors, metadata: metadata,
                               fileSize: r.fileSize, dataSize: dataSize, bytesRead: r.bytesRead, warnings: warnings)
    }
}

/// Writes safetensors headers for tests and sample renders. Data is zeros or a sparse hole.
public struct SafetensorsWriter {
    public struct Tensor {
        public var name: String
        public var dtype: String
        public var shape: [Int64]
        public init(_ name: String, _ dtype: String, _ shape: [Int64]) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
        }
    }

    public var tensors: [Tensor] = []
    public var metadata: [String: String] = [:]
    public init() {}

    public func header() throws -> (data: Data, totalSize: UInt64) {
        var root: [String: Any] = [:]
        var offset: Int64 = 0
        for t in tensors {
            let size = Int64(SafetensorsParser.dtypeSizes[t.dtype] ?? 1) * t.shape.reduce(1, *)
            root[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + size]] as [String: Any]
            offset += size
        }
        if !metadata.isEmpty { root["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        return (out, UInt64(out.count) + UInt64(offset))
    }

    public func write(to url: URL, fullSize: Bool = true) throws {
        let (data, total) = try header()
        try data.write(to: url)
        if fullSize, total > UInt64(data.count) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: total)
        }
    }
}
