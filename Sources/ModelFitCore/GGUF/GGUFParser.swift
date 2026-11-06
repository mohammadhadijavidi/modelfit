import Foundation

public struct GGUFTensorInfo: Sendable, Equatable {
    public var name: String
    /// ggml order: dimensions[0] is the fastest-varying (row length).
    public var dimensions: [UInt64]
    public var type: GGMLType
    /// Offset relative to the start of the tensor data section.
    public var offset: UInt64
    public var elementCount: UInt64
    /// Stored bytes. Exact for known types; estimated from neighbouring offsets otherwise.
    public var byteSize: UInt64
}

/// The parsed header of one GGUF file. Tensor data is never read.
public struct GGUFFile: Sendable {
    public var url: URL
    public var version: UInt32
    public var metadataKeys: [String]
    public var metadata: [String: GGUFValue]
    public var tensors: [GGUFTensorInfo]
    public var alignment: UInt64
    /// Absolute offset where the tensor data section begins.
    public var dataOffset: UInt64
    /// Bytes from the data section start to the end of the last tensor.
    public var tensorDataSize: UInt64
    public var fileSize: UInt64
    public var bytesRead: UInt64
    public var warnings: [String]

    public subscript(key: String) -> GGUFValue? { metadata[key] }
}

public enum GGUFParser {
    /// "GGUF" read as a little-endian UInt32.
    static let magic: UInt32 = 0x4655_4747
    /// Strings longer than this are skipped (only their length is kept).
    static let maxStoredString: UInt64 = 1 << 20
    /// Arrays longer than this keep only their count.
    static let maxStoredArray: UInt64 = 4096

    public static func parse(url: URL) throws -> GGUFFile {
        let r = try ByteReader(url: url)
        let file = r.fileName

        let magicValue = try r.u32("the GGUF magic number")
        guard magicValue == magic else { throw ModelFitError.notAModel(file: file) }

        let version = try r.u32("the GGUF version")
        if version & 0xFFFF == 0, version != 0, (1 ... 3).contains(version.byteSwapped) {
            throw ModelFitError.bigEndianGGUF(file: file)
        }
        guard (1 ... 3).contains(version) else {
            throw ModelFitError.unsupportedVersion(file: file, version: version)
        }
        let wide = version >= 2

        func readCount(_ context: @autoclosure () -> String) throws -> UInt64 {
            wide ? try r.u64(context()) : UInt64(try r.u32(context()))
        }

        let tensorCount = try readCount("the tensor count")
        let kvCount = try readCount("the metadata count")
        let lengthSize: UInt64 = wide ? 8 : 4

        // Each metadata entry needs at least a key length, a type and one value byte;
        // each tensor info at least a name length, dimension count, type and offset.
        guard kvCount <= 1_000_000, kvCount * (lengthSize + 5) <= r.remaining else {
            throw ModelFitError.corrupt(file: file, reason: "it declares \(kvCount) metadata entries, more than the file can hold")
        }
        guard tensorCount <= 10_000_000, tensorCount * (lengthSize + 16) <= r.remaining else {
            throw ModelFitError.corrupt(file: file, reason: "it declares \(tensorCount) tensors, more than the file can hold")
        }

        func readString(_ context: @autoclosure () -> String, cap: UInt64 = maxStoredString) throws -> GGUFValue {
            let ctx = context()
            let length = try readCount("the length of \(ctx)")
            guard length <= r.remaining else {
                throw ModelFitError.truncated(file: file, context: ctx, offset: r.position)
            }
            if length > cap {
                try r.skip(length, ctx)
                return .skippedString(length: length)
            }
            let bytes = try r.bytes(Int(length), ctx)
            return .string(String(decoding: bytes, as: UTF8.self))
        }

        func readValue(_ type: GGUFValueType, key: String, depth: Int) throws -> GGUFValue {
            let ctx = "metadata value '\(key)'"
            switch type {
            case .uint8: return .uint8(try r.u8(ctx))
            case .int8: return .int8(try r.i8(ctx))
            case .uint16: return .uint16(try r.u16(ctx))
            case .int16: return .int16(try r.i16(ctx))
            case .uint32: return .uint32(try r.u32(ctx))
            case .int32: return .int32(try r.i32(ctx))
            case .uint64: return .uint64(try r.u64(ctx))
            case .int64: return .int64(try r.i64(ctx))
            case .float32: return .float32(try r.f32(ctx))
            case .float64: return .float64(try r.f64(ctx))
            case .bool: return .bool(try r.u8(ctx) != 0)
            case .string: return try readString(ctx)
            case .array:
                guard depth < 4 else {
                    throw ModelFitError.corrupt(file: file, reason: "metadata '\(key)' nests arrays too deeply")
                }
                let rawElement = try r.u32(ctx)
                guard let element = GGUFValueType(rawValue: rawElement) else {
                    throw ModelFitError.corrupt(file: file, reason: "metadata '\(key)' has unknown array element type \(rawElement)")
                }
                let count = try readCount(ctx)
                let minimum = UInt64(element.fixedSize ?? (element == .string ? Int(lengthSize) : 12))
                let (needed, overflow) = count.multipliedReportingOverflow(by: minimum)
                guard !overflow, needed <= r.remaining else {
                    throw ModelFitError.truncated(file: file, context: ctx, offset: r.position)
                }
                if count > maxStoredArray {
                    if let size = element.fixedSize {
                        try r.skip(count * UInt64(size), ctx)
                    } else if element == .string {
                        for _ in 0 ..< count {
                            let length = try readCount(ctx)
                            try r.skip(length, ctx)
                        }
                    } else {
                        for _ in 0 ..< count { _ = try readValue(element, key: key, depth: depth + 1) }
                    }
                    return .array(GGUFArray(elementType: element, count: count, values: nil))
                }
                var values: [GGUFValue] = []
                values.reserveCapacity(Int(count))
                for _ in 0 ..< count {
                    values.append(try readValue(element, key: key, depth: depth + 1))
                }
                return .array(GGUFArray(elementType: element, count: count, values: values))
            }
        }

        var keys: [String] = []
        var metadata: [String: GGUFValue] = [:]
        keys.reserveCapacity(Int(kvCount))
        for index in 0 ..< kvCount {
            guard case let .string(key) = try readString("metadata key #\(index + 1)", cap: 65536) else {
                throw ModelFitError.corrupt(file: file, reason: "metadata key #\(index + 1) is longer than 64 KB")
            }
            let rawType = try r.u32("the type of metadata '\(key)'")
            guard let type = GGUFValueType(rawValue: rawType) else {
                throw ModelFitError.corrupt(file: file, reason: "metadata '\(key)' has unknown value type \(rawType)")
            }
            let value = try readValue(type, key: key, depth: 0)
            if metadata[key] == nil { keys.append(key) }
            metadata[key] = value
        }

        struct RawTensor { var name: String; var dims: [UInt64]; var type: GGMLType; var offset: UInt64; var elements: UInt64 }
        var raw: [RawTensor] = []
        raw.reserveCapacity(Int(tensorCount))
        for index in 0 ..< tensorCount {
            guard case let .string(name) = try readString("the name of tensor #\(index + 1)", cap: 65536) else {
                throw ModelFitError.corrupt(file: file, reason: "tensor #\(index + 1) has a name longer than 64 KB")
            }
            let ctx = "tensor '\(name)'"
            let nDims = try r.u32(ctx)
            guard nDims <= 8 else {
                throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' claims \(nDims) dimensions")
            }
            var dims: [UInt64] = []
            var elements: UInt64 = 1
            for _ in 0 ..< nDims {
                let d = try readCount(ctx)
                dims.append(d)
                let (product, overflow) = elements.multipliedReportingOverflow(by: d)
                guard !overflow else {
                    throw ModelFitError.corrupt(file: file, reason: "tensor '\(name)' has an impossible shape")
                }
                elements = product
            }
            let type = GGMLType(rawValue: try r.u32(ctx))
            let offset = try r.u64(ctx)
            raw.append(RawTensor(name: name, dims: dims, type: type, offset: offset, elements: elements))
        }

        var warnings: [String] = []
        var alignment: UInt64 = 32
        if let a = metadata["general.alignment"]?.intValue {
            if a > 0, a <= 1 << 20 { alignment = UInt64(a) } else {
                warnings.append("Ignored an invalid general.alignment of \(a); assumed 32.")
            }
        }
        let headerEnd = r.position
        let dataOffset = (headerEnd + alignment - 1) / alignment * alignment
        let dataAvailable = r.fileSize > dataOffset ? r.fileSize - dataOffset : 0

        // Sizes: exact from block layouts; offset gaps only for unknown types.
        let sortedOffsets = raw.map(\.offset).sorted()
        func gapSize(after offset: UInt64) -> UInt64 {
            var lo = 0, hi = sortedOffsets.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if sortedOffsets[mid] <= offset { lo = mid + 1 } else { hi = mid }
            }
            let next = lo < sortedOffsets.count ? sortedOffsets[lo] : max(dataAvailable, offset)
            return next > offset ? next - offset : 0
        }

        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(raw.count)
        var unknownTypes = Set<UInt32>()
        var dataEnd: UInt64 = 0
        for t in raw {
            let size: UInt64
            if t.type.isKnown {
                guard let exact = t.type.byteSize(elements: t.elements) else {
                    throw ModelFitError.corrupt(file: file, reason: "tensor '\(t.name)' has \(t.elements) elements, which doesn't fit \(t.type.name) blocks")
                }
                size = exact
            } else {
                unknownTypes.insert(t.type.rawValue)
                size = gapSize(after: t.offset)
            }
            let (end, overflow) = t.offset.addingReportingOverflow(size)
            guard !overflow else {
                throw ModelFitError.corrupt(file: file, reason: "tensor '\(t.name)' points past any possible file end")
            }
            dataEnd = max(dataEnd, end)
            tensors.append(GGUFTensorInfo(name: t.name, dimensions: t.dims, type: t.type, offset: t.offset, elementCount: t.elements, byteSize: size))
        }
        if !unknownTypes.isEmpty {
            let list = unknownTypes.sorted().map(String.init).joined(separator: ", ")
            warnings.append("Unknown tensor type \(list); those sizes were estimated from tensor offsets.")
        }
        let (describedEnd, endOverflow) = dataOffset.addingReportingOverflow(dataEnd)
        if endOverflow || describedEnd > r.fileSize, !tensors.isEmpty {
            warnings.append("The file is truncated: its header describes \(Format.bytes(Int64(clamping: endOverflow ? UInt64.max : describedEnd))) but only \(Format.bytes(Int64(clamping: r.fileSize))) are on disk. A runtime will refuse to load it.")
        }

        return GGUFFile(
            url: url, version: version, metadataKeys: keys, metadata: metadata, tensors: tensors,
            alignment: alignment, dataOffset: dataOffset, tensorDataSize: dataEnd,
            fileSize: r.fileSize, bytesRead: r.bytesRead, warnings: warnings
        )
    }
}
