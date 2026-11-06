import Foundation

/// A forward-only, chunk-buffered little-endian reader over a file.
///
/// It never maps or loads a whole file: it reads at most `chunkSize` bytes at a time,
/// and `skip` simply moves the cursor, so strings or arrays we don't need are never read.
/// Every read is bounds-checked against the real file size and throws `.truncated`
/// instead of trapping.
final class ByteReader {
    let fileName: String
    let fileSize: UInt64
    private let handle: FileHandle
    private let chunkSize: Int
    private var buffer: [UInt8] = []
    private var bufferStart: UInt64 = 0
    private(set) var position: UInt64 = 0
    /// Bytes actually pulled from disk. Used to prove we only read headers.
    private(set) var bytesRead: UInt64 = 0

    init(url: URL, chunkSize: Int = 1 << 20) throws {
        fileName = url.lastPathComponent
        do {
            handle = try FileHandle(forReadingFrom: url)
            fileSize = try handle.seekToEnd()
            try handle.seek(toOffset: 0)
        } catch {
            throw ModelFitError.unreadable(file: url.lastPathComponent, reason: error.localizedDescription)
        }
        self.chunkSize = max(4096, chunkSize)
    }

    deinit { try? handle.close() }

    var remaining: UInt64 { position >= fileSize ? 0 : fileSize - position }

    private func truncated(_ context: String) -> ModelFitError {
        .truncated(file: fileName, context: context, offset: position)
    }

    private func ensure(_ count: Int, _ context: @autoclosure () -> String) throws {
        guard count > 0 else { return }
        guard UInt64(count) <= remaining else { throw truncated(context()) }
        let bufferEnd = bufferStart + UInt64(buffer.count)
        if position >= bufferStart, position + UInt64(count) <= bufferEnd { return }
        let data: Data
        do {
            try handle.seek(toOffset: position)
            data = try handle.read(upToCount: max(count, chunkSize)) ?? Data()
        } catch {
            throw ModelFitError.unreadable(file: fileName, reason: error.localizedDescription)
        }
        bytesRead += UInt64(data.count)
        buffer = [UInt8](data)
        bufferStart = position
        guard buffer.count >= count else { throw truncated(context()) }
    }

    func bytes(_ count: Int, _ context: @autoclosure () -> String) throws -> [UInt8] {
        try ensure(count, context())
        let start = Int(position - bufferStart)
        position += UInt64(count)
        return Array(buffer[start ..< start + count])
    }

    func skip(_ count: UInt64, _ context: @autoclosure () -> String) throws {
        guard count <= remaining else { throw truncated(context()) }
        position += count
    }

    private func unsigned(_ size: Int, _ context: @autoclosure () -> String) throws -> UInt64 {
        try ensure(size, context())
        let start = Int(position - bufferStart)
        var value: UInt64 = 0
        for i in 0 ..< size {
            value |= UInt64(buffer[start + i]) << UInt64(8 * i)
        }
        position += UInt64(size)
        return value
    }

    func u8(_ context: @autoclosure () -> String) throws -> UInt8 { UInt8(try unsigned(1, context())) }
    func u16(_ context: @autoclosure () -> String) throws -> UInt16 { UInt16(try unsigned(2, context())) }
    func u32(_ context: @autoclosure () -> String) throws -> UInt32 { UInt32(try unsigned(4, context())) }
    func u64(_ context: @autoclosure () -> String) throws -> UInt64 { try unsigned(8, context()) }
    func i8(_ context: @autoclosure () -> String) throws -> Int8 { Int8(bitPattern: try u8(context())) }
    func i16(_ context: @autoclosure () -> String) throws -> Int16 { Int16(bitPattern: try u16(context())) }
    func i32(_ context: @autoclosure () -> String) throws -> Int32 { Int32(bitPattern: try u32(context())) }
    func i64(_ context: @autoclosure () -> String) throws -> Int64 { Int64(bitPattern: try u64(context())) }
    func f32(_ context: @autoclosure () -> String) throws -> Float { Float(bitPattern: try u32(context())) }
    func f64(_ context: @autoclosure () -> String) throws -> Double { Double(bitPattern: try u64(context())) }
}
