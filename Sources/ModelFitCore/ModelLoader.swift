import Foundation

/// Entry point: turns any dropped URL into one or more analyses.
public enum ModelLoader {
    /// Analyze a file or folder. A folder of GGUF files yields one analysis per model
    /// (split parts are grouped); a safetensors folder yields one.
    /// Reads headers only; safe to call off the main thread.
    public static func analyze(url: URL) throws -> [ModelAnalysis] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw ModelFitError.unreadable(file: url.lastPathComponent, reason: "the file doesn't exist")
        }
        if isDirectory.boolValue { return try analyzeFolder(url) }

        let name = url.lastPathComponent.lowercased()
        if name.hasSuffix(".safetensors.index.json") || name == "config.json" {
            return try analyzeFolder(url.deletingLastPathComponent())
        }
        if name.hasSuffix(".gguf") { return [try GGUFAnalyzer.analyze(url: url)] }
        if name.hasSuffix(".safetensors") { return [try FolderAnalyzer.analyze(file: url)] }

        switch try sniff(url) {
        case .gguf: return [try GGUFAnalyzer.analyze(url: url)]
        case .safetensors: return [try FolderAnalyzer.analyze(file: url)]
        default: throw ModelFitError.notAModel(file: url.lastPathComponent)
        }
    }

    static func analyzeFolder(_ folder: URL) throws -> [ModelAnalysis] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path))?.filter { !$0.hasPrefix(".") } ?? []
        if names.contains(where: { $0.hasSuffix(".safetensors") || $0.hasSuffix(".safetensors.index.json") }) {
            return [try FolderAnalyzer.analyze(folder: folder)]
        }
        let ggufs = names.filter { $0.lowercased().hasSuffix(".gguf") }.sorted()
        guard !ggufs.isEmpty else { throw ModelFitError.emptyFolder(folder: folder.lastPathComponent) }
        var results: [ModelAnalysis] = []
        var seenSplits = Set<String>()
        for g in ggufs {
            let url = folder.appendingPathComponent(g)
            if let split = GGUFAnalyzer.splitParts(of: url), split.count > 1 {
                guard seenSplits.insert(split.prefix).inserted else { continue }
            }
            results.append(try GGUFAnalyzer.analyze(url: url))
        }
        return results
    }

    /// Detect the format from the first bytes, for files without a telling extension.
    public static func sniff(_ url: URL) throws -> ModelFormat? {
        let r = try ByteReader(url: url, chunkSize: 4096)
        guard r.fileSize >= 9 else { return nil }
        let head = try r.bytes(9, "the file header")
        if head[0 ..< 4] == [0x47, 0x47, 0x55, 0x46] { return .gguf }
        var length: UInt64 = 0
        for i in 0 ..< 8 { length |= UInt64(head[i]) << UInt64(8 * i) }
        if length > 1, length <= SafetensorsParser.maxHeader, head[8] == UInt8(ascii: "{") { return .safetensors }
        return nil
    }
}
