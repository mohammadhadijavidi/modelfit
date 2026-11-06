import Foundation

enum GGUFAnalyzer {
    /// Matches llama.cpp split names: "Model-Q4_K_M-00001-of-00003.gguf".
    static func splitParts(of url: URL) -> (prefix: String, index: Int, count: Int)? {
        let name = url.lastPathComponent
        guard name.lowercased().hasSuffix(".gguf") else { return nil }
        let stem = String(name.dropLast(5))
        let parts = stem.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count >= 4,
              parts[parts.count - 2] == "of",
              let index = Int(parts[parts.count - 3]), let count = Int(parts[parts.count - 1]),
              parts[parts.count - 3].count == 5, parts[parts.count - 1].count == 5,
              index >= 1, count >= index
        else { return nil }
        return (parts.dropLast(3).joined(separator: "-"), index, count)
    }

    static func partURL(prefix: String, index: Int, count: Int, in directory: URL) -> URL {
        directory.appendingPathComponent(String(format: "%@-%05ld-of-%05ld.gguf", prefix, index, count))
    }

    static func analyze(url: URL) throws -> ModelAnalysis {
        var files: [GGUFFile] = []
        var warnings: [String] = []
        var primaryURL = url

        if let split = splitParts(of: url), split.count > 1 {
            let directory = url.deletingLastPathComponent()
            primaryURL = partURL(prefix: split.prefix, index: 1, count: split.count, in: directory)
            var missing: [Int] = []
            for i in 1 ... split.count {
                let part = partURL(prefix: split.prefix, index: i, count: split.count, in: directory)
                if part.standardizedFileURL == url.standardizedFileURL {
                    files.append(try GGUFParser.parse(url: part))
                } else if let parsed = try? GGUFParser.parse(url: part) {
                    files.append(parsed)
                } else {
                    missing.append(i)
                }
            }
            if !missing.isEmpty {
                let list = missing.map(String.init).joined(separator: ", ")
                warnings.append("Split model: part \(list) of \(split.count) couldn't be read, so totals cover only the parts found.")
            }
            files.sort { ($0["split.no"]?.intValue ?? 0) < ($1["split.no"]?.intValue ?? 0) }
        } else {
            files = [try GGUFParser.parse(url: url)]
        }

        // Architecture metadata lives in the first split; others only carry split.* keys.
        guard let main = files.first(where: { $0["general.architecture"] != nil }) ?? files.first else {
            throw ModelFitError.notAModel(file: url.lastPathComponent)
        }
        for f in files { warnings.append(contentsOf: f.warnings) }

        let md = main.metadata
        let arch = md["general.architecture"]?.stringValue
        func key(_ suffix: String) -> GGUFValue? {
            guard let arch else { return nil }
            return md["\(arch).\(suffix)"]
        }

        var info = ArchitectureInfo()
        info.architecture = arch
        info.layerCount = key("block_count")?.intValue
        info.embeddingLength = key("embedding_length")?.intValue
        info.headCount = key("attention.head_count")?.intList?.max()
        if let kv = key("attention.head_count_kv")?.intList {
            if kv.count > 1, Set(kv).count > 1 { info.kvHeadsPerLayer = kv } else { info.kvHeadCount = kv.first }
        }
        info.keyLength = key("attention.key_length")?.intValue
        info.valueLength = key("attention.value_length")?.intValue
        info.feedForwardLength = key("feed_forward_length")?.intList?.max()
        info.contextLength = key("context_length")?.intValue
        info.ropeFreqBase = key("rope.freq_base")?.doubleValue
        info.expertCount = key("expert_count")?.intValue
        info.expertUsedCount = key("expert_used_count")?.intValue
        info.slidingWindow = key("attention.sliding_window")?.intValue
        info.kvLoraRank = key("attention.kv_lora_rank")?.intValue
        info.ropeDimension = key("rope.dimension_count")?.intValue
        let tokens = md["tokenizer.ggml.tokens"]?.arrayValue
        info.vocabSize = key("vocab_size")?.intValue ?? tokens.map { Int($0.count) }
        info.source = arch == nil ? .unavailable : .ggufMetadata
        if arch == nil { warnings.append("This GGUF has no general.architecture key, so the KV cache can't be estimated.") }
        if info.isRecurrent { warnings.append("\(arch ?? "This") is a recurrent architecture: its state doesn't grow with context, so the KV cache is not modelled.") }
        if info.isHybrid { warnings.append("\(arch ?? "This") mixes attention and recurrent layers; the KV cache shown assumes every layer is attention, an upper bound.") }

        var tokenizer: TokenizerInfo?
        if md["tokenizer.ggml.model"] != nil || tokens != nil {
            var t = TokenizerInfo()
            t.model = md["tokenizer.ggml.model"]?.stringValue
            t.preTokenizer = md["tokenizer.ggml.pre"]?.stringValue
            t.tokenCount = tokens.map { Int($0.count) }
            t.mergeCount = md["tokenizer.ggml.merges"]?.arrayValue.map { Int($0.count) }
            t.bosTokenID = md["tokenizer.ggml.bos_token_id"]?.intValue
            t.eosTokenID = md["tokenizer.ggml.eos_token_id"]?.intValue
            t.hasChatTemplate = md["tokenizer.chat_template"]?.isNonEmptyString ?? false
            tokenizer = t
        }

        var shares: [TensorTypeShare] = []
        var params: Int64 = 0
        var weightBytes: Int64 = 0
        var tensorCount = 0
        for f in files {
            for t in f.tensors {
                let elements = Int64(clamping: t.elementCount)
                let bytes = Int64(clamping: t.byteSize)
                params = params.addingSaturating(elements)
                weightBytes = weightBytes.addingSaturating(bytes)
                tensorCount += 1
                shares.append(TensorTypeShare(name: t.type.name, tensorCount: 1, parameters: elements, bytes: bytes))
            }
        }
        let mix = [TensorTypeShare].combined(shares)

        let fileType = md["general.file_type"]?.intValue
        let quant = fileType.flatMap(GGUFFileType.name(for:)) ?? mix.first?.name ?? "Unknown"

        var facts: [ModelFact] = []
        if let s = md["general.size_label"]?.stringValue { facts.append(.init("Size label", s)) }
        if let s = md["general.organization"]?.stringValue { facts.append(.init("Organization", s)) }
        if let s = md["general.license"]?.stringValue { facts.append(.init("License", s)) }
        if let s = md["general.quantized_by"]?.stringValue { facts.append(.init("Quantized by", s)) }
        facts.append(.init("Metadata keys", Format.integer(main.metadataKeys.count)))
        facts.append(.init("Alignment", "\(main.alignment) bytes"))

        var name = md["general.name"]?.stringValue ?? ""
        if name.trimmingCharacters(in: .whitespaces).isEmpty {
            if let split = splitParts(of: primaryURL) { name = split.prefix } else {
                name = primaryURL.deletingPathExtension().lastPathComponent
            }
        }

        let detail = files.count > 1 ? "GGUF v\(main.version), \(files.count) parts" : "GGUF v\(main.version)"
        return ModelAnalysis(
            url: primaryURL,
            displayName: name,
            format: .gguf,
            formatDetail: detail,
            fileSize: files.reduce(0) { $0 + Int64(clamping: $1.fileSize) },
            weightBytes: weightBytes,
            parameterCount: params,
            tensorCount: tensorCount,
            quantizationLabel: quant,
            typeMix: mix,
            architecture: info,
            tokenizer: tokenizer,
            facts: facts,
            fileCount: files.count,
            headerBytesRead: files.reduce(0) { $0 + Int64(clamping: $1.bytesRead) },
            warnings: warnings,
            isSample: md["modelfit.sample"]?.boolValue ?? false
        )
    }
}
