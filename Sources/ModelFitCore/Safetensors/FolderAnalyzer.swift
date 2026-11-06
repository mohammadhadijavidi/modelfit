import Foundation

/// Quantization declared in config.json.
enum QuantizationConfig: Equatable {
    /// mlx-lm's `"quantization": {"group_size": 64, "bits": 4, "<module>": {...}}`.
    case mlx(bits: Int, groupSize: Int, overrides: [String: Int])
    /// transformers' `quantization_config` (GPTQ, AWQ, bitsandbytes, FP8...).
    case huggingFace(method: String, bits: Int?)

    func mlxBits(for weightName: String) -> Int? {
        guard case let .mlx(bits, _, overrides) = self else { return nil }
        let module = weightName.hasSuffix(".weight") ? String(weightName.dropLast(7)) : weightName
        return overrides[module] ?? bits
    }
}

struct ParsedConfig {
    var architecture = ArchitectureInfo()
    var quantization: QuantizationConfig?
    var torchDtype: String?
    var modelType: String?
}

enum ConfigParser {
    static func int(_ d: [String: Any], _ keys: String...) -> Int? {
        for k in keys {
            if let n = d[k] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() { return n.intValue }
        }
        return nil
    }

    static func parse(data: Data) -> ParsedConfig? {
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        var out = ParsedConfig()
        out.modelType = root["model_type"] as? String
        // Multimodal configs nest the language model under text_config / llm_config.
        var d = root
        if int(root, "num_hidden_layers", "n_layer", "n_layers") == nil {
            for nested in ["text_config", "llm_config", "language_config"] {
                if let t = root[nested] as? [String: Any] {
                    d = t
                    break
                }
            }
        }
        var a = ArchitectureInfo()
        a.architecture = (d["model_type"] as? String) ?? out.modelType ?? (root["architectures"] as? [String])?.first
        a.layerCount = int(d, "num_hidden_layers", "n_layer", "n_layers")
        a.embeddingLength = int(d, "hidden_size", "n_embd", "d_model")
        a.headCount = int(d, "num_attention_heads", "n_head", "n_heads")
        a.kvHeadCount = int(d, "num_key_value_heads", "n_head_kv", "num_kv_heads") ?? a.headCount
        a.keyLength = int(d, "head_dim")
        a.feedForwardLength = int(d, "intermediate_size", "n_inner", "ffn_dim")
        a.contextLength = int(d, "max_position_embeddings", "n_positions", "max_seq_len", "seq_length")
        a.vocabSize = int(d, "vocab_size")
        a.ropeFreqBase = (d["rope_theta"] as? NSNumber)?.doubleValue
        a.expertCount = int(d, "num_local_experts", "num_experts", "n_routed_experts")
        a.expertUsedCount = int(d, "num_experts_per_tok", "moe_top_k")
        a.slidingWindow = int(d, "sliding_window")
        a.kvLoraRank = int(d, "kv_lora_rank")
        a.ropeDimension = int(d, "qk_rope_head_dim")
        if a.kvLoraRank != nil {
            a.keyLength = nil
        }
        a.source = .configJSON
        out.architecture = a
        out.torchDtype = (d["torch_dtype"] as? String) ?? (root["torch_dtype"] as? String) ?? (root["dtype"] as? String)

        if let q = (root["quantization"] as? [String: Any]) ?? (root["quantization_config"] as? [String: Any]),
           let bits = int(q, "bits"), let group = int(q, "group_size"), q["quant_method"] == nil {
            var overrides: [String: Int] = [:]
            for (k, v) in q {
                if let m = v as? [String: Any], let b = int(m, "bits") { overrides[k] = b }
            }
            out.quantization = .mlx(bits: bits, groupSize: group, overrides: overrides)
        } else if let q = root["quantization_config"] as? [String: Any] {
            let method = (q["quant_method"] as? String) ?? "quantized"
            out.quantization = .huggingFace(method: method, bits: int(q, "bits", "weight_bits"))
        }
        return out
    }
}

enum TensorNameInference {
    static func layerIndex(in name: String) -> Int? {
        let parts = name.split(separator: ".")
        for (i, p) in parts.enumerated() where ["layers", "h", "blocks", "layer"].contains(String(p)) {
            if i + 1 < parts.count, let n = Int(parts[i + 1]) { return n }
        }
        return nil
    }

    /// Architecture from tensor names/shapes when there is no config.json.
    /// Works for Llama-family naming (`model.layers.N.self_attn.k_proj.weight`).
    static func infer(_ tensors: [SafetensorsTensor]) -> ArchitectureInfo? {
        var maxLayer = -1
        var kWidth: Int?, vWidth: Int?
        var a = ArchitectureInfo()
        for t in tensors {
            if let i = layerIndex(in: t.name) { maxLayer = max(maxLayer, i) }
            if layerIndex(in: t.name) == 0, let rows = t.shape.first {
                if t.name.hasSuffix("k_proj.weight") { kWidth = Int(rows) }
                if t.name.hasSuffix("v_proj.weight") { vWidth = Int(rows) }
            }
            if t.name.hasSuffix("embed_tokens.weight") || t.name.hasSuffix("wte.weight"), t.shape.count == 2 {
                a.vocabSize = Int(t.shape[0])
                if t.dtype != "U32" { a.embeddingLength = Int(t.shape[1]) }
            }
        }
        guard maxLayer >= 0 else { return nil }
        a.layerCount = maxLayer + 1
        if let k = kWidth, let v = vWidth { a.inferredKVWidthPerLayer = k + v }
        a.source = .inferredFromTensors
        return a
    }
}

enum FolderAnalyzer {
    static func readSmallJSON(_ url: URL, limit: Int = 8 << 20) -> Data? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attrs[.size] as? NSNumber)?.intValue, size <= limit
        else { return nil }
        return try? Data(contentsOf: url)
    }

    /// Analyze a Hugging Face / MLX model folder.
    static func analyze(folder: URL) throws -> ModelAnalysis {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: folder.path))?.filter { !$0.hasPrefix(".") }.sorted() ?? []
        var warnings: [String] = []

        var shardNames: [String]
        if let indexName = names.first(where: { $0.hasSuffix(".safetensors.index.json") }),
           let data = readSmallJSON(folder.appendingPathComponent(indexName)),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let map = root["weight_map"] as? [String: String] {
            shardNames = Array(Set(map.values)).sorted()
        } else {
            shardNames = names.filter { $0.hasSuffix(".safetensors") }
        }
        guard !shardNames.isEmpty else { throw ModelFitError.emptyFolder(folder: folder.lastPathComponent) }

        var files: [SafetensorsFile] = []
        var missing: [String] = []
        for shard in shardNames {
            let url = folder.appendingPathComponent(shard)
            if fm.fileExists(atPath: url.path) {
                files.append(try SafetensorsParser.parse(url: url))
            } else {
                missing.append(shard)
            }
        }
        if !missing.isEmpty {
            warnings.append("\(missing.count) shard\(missing.count == 1 ? "" : "s") listed in the index \(missing.count == 1 ? "is" : "are") missing: \(missing.prefix(3).joined(separator: ", ")). Totals cover the shards present.")
        }
        guard !files.isEmpty else { throw ModelFitError.emptyFolder(folder: folder.lastPathComponent) }

        let config = readSmallJSON(folder.appendingPathComponent("config.json")).flatMap(ConfigParser.parse(data:))
        if config == nil { warnings.append("No readable config.json: architecture was inferred from tensor names.") }

        var tokenizer: TokenizerInfo?
        if let data = readSmallJSON(folder.appendingPathComponent("tokenizer_config.json")),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            var t = TokenizerInfo()
            t.model = root["tokenizer_class"] as? String
            t.hasChatTemplate = root["chat_template"] != nil || names.contains("chat_template.jinja") || names.contains("chat_template.json")
            t.tokenCount = config?.architecture.vocabSize
            tokenizer = t
        }

        let isMLX: Bool = {
            if case .mlx = config?.quantization { return true }
            return files.contains { $0.metadata["format"] == "mlx" }
        }()
        return build(
            url: folder, displayName: folder.lastPathComponent, files: files, config: config,
            format: isMLX ? .mlxFolder : .huggingFaceFolder, tokenizer: tokenizer, warnings: warnings,
            totalFiles: names.count
        )
    }

    /// Analyze one .safetensors file. If it's a shard of an indexed folder, or a
    /// config.json sits beside it, use that context (when readable: Quick Look's
    /// sandbox only grants the previewed file).
    static func analyze(file: URL) throws -> ModelAnalysis {
        let folder = file.deletingLastPathComponent()
        let fm = FileManager.default
        if let names = try? fm.contentsOfDirectory(atPath: folder.path),
           let indexName = names.first(where: { $0.hasSuffix(".safetensors.index.json") }),
           let data = readSmallJSON(folder.appendingPathComponent(indexName)),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let map = root["weight_map"] as? [String: String],
           Set(map.values).count > 1, Set(map.values).contains(file.lastPathComponent),
           var whole = try? analyze(folder: folder) {
            whole.warnings.insert("\(file.lastPathComponent) is one shard of this model, so the whole folder was analyzed.", at: 0)
            return whole
        }
        let parsed = try SafetensorsParser.parse(url: file)
        let config = readSmallJSON(folder.appendingPathComponent("config.json")).flatMap(ConfigParser.parse(data:))
        var warnings: [String] = []
        if config != nil { warnings.append("Architecture taken from the config.json beside this file.") }
        let isMLX: Bool = {
            if case .mlx = config?.quantization { return true }
            return parsed.metadata["format"] == "mlx"
        }()
        return build(
            url: file, displayName: file.deletingPathExtension().lastPathComponent, files: [parsed],
            config: config, format: isMLX ? .mlxFolder : .safetensors, tokenizer: nil, warnings: warnings,
            totalFiles: 1, singleFile: true
        )
    }

    static func build(url: URL, displayName: String, files: [SafetensorsFile], config: ParsedConfig?,
                      format: ModelFormat, tokenizer: TokenizerInfo?, warnings initial: [String],
                      totalFiles: Int, singleFile: Bool = false) -> ModelAnalysis {
        var warnings = initial
        for f in files { warnings.append(contentsOf: f.warnings) }
        let all = files.flatMap(\.tensors)
        let names = Set(all.map(\.name))
        let quant = config?.quantization

        // MLX packs quantized weights into U32 words; a sibling ".scales" marks them.
        func isPackedWeight(_ t: SafetensorsTensor) -> Bool {
            guard t.dtype == "U32", t.name.hasSuffix(".weight") else { return false }
            return names.contains(String(t.name.dropLast(7)) + ".scales")
        }
        let packed = Set(all.filter(isPackedWeight).map(\.name))
        let mlxQuantized = !packed.isEmpty

        var params: Int64 = 0
        var shares: [TensorTypeShare] = []
        var bitsSeen = Set<Int>()
        for t in all {
            let bytes = Int64(clamping: t.byteSize)
            if t.name.hasSuffix(".scales") || t.name.hasSuffix(".biases"),
               packed.contains(String(t.name.dropLast(7)) + ".weight") {
                shares.append(.init(name: "Scales and biases", tensorCount: 1, parameters: 0, bytes: bytes))
                continue
            }
            if packed.contains(t.name) {
                let bits = quant?.mlxBits(for: t.name) ?? 4
                bitsSeen.insert(bits)
                let logical = t.elementCount * 32 / Int64(max(1, bits))
                params += logical
                shares.append(.init(name: "\(bits)-bit", tensorCount: 1, parameters: logical, bytes: bytes))
            } else {
                params += t.elementCount
                shares.append(.init(name: t.dtype, tensorCount: 1, parameters: t.elementCount, bytes: bytes))
            }
        }
        let mix = [TensorTypeShare].combined(shares)
        let weightBytes = all.reduce(Int64(0)) { $0 + Int64(clamping: $1.byteSize) }

        var label: String
        switch quant {
        case let .mlx(bits, group, _)?:
            label = bitsSeen.count > 1 ? "Mixed \(bitsSeen.sorted().map(String.init).joined(separator: "/"))-bit, group \(group)" : "\(bitsSeen.first ?? bits)-bit, group \(group)"
        case let .huggingFace(method, bits)?:
            label = method.uppercased() + (bits.map { " \($0)-bit" } ?? "")
            warnings.append("Parameters are counted from stored shapes; packed \(method.uppercased()) weights make that count low. Memory figures use real byte sizes and are unaffected.")
        case nil:
            if mlxQuantized { label = "\(bitsSeen.sorted().map(String.init).joined(separator: "/"))-bit (assumed)" } else {
                label = mix.first?.name ?? "Unknown"
            }
        }

        var arch = config?.architecture ?? TensorNameInference.infer(all) ?? ArchitectureInfo()
        if config == nil, arch.source == .unavailable {
            warnings.append("Couldn't infer layers or KV width from tensor names; only weights are estimated.")
        }
        if arch.vocabSize == nil { arch.vocabSize = TensorNameInference.infer(all)?.vocabSize }
        if arch.isRecurrent { warnings.append("Recurrent architecture: the KV cache is not modelled.") }

        var facts: [ModelFact] = []
        if let d = config?.torchDtype { facts.append(.init("Declared dtype", d)) }
        if let m = config?.modelType, m != arch.architecture { facts.append(.init("Model type", m)) }
        if let f = files.first?.metadata["format"] { facts.append(.init("Stored by", f)) }
        facts.append(.init("Files in folder", Format.integer(totalFiles)))

        let detail = singleFile ? "safetensors file" : "\(files.count) safetensors shard\(files.count == 1 ? "" : "s")"
        return ModelAnalysis(
            url: url, displayName: displayName, format: format, formatDetail: detail,
            fileSize: files.reduce(0) { $0 + Int64(clamping: $1.fileSize) },
            weightBytes: weightBytes, parameterCount: params, tensorCount: all.count,
            quantizationLabel: label, typeMix: mix, architecture: arch, tokenizer: tokenizer,
            facts: facts, fileCount: files.count,
            headerBytesRead: files.reduce(0) { $0 + Int64(clamping: $1.bytesRead) },
            warnings: warnings,
            isSample: files.contains { $0.metadata["modelfit.sample"] == "true" }
        )
    }
}
