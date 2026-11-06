import Foundation

/// Synthetic "sample header" models with the published architectures of well-known open
/// models. They are written as real files (header plus a sparse, empty data section) and
/// then read back through the normal parsers, so screenshots exercise the real pipeline.
/// Every sample is tagged `modelfit.sample` and the UI labels it as a sample.
public enum SampleModels {
    /// llama.cpp's `use_more_bits` rule: which layers Q4_K_M bumps to Q6_K.
    static func usesMoreBits(_ i: Int, _ n: Int) -> Bool {
        i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
    }

    /// Llama-family GGUF header with the Q4_K_M tensor-type mix.
    public static func llamaQ4KM(name: String, layers: Int, embedding: Int, heads: Int, kvHeads: Int,
                                 feedForward: Int, vocab: Int, context: Int, sizeLabel: String,
                                 tokenizerArrays: Bool = true) -> GGUFWriter {
        var w = GGUFWriter()
        let E = UInt64(embedding), F = UInt64(feedForward), V = UInt64(vocab)
        let headDim = embedding / heads
        let kvDim = UInt64(kvHeads * headDim)
        w.set("general.architecture", .string("llama"))
        w.set("general.type", .string("model"))
        w.set("general.name", .string(name))
        w.set("general.size_label", .string(sizeLabel))
        w.set("general.license", .string("llama3.1"))
        w.set("modelfit.sample", .bool(true))
        w.set("llama.block_count", .uint32(UInt32(layers)))
        w.set("llama.context_length", .uint32(UInt32(context)))
        w.set("llama.embedding_length", .uint32(UInt32(embedding)))
        w.set("llama.feed_forward_length", .uint32(UInt32(feedForward)))
        w.set("llama.attention.head_count", .uint32(UInt32(heads)))
        w.set("llama.attention.head_count_kv", .uint32(UInt32(kvHeads)))
        w.set("llama.rope.freq_base", .float32(500_000))
        w.set("llama.attention.layer_norm_rms_epsilon", .float32(1e-5))
        w.set("llama.rope.dimension_count", .uint32(UInt32(headDim)))
        w.set("llama.vocab_size", .uint32(UInt32(vocab)))
        w.set("general.file_type", .uint32(15))
        w.set("tokenizer.ggml.model", .string("gpt2"))
        w.set("tokenizer.ggml.pre", .string("llama-bpe"))
        if tokenizerArrays {
            let tokens = (0 ..< vocab).map { GGUFValue.string("t\($0)") }
            w.set("tokenizer.ggml.tokens", .array(GGUFArray(tokens, elementType: .string)))
            w.set("tokenizer.ggml.token_type", .array(GGUFArray(Array(repeating: .int32(1), count: vocab), elementType: .int32)))
            let merges = (0 ..< 280_147).map { GGUFValue.string("m \($0)") }
            w.set("tokenizer.ggml.merges", .array(GGUFArray(merges, elementType: .string)))
        }
        w.set("tokenizer.ggml.bos_token_id", .uint32(128_000))
        w.set("tokenizer.ggml.eos_token_id", .uint32(128_009))
        w.set("tokenizer.chat_template", .string("{# sample header: synthetic chat template #}{% for m in messages %}{{ m['content'] }}{% endfor %}"))
        w.set("general.quantization_version", .uint32(2))

        w.tensors.append(.init("token_embd.weight", [E, V], .q4_K))
        w.tensors.append(.init("rope_freqs.weight", [UInt64(headDim / 2)], .f32))
        for i in 0 ..< layers {
            let more = usesMoreBits(i, layers)
            w.tensors.append(.init("blk.\(i).attn_norm.weight", [E], .f32))
            w.tensors.append(.init("blk.\(i).attn_q.weight", [E, E], .q4_K))
            w.tensors.append(.init("blk.\(i).attn_k.weight", [E, kvDim], .q4_K))
            w.tensors.append(.init("blk.\(i).attn_v.weight", [E, kvDim], more ? .q6_K : .q4_K))
            w.tensors.append(.init("blk.\(i).attn_output.weight", [E, E], .q4_K))
            w.tensors.append(.init("blk.\(i).ffn_norm.weight", [E], .f32))
            w.tensors.append(.init("blk.\(i).ffn_gate.weight", [E, F], .q4_K))
            w.tensors.append(.init("blk.\(i).ffn_up.weight", [E, F], .q4_K))
            w.tensors.append(.init("blk.\(i).ffn_down.weight", [F, E], more ? .q6_K : .q4_K))
        }
        w.tensors.append(.init("output_norm.weight", [E], .f32))
        w.tensors.append(.init("output.weight", [E, V], .q6_K))
        return w
    }

    /// Llama 3.1 8B's published shape: 32 layers, 4096 wide, 32 heads, 8 KV heads, 128K context.
    public static func llama31_8B(tokenizerArrays: Bool = true) -> GGUFWriter {
        llamaQ4KM(name: "Llama 3.1 8B Instruct", layers: 32, embedding: 4096, heads: 32, kvHeads: 8,
                  feedForward: 14336, vocab: 128_256, context: 131_072, sizeLabel: "8B", tokenizerArrays: tokenizerArrays)
    }

    /// Llama 3.1 70B's published shape: 80 layers, 8192 wide, 64 heads, 8 KV heads.
    public static func llama31_70B() -> GGUFWriter {
        llamaQ4KM(name: "Llama 3.1 70B Instruct", layers: 80, embedding: 8192, heads: 64, kvHeads: 8,
                  feedForward: 28672, vocab: 128_256, context: 131_072, sizeLabel: "70B", tokenizerArrays: false)
    }

    /// Hugging Face Llama-style linear layer names for one decoder block.
    static func hfLayer(_ i: Int, hidden: Int64, kvDim: Int64, ffn: Int64, qkvBias: Bool) -> [(String, [Int64])] {
        let p = "model.layers.\(i)."
        var out: [(String, [Int64])] = [
            (p + "input_layernorm.weight", [hidden]),
            (p + "self_attn.q_proj.weight", [hidden, hidden]),
            (p + "self_attn.k_proj.weight", [kvDim, hidden]),
            (p + "self_attn.v_proj.weight", [kvDim, hidden]),
            (p + "self_attn.o_proj.weight", [hidden, hidden]),
            (p + "post_attention_layernorm.weight", [hidden]),
            (p + "mlp.gate_proj.weight", [ffn, hidden]),
            (p + "mlp.up_proj.weight", [ffn, hidden]),
            (p + "mlp.down_proj.weight", [hidden, ffn]),
        ]
        if qkvBias {
            out += [(p + "self_attn.q_proj.bias", [hidden]), (p + "self_attn.k_proj.bias", [kvDim]), (p + "self_attn.v_proj.bias", [kvDim])]
        }
        return out
    }

    static func writeIndex(_ shards: [(String, SafetensorsWriter)], to folder: URL) throws {
        var map: [String: String] = [:]
        var total: Int64 = 0
        for (file, w) in shards {
            for t in w.tensors {
                map[t.name] = file
                total += Int64(SafetensorsParser.dtypeSizes[t.dtype] ?? 1) * t.shape.reduce(1, *)
            }
        }
        let root: [String: Any] = ["metadata": ["total_size": total], "weight_map": map]
        try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
            .write(to: folder.appendingPathComponent("model.safetensors.index.json"))
    }

    static func writeJSON(_ object: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    }

    /// Mistral 7B v0.3's published shape as a BF16 Hugging Face folder with 3 shards.
    public static func writeMistral7B(to folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let hidden: Int64 = 4096, kvDim: Int64 = 1024, ffn: Int64 = 14336, vocab: Int64 = 32768
        var tensors: [[(String, [Int64])]] = [[], [], []]
        tensors[0].append(("model.embed_tokens.weight", [vocab, hidden]))
        for i in 0 ..< 32 {
            tensors[i < 11 ? 0 : (i < 22 ? 1 : 2)] += hfLayer(i, hidden: hidden, kvDim: kvDim, ffn: ffn, qkvBias: false)
        }
        tensors[2] += [("model.norm.weight", [hidden]), ("lm_head.weight", [vocab, hidden])]
        var shards: [(String, SafetensorsWriter)] = []
        for (n, list) in tensors.enumerated() {
            var w = SafetensorsWriter()
            w.metadata = ["format": "pt", "modelfit.sample": "true"]
            w.tensors = list.map { .init($0.0, "BF16", $0.1) }
            let file = String(format: "model-%05ld-of-%05ld.safetensors", n + 1, 3)
            try w.write(to: folder.appendingPathComponent(file))
            shards.append((file, w))
        }
        try writeIndex(shards, to: folder)
        try writeJSON([
            "architectures": ["MistralForCausalLM"], "model_type": "mistral",
            "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "max_position_embeddings": 32768, "vocab_size": 32768, "rope_theta": 1_000_000.0,
            "torch_dtype": "bfloat16", "sliding_window": NSNull(), "tie_word_embeddings": false,
        ], to: folder.appendingPathComponent("config.json"))
        try writeJSON(["tokenizer_class": "LlamaTokenizer", "chat_template": "{# sample header #}"],
                      to: folder.appendingPathComponent("tokenizer_config.json"))
    }

    /// Qwen2.5 14B's published shape as an mlx-lm 4-bit folder (group size 64).
    public static func writeQwen14BMLX(to folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let hidden: Int64 = 5120, kvDim: Int64 = 1024, ffn: Int64 = 13824, vocab: Int64 = 152_064
        let bits: Int64 = 4, group: Int64 = 64
        func quantized(_ name: String, _ shape: [Int64]) -> [SafetensorsWriter.Tensor] {
            let base = String(name.dropLast(7))
            let (out, inp) = (shape[0], shape[1])
            return [
                .init(name, "U32", [out, inp * bits / 32]),
                .init(base + ".scales", "BF16", [out, inp / group]),
                .init(base + ".biases", "BF16", [out, inp / group]),
            ]
        }
        func convert(_ list: [(String, [Int64])]) -> [SafetensorsWriter.Tensor] {
            list.flatMap { (entry: (String, [Int64])) -> [SafetensorsWriter.Tensor] in
                entry.1.count == 2 ? quantized(entry.0, entry.1) : [SafetensorsWriter.Tensor(entry.0, "BF16", entry.1)]
            }
        }
        var halves: [[(String, [Int64])]] = [[("model.embed_tokens.weight", [vocab, hidden])], []]
        for i in 0 ..< 48 {
            halves[i < 24 ? 0 : 1] += hfLayer(i, hidden: hidden, kvDim: kvDim, ffn: ffn, qkvBias: true)
        }
        halves[1] += [("model.norm.weight", [hidden]), ("lm_head.weight", [vocab, hidden])]
        var shards: [(String, SafetensorsWriter)] = []
        for (n, list) in halves.enumerated() {
            var w = SafetensorsWriter()
            w.metadata = ["format": "mlx", "modelfit.sample": "true"]
            w.tensors = convert(list)
            let file = String(format: "model-%05ld-of-%05ld.safetensors", n + 1, 2)
            try w.write(to: folder.appendingPathComponent(file))
            shards.append((file, w))
        }
        try writeIndex(shards, to: folder)
        try writeJSON([
            "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "hidden_size": 5120, "intermediate_size": 13824, "num_hidden_layers": 48,
            "num_attention_heads": 40, "num_key_value_heads": 8,
            "max_position_embeddings": 32768, "vocab_size": 152_064, "rope_theta": 1_000_000.0,
            "torch_dtype": "bfloat16", "tie_word_embeddings": false,
            "quantization": ["group_size": 64, "bits": 4],
        ], to: folder.appendingPathComponent("config.json"))
        try writeJSON(["tokenizer_class": "Qwen2Tokenizer", "chat_template": "{# sample header #}"],
                      to: folder.appendingPathComponent("tokenizer_config.json"))
    }

    /// Writes a GGUF as two llama.cpp-style split parts.
    public static func writeSplit(_ writer: GGUFWriter, prefix: String, in folder: URL) throws -> URL {
        let half = writer.tensors.count / 2
        var first = writer
        first.tensors = Array(writer.tensors[..<half])
        first.set("split.no", .uint16(0))
        first.set("split.count", .uint16(2))
        first.set("split.tensors.count", .int32(Int32(writer.tensors.count)))
        var second = GGUFWriter()
        second.set("split.no", .uint16(1))
        second.set("split.count", .uint16(2))
        second.set("split.tensors.count", .int32(Int32(writer.tensors.count)))
        second.tensors = Array(writer.tensors[half...])
        let a = folder.appendingPathComponent("\(prefix)-00001-of-00002.gguf")
        try first.write(to: a)
        try second.write(to: folder.appendingPathComponent("\(prefix)-00002-of-00002.gguf"))
        return a
    }

    /// Writes every sample into `folder` and returns the URLs to open, in display order.
    public static func writeAll(to folder: URL) throws -> [URL] {
        let fm = FileManager.default
        try? fm.removeItem(at: folder)
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        let llama8 = folder.appendingPathComponent("Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        try llama31_8B().write(to: llama8)
        let qwen = folder.appendingPathComponent("Qwen2.5-14B-Instruct-4bit")
        try writeQwen14BMLX(to: qwen)
        let mistral = folder.appendingPathComponent("Mistral-7B-Instruct-v0.3")
        try writeMistral7B(to: mistral)
        let seventy = try writeSplit(llama31_70B(), prefix: "Llama-3.1-70B-Instruct-Q4_K_M", in: folder)
        return [llama8, qwen, mistral, seventy]
    }

    /// A GGUF cut off in the middle of its metadata, for the error-state screenshot.
    public static func writeTruncated(to url: URL) throws {
        let (data, _) = llama31_8B(tokenizerArrays: false).header()
        try data.prefix(600).write(to: url)
    }
}
