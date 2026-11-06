import SwiftUI
import ModelFitCore

/// The full analysis of one model. Used by the app's detail column and, unchanged,
/// by the screenshot renderer.
struct DetailContent: View {
    let analysis: ModelAnalysis
    let inputs: FitInputs
    @Binding var kvType: KVCacheType
    @Binding var showCalculation: Bool

    var body: some View {
        let estimates = MemoryEstimator.ladder(analysis, settings: inputs.settings, profile: inputs.profile)
        let summary = FitSummary(estimates: estimates, profile: inputs.profile)
        VStack(alignment: .leading, spacing: 22) {
            ModelHeader(analysis: analysis)
            if !analysis.warnings.isEmpty { WarningsBanner(warnings: analysis.warnings) }
            FitCard(analysis: analysis, estimates: estimates, summary: summary, inputs: inputs, kvType: $kvType)
            HStack(alignment: .top, spacing: 20) {
                ArchitectureCard(analysis: analysis, kvType: inputs.settings.kvCacheType)
                QuantizationCard(analysis: analysis)
            }
            HStack(alignment: .top, spacing: 20) {
                TokenizerCard(analysis: analysis)
                FileCard(analysis: analysis)
            }
            CalculationCard(analysis: analysis, estimates: estimates, inputs: inputs, expanded: $showCalculation)
        }
    }
}

struct FitCard: View {
    let analysis: ModelAnalysis
    let estimates: [MemoryEstimate]
    let summary: FitSummary
    let inputs: FitInputs
    @Binding var kvType: KVCacheType

    var body: some View {
        Card(padding: 22) {
            VStack(alignment: .leading, spacing: 22) {
                HStack(alignment: .top, spacing: 16) {
                    VerdictHeadline(summary: summary)
                    Spacer(minLength: 12)
                    VStack(alignment: .trailing, spacing: 6) {
                        Text("KV cache precision")
                            .font(Typo.label)
                            .foregroundStyle(.secondary)
                        PillPicker(options: KVCacheType.allCases, selection: $kvType, label: \.label,
                                   accessibilityTitle: "KV cache precision")
                    }
                }
                FitLadderView(estimates: estimates, profile: inputs.profile)
                HStack(alignment: .firstTextBaseline) {
                    MemoryLegend(kvLabel: inputs.settings.kvCacheType.label)
                    Spacer(minLength: 12)
                    Text(inputs.profile.isSample ? "Estimates for a sample 16 GB Mac" : "Estimates, not measurements")
                        .font(Typo.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}

struct WarningsBanner: View {
    let warnings: [String]
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        VStack(alignment: .leading, spacing: 6) {
            ForEach(warnings, id: \.self) { w in
                Label {
                    Text(w).fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(theme.tight)
                }
                .font(Typo.body)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(theme.tight.opacity(0.10), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(theme.tight.opacity(0.25)))
    }
}

struct ArchitectureCard: View {
    let analysis: ModelAnalysis
    let kvType: KVCacheType

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                CardTitle("Architecture", symbol: "square.stack.3d.down.right")
                FactGrid(facts: facts)
                Text(analysis.architecture.source.label)
                    .font(Typo.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    var facts: [ModelFact] {
        let a = analysis.architecture
        var f: [ModelFact] = []
        func add(_ label: String, _ value: Int?, suffix: String = "") {
            if let value { f.append(.init(label, Format.integer(value) + suffix)) }
        }
        f.append(.init("Architecture", a.architecture ?? "Unknown"))
        f.append(.init("Parameters", Format.integer(analysis.parameterCount)))
        add("Layers", a.layerCount)
        add("Embedding width", a.embeddingLength)
        add("Attention heads", a.headCount)
        if let kv = a.kvHeadCount {
            var text = Format.integer(kv)
            if let h = a.headCount, kv > 0, h % kv == 0, h / kv > 1 { text += " (GQA \(h / kv):1)" }
            f.append(.init("KV heads", text))
        } else if let per = a.kvHeadsPerLayer {
            f.append(.init("KV heads", "\(per.min() ?? 0)–\(per.max() ?? 0) per layer"))
        }
        if a.kvLoraRank == nil { add("Head dimension", a.headDimension) }
        add("Latent KV rank", a.kvLoraRank)
        add("Feed-forward width", a.feedForwardLength)
        add("Context length", a.contextLength, suffix: " tokens")
        add("Vocabulary", a.vocabSize)
        if let r = a.ropeFreqBase {
            f.append(.init("RoPE base", r == r.rounded() && r < 1e15 ? Format.integer(Int64(r)) : Format.fixed(r, decimals: 1)))
        }
        if let e = a.expertCount {
            f.append(.init("Experts", a.expertUsedCount.map { "\(e), \($0) active per token" } ?? "\(e)"))
        }
        add("Sliding window", a.slidingWindow, suffix: " tokens")
        if let shape = a.kvCacheShape {
            let bytes = Double(shape.elementsPerToken) * kvType.bytesPerElementValue
            f.append(.init("KV cache per token", "\(Format.bytes(Int64(bytes))) (\(kvType.label))"))
        }
        return f
    }
}

struct QuantizationCard: View {
    let analysis: ModelAnalysis
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        let mix = Array(analysis.typeMix.prefix(5))
        let total = Double(max(1, analysis.weightBytes))
        Card {
            VStack(alignment: .leading, spacing: 14) {
                CardTitle("Quantization", symbol: "dial.medium")
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(analysis.quantizationLabel)
                        .font(Typo.number(24))
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    if let bpw = analysis.bitsPerWeight {
                        Text("\(Format.fixed(bpw, decimals: 2)) bits per weight")
                            .font(Typo.body)
                            .foregroundStyle(.secondary)
                    }
                }
                GeometryReader { geo in
                    HStack(spacing: 2) {
                        ForEach(Array(mix.enumerated()), id: \.element.id) { index, share in
                            Rectangle()
                                .fill(theme.mixColor(index))
                                .frame(width: max(1.5, (geo.size.width - CGFloat(mix.count - 1) * 2) * CGFloat(Double(share.bytes) / total)))
                        }
                    }
                    .frame(width: geo.size.width, alignment: .leading)
                    .clipShape(Capsule())
                }
                .frame(height: 10)
                .accessibilityHidden(true)

                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                    ForEach(Array(mix.enumerated()), id: \.element.id) { index, share in
                        GridRow {
                            HStack(spacing: 7) {
                                RoundedRectangle(cornerRadius: 2.5).fill(theme.mixColor(index)).frame(width: 10, height: 10)
                                Text(share.name).font(.system(size: 13, weight: .medium))
                            }
                            Text("\(Format.integer(share.tensorCount)) tensors")
                                .font(Typo.caption).foregroundStyle(.secondary)
                            Text(Format.bytes(share.bytes))
                                .font(.system(size: 12.5).monospacedDigit())
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            Text(Format.percent(Double(share.bytes) / total))
                                .font(.system(size: 12.5, weight: .semibold).monospacedDigit())
                                .frame(width: 40, alignment: .trailing)
                        }
                    }
                }
                if analysis.typeMix.count > mix.count {
                    Text("Plus \(analysis.typeMix.count - mix.count) smaller types")
                        .font(Typo.caption).foregroundStyle(.tertiary)
                }
            }
        }
    }
}

struct TokenizerCard: View {
    let analysis: ModelAnalysis

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                CardTitle("Tokenizer", symbol: "textformat.abc")
                if let t = analysis.tokenizer {
                    FactGrid(facts: facts(t))
                } else {
                    Text("This model's files don't include tokenizer metadata.")
                        .font(Typo.body).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func facts(_ t: TokenizerInfo) -> [ModelFact] {
        var f: [ModelFact] = []
        if let m = t.model { f.append(.init("Model", m)) }
        if let p = t.preTokenizer { f.append(.init("Pre-tokenizer", p)) }
        if let n = t.tokenCount { f.append(.init("Tokens", Format.integer(n))) }
        if let n = t.mergeCount { f.append(.init("Merges", Format.integer(n))) }
        if let b = t.bosTokenID { f.append(.init("BOS token ID", "\(b)")) }
        if let e = t.eosTokenID { f.append(.init("EOS token ID", "\(e)")) }
        f.append(.init("Chat template", t.hasChatTemplate ? "Included" : "Not included"))
        return f
    }
}

struct FileCard: View {
    let analysis: ModelAnalysis

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                CardTitle("Files", symbol: "doc.on.doc")
                FactGrid(facts: facts)
            }
        }
    }

    var facts: [ModelFact] {
        let a = analysis
        var f: [ModelFact] = [
            .init("Format", a.formatDetail),
            .init("On disk", Format.bytes(a.fileSize)),
            .init("Tensor data", Format.bytes(a.weightBytes)),
            .init("Tensors", Format.integer(a.tensorCount)),
            .init("Header bytes read", Format.bytes(a.headerBytesRead)),
        ]
        f.append(contentsOf: a.facts)
        return f
    }
}

/// "How we calculate", expandable, with this model's numbers in each formula.
struct CalculationCard: View {
    let analysis: ModelAnalysis
    let estimates: [MemoryEstimate]
    let inputs: FitInputs
    @Binding var expanded: Bool
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var reference: MemoryEstimate? {
        estimates.first { $0.contextLength == 8192 } ?? estimates.first
    }

    var body: some View {
        let theme = Theme(scheme)
        Card(padding: 0) {
            VStack(alignment: .leading, spacing: 0) {
                Button {
                    if reduceMotion { expanded.toggle() } else {
                        withAnimation(.snappy(duration: 0.25)) { expanded.toggle() }
                    }
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "function")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text("How we calculate").font(Typo.title)
                        if let r = reference {
                            Text("at \(Format.context(r.contextLength)) context")
                                .font(Typo.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(.secondary)
                            .rotationEffect(.degrees(expanded ? 90 : 0))
                    }
                    .padding(20)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(expanded ? "Hide how we calculate" : "Show how we calculate")

                if expanded, let r = reference {
                    Rectangle().fill(theme.hairline).frame(height: 1)
                    VStack(alignment: .leading, spacing: 18) {
                        Text("Total = weights + KV cache + compute buffers + runtime. Every number below is an estimate from the file's header; nothing is loaded or run.")
                            .font(Typo.body)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        ForEach(CalculationExplainer.steps(for: analysis, at: r, settings: inputs.settings, profile: inputs.profile)) { step in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(step.title).font(.system(size: 13, weight: .semibold))
                                Text(step.formula)
                                    .font(Typo.formula)
                                    .textSelection(.enabled)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 7)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(theme.inset, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                                Text(step.explanation)
                                    .font(Typo.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                    }
                    .padding(20)
                    .transition(.opacity)
                }
            }
        }
    }
}
