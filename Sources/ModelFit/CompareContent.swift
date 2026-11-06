import SwiftUI
import ModelFitCore

/// Every open model against this Mac at one context length, plus a verdict table.
struct CompareContent: View {
    let models: [ModelAnalysis]
    let inputs: FitInputs
    @Binding var context: Int
    @Environment(\.colorScheme) private var scheme

    private let nameWidth: CGFloat = 230
    private let trailingWidth: CGFloat = 190
    private let spacing: CGFloat = 16

    private func estimate(_ m: ModelAnalysis) -> MemoryEstimate {
        MemoryEstimator.estimate(m, context: min(context, m.maxContext ?? context), settings: inputs.settings, profile: inputs.profile)
    }

    var body: some View {
        let estimates = models.map(estimate)
        let scaleMax = max(Double(inputs.profile.physicalMemory), Double(estimates.map(\.total).max() ?? 0)) * 1.05
        VStack(alignment: .leading, spacing: 22) {
            HStack(alignment: .bottom) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Compare").font(Typo.display)
                    Text("\(models.count) models against \(inputs.profile.isSample ? "a sample" : "this Mac's") \(Format.bytes(inputs.profile.physicalMemory)) of memory")
                        .font(Typo.body)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    Text("Context length").font(Typo.label).foregroundStyle(.secondary)
                    PillPicker(options: MemoryEstimator.standardContexts, selection: $context, label: Format.context,
                               accessibilityTitle: "Context length")
                }
            }

            Card(padding: 22) {
                VStack(alignment: .leading, spacing: 16) {
                    CardTitle(title: "Memory at \(Format.context(context)) context", symbol: "chart.bar.xaxis") {
                        Text(inputs.settings.kvCacheType.label + " KV cache")
                            .font(Typo.caption).foregroundStyle(.secondary)
                    }
                    LimitLabels(profile: inputs.profile, scaleMax: scaleMax,
                                leadingInset: nameWidth + spacing, trailingInset: trailingWidth + spacing)
                    VStack(spacing: 16) {
                        ForEach(models.indices, id: \.self) { i in
                            compareRow(models[i], estimates[i], scaleMax: scaleMax)
                        }
                    }
                    .overlay(LimitLines(profile: inputs.profile, scaleMax: scaleMax,
                                        leadingInset: nameWidth + spacing, trailingInset: trailingWidth + spacing))
                    MemoryLegend(kvLabel: inputs.settings.kvCacheType.label)
                }
            }

            Card(padding: 22) {
                VStack(alignment: .leading, spacing: 14) {
                    CardTitle("Side by side", symbol: "tablecells")
                    table
                }
            }
        }
    }

    private func compareRow(_ m: ModelAnalysis, _ e: MemoryEstimate, scaleMax: Double) -> some View {
        let (number, unit) = Format.bytesParts(e.total)
        return HStack(spacing: spacing) {
            HStack(spacing: 10) {
                FormatBadge(format: m.format, size: 26)
                VStack(alignment: .leading, spacing: 1) {
                    Text(m.displayName).font(.system(size: 12.5, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                    Text(e.contextLength < context ? "\(m.quantizationLabel), max \(Format.context(e.contextLength))" : m.quantizationLabel)
                        .font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            .frame(width: nameWidth, alignment: .leading)
            MemoryBar(estimate: e, scaleMax: scaleMax, height: 18)
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                HStack(alignment: .firstTextBaseline, spacing: 2) {
                    Text(number).font(Typo.number(15))
                    Text(unit).font(Typo.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                VerdictPill(verdict: e.verdict)
            }
            .frame(width: trailingWidth)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(m.displayName): about \(Format.bytes(e.total)). \(e.verdict.label).")
    }

    private var table: some View {
        let theme = Theme(scheme)
        let contexts = MemoryEstimator.standardContexts
        return Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
            GridRow {
                ForEach(["Model", "Params", "Weights", "KV per 1K", "Max context"], id: \.self) { header in
                    Text(header).font(Typo.label).foregroundStyle(.secondary)
                }
                ForEach(contexts, id: \.self) { c in
                    Text(Format.context(c)).font(Typo.label).foregroundStyle(.secondary)
                        .gridColumnAlignment(.center)
                }
            }
            Rectangle().fill(theme.hairline).frame(height: 1).gridCellUnsizedAxes(.horizontal)
            ForEach(models) { m in
                GridRow {
                    Text(m.displayName).font(.system(size: 12.5, weight: .medium)).lineLimit(1).truncationMode(.middle)
                        .frame(maxWidth: 230, alignment: .leading)
                    Text(Format.parameters(m.parameterCount)).font(Typo.number(12.5, weight: .medium))
                    Text(Format.bytes(m.weightBytes)).font(Typo.number(12.5, weight: .medium))
                    Text(kvPer1K(m)).font(Typo.number(12.5, weight: .medium))
                    Text(m.maxContext.map(Format.context) ?? "Unknown").font(Typo.number(12.5, weight: .medium))
                    ForEach(contexts, id: \.self) { c in
                        if let max = m.maxContext, c > max {
                            Text("Beyond max").font(Typo.caption).foregroundStyle(.tertiary)
                        } else {
                            let v = MemoryEstimator.estimate(m, context: c, settings: inputs.settings, profile: inputs.profile).verdict
                            Image(systemName: v.symbolName)
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(theme.color(for: v))
                                .help(v.label)
                                .accessibilityLabel("\(Format.context(c)): \(v.label)")
                        }
                    }
                }
            }
        }
    }

    private func kvPer1K(_ m: ModelAnalysis) -> String {
        guard let shape = m.architecture.kvCacheShape else { return "n/a" }
        return Format.bytes(MemoryEstimator.kvCacheBytes(shape: shape, context: 1024, type: inputs.settings.kvCacheType))
    }
}
