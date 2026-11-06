import SwiftUI
import ModelFitCore

/// A grouped surface. Radius is larger for primary content than for nested insets.
struct Card<Content: View>: View {
    var padding: CGFloat = 20
    @ViewBuilder var content: Content
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(theme.card, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(theme.cardStroke))
            .shadow(color: .black.opacity(theme.isDark ? 0.25 : 0.04), radius: 8, y: 2)
    }
}

/// Small title for a card, with an optional trailing accessory.
struct CardTitle<Trailing: View>: View {
    let title: String
    var symbol: String?
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if let symbol {
                Image(systemName: symbol)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            Text(title).font(Typo.title)
            Spacer(minLength: 8)
            trailing
        }
    }
}

extension CardTitle where Trailing == EmptyView {
    init(_ title: String, symbol: String? = nil) {
        self.init(title: title, symbol: symbol) { EmptyView() }
    }
}

/// Rounded-square format glyph used in the sidebar and headers.
struct FormatBadge: View {
    let format: ModelFormat
    var size: CGFloat = 28
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let color = Theme(scheme).color(for: format)
        RoundedRectangle(cornerRadius: size * 0.28, style: .continuous)
            .fill(LinearGradient(colors: [color.opacity(0.95), color.opacity(0.75)], startPoint: .top, endPoint: .bottom))
            .frame(width: size, height: size)
            .overlay(
                Image(systemName: format.symbolName)
                    .font(.system(size: size * 0.46, weight: .semibold))
                    .foregroundStyle(.white)
            )
            .accessibilityLabel("\(format.displayName) model")
    }
}

/// A fact chip in a model header: icon plus value.
struct Chip: View {
    let symbol: String
    let text: String
    var tint: Color?
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        HStack(spacing: 5) {
            Image(systemName: symbol)
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(tint ?? .secondary)
            Text(text)
                .font(.system(size: 12, weight: .medium))
                .monospacedDigit()
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 4.5)
        .background(tint.map { $0.opacity(0.14) } ?? theme.inset, in: Capsule())
        .fixedSize()
    }
}

struct VerdictPill: View {
    let verdict: FitVerdict
    var compact = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let color = Theme(scheme).color(for: verdict)
        HStack(spacing: 4) {
            Image(systemName: verdict.symbolName)
                .font(.system(size: compact ? 9.5 : 10.5, weight: .bold))
            Text(compact ? verdict.shortLabel : verdict.label)
                .font(.system(size: compact ? 11 : 12, weight: .semibold))
        }
        .foregroundStyle(color)
        .padding(.horizontal, compact ? 7 : 9)
        .padding(.vertical, compact ? 3 : 4)
        .background(color.opacity(scheme == .dark ? 0.16 : 0.12), in: Capsule())
        .fixedSize()
        .accessibilityElement(children: .combine)
        .accessibilityLabel(verdict.label)
    }
}

struct LegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                .fill(color)
                .frame(width: 10, height: 10)
            Text(label)
                .font(Typo.caption)
                .foregroundStyle(.secondary)
        }
        .fixedSize()
    }
}

/// A segmented control drawn in SwiftUI (so it renders offscreen too), with a sliding
/// selection that respects Reduce Motion.
struct PillPicker<Value: Hashable>: View {
    let options: [Value]
    @Binding var selection: Value
    let label: (Value) -> String
    var accessibilityTitle: String
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Namespace private var namespace

    var body: some View {
        let theme = Theme(scheme)
        HStack(spacing: 2) {
            ForEach(options, id: \.self) { option in
                let selected = option == selection
                Button {
                    if reduceMotion { selection = option } else {
                        withAnimation(.spring(response: 0.32, dampingFraction: 0.86)) { selection = option }
                    }
                } label: {
                    Text(label(option))
                        .font(.system(size: 11.5, weight: selected ? .semibold : .medium))
                        .monospacedDigit()
                        .foregroundStyle(selected ? Color.primary : Color.secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background {
                            if selected {
                                Capsule()
                                    .fill(theme.isDark ? Color.white.opacity(0.14) : Color.white)
                                    .shadow(color: .black.opacity(theme.isDark ? 0 : 0.12), radius: 1.5, y: 0.5)
                                    .matchedGeometryEffect(id: "selection", in: namespace)
                            }
                        }
                        .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(label(option))
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(2)
        .background(theme.inset, in: Capsule())
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilityTitle)
    }
}

/// Label/value rows laid out in two columns.
struct FactGrid: View {
    let facts: [ModelFact]

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 9) {
            ForEach(facts) { fact in
                GridRow {
                    Text(fact.label)
                        .font(Typo.body)
                        .foregroundStyle(.secondary)
                    Text(fact.value)
                        .font(.system(size: 13, weight: .medium))
                        .monospacedDigit()
                        .textSelection(.enabled)
                        .lineLimit(2)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
        }
    }
}

/// "This Mac" summary used by the sidebar footer and Quick Look.
struct MacSummaryLine: View {
    let profile: MacProfile

    var body: some View {
        let name = profile.isSample ? "Sample Mac" : profile.chipName
        Text("\(name), \(Format.bytes(profile.physicalMemory)) memory, GPU limit \(Format.bytes(profile.gpuWorkingSet))")
    }
}

/// Wraps children onto new lines when they don't fit, like text.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6
    var lineSpacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, lineHeight: CGFloat = 0, widest: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > 0, x + size.width > maxWidth {
                y += lineHeight + lineSpacing
                x = 0
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
            widest = max(widest, x - spacing)
        }
        return CGSize(width: proposal.width ?? widest, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, lineHeight: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                y += lineHeight + lineSpacing
                x = bounds.minX
                lineHeight = 0
            }
            view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
