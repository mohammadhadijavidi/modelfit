import SwiftUI
import ModelFitCore

/// What a sidebar row shows, independent of List so the renderer can draw it too.
struct SidebarRowModel: Identifiable {
    enum Status { case loading, failed, loaded(FitVerdict) }
    let id: UUID
    var name: String
    var subtitle: String
    var format: ModelFormat?
    var status: Status

    /// The verdict at 8K context (or the model's maximum if smaller) labels the row.
    static let referenceContext = 8192

    init(item: ModelItem, inputs: FitInputs) {
        id = item.id
        name = item.name
        switch item.state {
        case .loading:
            subtitle = "Reading header…"
            status = .loading
        case .failed:
            subtitle = "Couldn't read this file"
            status = .failed
        case let .loaded(a):
            subtitle = "\(a.quantizationLabel), \(Format.bytes(a.weightBytes))"
            format = a.format
            let ctx = min(Self.referenceContext, a.maxContext ?? Self.referenceContext)
            status = .loaded(MemoryEstimator.estimate(a, context: ctx, settings: inputs.settings, profile: inputs.profile).verdict)
        }
    }
}

struct SidebarRow: View {
    let model: SidebarRowModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        HStack(spacing: 10) {
            Group {
                switch model.status {
                case .loading:
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(theme.inset)
                        .overlay(Image(systemName: "hourglass").font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary))
                case .failed:
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(theme.wontFit.opacity(0.15))
                        .overlay(Image(systemName: "exclamationmark").font(.system(size: 12, weight: .bold)).foregroundStyle(theme.wontFit))
                case .loaded:
                    FormatBadge(format: model.format ?? .gguf, size: 26)
                }
            }
            .frame(width: 26, height: 26)

            VStack(alignment: .leading, spacing: 1) {
                Text(model.name)
                    .font(.system(size: 12.5, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(model.subtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 4)
            if case let .loaded(verdict) = model.status {
                Circle()
                    .fill(theme.color(for: verdict))
                    .frame(width: 7, height: 7)
                    .help("\(verdict.label) at \(Format.context(SidebarRowModel.referenceContext)) context")
                    .accessibilityLabel("\(verdict.label) at 8K context")
            }
        }
        .padding(.vertical, 3)
    }
}

/// "This Mac" card at the bottom of the sidebar.
struct MacFooter: View {
    let profile: MacProfile
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "laptopcomputer")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(profile.isSample ? "Sample Mac" : "This Mac")
                    .font(.system(size: 11.5, weight: .semibold))
                Spacer(minLength: 0)
            }
            if !profile.isSample {
                Text(profile.chipName)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(Format.bytesParts(profile.physicalMemory).0).font(Typo.number(18))
                Text("GB memory").font(.system(size: 11)).foregroundStyle(.secondary)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(theme.track)
                    Capsule().fill(theme.weights.opacity(0.85))
                        .frame(width: geo.size.width * CGFloat(Double(profile.gpuWorkingSet) / Double(max(1, profile.physicalMemory))))
                }
            }
            .frame(height: 5)
            Text("GPU limit \(Format.bytes(profile.gpuWorkingSet)), \(profile.gpuLimitSource.label)")
                .font(.system(size: 10.5))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(theme.inset, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

struct SidebarView: View {
    @Bindable var store: ModelStore
    let inputs: FitInputs

    var body: some View {
        List(selection: $store.selection) {
            if store.analyses.count >= 2 {
                Section {
                    Label("Compare", systemImage: "chart.bar.xaxis")
                        .tag(SidebarSelection.compare)
                }
            }
            Section("Models") {
                if store.items.isEmpty {
                    Text("Drop .gguf, .safetensors or model folders here.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(.secondary)
                        .selectionDisabled()
                }
                ForEach(store.items) { item in
                    SidebarRow(model: SidebarRowModel(item: item, inputs: inputs))
                        .tag(SidebarSelection.model(item.id))
                        .contextMenu {
                            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                            Divider()
                            Button("Remove from List") { store.remove(item.id) }
                        }
                }
            }
        }
        .listStyle(.sidebar)
        .onDeleteCommand { store.removeSelected() }
        .safeAreaInset(edge: .bottom) {
            MacFooter(profile: inputs.profile).padding(10)
        }
    }
}
