import SwiftUI
import ModelFitCore

/// Estimation inputs derived from preferences, passed explicitly to every screen so the
/// screenshot renderer can supply a fixed sample Mac instead.
struct FitInputs {
    var profile: MacProfile
    var settings: EstimationSettings
}

struct ContentView: View {
    @Bindable var store: ModelStore
    @AppStorage(PrefKey.kvCache) private var kvRaw = KVCacheType.f16.rawValue
    @AppStorage(PrefKey.customGPULimit) private var customGPU = false
    @AppStorage(PrefKey.gpuPercent) private var gpuPercent = 75.0
    @AppStorage(PrefKey.overheadMB) private var overheadMB = 512
    @State private var compareContext = 8192

    private var kvBinding: Binding<KVCacheType> {
        Binding(get: { KVCacheType(rawValue: kvRaw) ?? .f16 }, set: { kvRaw = $0.rawValue })
    }

    private var inputs: FitInputs {
        FitInputs(
            profile: .current(policy: customGPU ? .fraction(gpuPercent / 100) : .automatic),
            settings: EstimationSettings(kvCacheType: kvBinding.wrappedValue, runtimeOverheadBytes: Int64(max(0, overheadMB)) << 20)
        )
    }

    var body: some View {
        let inputs = inputs
        NavigationSplitView {
            SidebarView(store: store, inputs: inputs)
                .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 340)
        } detail: {
            detail(inputs)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    store.presentOpenPanel()
                } label: {
                    Label("Open Model", systemImage: "plus")
                }
                .help("Open a model file or folder (⌘O)")
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            store.open(urls)
            return !urls.isEmpty
        } isTargeted: { store.isDropTargeted = $0 }
        .overlay {
            if store.isDropTargeted { DropOverlay().transition(.opacity) }
        }
        .animation(.easeOut(duration: 0.15), value: store.isDropTargeted)
    }

    @ViewBuilder
    private func detail(_ inputs: FitInputs) -> some View {
        switch store.selection {
        case .compare?:
            ScrollView {
                CompareContent(models: store.analyses, inputs: inputs, context: $compareContext)
                    .padding(28)
                    .frame(maxWidth: 1100)
                    .frame(maxWidth: .infinity)
            }
            .background(CanvasBackground())
            .navigationTitle("Compare")
            .navigationSubtitle("\(store.analyses.count) models")
        case let .model(id)?:
            if let item = store.items.first(where: { $0.id == id }) {
                ModelScreen(item: item, inputs: inputs, kvType: kvBinding, store: store)
            } else {
                EmptyStateView(profile: inputs.profile) { store.presentOpenPanel() }
            }
        case nil:
            EmptyStateView(profile: inputs.profile) { store.presentOpenPanel() }
                .background(CanvasBackground())
        }
    }
}

struct CanvasBackground: View {
    @Environment(\.colorScheme) private var scheme
    var body: some View { Theme(scheme).canvas.ignoresSafeArea() }
}

/// The detail column for one model, in whatever state it's in.
struct ModelScreen: View {
    let item: ModelItem
    let inputs: FitInputs
    @Binding var kvType: KVCacheType
    let store: ModelStore
    @State private var showCalculation = false

    var body: some View {
        Group {
            switch item.state {
            case .loading:
                VStack(spacing: 12) {
                    ProgressView().controlSize(.large)
                    Text("Reading the header of \(item.name)…")
                        .font(Typo.body)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case let .failed(error):
                ErrorScreen(error: error, fileName: item.url.lastPathComponent) {
                    NSWorkspace.shared.activateFileViewerSelecting([item.url])
                } remove: {
                    store.remove(item.id)
                }
            case let .loaded(analysis):
                ScrollView {
                    DetailContent(analysis: analysis, inputs: inputs, kvType: $kvType, showCalculation: $showCalculation)
                        .padding(28)
                        .frame(maxWidth: 1040)
                        .frame(maxWidth: .infinity)
                }
                .toolbar {
                    ToolbarItemGroup {
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([analysis.url])
                        } label: {
                            Label("Show in Finder", systemImage: "folder")
                        }
                        .help("Show in Finder (⇧⌘R)")
                        Button {
                            store.remove(item.id)
                        } label: {
                            Label("Remove from List", systemImage: "minus.circle")
                        }
                        .help("Remove from the list (⌘⌫). The file is not touched.")
                    }
                }
            }
        }
        .background(CanvasBackground())
        .navigationTitle(item.name)
        .navigationSubtitle(item.analysis?.formatDetail ?? "")
    }
}

struct ErrorScreen: View {
    let error: ModelFitError
    let fileName: String
    var reveal: () -> Void
    var remove: () -> Void

    var body: some View {
        VStack(spacing: 4) {
            ModelErrorView(error: error, fileName: fileName)
            HStack(spacing: 10) {
                Button("Show in Finder", action: reveal).buttonStyle(QuietButtonStyle())
                Button("Remove from List", action: remove).buttonStyle(QuietButtonStyle())
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct DropOverlay: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let accent = Theme(scheme).weights
        ZStack {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(accent.opacity(0.08))
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(accent, style: StrokeStyle(lineWidth: 2, dash: [9, 7]))
            Label("Drop to analyze", systemImage: "arrow.down.circle.fill")
                .font(Typo.headline)
                .foregroundStyle(accent)
        }
        .padding(14)
        .allowsHitTesting(false)
    }
}

/// A filled capsule button drawn in SwiftUI (renders identically offscreen).
struct ProminentButtonStyle: ButtonStyle {
    @Environment(\.colorScheme) private var scheme

    func makeBody(configuration: Configuration) -> some View {
        let accent = Theme(scheme).weights
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 8)
            .background(accent.opacity(configuration.isPressed ? 0.8 : 1), in: Capsule())
            .contentShape(Capsule())
    }
}

struct QuietButtonStyle: ButtonStyle {
    @Environment(\.colorScheme) private var scheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12.5, weight: .medium))
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(Theme(scheme).inset.opacity(configuration.isPressed ? 2 : 1), in: Capsule())
            .overlay(Capsule().strokeBorder(Theme(scheme).hairline))
            .contentShape(Capsule())
    }
}
