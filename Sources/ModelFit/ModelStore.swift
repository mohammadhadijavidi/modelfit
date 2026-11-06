import AppKit
import Observation
import SwiftUI
import ModelFitCore

/// A model in the sidebar: loading, analyzed, or failed.
struct ModelItem: Identifiable, Equatable {
    enum State: Equatable {
        case loading
        case loaded(ModelAnalysis)
        case failed(ModelFitError)
    }

    let id: UUID
    var url: URL
    var name: String
    var state: State

    var analysis: ModelAnalysis? {
        if case let .loaded(a) = state { return a }
        return nil
    }
}

enum SidebarSelection: Hashable {
    case compare
    case model(UUID)
}

/// Preference keys shared by the app, the settings window and the menu bar commands.
enum PrefKey {
    static let kvCache = "kvCacheType"
    static let customGPULimit = "useCustomGPULimit"
    static let gpuPercent = "gpuLimitPercent"
    static let overheadMB = "runtimeOverheadMB"
}

@MainActor
@Observable
final class ModelStore {
    static let shared = ModelStore()

    var items: [ModelItem] = []
    var selection: SidebarSelection?
    var isDropTargeted = false

    var analyses: [ModelAnalysis] { items.compactMap(\.analysis) }

    var selectedItem: ModelItem? {
        guard case let .model(id)? = selection else { return nil }
        return items.first { $0.id == id }
    }

    /// Adds models and analyzes them off the main thread. Re-opening a model selects it.
    func open(_ urls: [URL]) {
        for url in urls {
            let key = url.standardizedFileURL.path
            if let existing = items.first(where: { $0.url.standardizedFileURL.path == key }) {
                selection = .model(existing.id)
                continue
            }
            let id = UUID()
            items.append(ModelItem(id: id, url: url, name: url.lastPathComponent, state: .loading))
            selection = .model(id)
            Task { await load(id: id, url: url) }
        }
    }

    private func load(id: UUID, url: URL) async {
        let result = await Task.detached(priority: .userInitiated) { () -> Result<[ModelAnalysis], ModelFitError> in
            do {
                return .success(try ModelLoader.analyze(url: url))
            } catch let error as ModelFitError {
                return .failure(error)
            } catch {
                return .failure(.unreadable(file: url.lastPathComponent, reason: error.localizedDescription))
            }
        }.value
        apply(result, to: id)
    }

    /// Applies a finished analysis. Also used synchronously by the screenshot renderer.
    func apply(_ result: Result<[ModelAnalysis], ModelFitError>, to id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        switch result {
        case let .failure(error):
            items[index].state = .failed(error)
        case let .success(list):
            guard let first = list.first else {
                items[index].state = .failed(.notAModel(file: items[index].url.lastPathComponent))
                return
            }
            // Opening a second part of a split model resolves to one already open.
            if let duplicate = items.first(where: { $0.id != id && $0.analysis?.url.standardizedFileURL == first.url.standardizedFileURL }) {
                items.remove(at: index)
                selection = .model(duplicate.id)
                return
            }
            items[index].state = .loaded(first)
            items[index].name = first.displayName
            items[index].url = first.url
            for extra in list.dropFirst() where !items.contains(where: { $0.url.standardizedFileURL == extra.url.standardizedFileURL }) {
                items.append(ModelItem(id: UUID(), url: extra.url, name: extra.displayName, state: .loaded(extra)))
            }
        }
    }

    func remove(_ id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items.remove(at: index)
        if selection == .model(id) {
            if items.isEmpty {
                selection = nil
            } else {
                selection = .model(items[min(index, items.count - 1)].id)
            }
        }
        if selection == .compare, analyses.count < 2 { selection = items.first.map { .model($0.id) } }
    }

    func removeSelected() {
        if let item = selectedItem { remove(item.id) }
    }

    func revealSelected() {
        if let item = selectedItem { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
    }

    func presentOpenPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open a Model"
        panel.message = "Choose .gguf or .safetensors files, or an MLX / Hugging Face model folder."
        panel.prompt = "Analyze"
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        if panel.runModal() == .OK { open(panel.urls) }
    }
}
