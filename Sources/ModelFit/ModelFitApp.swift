import AppKit
import SwiftUI
import ModelFitCore

@main
enum ModelFitMain {
    @MainActor
    static func main() {
        let args = CommandLine.arguments
        if let flag = args.firstIndex(of: "--render-assets") {
            let path = args.count > flag + 1 ? args[flag + 1] : "renders"
            exit(AssetRenderer.run(output: URL(fileURLWithPath: path, isDirectory: true)))
        }
        ModelFitApp.main()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, open urls: [URL]) {
        ModelStore.shared.open(urls)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct ModelFitApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = ModelStore.shared
    @AppStorage(PrefKey.kvCache) private var kvRaw = KVCacheType.f16.rawValue

    var body: some Scene {
        Window("ModelFit", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 920, minHeight: 620)
        }
        .defaultSize(width: 1240, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Model…") { store.presentOpenPanel() }
                    .keyboardShortcut("o")
            }
            CommandMenu("Model") {
                Button("Compare Models") { store.selection = .compare }
                    .keyboardShortcut("k")
                    .disabled(store.analyses.count < 2)
                Picker("KV Cache Precision", selection: Binding(
                    get: { KVCacheType(rawValue: kvRaw) ?? .f16 },
                    set: { kvRaw = $0.rawValue }
                )) {
                    ForEach(KVCacheType.allCases) { Text($0.label).tag($0) }
                }
                Divider()
                Button("Show in Finder") { store.revealSelected() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(store.selectedItem == nil)
                Button("Remove from List") { store.removeSelected() }
                    .keyboardShortcut(.delete, modifiers: [.command])
                    .disabled(store.selectedItem == nil)
            }
        }

        Settings {
            SettingsView()
        }
    }
}
