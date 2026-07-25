// hf-resume progress window — a pure reader of the downloader's state JSON.
//
// Build:  swiftc -O Window.swift -o hf-resume-window
// Run:    hf-resume-window /path/to/state.json
//
// This process owns no download. Closing it stops nothing; the daemon keeps going.
//
// WHY AppKit instead of a SwiftUI `App`/`Window` scene (measured, not preference):
// in a bare non-bundled binary, a SwiftUI Scene renders fine while the App struct holds only
// value `@State`, but as soon as the App owns a model object — `@StateObject` on an
// ObservableObject, or `@State` on an `@Observable` class, with or without `@MainActor` — the
// process launches, idles in a healthy event loop, and creates NO window at all (0 windows in
// CGWindowListCopyWindowInfo, no crash, nothing in the unified log). Creating the NSWindow
// explicitly and hosting the SwiftUI tree in an NSHostingView works every time.
// Consequence: this file is top-level code, so it must be compiled WITHOUT -parse-as-library.

import AppKit
import SwiftUI

// MARK: - State model (mirrors downloader.py; every field tolerant of schema drift)

struct FileRow: Codable, Identifiable, Equatable {
    var path: String
    var size: Int64?
    var done: Int64?
    var state: String?
    var speed_bps: Double?
    var resumed_from: Int64?

    var id: String { path }
    var bytes: Int64 { done ?? 0 }
    var total: Int64 { size ?? 0 }
    var fraction: Double { total > 0 ? min(1.0, Double(bytes) / Double(total)) : 0 }
    var status: String { state ?? "queued" }
}

struct Snapshot: Codable, Equatable {
    var repo_id: String?
    var repo_type: String?
    var dest: String?
    var status: String?
    var error: String?
    var notes: [String]?
    var attempt: Int?
    var total_bytes: Int64?
    var done_bytes: Int64?
    var resumed_bytes: Int64?
    var speed_bps: Double?
    var eta_seconds: Double?
    var files: [FileRow]?

    var fraction: Double {
        let t = total_bytes ?? 0
        return t > 0 ? min(1.0, Double(done_bytes ?? 0) / Double(t)) : 0
    }
}

// MARK: - Formatting

func humanBytes(_ n: Int64) -> String {
    let x = Double(n)
    if x >= 1e9 { return String(format: "%.2f GB", x / 1e9) }
    if x >= 1e6 { return String(format: "%.1f MB", x / 1e6) }
    if x >= 1e3 { return String(format: "%.0f KB", x / 1e3) }
    return "\(n) B"
}

func humanSpeed(_ bps: Double) -> String {
    if bps >= 1e6 { return String(format: "%.1f MB/s", bps / 1e6) }
    if bps >= 1e3 { return String(format: "%.0f KB/s", bps / 1e3) }
    return bps > 0 ? String(format: "%.0f B/s", bps) : "—"
}

func humanETA(_ s: Double?) -> String {
    guard let s, s.isFinite, s > 0 else { return "—" }
    let t = Int(s.rounded())
    if t >= 3600 { return "\(t / 3600)h \((t % 3600) / 60)m" }
    if t >= 60 { return "\(t / 60)m \(t % 60)s" }
    return "\(t)s"
}

// MARK: - Poller

final class Model: ObservableObject {
    @Published var snap = Snapshot()
    @Published var missing = true
    @Published var showDone = false

    weak var window: NSWindow?          // set by the delegate, used only for the title
    private let url: URL
    private var timer: Timer?

    init(path: String) {
        self.url = URL(fileURLWithPath: path)
        reload()
        // 0.35s matches the downloader's publish cadence; the file is a few KB.
        let t = Timer(timeInterval: 0.35, repeats: true) { [weak self] _ in self?.reload() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func reload() {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(Snapshot.self, from: data)
        else { missing = true; return }
        missing = false
        if decoded != snap { snap = decoded }
        if let repo = decoded.repo_id {
            window?.title = "\(repo) — \(Int((decoded.fraction * 100).rounded()))%"
        }
    }

    var active: [FileRow] {
        (snap.files ?? []).filter { $0.status == "active" }.sorted { $0.fraction > $1.fraction }
    }
    var queued: [FileRow] {
        (snap.files ?? []).filter { $0.status == "queued" }.sorted { $0.total > $1.total }
    }
    var done: [FileRow] { (snap.files ?? []).filter { $0.status == "done" } }
}

// MARK: - Views

struct StatusPill: View {
    let status: String
    var color: Color {
        switch status {
        case "done": return .green
        case "error": return .red
        case "retrying": return .orange
        case "stopped": return .gray
        default: return .accentColor
        }
    }
    var body: some View {
        Text(status.uppercased())
            .font(.system(size: 10, weight: .bold))
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}

struct FileRowView: View {
    let row: FileRow
    let showBar: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(row.path)
                    .font(.system(size: 11, design: .monospaced))
                    .lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 8)
                if row.status == "done" {
                    Text(humanBytes(row.total)).font(.system(size: 10)).foregroundStyle(.secondary)
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.system(size: 10))
                } else if row.status == "active" {
                    Text("\(Int(row.fraction * 100))%")
                        .font(.system(size: 10, weight: .medium)).monospacedDigit()
                    Text(humanSpeed(row.speed_bps ?? 0))
                        .font(.system(size: 10)).foregroundStyle(.secondary)
                        .frame(width: 62, alignment: .trailing).monospacedDigit()
                } else {
                    Text(humanBytes(row.total)).font(.system(size: 10)).foregroundStyle(.tertiary)
                }
            }
            if showBar, row.status == "active" {
                ProgressView(value: row.fraction).progressViewStyle(.linear).controlSize(.small)
                if let r = row.resumed_from, r > 0 {
                    Text("resumed from \(humanBytes(r))")
                        .font(.system(size: 9)).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
    }
}

struct FileSection<Content: View>: View {
    let title: String
    let count: Int
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(title).font(.system(size: 11, weight: .semibold))
                Text("\(count)").font(.system(size: 10)).foregroundStyle(.secondary)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
            }
            content
        }
    }
}

struct RootView: View {
    @ObservedObject var model: Model

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            if model.missing {
                VStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("waiting for the downloader to publish state…")
                        .font(.system(size: 11)).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, minHeight: 120)
            } else {
                lists
            }
            Divider()
            footer
        }
        .padding(16)
        .frame(minWidth: 560, minHeight: 420)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(model.snap.repo_id ?? "hf-resume")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .lineLimit(1).truncationMode(.middle)
                StatusPill(status: model.snap.status ?? "…")
                if let a = model.snap.attempt, a > 1 {
                    Text("attempt \(a)").font(.system(size: 10)).foregroundStyle(.orange)
                }
                Spacer()
            }
            ProgressView(value: model.snap.fraction).progressViewStyle(.linear)
            HStack(spacing: 14) {
                Text("\(Int((model.snap.fraction * 100).rounded()))%")
                    .font(.system(size: 12, weight: .semibold)).monospacedDigit()
                Text("\(humanBytes(model.snap.done_bytes ?? 0)) / \(humanBytes(model.snap.total_bytes ?? 0))")
                    .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
                Label(humanSpeed(model.snap.speed_bps ?? 0), systemImage: "arrow.down.circle")
                    .font(.system(size: 11)).monospacedDigit()
                Label("ETA \(humanETA(model.snap.eta_seconds))", systemImage: "clock")
                    .font(.system(size: 11)).foregroundStyle(.secondary).monospacedDigit()
                Spacer()
                Text("\(model.active.count) active")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            if let r = model.snap.resumed_bytes, r > 0 {
                Label("resumed from \(humanBytes(r)) already on disk", systemImage: "arrow.clockwise")
                    .font(.system(size: 10)).foregroundStyle(.green)
            }
            if let e = model.snap.error, !e.isEmpty {
                Text(e).font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.red).lineLimit(3)
            }
        }
    }

    private var lists: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 14) {
                if !model.active.isEmpty {
                    FileSection(title: "Downloading", count: model.active.count) {
                        ForEach(model.active) { FileRowView(row: $0, showBar: true) }
                    }
                }
                if !model.queued.isEmpty {
                    FileSection(title: "Queued", count: model.queued.count) {
                        ForEach(model.queued) { FileRowView(row: $0, showBar: false) }
                    }
                }
                if !model.done.isEmpty {
                    FileSection(title: "Complete", count: model.done.count) {
                        DisclosureGroup(isExpanded: $model.showDone) {
                            ForEach(model.done) { FileRowView(row: $0, showBar: false) }
                        } label: {
                            Text(model.showDone ? "hide" : "show")
                                .font(.system(size: 10)).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(.trailing, 4)
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            if let d = model.snap.dest, !d.isEmpty {
                Button {
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: d)
                } label: {
                    Label("Reveal", systemImage: "folder").font(.system(size: 11))
                }.buttonStyle(.link)
            }
            Text("closing this window does not stop the download")
                .font(.system(size: 10)).foregroundStyle(.tertiary)
            Spacer()
            if let notes = model.snap.notes, !notes.isEmpty {
                Text(notes.joined(separator: " · "))
                    .font(.system(size: 9)).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.tail)
            }
        }
    }
}

// MARK: - App

final class Delegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    private let model: Model

    init(model: Model) {
        self.model = model
        super.init()
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        // A bare (non-bundled) binary is .prohibited by default — it would never show a window.
        NSApp.setActivationPolicy(.regular)

        let w = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 660, height: 520),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false
        )
        w.title = model.snap.repo_id ?? "hf-resume"
        w.contentView = NSHostingView(rootView: RootView(model: model))
        w.center()
        w.makeKeyAndOrderFront(nil)
        window = w
        model.window = w

        NSApp.activate(ignoringOtherApps: true)
    }

    // The window is the whole UI; closing it should end this reader process (never the daemon).
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

let statePath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : NSString(string: "~/.cache/hf-resume/state.json").expandingTildeInPath

let application = NSApplication.shared
let appDelegate = Delegate(model: Model(path: statePath))
application.delegate = appDelegate
application.run()
