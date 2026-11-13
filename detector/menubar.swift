import AppKit
import AVFoundation
import CoreAudio
import Foundation

// The menu bar front end. It owns the detector rather than watching it: one
// thing to launch, one thing to quit, and no terminal. meetingd runs as a child
// with --json and reports every state change on stdout; capture is still its own
// app because its TCC identity has to be separate.

// ── audio devices ───────────────────────────────────────────────────────
// A compact copy of capture.swift's enumeration. Shared code would mean a
// module and a build system; these are two dozen lines that rarely change.
let sysObj = AudioObjectID(kAudioObjectSystemObject)

func paddr(_ sel: AudioObjectPropertySelector,
           _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func u32Prop(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> UInt32? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<UInt32>.size)
    var v: UInt32 = 0
    guard AudioObjectGetPropertyData(o, &a, 0, nil, &size, &v) == noErr else { return nil }
    return v
}

func stringProp(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var out: Unmanaged<CFString>?
    let st = withUnsafeMutablePointer(to: &out) { AudioObjectGetPropertyData(o, &a, 0, nil, &size, $0) }
    guard st == noErr, let v = out?.takeRetainedValue() else { return nil }
    return v as String
}

struct Mic { let name: String; let external: Bool }

func inputDevices() -> [Mic] {
    var a = paddr(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(sysObj, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(sysObj, &a, 0, nil, &size, &ids) == noErr else { return [] }

    var out: [Mic] = []
    for d in ids {
        var ca = paddr(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
        var csz: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(d, &ca, 0, nil, &csz) == noErr, csz > 0 else { continue }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(csz), alignment: 16)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(d, &ca, 0, nil, &csz, raw) == noErr else { continue }
        let abl = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        guard abl.reduce(0, { $0 + Int($1.mNumberChannels) }) > 0 else { continue }
        let t = u32Prop(d, kAudioDevicePropertyTransportType) ?? kAudioDeviceTransportTypeUnknown
        let external = ![kAudioDeviceTransportTypeBuiltIn, kAudioDeviceTransportTypeVirtual,
                         kAudioDeviceTransportTypeAggregate, kAudioDeviceTransportTypeUnknown].contains(t)
        out.append(Mic(name: stringProp(d, kAudioObjectPropertyName) ?? "device \(d)", external: external))
    }
    return out
}

// ── where things live ───────────────────────────────────────────────────
let appDir = Bundle.main.bundleURL.deletingLastPathComponent()
let repoRoot = appDir.deletingLastPathComponent()
let configURL = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".meetingd/config.json")
let appLogURL = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".meetingd/app.log")

/// Everything the detector says, kept where it can be read later. A menu bar
/// app has nowhere to print, so without this a line like "calendar DENIED"
/// vanishes and the app just looks idle for reasons nobody can see.
func logLine(_ s: String) {
    let stamped = ISO8601DateFormatter().string(from: Date()) + "  " + s + "\n"
    guard let data = stamped.data(using: .utf8) else { return }
    if let fh = try? FileHandle(forWritingTo: appLogURL) {
        fh.seekToEndOfFile(); fh.write(data); try? fh.close()
    } else {
        try? FileManager.default.createDirectory(at: appLogURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try? data.write(to: appLogURL)
    }
}

struct Config {
    var mic: String?
    var me: String

    static func load() -> Config {
        guard let raw = try? Data(contentsOf: configURL),
              let d = (try? JSONSerialization.jsonObject(with: raw)) as? [String: Any] else {
            return Config(mic: nil, me: "Bill")
        }
        return Config(mic: d["mic"] as? String, me: (d["me"] as? String) ?? "Bill")
    }

    func save() {
        var d: [String: Any] = ["me": me]
        if let mic { d["mic"] = mic }
        try? FileManager.default.createDirectory(at: configURL.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        if let out = try? JSONSerialization.data(withJSONObject: d, options: .prettyPrinted) {
            try? out.write(to: configURL)
        }
    }
}

// ── the app ─────────────────────────────────────────────────────────────
final class Controller: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var detector: Process?
    var config = Config.load()

    // What the menu is showing. Everything here is set from meetingd's events.
    var recording = false
    var title = ""
    var meetURL: String?
    var sessionDir: URL?
    var micMuted = false
    var transcribing = false
    var lastNote: String?
    var detectorNote: String?      // why the detector is not running, if it isn't
    var quitting = false
    var loggedSymbol = false
    var restarts = 0

    // kill(1) and a crash both skip applicationWillTerminate, so catch the
    // signal too - an orphaned detector keeps the lock and blocks the next launch.
    var signalSources: [DispatchSourceSignal] = []

    func trapSignals() {
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler { [weak self] in
                self?.quitting = true
                self?.detector?.terminate()
                if let pid = self?.capturePID() { kill(pid, SIGTERM) }
                exit(0)
            }
            src.resume()
            signalSources.append(src)
        }
    }

    func applicationDidFinishLaunching(_: Notification) {
        trapSignals()
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.menu = NSMenu()
        statusItem.menu?.delegate = self
        statusItem.isVisible = true
        statusItem.behavior = []          // never let a drag remove it
        logLine("status item created: button=\(statusItem.button != nil) " +
                "visible=\(statusItem.isVisible) bar=\(NSStatusBar.system.thickness)pt")
        redraw()
        startDetector()
    }

    func applicationWillTerminate(_: Notification) {
        quitting = true
        detector?.terminate()
        if let pid = capturePID() { kill(pid, SIGTERM) }
    }

    // ── detector child ──────────────────────────────────────────────────
    func startDetector() {
        let exe = appDir.appendingPathComponent("meetingd")
        guard FileManager.default.isExecutableFile(atPath: exe.path) else {
            detectorNote = "meetingd not found next to the app"
            redraw(); return
        }
        var argv = ["--json", "--exit-with-parent", "--confirm", "30",
                    "--capture-app", appDir.appendingPathComponent("Capture.app").path]
        if let mic = config.mic { argv += ["--mic", mic] }

        let p = Process()
        p.executableURL = exe
        p.arguments = argv
        let pipe = Pipe()
        p.standardOutput = pipe
        let errPipe = Pipe()
        p.standardError = errPipe
        errPipe.fileHandleForReading.readabilityHandler = { fh in
            let d = fh.availableData
            if !d.isEmpty, let t = String(data: d, encoding: .utf8) {
                logLine(t.trimmingCharacters(in: .newlines))
            }
        }

        // Lines arrive as they are written; a partial line is held until the
        // rest of it turns up, because a split JSON row parses as nothing.
        var buffer = Data()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] fh in
            buffer.append(fh.availableData)
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[..<nl]
                buffer.removeSubrange(...nl)
                guard let text = String(data: line, encoding: .utf8) else { continue }
                logLine(text)
                DispatchQueue.main.async { self?.handle(text) }
            }
        }
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self, !self.quitting else { return }
                self.detector = nil
                self.recording = false
                // Status 1 is the lock refusing: another detector owns the
                // machine, and restarting would just lose the race again.
                if proc.terminationStatus == 1 {
                    self.detectorNote = "another detector is already running"
                } else if self.restarts < 5 {
                    self.restarts += 1
                    self.detectorNote = "detector stopped — restarting"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.startDetector() }
                } else {
                    self.detectorNote = "detector keeps stopping — check Console"
                }
                self.redraw()
            }
        }
        do {
            try p.run()
            detector = p
            detectorNote = nil
        } catch {
            detectorNote = "could not start the detector: \(error.localizedDescription)"
        }
        logLine("detector started: \(argv.joined(separator: " "))")
        redraw()
    }

    func handle(_ line: String) {
        guard line.hasPrefix("{"),
              let data = line.data(using: .utf8),
              let row = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let kind = row["type"] as? String else { return }

        switch kind {
        case "MEETING", "SPLIT HINT":
            title = (row["title"] as? String) ?? "Meeting"
            meetURL = row["meet_url"] as? String
        case "CAPTURE":
            if let dir = row["dir"] as? String {
                sessionDir = URL(fileURLWithPath: dir)
                recording = true
                micMuted = false
            }
        case "END":
            recording = false
            if let dir = sessionDir { transcribe(dir) }
            sessionDir = nil
            title = ""
            meetURL = nil
        case "DROPPED":
            recording = false
            title = ""
        default:
            break
        }
        redraw()
    }

    // ── mute ────────────────────────────────────────────────────────────
    func capturePID() -> pid_t? {
        guard let dir = sessionDir,
              let t = try? String(contentsOf: dir.appendingPathComponent("capture.pid"), encoding: .utf8),
              let pid = pid_t(t.trimmingCharacters(in: .whitespacesAndNewlines)) else { return nil }
        return kill(pid, 0) == 0 ? pid : nil
    }

    @objc func toggleMute() {
        guard let pid = capturePID() else { return }
        kill(pid, SIGUSR1)
        micMuted.toggle()
        redraw()
    }

    // ── transcription ───────────────────────────────────────────────────
    // One shot per finished session rather than a resident watcher: the models
    // are several gigabytes and would sit in memory all day between meetings.
    func transcribe(_ dir: URL) {
        let python = repoRoot.appendingPathComponent(".venv/bin/python")
        let script = repoRoot.appendingPathComponent("session_watch.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: script.path) else {
            lastNote = "transcriber not found in \(repoRoot.lastPathComponent)"
            redraw(); return
        }
        let p = Process()
        p.executableURL = python
        p.arguments = [script.path, "--once", "--me", config.me]
        p.currentDirectoryURL = repoRoot
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { [weak self] proc in
            let written = (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?
                .first { $0.hasSuffix(".md") }
            DispatchQueue.main.async {
                self?.transcribing = false
                self?.lastNote = proc.terminationStatus == 0
                    ? (written ?? "notes written")
                    : "transcription failed"
                self?.redraw()
            }
        }
        do {
            try p.run()
            transcribing = true
        } catch {
            lastNote = "could not start the transcriber"
        }
        redraw()
    }

    // ── menu ────────────────────────────────────────────────────────────
    func redraw() {
        guard let button = statusItem.button else { return }
        // An image as well as text: a title alone is easy to lose among other
        // menu bar items, and a template image gets the system's own contrast
        // handling in both appearances.
        let symbol = recording ? "record.circle.fill" : "waveform.circle"
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: "Meeting notes")
        image?.isTemplate = !recording
        button.image = image
        button.imagePosition = image == nil ? .noImage : .imageLeading
        if !loggedSymbol {
            loggedSymbol = true
            // Where the system actually put it. A frame off the right edge of
            // the screen, or zero width, means the bar had no room and macOS
            // hid it rather than telling anyone.
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                let f = button.window?.frame ?? .zero
                let screen = NSScreen.main?.frame ?? .zero
                logLine("symbol '\(symbol)' loaded: \(image != nil); " +
                        "item frame \(Int(f.origin.x)),\(Int(f.origin.y)) " +
                        "\(Int(f.width))x\(Int(f.height)); screen \(Int(screen.width))x\(Int(screen.height))")
            }
        }
        // Always carry a title. An item with no image and no title is zero
        // points wide, which looks exactly like macOS having hidden it.
        if recording {
            button.contentTintColor = .systemRed
            let short = title.count > 24 ? String(title.prefix(23)) + "…" : title
            button.title = short.isEmpty ? " REC" : " " + short
        } else {
            button.contentTintColor = nil
            button.title = image == nil ? "◉" : ""
        }
        buildMenu()
    }

    func buildMenu() {
        let menu = statusItem.menu!
        menu.removeAllItems()

        func info(_ s: String) {
            let i = NSMenuItem(title: s, action: nil, keyEquivalent: "")
            i.isEnabled = false
            menu.addItem(i)
        }

        if recording {
            info("RECORDING")
            if !title.isEmpty { info(title) }
            if let u = meetURL { info(u.replacingOccurrences(of: "https://", with: "")) }
        } else if transcribing {
            info("Transcribing…")
        } else if let note = detectorNote {
            info(note)
        } else {
            info("Waiting for a meeting")
        }
        if let n = lastNote, !recording { info("Last note: \(n)") }

        menu.addItem(.separator())

        let mute = NSMenuItem(title: "Mute mic", action: #selector(toggleMute), keyEquivalent: "")
        mute.target = self
        mute.state = micMuted ? .on : .off
        mute.isEnabled = recording && capturePID() != nil
        menu.addItem(mute)
        if micMuted {
            info("the far end still hears you")
        }

        let micMenu = NSMenu()
        let auto = NSMenuItem(title: "Automatic (prefer external)",
                              action: #selector(pickMic(_:)), keyEquivalent: "")
        auto.target = self
        auto.state = config.mic == nil ? .on : .off
        micMenu.addItem(auto)
        micMenu.addItem(.separator())
        for d in inputDevices() {
            let i = NSMenuItem(title: d.name + (d.external ? "  ·  external" : ""),
                               action: #selector(pickMic(_:)), keyEquivalent: "")
            i.target = self
            i.representedObject = d.name
            i.state = config.mic == d.name ? .on : .off
            micMenu.addItem(i)
        }
        let micItem = NSMenuItem(title: "Microphone", action: nil, keyEquivalent: "")
        micItem.submenu = micMenu
        menu.addItem(micItem)
        if recording {
            info("mic choice applies to the next meeting")
        }

        menu.addItem(.separator())
        let open = NSMenuItem(title: "Open recordings", action: #selector(openRecordings),
                              keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        let log = NSMenuItem(title: "Open log", action: #selector(openLog), keyEquivalent: "")
        log.target = self
        menu.addItem(log)
        let quit = NSMenuItem(title: "Quit", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
    }

    @objc func pickMic(_ sender: NSMenuItem) {
        config.mic = sender.representedObject as? String
        config.save()
        // The detector passes --mic when it launches capture, so a change takes
        // effect at the next meeting. Restarting mid-call would stop the
        // recording, which is worse than waiting.
        if !recording {
            detector?.terminationHandler = nil
            detector?.terminate()
            detector = nil
            startDetector()
        }
        redraw()
    }

    @objc func openRecordings() {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/Recordings/meetingd")
        NSWorkspace.shared.open(sessionDir ?? root)
    }

    @objc func openLog() { NSWorkspace.shared.open(appLogURL) }

    @objc func quit() { NSApp.terminate(nil) }
}

extension Controller: NSMenuDelegate {
    func menuNeedsUpdate(_ menu: NSMenu) { buildMenu() }
}

let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.setActivationPolicy(.accessory)
app.run()
