import CoreAudio
import EventKit
import Foundation

// ───────────────────────── CoreAudio: who is holding the mic ─────────────────────────

private let systemObj = AudioObjectID(kAudioObjectSystemObject)

private func paddr(_ sel: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel,
                               mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

private func processObjects() -> [AudioObjectID] {
    var a = paddr(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(systemObj, &a, 0, nil, &size) == noErr else { return [] }
    let n = Int(size) / MemoryLayout<AudioObjectID>.size
    guard n > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: n)
    guard AudioObjectGetPropertyData(systemObj, &a, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

private func stringProp(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var out: Unmanaged<CFString>?
    let st = withUnsafeMutablePointer(to: &out) {
        AudioObjectGetPropertyData(o, &a, 0, nil, &size, $0)
    }
    guard st == noErr, let v = out?.takeRetainedValue() else { return nil }
    return v as String
}

private func u32Prop(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> UInt32? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<UInt32>.size)
    var v: UInt32 = 0
    guard AudioObjectGetPropertyData(o, &a, 0, nil, &size, &v) == noErr else { return nil }
    return v
}

private func pidProp(_ o: AudioObjectID) -> pid_t? {
    var a = paddr(kAudioProcessPropertyPID)
    var size = UInt32(MemoryLayout<pid_t>.size)
    var v: pid_t = 0
    guard AudioObjectGetPropertyData(o, &a, 0, nil, &size, &v) == noErr else { return nil }
    return v
}

struct Holder: Hashable { let bundle: String; let pid: pid_t }

func micHolders() -> [Holder] {
    processObjects().compactMap { o in
        guard (u32Prop(o, kAudioProcessPropertyIsRunningInput) ?? 0) != 0 else { return nil }
        return Holder(bundle: stringProp(o, kAudioProcessPropertyBundleID) ?? "unknown",
                      pid: pidProp(o) ?? -1)
    }
}

// ───────────────────────── Calendar: what meeting is this ─────────────────────────

struct Meeting {
    let title: String
    let meetURL: String?
    let start: Date
    let end: Date
    let attendees: Int
    var key: String { (meetURL ?? title) + "@" + String(Int(start.timeIntervalSince1970)) }
}

final class CalendarSource {
    private let store = EKEventStore()
    private(set) var available = false

    func requestAccess() -> Bool {
        let sem = DispatchSemaphore(value: 0)
        var ok = false
        store.requestFullAccessToEvents { granted, _ in ok = granted; sem.signal() }
        if sem.wait(timeout: .now() + 60) == .timedOut { return false }
        available = ok
        return ok
    }

    private func meetURL(_ e: EKEvent) -> String? {
        for s in [e.notes, e.url?.absoluteString, e.location].compactMap({ $0 }) {
            if let r = s.range(of: #"https://meet\.google\.com/[a-z]+-[a-z]+-[a-z]+"#,
                               options: .regularExpression) { return String(s[r]) }
        }
        return nil
    }

    /// Events in progress right now, deduplicated (the same meeting often appears
    /// on both a group calendar and your own).
    func inProgress(at now: Date) -> [Meeting] {
        guard available else { return [] }
        let from = now.addingTimeInterval(-4 * 3600)
        let to = now.addingTimeInterval(4 * 3600)
        let raw = store.events(matching: store.predicateForEvents(withStart: from, end: to, calendars: nil))
        var seen = Set<String>()
        var out: [Meeting] = []
        for e in raw {
            guard let s = e.startDate, let t = e.endDate, s <= now, now <= t else { continue }
            if e.isAllDay { continue }
            let m = Meeting(title: e.title ?? "(untitled)", meetURL: meetURL(e),
                            start: s, end: t, attendees: e.attendees?.count ?? 0)
            if seen.insert(m.key).inserted { out.append(m) }
        }
        // A real video call beats a focus block; a later start beats an earlier one.
        return out.sorted {
            ($0.meetURL != nil ? 1 : 0, $0.start) > ($1.meetURL != nil ? 1 : 0, $1.start)
        }
    }
}

// ───────────────────────── Chrome: which Meet room am I actually in ─────────────────────────

struct LiveCall { let url: String; let title: String }

func meetCode(_ s: String) -> String? {
    guard let r = s.range(of: #"[a-z]{3}-[a-z]{4}-[a-z]{3}"#, options: .regularExpression)
    else { return nil }
    return String(s[r])
}

var tabReadFailed = false

func liveMeetTabs() -> [LiveCall] {
    let src = """
    tell application "Google Chrome"
      set out to ""
      repeat with w in windows
        repeat with t in tabs of w
          set u to URL of t
          if u contains "meet.google.com/" then
            set out to out & u & "<|>" & (title of t) & linefeed
          end if
        end repeat
      end repeat
      return out
    end tell
    """
    // osascript as a subprocess, not NSAppleScript: NSAppleScript is not thread
    // safe and fails silently off the main thread, which quietly demoted every
    // meeting to a calendar-only guess.
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    p.arguments = ["-e", src]
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = FileHandle.nullDevice
    do { try p.run() } catch { tabReadFailed = true; return [] }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    guard p.terminationStatus == 0, let text = String(data: data, encoding: .utf8) else {
        tabReadFailed = true
        return []
    }
    tabReadFailed = false
    var out: [LiveCall] = []
    for line in text.split(separator: "\n") {
        let parts = line.components(separatedBy: "<|>")
        guard let u = parts.first, meetCode(String(u)) != nil else { continue }
        out.append(LiveCall(url: String(u), title: parts.count > 1 ? String(parts[1]) : ""))
    }
    return out
}

// Reading Chrome's tabs over AppleScript takes seconds. Polled inline it stalls
// the whole loop, which is how a short meeting got missed entirely: by the time
// the state machine looked again, the call was over. Refresh it off to one side.
final class TabCache: @unchecked Sendable {
    private let lock = NSLock()
    private var tabs: [LiveCall] = []
    var current: [LiveCall] {
        get { lock.lock(); defer { lock.unlock() }; return tabs }
        set { lock.lock(); tabs = newValue; lock.unlock() }
    }
}
let tabCache = TabCache()

func startTabWatcher() {
    Thread.detachNewThread {
        while true {
            tabCache.current = liveMeetTabs()
            Thread.sleep(forTimeInterval: 2)
        }
    }
}

struct Resolved {
    let title: String
    let meetURL: String?
    let attendees: Int
    let source: String      // meet_url | tab_title | calendar_only
    let confident: Bool
    let key: String
}

// ───────────────────────── Output ─────────────────────────

let logURL: URL = {
    let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".meetingd")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent("events.jsonl")
}()

let clock: DateFormatter = { let f = DateFormatter(); f.dateFormat = "HH:mm:ss"; return f }()
let iso = ISO8601DateFormatter()

func emit(_ kind: String, _ human: String, _ fields: [String: Any] = [:]) {
    print("[\(clock.string(from: Date()))] \(kind.padding(toLength: 11, withPad: " ", startingAt: 0))\(human)")
    fflush(stdout)
    var row: [String: Any] = fields
    row["type"] = kind
    row["ts"] = iso.string(from: Date())
    guard let data = try? JSONSerialization.data(withJSONObject: row),
          var line = String(data: data, encoding: .utf8) else { return }
    line += "\n"
    if let fh = try? FileHandle(forWritingTo: logURL) {
        fh.seekToEndOfFile(); fh.write(line.data(using: .utf8)!); try? fh.close()
    } else {
        try? line.write(to: logURL, atomically: true, encoding: .utf8)
    }
}

func hms(_ s: TimeInterval) -> String {
    let t = Int(s); return String(format: "%dm%02ds", t / 60, t % 60)
}

// ───────────────────────── State machine ─────────────────────────

setvbuf(stdout, nil, _IOLBF, 0)   // line-buffer so output survives being killed

let args = CommandLine.arguments
func argValue(_ k: String) -> String? {
    guard let i = args.firstIndex(of: k), i + 1 < args.count else { return nil }
    return args[i + 1]
}
let useCalendar = !args.contains("--no-calendar")
let confirmAfter: TimeInterval = {
    if let i = args.firstIndex(of: "--confirm"), i + 1 < args.count,
       let v = Double(args[i + 1]) { return v }
    return 45
}()

if args.contains("--once") {
    let h = micHolders()
    print(h.isEmpty ? "no process is holding the mic"
                    : h.map { "\($0.bundle) (pid \($0.pid)) holds the mic" }.joined(separator: "\n"))
    exit(0)
}

// ───────────────────────── Capture handoff ─────────────────────────
// Capture has to be launched through launchd, not spawned as a child: a child
// inherits our responsible process and gets no TCC identity, so its process tap
// would silently record zeros.

let captureApp = argValue("--capture-app")
    ?? URL(fileURLWithPath: CommandLine.arguments[0])
        .deletingLastPathComponent().appendingPathComponent("Capture.app").path
let captureMic = argValue("--mic")
let recordingRoot = argValue("--recordings")
    ?? FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/Recordings/meetingd").path
let captureEnabled = !args.contains("--no-capture")

var sessionDir: URL?
var captureGeneration = 0

func slug(_ s: String) -> String {
    let ok = s.unicodeScalars.map { CharacterSet.alphanumerics.contains($0) ? Character($0) : "-" }
    return String(ok).replacingOccurrences(of: "-+", with: "-", options: .regularExpression)
        .trimmingCharacters(in: CharacterSet(charactersIn: "-")).prefix(60).description
}

func launchCapture(dir: URL, appName: String) {
    var a = ["-n", "-a", captureApp, "--args",
             "--session", dir.path,
             "--bundle", appName,
             "--out", dir.appendingPathComponent("capture.log").path]
    if let captureMic { a += ["--mic", captureMic] }
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    p.arguments = a
    do { try p.run() } catch { emit("CAPTURE", "failed to launch: \(error.localizedDescription)") }
}

/// Capture is headless, and macOS kills a background app the moment you grant it
/// a permission. Without a check, that failure is completely invisible - so
/// confirm it actually came up, and say so loudly if it did not.
func confirmCaptureStarted(dir: URL, appName: String, attempt: Int = 1) {
    let generation = captureGeneration
    DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
        guard generation == captureGeneration else { return }   // meeting already ended
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("capture.pid").path) {
            if attempt > 1 { emit("CAPTURE", "recording after retry") }
            return
        }
        if attempt == 1 {
            emit("CAPTURE", "did not start — retrying (macOS kills the app when a permission is granted)")
            launchCapture(dir: dir, appName: appName)
            confirmCaptureStarted(dir: dir, appName: appName, attempt: 2)
        } else {
            emit("CAPTURE", "NOT RECORDING — check Privacy & Security > Microphone and Screen & System Audio Recording")
        }
    }
}

func startCapture(_ r: Resolved?, holder: String) {
    guard captureEnabled else { return }
    let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmmss"
    let name = f.string(from: Date()) + (r.map { "-" + slug($0.title) } ?? "")
    let dir = URL(fileURLWithPath: recordingRoot).appendingPathComponent(name)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    sessionDir = dir
    captureGeneration += 1

    if let r {
        var meta: [String: Any] = ["title": r.title, "attendees": r.attendees,
                                   "source": r.source, "confident": r.confident]
        if let u = r.meetURL { meta["meet_url"] = u }
        if let d = try? JSONSerialization.data(withJSONObject: meta, options: .prettyPrinted) {
            try? d.write(to: dir.appendingPathComponent("meeting.json"))
        }
    }

    // Tap whatever app took the mic, not a hardcoded browser.
    let appName = holder.components(separatedBy: ".").dropLast().last ?? "Chrome"
    launchCapture(dir: dir, appName: appName)
    emit("CAPTURE", "recording to \(dir.lastPathComponent)", ["dir": dir.path])
    confirmCaptureStarted(dir: dir, appName: appName)
}

func stopCapture() {
    guard let dir = sessionDir else { return }
    sessionDir = nil
    captureGeneration += 1
    let pidFile = dir.appendingPathComponent("capture.pid")
    guard let text = try? String(contentsOf: pidFile, encoding: .utf8),
          let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)) else {
        emit("CAPTURE", "no pidfile — capture may still be running")
        return
    }
    kill(pid, SIGTERM)
    emit("CAPTURE", "stopped (pid \(pid))")
}

func addMark(_ kind: String, _ fields: [String: Any]) {
    guard let dir = sessionDir else { return }
    var row = fields
    row["mark"] = kind
    row["ts"] = ISO8601DateFormatter().string(from: Date())
    guard let d = try? JSONSerialization.data(withJSONObject: row) else { return }
    let url = dir.appendingPathComponent("marks.jsonl")
    if let fh = try? FileHandle(forWritingTo: url) {
        fh.seekToEndOfFile(); fh.write(d); fh.write("\n".data(using: .utf8)!); try? fh.close()
    } else {
        try? (String(data: d, encoding: .utf8)! + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

let cal = CalendarSource()
if useCalendar {
    print("requesting calendar access…")
    if !cal.requestAccess() {
        print("calendar DENIED — continuing without it (run from Terminal.app to grant)")
    }
}

if args.contains("--tabs") {
    let t = liveMeetTabs()
    if tabReadFailed { print("could not read Chrome tabs — Automation permission denied?") }
    else if t.isEmpty { print("no Google Meet tab open") }
    for x in t { print("\(x.url)   |   \(x.title)") }
    exit(0)
}

if args.contains("--match-now") {
    let events = cal.inProgress(at: Date())
    if events.isEmpty { print("no calendar event in progress right now") }
    for (i, m) in events.enumerated() {
        let mark = i == 0 ? "picked →" : "        "
        print("\(mark) \(m.title)  [\(m.attendees) attendees]  \(m.meetURL ?? "(no meet url)")")
    }
    if events.filter({ $0.meetURL != nil }).count > 1 {
        print("\nAMBIGUOUS: more than one video call overlaps — only the live Meet URL can resolve this.")
    }
    exit(0)
}

enum State { case idle, candidate(Date), meeting(Date) }
var state: State = .idle
var micMissingSince: Date?
let endGrace: TimeInterval = Double(argValue("--end-grace") ?? "") ?? 6
var currentKey: String?
var segments = 0

func cleanTitle(_ t: String) -> String {
    var s = t
    for p in ["Meet \u{2013} ", "Meet - ", "Meet \u{2014} "] {
        s = s.replacingOccurrences(of: p, with: "")
    }
    s = s.replacingOccurrences(of: " - Google Chrome", with: "")
    return s.trimmingCharacters(in: .whitespaces)
}

/// Identity, best source first: the live Meet room beats anything the calendar guesses.
func resolve(at now: Date) -> Resolved? {
    let tabs = tabCache.current
    let events = useCalendar ? cal.inProgress(at: now) : []

    for t in tabs {
        guard let code = meetCode(t.url) else { continue }
        if let m = events.first(where: { $0.meetURL.flatMap(meetCode) == code }) {
            return Resolved(title: m.title, meetURL: t.url, attendees: m.attendees,
                            source: "meet_url", confident: true, key: code)
        }
    }
    if let t = tabs.first, let code = meetCode(t.url) {
        let name = cleanTitle(t.title)
        return Resolved(title: name.isEmpty ? "Ad-hoc call" : name, meetURL: t.url,
                        attendees: 0, source: "tab_title", confident: true, key: code)
    }
    if let m = events.first(where: { $0.meetURL != nil }) {
        return Resolved(title: m.title, meetURL: m.meetURL, attendees: m.attendees,
                        source: "calendar_only", confident: false, key: m.key)
    }
    return nil
}

func fields(_ r: Resolved) -> [String: Any] {
    var f: [String: Any] = ["title": r.title, "attendees": r.attendees,
                            "source": r.source, "confident": r.confident]
    if let u = r.meetURL { f["meet_url"] = u }
    return f
}

func describe(_ r: Resolved) -> String {
    let who = r.attendees > 0 ? ", \(r.attendees) attendees" : ""
    let u = r.meetURL.map { " (\($0.replacingOccurrences(of: "https://", with: ""))\(who))" } ?? ""
    return r.title + u + (r.confident ? "" : "   [low confidence: \(r.source)]")
}

// One detector per machine. Two of them double every event, lose each other's
// writes to the shared log - seekToEndOfFile is not atomic across processes -
// and reap each other's captures. An instance left running overnight is easy to
// forget, so refuse rather than quietly join in.
let lockURL = logURL.deletingLastPathComponent().appendingPathComponent("meetingd.lock")
let lockFD = open(lockURL.path, O_CREAT | O_RDWR, 0o644)
if lockFD < 0 {
    print("cannot lock \(lockURL.path) — continuing unlocked")
} else if flock(lockFD, LOCK_EX | LOCK_NB) != 0 {
    let other = ((try? String(contentsOf: lockURL, encoding: .utf8)) ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    print("meetingd is already running" + (other.isEmpty ? "" : " as pid \(other)") + " — exiting.")
    print(other.isEmpty ? "Stop it first." : "Stop it first:  kill \(other)")
    exit(1)
} else {
    ftruncate(lockFD, 0)
    let mine = "\(getpid())\n"
    _ = mine.withCString { write(lockFD, $0, strlen($0)) }
}

// A capture launched by a previous run survives our restart. If it is still
// recording the meeting that is live right now, adopt it: killing it would
// split one meeting across two folders and lose the audio in between. Anything
// else - a finished call, a crash, a different room - is debris.
if captureEnabled, let dirs = try? FileManager.default.contentsOfDirectory(
        at: URL(fileURLWithPath: recordingRoot), includingPropertiesForKeys: nil) {
    let liveCodes = Set(liveMeetTabs().compactMap { meetCode($0.url) })
    var running: [(dir: URL, pid: pid_t, code: String?, started: Date)] = []

    for d in dirs {
        let pf = d.appendingPathComponent("capture.pid")
        guard let t = try? String(contentsOf: pf, encoding: .utf8),
              let pid = pid_t(t.trimmingCharacters(in: .whitespacesAndNewlines)) else { continue }
        guard kill(pid, 0) == 0 else {            // pidfile left behind by a dead process
            try? FileManager.default.removeItem(at: pf)
            continue
        }
        var code: String?
        if let md = try? Data(contentsOf: d.appendingPathComponent("meeting.json")),
           let m = try? JSONSerialization.jsonObject(with: md) as? [String: Any],
           let u = m["meet_url"] as? String { code = meetCode(u) }
        let started = (try? pf.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? Date()
        running.append((d, pid, code, started))
    }

    if let keep = running.filter({ $0.code.map(liveCodes.contains) ?? false })
                         .max(by: { $0.started < $1.started }) {
        sessionDir = keep.dir
        state = .meeting(keep.started)
        currentKey = keep.code
        let marks = (try? String(contentsOf: keep.dir.appendingPathComponent("marks.jsonl"),
                                 encoding: .utf8)) ?? ""
        segments = 1 + marks.split(separator: "\n").filter { !$0.isEmpty }.count
        emit("ADOPTED", "\(keep.dir.lastPathComponent) is still recording this call (pid \(keep.pid))",
             ["dir": keep.dir.path, "pid": Int(keep.pid), "segments": segments])
    }

    var reaped = 0
    for r in running where r.dir != sessionDir {
        kill(r.pid, SIGTERM)
        try? FileManager.default.removeItem(at: r.dir.appendingPathComponent("capture.pid"))
        reaped += 1
    }
    if reaped > 0 { print("stopped \(reaped) orphaned capture(s)") }
}

// Clear the TCC prompts now rather than in front of a real meeting.
if captureEnabled && !args.contains("--no-warmup") {
    let w = Process()
    w.executableURL = URL(fileURLWithPath: "/usr/bin/open")
    w.arguments = ["-n", "-a", captureApp, "--args", "--warmup",
                   "--out", NSTemporaryDirectory() + "meetingd-warmup.log"]
    try? w.run()
    print("warming up capture permissions…")
    DispatchQueue.global().asyncAfter(deadline: .now() + 6) {
        let w2 = Process()
        w2.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        w2.arguments = w.arguments
        try? w2.run()
    }
}

startTabWatcher()
Thread.sleep(forTimeInterval: 0.5)   // let the first tab read land
print("meetingd watching (confirm after \(Int(confirmAfter))s)…  log: \(logURL.path)")

while true {
    let now = Date()
    let ignored = ["com.apple.Sound-Settings", "com.apple.systempreferences",
                   "com.bgilpin.capture", "com.bgilpin.meetingd"]
    let live = micHolders().filter { h in
        h.bundle != "unknown" && !ignored.contains(where: { h.bundle.hasPrefix($0) })
    }

    switch state {
    case .idle:
        if let h = live.first {
            state = .candidate(now)
            emit("CANDIDATE", "\(h.bundle) took the mic", ["bundle": h.bundle, "pid": Int(h.pid)])
        }

    case .candidate(let since):
        if live.isEmpty {
            state = .idle
            emit("DROPPED", "mic released after \(hms(now.timeIntervalSince(since))) — not a meeting")
        } else {
            let r = resolve(at: now)
            let heldLongEnough = now.timeIntervalSince(since) >= confirmAfter
            if let r, r.confident {
                state = .meeting(since); segments = 1; currentKey = r.key
                emit("MEETING", describe(r), fields(r))
                startCapture(r, holder: live.first?.bundle ?? "com.google.Chrome")
            } else if heldLongEnough {
                state = .meeting(since); segments = 1
                if let r {
                    currentKey = r.key
                    emit("MEETING", describe(r), fields(r))
                } else {
                    currentKey = nil
                    emit("MEETING", "unidentified call (no Meet tab, no calendar match)",
                         ["source": "duration", "confident": false])
                }
                startCapture(r, holder: live.first?.bundle ?? "com.google.Chrome")
                if tabReadFailed {
                    emit("NO TABS", "could not read Chrome tabs — grant Automation access for accurate naming")
                }
            }
        }

    case .meeting(let since):
        if live.isEmpty {
            // Chrome's audio processes churn mid-call. Ending on a single empty
            // poll cut a meeting one second after it started.
            if micMissingSince == nil { micMissingSince = now }
            guard let gone = micMissingSince, now.timeIntervalSince(gone) >= endGrace else { break }
            let ended = gone
            state = .idle
            emit("END", "\(hms(ended.timeIntervalSince(since)))  →  \(segments) segment(s)",
                 ["duration_s": Int(ended.timeIntervalSince(since)), "segments": segments])
            stopCapture()
            currentKey = nil; segments = 0; micMissingSince = nil
        } else {
            micMissingSince = nil          // mic is back; it was only a blip
            if let r = resolve(at: now), r.confident, r.key != currentKey {
                currentKey = r.key; segments += 1
                var f = fields(r); f["segment"] = segments
                emit("SPLIT HINT", "moved to: \(describe(r))", f)
                addMark("split", f)
            }
        }
    }

    Thread.sleep(forTimeInterval: 1.0)
}
