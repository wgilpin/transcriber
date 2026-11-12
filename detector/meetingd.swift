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
            set out to out & u & "\t" & (title of t) & "\n"
          end if
        end repeat
      end repeat
      return out
    end tell
    """
    guard let script = NSAppleScript(source: src) else { return [] }
    var err: NSDictionary?
    let res = script.executeAndReturnError(&err)
    if err != nil { tabReadFailed = true; return [] }
    tabReadFailed = false
    guard let text = res.stringValue else { return [] }
    var out: [LiveCall] = []
    for line in text.split(separator: "\n") {
        let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
        guard let u = parts.first, meetCode(String(u)) != nil else { continue }
        out.append(LiveCall(url: String(u), title: parts.count > 1 ? String(parts[1]) : ""))
    }
    return out
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
    let tabs = liveMeetTabs()
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

print("meetingd watching (confirm after \(Int(confirmAfter))s)…  log: \(logURL.path)")

while true {
    let now = Date()
    let live = micHolders().filter { $0.bundle != "unknown" }

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
                if tabReadFailed {
                    emit("NO TABS", "could not read Chrome tabs — grant Automation access for accurate naming")
                }
            }
        }

    case .meeting(let since):
        if live.isEmpty {
            state = .idle
            emit("END", "\(hms(now.timeIntervalSince(since)))  →  \(segments) segment(s)",
                 ["duration_s": Int(now.timeIntervalSince(since)), "segments": segments])
            currentKey = nil; segments = 0
        } else if let r = resolve(at: now), r.confident, r.key != currentKey {
            currentKey = r.key; segments += 1
            var f = fields(r); f["segment"] = segments
            emit("SPLIT HINT", "moved to: \(describe(r))", f)
        }
    }

    Thread.sleep(forTimeInterval: 1.0)
}
