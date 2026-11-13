import AppKit
import Foundation

// The notes browser. A folder of folders is not a way to find a meeting, so
// this reads the recordings directory into a list grouped by day, newest first,
// and renders the note itself rather than handing you a markdown file.

// ── the index ───────────────────────────────────────────────────────────
struct Note {
    let url: URL              // the .md file
    let session: URL          // the folder it lives in
    let title: String
    let start: Date           // when the meeting began, not when the file was written
    let duration: TimeInterval
    let meetURL: String?
    let part: Int?            // "1 - x.md" is part one of a split recording
    let parts: Int

    var time: String {
        let f = DateFormatter(); f.dateFormat = "HH:mm"
        return f.string(from: start)
    }

    var length: String {
        let m = Int(duration) / 60
        return m >= 60 ? "\(m / 60)h \(m % 60)m" : "\(m)m"
    }
}

/// Parsed from the folder name when there is no session.json to say better.
private func startFromName(_ name: String) -> Date? {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd-HHmmss"
    return f.date(from: String(name.prefix(17)))
}

func loadNotes(root: URL) -> [Note] {
    let fm = FileManager.default
    guard let dirs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
        return []
    }
    let iso = ISO8601DateFormatter()
    var out: [Note] = []

    for d in dirs where (try? d.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
        guard let files = try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil) else {
            continue
        }
        let mds = files.filter { $0.pathExtension == "md" }
        guard !mds.isEmpty else { continue }

        let session = (try? Data(contentsOf: d.appendingPathComponent("session.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? nil
        let meeting = (try? Data(contentsOf: d.appendingPathComponent("meeting.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? nil

        let start = (session?["started"] as? String).flatMap(iso.date(from:))
            ?? startFromName(d.lastPathComponent)
            ?? (try? d.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            ?? Date.distantPast
        let duration = (session?["duration_s"] as? Double) ?? 0

        // A split recording writes "1 - title.md", "2 - title.md". Those are
        // separate meetings and belong in the list separately, in order.
        let numbered = mds.compactMap { url -> (Int, URL)? in
            let name = url.deletingPathExtension().lastPathComponent
            guard let dash = name.range(of: " - "), let n = Int(name[..<dash.lowerBound]) else {
                return nil
            }
            return (n, url)
        }.sorted { $0.0 < $1.0 }

        if numbered.count > 1 {
            // Each part already states where it begins in the recording, so
            // read that rather than dividing the running time evenly - an
            // uneven split would otherwise put both parts at the midpoint.
            let offsets = numbered.map { audioOffset(of: $0.1) }
            for (i, (n, url)) in numbered.enumerated() {
                let name = url.deletingPathExtension().lastPathComponent
                let title = name.range(of: " - ").map { String(name[$0.upperBound...]) } ?? name
                let from = offsets[i] ?? duration / Double(numbered.count) * Double(n - 1)
                let to = i + 1 < offsets.count ? (offsets[i + 1] ?? duration) : duration
                out.append(Note(url: url, session: d, title: title,
                                start: start.addingTimeInterval(from),
                                duration: max(to - from, 0),
                                meetURL: meeting?["meet_url"] as? String,
                                part: n, parts: numbered.count))
            }
        } else if let url = mds.first {
            out.append(Note(url: url, session: d,
                            title: url.deletingPathExtension().lastPathComponent,
                            start: start, duration: duration,
                            meetURL: meeting?["meet_url"] as? String,
                            part: nil, parts: 1))
        }
    }
    return out.sorted { $0.start > $1.start }
}

/// Seconds into the recording that a note's meeting starts, from the line the
/// note carries for the reader: "audio from 19:11".
private func audioOffset(of note: URL) -> TimeInterval? {
    guard let head = try? String(contentsOf: note, encoding: .utf8) else { return nil }
    for line in head.components(separatedBy: "\n").prefix(12) {
        guard let r = line.range(of: "audio from ") else { continue }
        let stamp = line[r.upperBound...].prefix { "0123456789:".contains($0) }
        let parts = stamp.components(separatedBy: ":").compactMap { Double($0) }
        guard !parts.isEmpty else { continue }
        return parts.reduce(0) { $0 * 60 + $1 }
    }
    return nil
}

/// Notes grouped into days, newest day first, newest meeting first within a day.
func groupByDay(_ notes: [Note]) -> [(day: Date, label: String, notes: [Note])] {
    let cal = Calendar.current
    var buckets: [Date: [Note]] = [:]
    for n in notes { buckets[cal.startOfDay(for: n.start), default: []].append(n) }

    let heading = DateFormatter()
    heading.dateFormat = "EEEE d MMMM"
    let withYear = DateFormatter()
    withYear.dateFormat = "EEEE d MMMM yyyy"

    return buckets.keys.sorted(by: >).map { day in
        let label: String
        if cal.isDateInToday(day) { label = "Today" }
        else if cal.isDateInYesterday(day) { label = "Yesterday" }
        else if cal.isDate(day, equalTo: Date(), toGranularity: .year) { label = heading.string(from: day) }
        else { label = withYear.string(from: day) }
        return (day, label, buckets[day]!.sorted { $0.start > $1.start })
    }
}

// ── rendering ───────────────────────────────────────────────────────────
// A general markdown renderer would be the wrong amount of work: these notes
// are written by this project, so the subset is known and small.
func render(_ markdown: String) -> NSAttributedString {
    let out = NSMutableAttributedString()
    let body = NSFont.systemFont(ofSize: 13)
    let mono = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)

    func para(_ spacing: CGFloat, _ head: CGFloat = 0) -> NSParagraphStyle {
        let p = NSMutableParagraphStyle()
        p.paragraphSpacing = spacing
        p.lineSpacing = 2
        p.headIndent = head
        p.firstLineHeadIndent = head
        return p
    }

    func add(_ s: String, font: NSFont, color: NSColor = .labelColor,
             style: NSParagraphStyle = para(6)) {
        out.append(NSAttributedString(string: s + "\n", attributes: [
            .font: font, .foregroundColor: color, .paragraphStyle: style,
        ]))
    }

    for raw in markdown.components(separatedBy: "\n") {
        let line = raw.trimmingCharacters(in: .whitespaces)

        if line.isEmpty {
            add("", font: NSFont.systemFont(ofSize: 4))
        } else if line == "---" {
            let rule = NSMutableAttributedString(string: "\n")
            rule.append(NSAttributedString(string: String(repeating: " ", count: 200) + "\n",
                                           attributes: [.font: NSFont.systemFont(ofSize: 1),
                                                        .strikethroughStyle: NSUnderlineStyle.single.rawValue,
                                                        .strikethroughColor: NSColor.separatorColor]))
            out.append(rule)
        } else if line.hasPrefix("### ") {
            add(String(line.dropFirst(4)), font: .boldSystemFont(ofSize: 13), style: para(4))
        } else if line.hasPrefix("## ") {
            add(String(line.dropFirst(3)), font: .boldSystemFont(ofSize: 16), style: para(8))
        } else if line.hasPrefix("# ") {
            add(String(line.dropFirst(2)), font: .boldSystemFont(ofSize: 22), style: para(10))
        } else if line.hasPrefix("- ") || line.hasPrefix("* ") {
            add("•  " + strip(String(line.dropFirst(2))), font: body, style: para(3, 16))
        } else if line.hasPrefix("*") && line.hasSuffix("*") && !line.contains("**") {
            add(strip(line), font: NSFont.systemFont(ofSize: 11),
                color: .secondaryLabelColor, style: para(6))
        } else if let turn = speakerTurn(line) {
            // The transcript is most of the file, and a wall of it is unreadable
            // without the speaker standing out from what they said.
            let p = NSMutableParagraphStyle()
            p.paragraphSpacing = 6
            p.lineSpacing = 2
            p.headIndent = 12
            let l = NSMutableAttributedString()
            if let t = turn.time {
                l.append(NSAttributedString(string: t + "  ", attributes: [
                    .font: mono, .foregroundColor: NSColor.tertiaryLabelColor,
                ]))
            }
            l.append(NSAttributedString(string: turn.who + "  ", attributes: [
                .font: NSFont.boldSystemFont(ofSize: 13),
                .foregroundColor: turn.who == "UNKNOWN" ? NSColor.secondaryLabelColor : NSColor.labelColor,
            ]))
            l.append(NSAttributedString(string: turn.text + "\n", attributes: [
                .font: body, .foregroundColor: NSColor.labelColor,
            ]))
            l.addAttribute(.paragraphStyle, value: p, range: NSRange(location: 0, length: l.length))
            out.append(l)
        } else {
            add(strip(line), font: body)
        }
    }
    return out
}

/// Markdown emphasis markers, removed once the styling they asked for is applied.
private func strip(_ s: String) -> String {
    s.replacingOccurrences(of: "**", with: "").replacingOccurrences(of: "*", with: "")
}

private func speakerTurn(_ line: String) -> (time: String?, who: String, text: String)? {
    // [0:01:02.5] **NAME**: words
    guard let nameStart = line.range(of: "**"),
          let nameEnd = line.range(of: "**:", range: nameStart.upperBound..<line.endIndex)
    else { return nil }

    var time: String?
    if line.hasPrefix("["), let close = line.firstIndex(of: "]") {
        let stamp = String(line[line.index(after: line.startIndex)..<close])
        time = stamp.components(separatedBy: ".").first
    }
    let who = String(line[nameStart.upperBound..<nameEnd.lowerBound])
    let text = String(line[nameEnd.upperBound...]).trimmingCharacters(in: .whitespaces)
    return (time, who, text)
}

// ── the window ──────────────────────────────────────────────────────────
final class NotesWindow: NSWindowController, NSOutlineViewDataSource, NSOutlineViewDelegate {
    private let outline = NSOutlineView()
    private let text = NSTextView()
    private var groups: [(day: Date, label: String, notes: [Note])] = []
    private var selected: Note?
    private let root: URL

    init(root: URL) {
        self.root = root
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 980, height: 640),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = "Meetings"
        window.setFrameAutosaveName("MeetingNotesBrowser")
        super.init(window: window)
        build()
        reload()
    }

    required init?(coder: NSCoder) { fatalError() }

    private func build() {
        let listScroll = NSScrollView()
        listScroll.hasVerticalScroller = true
        listScroll.documentView = outline
        outline.headerView = nil
        outline.rowSizeStyle = .custom
        outline.rowHeight = 38
        outline.floatsGroupRows = false
        outline.indentationPerLevel = 0
        outline.style = .sourceList
        outline.dataSource = self
        outline.delegate = self
        let col = NSTableColumn(identifier: .init("meeting"))
        col.width = 280
        outline.addTableColumn(col)
        outline.outlineTableColumn = col

        let textScroll = NSScrollView()
        textScroll.hasVerticalScroller = true
        textScroll.documentView = text
        text.isEditable = false
        text.isSelectable = true
        text.drawsBackground = true
        text.backgroundColor = .textBackgroundColor
        text.textContainerInset = NSSize(width: 28, height: 24)
        text.autoresizingMask = [.width]
        text.textContainer?.widthTracksTextView = true

        let split = NSSplitView()
        split.isVertical = true
        split.dividerStyle = .thin
        split.addArrangedSubview(listScroll)
        split.addArrangedSubview(textScroll)
        split.autoresizingMask = [.width, .height]
        split.frame = window!.contentView!.bounds
        window?.contentView?.addSubview(split)
        split.setPosition(300, ofDividerAt: 0)
    }

    func reload() {
        groups = groupByDay(loadNotes(root: root))
        outline.reloadData()
        for g in groups { outline.expandItem(g.label) }
        if selected == nil, let first = groups.first?.notes.first {
            show(first)
            outline.selectRowIndexes([1], byExtendingSelection: false)
        }
    }

    private func show(_ note: Note) {
        selected = note
        let md = (try? String(contentsOf: note.url, encoding: .utf8)) ?? "Could not read this note."
        text.textStorage?.setAttributedString(render(md))
        text.scroll(NSPoint.zero)
        window?.title = note.title
        window?.subtitle = note.time + " · " + note.length
    }

    // ── outline plumbing ────────────────────────────────────────────────
    func outlineView(_ v: NSOutlineView, numberOfChildrenOfItem item: Any?) -> Int {
        guard let item else { return groups.count }
        return groups.first { $0.label == item as? String }?.notes.count ?? 0
    }

    func outlineView(_ v: NSOutlineView, child i: Int, ofItem item: Any?) -> Any {
        guard let item else { return groups[i].label }
        return groups.first { $0.label == item as? String }!.notes[i]
    }

    func outlineView(_ v: NSOutlineView, isItemExpandable item: Any) -> Bool { item is String }
    func outlineView(_ v: NSOutlineView, isGroupItem item: Any) -> Bool { item is String }
    func outlineView(_ v: NSOutlineView, shouldSelectItem item: Any) -> Bool { item is Note }

    func outlineView(_ v: NSOutlineView, heightOfRowByItem item: Any) -> CGFloat {
        item is String ? 26 : 44
    }

    func outlineView(_ v: NSOutlineView, viewFor col: NSTableColumn?, item: Any) -> NSView? {
        if let day = item as? String {
            let label = NSTextField(labelWithString: day.uppercased())
            label.font = .systemFont(ofSize: 10, weight: .semibold)
            label.textColor = .secondaryLabelColor
            let cell = NSTableCellView()
            cell.addSubview(label)
            label.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                label.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 10),
                label.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
            ])
            return cell
        }
        guard let note = item as? Note else { return nil }

        let time = NSTextField(labelWithString: note.time)
        time.font = .monospacedDigitSystemFont(ofSize: 11, weight: .regular)
        time.textColor = .secondaryLabelColor

        var name = note.title
        if let p = note.part { name = "\(p)/\(note.parts)  " + name }
        let title = NSTextField(labelWithString: name)
        title.font = .systemFont(ofSize: 13)
        title.lineBreakMode = .byTruncatingTail

        let length = NSTextField(labelWithString: note.length)
        length.font = .systemFont(ofSize: 10)
        length.textColor = .tertiaryLabelColor

        let cell = NSTableCellView()
        for sub in [title, time, length] {
            cell.addSubview(sub)
            sub.translatesAutoresizingMaskIntoConstraints = false
        }
        NSLayoutConstraint.activate([
            title.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 10),
            title.trailingAnchor.constraint(lessThanOrEqualTo: cell.trailingAnchor, constant: -8),
            title.topAnchor.constraint(equalTo: cell.topAnchor, constant: 5),
            time.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            time.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 2),
            length.leadingAnchor.constraint(equalTo: time.trailingAnchor, constant: 8),
            length.centerYAnchor.constraint(equalTo: time.centerYAnchor),
        ])
        return cell
    }

    func outlineViewSelectionDidChange(_ n: Notification) {
        let row = outline.selectedRow
        guard row >= 0, let note = outline.item(atRow: row) as? Note else { return }
        show(note)
    }
}
