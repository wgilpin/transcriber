import AVFoundation
import CoreAudio
import Foundation

// Two-track capture: your mic on one file, the meeting app's output on another.
// Separate tracks mean "me" is free — diarization only has to split the far end.

let sysObj = AudioObjectID(kAudioObjectSystemObject)

func paddr(_ sel: AudioObjectPropertySelector,
           _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func processObjects() -> [AudioObjectID] {
    var a = paddr(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(sysObj, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(sysObj, &a, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

func stringProp(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> String? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var out: Unmanaged<CFString>?
    let st = withUnsafeMutablePointer(to: &out) { AudioObjectGetPropertyData(o, &a, 0, nil, &size, $0) }
    guard st == noErr, let v = out?.takeRetainedValue() else { return nil }
    return v as String
}

func u32Prop(_ o: AudioObjectID, _ sel: AudioObjectPropertySelector) -> UInt32? {
    var a = paddr(sel)
    var size = UInt32(MemoryLayout<UInt32>.size)
    var v: UInt32 = 0
    guard AudioObjectGetPropertyData(o, &a, 0, nil, &size, &v) == noErr else { return nil }
    return v
}

/// Devices that actually have input channels, with their names.
func inputDevices() -> [(id: AudioObjectID, name: String)] {
    var a = paddr(kAudioHardwarePropertyDevices)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(sysObj, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(sysObj, &a, 0, nil, &size, &ids) == noErr else { return [] }

    var out: [(AudioObjectID, String)] = []
    for d in ids {
        var ca = paddr(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
        var csz: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(d, &ca, 0, nil, &csz) == noErr, csz > 0 else { continue }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(csz), alignment: 16)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(d, &ca, 0, nil, &csz, raw) == noErr else { continue }
        let abl = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        let channels = abl.reduce(0) { $0 + Int($1.mNumberChannels) }
        guard channels > 0 else { continue }
        let name = stringProp(d, kAudioObjectPropertyName) ?? "device \(d)"
        out.append((d, name))
    }
    return out
}

func defaultOutputUID() -> String? {
    for sel in [kAudioHardwarePropertyDefaultSystemOutputDevice,
                kAudioHardwarePropertyDefaultOutputDevice] {
        var a = paddr(sel)
        var dev = AudioObjectID(kAudioObjectUnknown)
        var sz = UInt32(MemoryLayout<AudioObjectID>.size)
        if AudioObjectGetPropertyData(sysObj, &a, 0, nil, &sz, &dev) == noErr,
           dev != kAudioObjectUnknown, let uid = stringProp(dev, kAudioDevicePropertyDeviceUID) {
            return uid
        }
    }
    return nil
}

// ── args ────────────────────────────────────────────────────────────────
let args = CommandLine.arguments
func argValue(_ k: String) -> String? {
    guard let i = args.firstIndex(of: k), i + 1 < args.count else { return nil }
    return args[i + 1]
}
if let path = argValue("--out") {
    FileManager.default.createFile(atPath: path, contents: nil)
    freopen(path, "w", stdout); freopen(path, "a", stderr)
}
setvbuf(stdout, nil, _IOLBF, 0)

if args.contains("--list-mics") {
    for d in inputDevices() { print("\(d.name)") }
    exit(0)
}

let wantMic = argValue("--mic")
// Opt-in only: macOS voice processing (VPIO) seizes the output device as its
// AEC reference and ducks it hard - on speakers it silences the meeting you
// are trying to hear, and it reported a bogus 8-channel input format here.
// Echo bleed is handled downstream instead, against the far track.
let useAEC = args.contains("--aec")
let wantBundle = argValue("--bundle") ?? "Chrome"
let seconds = Double(argValue("--seconds") ?? "") ?? 0     // 0 = until SIGINT
let title = argValue("--title")
let meetURL = argValue("--meet-url")
let baseDir = argValue("--dir")
    ?? FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/Recordings/meetingd").path

let stamp: String = {
    let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd-HHmmss"
    return f.string(from: Date())
}()
// Warm-up exists only to trigger TCC prompts; it must not litter Recordings.
let sessionDir: URL = args.contains("--warmup")
    ? URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("meetingd-warmup")
    : (argValue("--session").map { URL(fileURLWithPath: $0) }
       ?? URL(fileURLWithPath: baseDir).appendingPathComponent(stamp))
try? FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
let farURL = sessionDir.appendingPathComponent("far.wav")
let micURL = sessionDir.appendingPathComponent("mic.wav")
print("session: \(sessionDir.path)")
let pidFile = sessionDir.appendingPathComponent("capture.pid")
try? String(ProcessInfo.processInfo.processIdentifier).write(to: pidFile, atomically: true, encoding: .utf8)

let engine = AVAudioEngine()
var micFile: AVAudioFile?
var micAllowed = false
let writeQ = DispatchQueue(label: "meetingd.write")
var micStarted: Date?
var farStarted: Date?

// ── far-end tap ─────────────────────────────────────────────────────────
struct Proc { let obj: AudioObjectID; let bundle: String; let output: Bool }
var matches: [Proc] = []
for o in processObjects() {
    let b = stringProp(o, kAudioProcessPropertyBundleID) ?? "unknown"
    guard b.localizedCaseInsensitiveContains(wantBundle) else { continue }
    matches.append(Proc(obj: o, bundle: b,
                        output: (u32Prop(o, kAudioProcessPropertyIsRunningOutput) ?? 0) != 0))
}
let playing = matches.filter { $0.output }
let chosen = playing.isEmpty ? matches : playing
let warmup = args.contains("--warmup")
if chosen.isEmpty && !warmup {
    print("no audio process matching '\(wantBundle)' — nothing to tap")
    exit(1)
}

// A global tap still triggers the audio-capture prompt when nothing happens to
// be playing, which is what warm-up needs.
let tapDesc: CATapDescription
if chosen.isEmpty {
    print("far track: nothing playing — global tap for warm-up only")
    tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
} else {
    print("far track: tapping \(chosen.count) \(wantBundle) process(es)")
    tapDesc = CATapDescription(stereoMixdownOfProcesses: chosen.map { $0.obj })
}
tapDesc.name = "meetingd-farend"
tapDesc.isPrivate = true
tapDesc.muteBehavior = .unmuted

var tapID = AudioObjectID(kAudioObjectUnknown)
guard AudioHardwareCreateProcessTap(tapDesc, &tapID) == noErr, tapID != kAudioObjectUnknown else {
    print("process tap failed — launch this from the .app via `open` so macOS can grant audio capture")
    exit(2)
}

let aggDict: [String: Any] = [
    kAudioAggregateDeviceNameKey: "meetingd-capture",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceMainSubDeviceKey: defaultOutputUID() ?? "",
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [] as [[String: Any]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapUIDKey: tapDesc.uuid.uuidString,
        kAudioSubTapDriftCompensationKey: true,
    ]],
]
var aggID = AudioObjectID(kAudioObjectUnknown)
guard AudioHardwareCreateAggregateDevice(aggDict as CFDictionary, &aggID) == noErr else {
    print("aggregate device failed"); AudioHardwareDestroyProcessTap(tapID); exit(3)
}

var asbd = AudioStreamBasicDescription()
var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var fa = paddr(kAudioDevicePropertyStreamFormat, kAudioObjectPropertyScopeInput)
AudioObjectGetPropertyData(aggID, &fa, 0, nil, &fsz, &asbd)

guard let farFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                    sampleRate: asbd.mSampleRate,
                                    channels: AVAudioChannelCount(max(asbd.mChannelsPerFrame, 1)),
                                    interleaved: true) else {
    print("bad far format"); exit(4)
}
var farFile: AVAudioFile?
do {
    farFile = try AVAudioFile(forWriting: farURL, settings: farFormat.settings,
                              commonFormat: .pcmFormatFloat32, interleaved: true)
    print(String(format: "far track: %.0f Hz, %d ch", asbd.mSampleRate, asbd.mChannelsPerFrame))
} catch {
    print("far file failed: \(error.localizedDescription)"); exit(5)
}

final class Counters: @unchecked Sendable {
    let lock = NSLock()
    var farFrames: Int64 = 0
    var farPeak: Float = 0
    var micFrames: Int64 = 0
    var micPeak: Float = 0
    var micEverNonZero = false
}
let counters = Counters()

var procID: AudioDeviceIOProcID?
guard AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil, { _, inData, _, _, _ in
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
    guard let b = abl.first, let raw = b.mData, b.mDataByteSize > 0 else { return }
    let bytesPerFrame = 4 * Int(max(b.mNumberChannels, 1))
    let frames = Int(b.mDataByteSize) / bytesPerFrame
    guard frames > 0,
          let buf = AVAudioPCMBuffer(pcmFormat: farFormat, frameCapacity: AVAudioFrameCount(frames))
    else { return }
    buf.frameLength = AVAudioFrameCount(frames)
    memcpy(buf.audioBufferList.pointee.mBuffers.mData, raw, Int(b.mDataByteSize))

    var peak: Float = 0
    let s = raw.assumingMemoryBound(to: Float.self)
    for i in 0..<(frames * Int(b.mNumberChannels)) { peak = max(peak, abs(s[i])) }
    counters.lock.lock()
    counters.farFrames += Int64(frames); counters.farPeak = max(counters.farPeak, peak)
    counters.lock.unlock()

    writeQ.async { try? farFile?.write(from: buf) }
}) == noErr, let procID else {
    print("IOProc failed"); exit(6)
}

farStarted = Date()
AudioDeviceStart(aggID, procID)

// The far end is already recording by now. Asking for microphone permission can
// block on a dialog for as long as the user takes to notice it, so it must not
// sit in front of the tap - that cost us the opening of a real meeting.
// ── microphone permission + engine ──────────────────────────────────────

func setUpMic() {
    micStarted = Date()
    let input = engine.inputNode
    // The system default input is often the built-in mic, which returns digital
    // silence when the lid is shut. Pick the device explicitly.
    if let want = wantMic {
        if let match = inputDevices().first(where: { $0.name.localizedCaseInsensitiveContains(want) }) {
            var devID = match.id
            if let unit = input.audioUnit {
                let st = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice,
                                              kAudioUnitScope_Global, 0, &devID,
                                              UInt32(MemoryLayout<AudioDeviceID>.size))
                print(st == noErr ? "mic device: \(match.name)"
                                  : "could not select '\(match.name)': OSStatus \(st)")
            }
        } else {
            print("no input device matching '\(want)'. Try --list-mics")
        }
    }
    // Speakers mean the far end bleeds into the mic. macOS's own voice
    // processing cancels it, using system output as the reference signal.
    // It forces mono and adds AGC + noise suppression, so --no-aec exists
    // to record raw for comparison.
    if useAEC {
        do {
            try input.setVoiceProcessingEnabled(true)
            print("echo cancellation: on — WARNING: this ducks your speaker output")
        } catch {
            print("echo cancellation unavailable: \(error.localizedDescription)")
        }
    } else {
        print("echo cancellation: off (bleed handled downstream)")
    }

    // inputFormat is the hardware format. outputFormat returns AVAudioEngine's
    // 44100/1ch default when the input chain was never realised - which yields
    // a tap that fires at the right rate but delivers pure silence.
    var micFormat = input.inputFormat(forBus: 0)
    if micFormat.sampleRate == 0 || micFormat.channelCount == 0 {
        micFormat = input.outputFormat(forBus: 0)
    }
    if wantMic == nil {
        print("mic device: \(AVCaptureDevice.default(for: .audio)?.localizedName ?? "unknown") (system default)")
    }
    do {
        micFile = try AVAudioFile(forWriting: micURL,
                                  settings: micFormat.settings,
                                  commonFormat: .pcmFormatFloat32,
                                  interleaved: micFormat.isInterleaved)
        input.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buf, _ in
            var peak: Float = 0
            if let ch = buf.floatChannelData {
                for c in 0..<Int(buf.format.channelCount) {
                    for i in 0..<Int(buf.frameLength) { peak = max(peak, abs(ch[c][i])) }
                }
            }
            counters.lock.lock()
            counters.micPeak = max(counters.micPeak, peak)
            if peak > 0 { counters.micEverNonZero = true }
            counters.micFrames += Int64(buf.frameLength)
            counters.lock.unlock()
            writeQ.async { try? micFile?.write(from: buf) }
        }
        NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { _ in
            print("audio config changed — restarting mic engine")
            try? engine.start()
        }
        print(String(format: "mic track: %.0f Hz, %d ch", micFormat.sampleRate, micFormat.channelCount))
    } catch {
        print("mic file/tap failed: \(error.localizedDescription)")
        micFile = nil
    }
    if micFile != nil {
        do { try engine.start() }
        catch {
            print("engine.start failed: \(error.localizedDescription)")
            micFile = nil
        }
    }
}



AVCaptureDevice.requestAccess(for: .audio) { ok in
    micAllowed = ok
    print("microphone: \(ok ? "granted" : "DENIED")")
    if ok { setUpMic() } else { print("recording far end only") }
}

if warmup {
    // Clear both prompts ahead of time so a real meeting never waits on a dialog.
    Thread.sleep(forTimeInterval: 3)
    print("warmup done")
    try? FileManager.default.removeItem(at: sessionDir)
    AudioDeviceStop(aggID, procID)
    AudioDeviceDestroyIOProcID(aggID, procID)
    AudioHardwareDestroyAggregateDevice(aggID)
    AudioHardwareDestroyProcessTap(tapID)
    exit(0)
}


// ── run ─────────────────────────────────────────────────────────────────
nonisolated(unsafe) var stopFlag: sig_atomic_t = 0
signal(SIGINT) { _ in stopFlag = 1 }
signal(SIGTERM) { _ in stopFlag = 1 }

var micZeroWarned = false
let started = Date()
print("recording… (ctrl-C to stop)\n")
while stopFlag == 0 {
    if seconds > 0, Date().timeIntervalSince(started) >= seconds { break }
    Thread.sleep(forTimeInterval: 0.5)
    counters.lock.lock()
    let fp = counters.farPeak; counters.farPeak = 0
    let mp = counters.micPeak; counters.micPeak = 0
    counters.lock.unlock()
    func bar(_ v: Float) -> String {
        String(repeating: "█", count: Int(min(v, 1.0) * 20))
            .padding(toLength: 20, withPad: " ", startingAt: 0)
    }
    print(String(format: "far %.3f |%@|   mic %.3f |%@|", fp, bar(fp), mp, bar(mp)))
    // A live mic always has a noise floor. Frames arriving with peak exactly
    // zero means macOS is zeroing them - usually a TCC grant left stale by an
    // ad-hoc re-sign, which still reports "authorized".
    if !micZeroWarned, micAllowed, Date().timeIntervalSince(started) > 4 {
        counters.lock.lock(); let frames = counters.micFrames; counters.lock.unlock()
        if frames > 0 && counters.micEverNonZero == false {
            micZeroWarned = true
            print("""

                  WARNING: mic frames are arriving but every sample is zero.
                  Either the engine never bound to the hardware (check the
                  reported format matches the device), or the TCC grant is stale
                  after an ad-hoc re-sign: tccutil reset Microphone com.bgilpin.capture

                  """)
        }
    }
}

AudioDeviceStop(aggID, procID)
AudioDeviceDestroyIOProcID(aggID, procID)
AudioHardwareDestroyAggregateDevice(aggID)
AudioHardwareDestroyProcessTap(tapID)
if micAllowed { engine.inputNode.removeTap(onBus: 0); engine.stop() }
writeQ.sync {}
farFile = nil
micFile = nil

try? FileManager.default.removeItem(at: pidFile)
let ended = Date()
let iso = ISO8601DateFormatter()
var sidecar: [String: Any] = [
    "started": iso.string(from: started),
    "ended": iso.string(from: ended),
    "duration_s": Int(ended.timeIntervalSince(started)),
    "far_track": "far.wav",
    "far_frames": counters.farFrames,
    "mic_track": micAllowed ? "mic.wav" : NSNull(),
    "tapped_bundle": wantBundle,
    "aec": useAEC,
    "mic_frames": counters.micFrames,
    "mic_started": micStarted.map { iso.string(from: $0) } ?? NSNull(),
    "far_started": farStarted.map { iso.string(from: $0) } ?? NSNull(),
]
if let title { sidecar["title"] = title }
if let meetURL { sidecar["meet_url"] = meetURL }
if let d = try? JSONSerialization.data(withJSONObject: sidecar, options: .prettyPrinted) {
    try? d.write(to: sessionDir.appendingPathComponent("session.json"))
}

func sizeOf(_ u: URL) -> String {
    let n = ((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? Int) ?? 0
    return String(format: "%.1f MB", Double(n) / 1_048_576)
}
print("\nfar.wav  \(sizeOf(farURL))")
if micAllowed { print("mic.wav  \(sizeOf(micURL))") }
print("session.json written")
