import AVFoundation
import CoreAudio
import Foundation

// Milestone 1: can we tap a single process's audio output at all?
// Prints level meters. No file writing yet.

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

// ── pick the process to tap ──────────────────────────────────────────────
let args = CommandLine.arguments
let wantBundle: String? = {
    if let i = args.firstIndex(of: "--bundle"), i + 1 < args.count { return args[i + 1] }
    return nil
}()
if let i = args.firstIndex(of: "--out"), i + 1 < args.count {
    let path = args[i + 1]
    FileManager.default.createFile(atPath: path, contents: nil)
    freopen(path, "w", stdout)
    freopen(path, "a", stderr)
    setvbuf(stdout, nil, _IOLBF, 0)
}

let seconds: Double = {
    if let i = args.firstIndex(of: "--seconds"), i + 1 < args.count, let v = Double(args[i + 1]) { return v }
    return 15
}()

struct Proc { let obj: AudioObjectID; let bundle: String; let input: Bool; let output: Bool }
var matches: [Proc] = []
for o in processObjects() {
    let bundle = stringProp(o, kAudioProcessPropertyBundleID) ?? "unknown"
    let input = (u32Prop(o, kAudioProcessPropertyIsRunningInput) ?? 0) != 0
    let output = (u32Prop(o, kAudioProcessPropertyIsRunningOutput) ?? 0) != 0
    if let want = wantBundle {
        if bundle.localizedCaseInsensitiveContains(want) {
            matches.append(Proc(obj: o, bundle: bundle, input: input, output: output))
        }
    } else if input {
        matches.append(Proc(obj: o, bundle: bundle, input: input, output: output))
    }
}

guard !matches.isEmpty else {
    print("no matching audio process. Play audio in Chrome, or pass --bundle Chrome")
    exit(1)
}

print("matching audio processes:")
for m in matches { print("   obj \(m.obj)  \(m.bundle)  mic=\(m.input)  speaker=\(m.output)") }

// Chrome spreads audio across several helpers; tap every one that is playing,
// and fall back to all matches if none reports output yet.
let playing = matches.filter { $0.output }
let chosen = playing.isEmpty ? matches : playing
print("tapping \(chosen.count) process(es): \(chosen.map { String($0.obj) }.joined(separator: ", "))")

// ── create the process tap ───────────────────────────────────────────────
let tapDesc = CATapDescription(stereoMixdownOfProcesses: chosen.map { $0.obj })
tapDesc.name = "meetingd-farend"
tapDesc.isPrivate = true
tapDesc.muteBehavior = .unmuted          // keep the user hearing the call

var tapID = AudioObjectID(kAudioObjectUnknown)
let tapErr = AudioHardwareCreateProcessTap(tapDesc, &tapID)
guard tapErr == noErr, tapID != kAudioObjectUnknown else {
    print("AudioHardwareCreateProcessTap failed: OSStatus \(tapErr)")
    print("(macOS may require audio-recording permission for this)")
    exit(2)
}
print("tap created (\(tapID))")

// ── wrap it in a private aggregate device so we can read from it ─────────
func defaultOutputUID() -> String? {
    for sel in [kAudioHardwarePropertyDefaultSystemOutputDevice,
                kAudioHardwarePropertyDefaultOutputDevice] {
        var a = paddr(sel)
        var dev = AudioObjectID(kAudioObjectUnknown)
        var sz = UInt32(MemoryLayout<AudioObjectID>.size)
        if AudioObjectGetPropertyData(sysObj, &a, 0, nil, &sz, &dev) == noErr,
           dev != kAudioObjectUnknown,
           let uid = stringProp(dev, kAudioDevicePropertyDeviceUID) {
            return uid
        }
    }
    return nil
}

let clockUID = defaultOutputUID()
print("clock source: \(clockUID ?? "none found")")

let aggUID = UUID().uuidString
let aggDict: [String: Any] = [
    kAudioAggregateDeviceNameKey: "meetingd-capture",
    kAudioAggregateDeviceUIDKey: aggUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceMainSubDeviceKey: clockUID ?? "",
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [] as [[String: Any]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapUIDKey: tapDesc.uuid.uuidString,
        kAudioSubTapDriftCompensationKey: true,
    ]],
]

var aggID = AudioObjectID(kAudioObjectUnknown)
let aggErr = AudioHardwareCreateAggregateDevice(aggDict as CFDictionary, &aggID)
guard aggErr == noErr, aggID != kAudioObjectUnknown else {
    print("AudioHardwareCreateAggregateDevice failed: OSStatus \(aggErr)")
    AudioHardwareDestroyProcessTap(tapID)
    exit(3)
}

var fmt = AudioStreamBasicDescription()
var fsz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
var fa = paddr(kAudioDevicePropertyStreamFormat, kAudioObjectPropertyScopeInput)
AudioObjectGetPropertyData(aggID, &fa, 0, nil, &fsz, &fmt)
let isFloat = fmt.mFormatFlags & kAudioFormatFlagIsFloat != 0
let isInt = fmt.mFormatFlags & kAudioFormatFlagIsSignedInteger != 0
print(String(format: "aggregate ready: %.0f Hz, %d ch, %d bits, float=%@ int=%@",
             fmt.mSampleRate, fmt.mChannelsPerFrame, fmt.mBitsPerChannel,
             isFloat ? "yes" : "no", isInt ? "yes" : "no"))

// ── read ─────────────────────────────────────────────────────────────────
final class Meter: @unchecked Sendable {
    var peak: Float = 0
    var sum: Double = 0
    var count: Int = 0
    var callbacks: Int = 0
    let lock = NSLock()
}
let meter = Meter()
nonisolated(unsafe) var dumpedShape = false

var procID: AudioDeviceIOProcID?
let ioErr = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { _, inData, _, _, _ in
    let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
    if !dumpedShape {
        dumpedShape = true
        var msg = "first callback: \(abl.count) buffer(s)"
        for (i, b) in abl.enumerated() {
            msg += "\n   buf \(i): \(b.mNumberChannels) ch, \(b.mDataByteSize) bytes"
            if let raw = b.mData, b.mDataByteSize >= 16 {
                let bytes = raw.assumingMemoryBound(to: UInt8.self)
                let hex = (0..<16).map { String(format: "%02x", bytes[$0]) }.joined(separator: " ")
                msg += "  first16: \(hex)"
            }
        }
        print(msg); fflush(stdout)
    }
    var peak: Float = 0
    var sum: Double = 0
    var n = 0
    for buf in abl {
        guard let raw = buf.mData else { continue }
        let frames = Int(buf.mDataByteSize) / MemoryLayout<Float>.size
        let samples = raw.assumingMemoryBound(to: Float.self)
        for i in 0..<frames {
            let v = abs(samples[i])
            if v > peak { peak = v }
            sum += Double(v * v)
            n += 1
        }
    }
    meter.lock.lock()
    meter.callbacks += 1
    meter.peak = max(meter.peak, peak)
    meter.sum += sum
    meter.count += n
    meter.lock.unlock()
}
guard ioErr == noErr, let procID else {
    print("AudioDeviceCreateIOProcIDWithBlock failed: OSStatus \(ioErr)")
    exit(4)
}

AudioDeviceStart(aggID, procID)
print("listening for \(Int(seconds))s — play some audio in the call\n")

let deadline = Date().addingTimeInterval(seconds)
var silentTicks = 0
var totalCallbacks = 0
while Date() < deadline {
    Thread.sleep(forTimeInterval: 0.5)
    meter.lock.lock()
    let peak = meter.peak, rms = meter.count > 0 ? sqrt(meter.sum / Double(meter.count)) : 0
    let cbs = meter.callbacks; meter.callbacks = 0
    meter.peak = 0; meter.sum = 0; meter.count = 0
    meter.lock.unlock()
    totalCallbacks += cbs
    if peak < 0.0001 { silentTicks += 1 }
    let bars = Int(min(peak, 1.0) * 40)
    print(String(format: "cb %3d  peak %.4f  rms %.4f  |%@", cbs, peak, rms,
                 String(repeating: "█", count: bars).padding(toLength: 30, withPad: " ", startingAt: 0)))
    fflush(stdout)
}

AudioDeviceStop(aggID, procID)
AudioDeviceDestroyIOProcID(aggID, procID)
AudioHardwareDestroyAggregateDevice(aggID)
AudioHardwareDestroyProcessTap(tapID)

if totalCallbacks == 0 {
    print("\nIOProc NEVER FIRED — the aggregate device is not running (clock problem).")
} else if silentTicks > Int(seconds * 2) - 2 {
    print("\nIOProc fired \(totalCallbacks)x but every buffer was silent.")
} else {
    print("\nGOT AUDIO — process tap works.")
}
