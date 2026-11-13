#!/usr/bin/env python3
"""Turn finished meetingd sessions into notes.

A session folder holds two tracks - mic.wav is you, far.wav is everyone else -
plus session.json describing them. Because the tracks are separate, your own
speech needs no diarization at all; only the far track does.

Imports transcribe.py rather than modifying it, so that file stays yours.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import transcribe as T

DEFAULT_ROOT = Path.home() / "Documents" / "Recordings" / "meetingd"
UTTERANCE_GAP = 0.8      # seconds of silence that ends an utterance
BLEED_WINDOW = 2.0       # how far to look for the same words on the far track
BLEED_OVERLAP = 0.6      # token overlap that counts as "the same words"
DECISIVE_RATIO = 0.15    # mic this much quieter than the far track is not you
DECISIVE_OVERLAP = 0.3   # ...and needs only a partial text match to convict
MIC_PCT = 50             # the mic is continuous: its median is representative
FAR_PCT = 90             # the tap is gappy: only an upper percentile shows speech
DEAD_RUN = 5.0           # exact silence this long on the far track means the tap died
FAR_ALIVE_MIN = 0.10     # below this the far track never worked, so nothing can be dead
# Measured: a real call with people talking runs ~72% active, because Chrome
# streams continuously once you are in a room. A solo call is ~1.7% - stray
# join and leave sounds. 10% sits in open space between the two.


# --- models -----------------------------------------------------------------
# Thin wrappers over transcribe.py, so both entry points load the same models.
def load_models():
    if torch.cuda.is_available():
        pyannote_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        pyannote_device = torch.device("mps")
    else:
        pyannote_device = torch.device("cpu")

    logging.info("pyannote on %s, ASR on MLX", pyannote_device)
    return T.load_asr(), T.load_diarizer(pyannote_device)


def transcribe_words(model, path):
    return T.transcribe_words(model, path)


def diarize(pipeline, path):
    return T.diarize(pipeline, path)


# --- echo bleed -------------------------------------------------------------
def rms_envelope(path, hop=0.1):
    """Coarse RMS over time, used to tell your voice from speaker bleed."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    n = max(int(sr * hop), 1)
    frames = len(mono) // n
    if frames == 0:
        return np.zeros(1), hop
    block = mono[: frames * n].reshape(frames, n)
    return np.sqrt((block ** 2).mean(axis=1)), hop


def level(env, hop, start, end, pct=50):
    """How loud a track is over a span, as a percentile of its RMS blocks.

    Never the peak: one thump on the desk outweighs a whole sentence of bleed.
    The two tracks need different percentiles, because they fail differently.
    The mic is continuous, so its median is what the span really sounds like.
    The far track arrives in bursts from the process tap and is better than
    two-thirds exact zeros even mid-sentence, so its median is 0 and only an
    upper percentile says whether anyone was talking.
    """
    a, b = int(start / hop), max(int(end / hop), int(start / hop) + 1)
    seg = env[a:b]
    return float(np.percentile(seg, pct)) if len(seg) else 0.0


def dead_spans(env, hop, minimum=DEAD_RUN):
    """Stretches where the far track is not quiet but absent.

    A process tap can stop delivering - Chrome moves its audio to a helper the
    tap never knew about - and what lands on disk is exact zeros, not a room
    tone. Speech reaching the mic during such a stretch came through the
    speakers with no second copy to compare it against, so it cannot be
    attributed to anyone. A real pause in the conversation still has a noise
    floor and will not match this.
    """
    silent = env < 1e-5
    spans, run = [], None
    for i, q in enumerate(silent):
        if q and run is None:
            run = i
        elif not q and run is not None:
            if (i - run) * hop >= minimum:
                spans.append((run * hop, i * hop))
            run = None
    if run is not None and (len(silent) - run) * hop >= minimum:
        spans.append((run * hop, len(silent) * hop))
    return spans


def in_span(spans, start, end):
    """True when most of an utterance falls inside one of the spans."""
    covered = sum(max(0.0, min(end, b) - max(start, a)) for a, b in spans)
    return covered > (end - start) / 2


def group_utterances(words, gap=UTTERANCE_GAP):
    out, cur = [], None
    for w in words:
        if cur and w["start"] - cur["end"] <= gap and w.get("speaker") == cur.get("speaker"):
            cur["end"] = w["end"]
            cur["words"].append(w)
        else:
            if cur:
                out.append(cur)
            cur = {"start": w["start"], "end": w["end"],
                   "speaker": w.get("speaker"), "words": [w]}
    if cur:
        out.append(cur)
    for u in out:
        u["text"] = "".join(x["text"] for x in u["words"])
    return out


def tokens(text):
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t}


def drop_bleed(mic_utts, far_utts, mic_env, far_env, hop, ratio_threshold):
    """Remove far-end speech that leaked into the mic through the speakers.

    Text match alone would also delete you talking over someone. So a matching
    utterance is only dropped when the mic is quiet relative to the far track -
    if you were speaking too, the mic is louder than bleed can account for.
    """
    kept, dropped = [], []
    for u in mic_utts:
        tok = tokens(u["text"])
        if not tok:
            kept.append(u)
            continue
        near = [f for f in far_utts
                if f["end"] > u["start"] - BLEED_WINDOW and f["start"] < u["end"] + BLEED_WINDOW]
        far_tok = set().union(*(tokens(f["text"]) for f in near)) if near else set()
        overlap = len(tok & far_tok) / len(tok)

        mic_level = level(mic_env, hop, u["start"], u["end"], MIC_PCT)
        far_level = level(far_env, hop, u["start"], u["end"], FAR_PCT)
        ratio = mic_level / far_level if far_level > 1e-6 else float("inf")

        # The recognizer rarely transcribes the two copies of a sentence identically,
        # so strong level evidence is allowed to convict on a weaker text match.
        decisive = ratio < DECISIVE_RATIO and overlap >= DECISIVE_OVERLAP
        if decisive or (overlap >= BLEED_OVERLAP and ratio < ratio_threshold):
            dropped.append(u)
            continue
        kept.append(u)
    return kept, dropped


# --- segments ---------------------------------------------------------------
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def ts_offset(ts, base):
    """Seconds between an absolute ISO timestamp and the session start."""
    return (datetime.strptime(ts, TS_FORMAT) - datetime.strptime(base, TS_FORMAT)).total_seconds()


def clock(seconds):
    s = int(max(seconds, 0))
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


def segments_for(session, meeting, marks):
    """One entry per meeting inside the recording.

    The detector writes a split mark whenever the live Meet room changes while
    the mic stays held, so back-to-back calls are a single recording but several
    meetings. Each segment carries the identity that was live at the time.
    """
    base = session.get("started")
    duration = float(session.get("duration_s") or 0) or float("inf")
    segs = [{"start": 0.0, "title": meeting.get("title"),
             "meet_url": meeting.get("meet_url"), "attendees": meeting.get("attendees")}]
    for m in marks:
        if m.get("mark") != "split" or not base or not m.get("ts"):
            continue
        try:
            at = ts_offset(m["ts"], base)
        except ValueError:
            continue
        if at <= segs[-1]["start"]:
            continue
        segs.append({"start": at, "title": m.get("title"),
                     "meet_url": m.get("meet_url"), "attendees": m.get("attendees")})
    for i, s in enumerate(segs):
        s["end"] = segs[i + 1]["start"] if i + 1 < len(segs) else duration
    return segs


def write_note(d: Path, i: int, segs: list, seg: dict, words: list, dropped: int, marks: list):
    """One note for one meeting. Word times are already rebased to its start."""
    transcript = T.format_final_transcript(words)
    summary = T.get_llm_summary_and_actions(transcript)

    title = seg.get("title")
    if not title or title == "Ad-hoc call" or T.sanitize_filename(title) != title:
        title = T.get_llm_title(summary or transcript)

    name = T.sanitize_filename(title)
    if len(segs) > 1:
        name = f"{i} - {name}"
    out = d / f"{name}.md"

    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if len(segs) > 1:
            f.write(f"*Meeting {i} of {len(segs)} in this recording, "
                    f"audio from {clock(seg['start'])}*\n\n")
        if seg.get("meet_url"):
            f.write(f"*{seg['meet_url']}*\n\n")
        f.write(f"{summary}\n\n")
        other = [m for m in marks if m.get("mark") != "split"]
        if other:
            f.write("## Markers\n\n")
            for m in other:
                f.write(f"- `{m.get('ts','')}` {m.get('mark','')}: {m.get('title','')}\n")
            f.write("\n")
        if dropped:
            f.write(f"*{dropped} mic utterance(s) removed as speaker bleed.*\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(transcript)
        f.write("\n")

    logging.info("wrote %s", out)
    return out


# --- one session ------------------------------------------------------------
def transcribe_tracks(d: Path, session: dict, model, pipeline, me: str):
    """The expensive half: both tracks to timed words, the far one diarized."""
    mic, far = d / "mic.wav", d / "far.wav"
    base = session.get("started")

    def offset(key):
        if not base or not session.get(key):
            return 0.0
        return ts_offset(session[key], base)

    logging.info("transcribing far track...")
    far_words = transcribe_words(model, far) if far.exists() else []
    far_shift = offset("far_started")
    for w in far_words:
        w["start"] += far_shift
        w["end"] += far_shift
    far_words = T.combine_transcription_and_diarization(
        far_words, T.apply_collar_fix(diarize(pipeline, far)))
    for _ in range(5):
        before = [w.copy() for w in far_words]
        far_words = T.fix_trailing_unknowns(T.fix_leading_unknowns(far_words))
        if far_words == before:
            break

    logging.info("transcribing mic track (speaker is known: %s)...", me)
    mic_words = transcribe_words(model, mic) if mic.exists() else []
    mic_shift = offset("mic_started")
    for w in mic_words:
        w["start"] += mic_shift
        w["end"] += mic_shift
        w["speaker"] = me

    return far_words, group_utterances(mic_words)


def process_session(d: Path, model, pipeline, me: str, ratio_threshold: float):
    session = json.loads((d / "session.json").read_text())
    meeting = {}
    if (d / "meeting.json").exists():
        meeting = json.loads((d / "meeting.json").read_text())

    mic, far = d / "mic.wav", d / "far.wav"
    cache = d / "words.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}

    if "far_words" in cached:
        logging.info("reusing cached transcription")
        far_words, mic_utts = cached["far_words"], cached["mic_utts"]
    else:
        far_words, mic_utts = transcribe_tracks(d, session, model, pipeline, me)
        cache.write_text(json.dumps({"far_words": far_words, "mic_utts": mic_utts}))

    # Filtering happens after the cache, so changing --bleed-ratio is seconds.
    dropped = []
    far_utts = group_utterances(far_words)
    if mic_utts and far_utts and mic.exists() and far.exists():
        mic_env, hop = rms_envelope(mic)
        far_env, _ = rms_envelope(far)
        mic_utts, dropped = drop_bleed(mic_utts, far_utts, mic_env, far_env, hop, ratio_threshold)
        logging.info("dropped %d mic utterance(s) as speaker bleed", len(dropped))

        # Where the far track died, mic speech has no counterpart to test it
        # against. Most of it is bleed, some of it is you, and nothing here can
        # tell them apart - so say so rather than put your name to all of it.
        # A far track that was never alive is not a failed tap - it is a call
        # with nobody else talking. Nothing bled through speakers that played
        # nothing, so the mic is unambiguously you and relabelling it UNKNOWN
        # would be worse than saying nothing.
        far_alive = float((far_env >= 1e-5).mean())
        dead = dead_spans(far_env, hop) if far_alive > FAR_ALIVE_MIN else []
        if far_alive <= FAR_ALIVE_MIN:
            logging.info("far track silent throughout (%.1f%% active) - treating as "
                         "a call with no one else speaking", far_alive * 100)
        if dead:
            total = sum(b - a for a, b in dead)
            orphaned = 0
            for u in mic_utts:
                if in_span(dead, u["start"], u["end"]):
                    orphaned += 1
                    for w in u["words"]:
                        w["speaker"] = "UNKNOWN"
            logging.warning("far track dead for %.0fs in %d span(s) - %d utterance(s) "
                            "could not be attributed", total, len(dead), orphaned)

    merged = [w for u in mic_utts for w in u["words"]] + far_words
    merged.sort(key=lambda w: w["start"])
    return write_notes(d, session, meeting, merged, dropped)


def write_notes(d: Path, session: dict, meeting: dict, merged: list, dropped: list):
    marks = []
    mf = d / "marks.jsonl"
    if mf.exists():
        marks = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]

    segs = segments_for(session, meeting, marks)
    if len(segs) > 1:
        logging.info("%d meetings in this recording", len(segs))

    notes = []
    for i, seg in enumerate(segs, 1):
        words = [dict(w, start=w["start"] - seg["start"], end=w["end"] - seg["start"])
                 for w in merged if seg["start"] <= w["start"] < seg["end"]]
        if not words:
            logging.info("segment %d (%s): no speech, skipped", i, seg.get("title") or "untitled")
            continue
        n = sum(1 for u in dropped if seg["start"] <= u["start"] < seg["end"])
        notes.append(write_note(d, i, segs, seg, words, n, marks))

    (d / ".processed").write_text("\n".join(x.name for x in notes))
    return notes


OLLAMA = "http://127.0.0.1:11434/api/tags"


def ensure_ollama(wait=40):
    """Start Ollama if it is not up, and wait for it to answer.

    The app runs this unattended, so a quit menu bar app should not be the
    difference between a note with a summary and a note with an error in place
    of one. Starting the server is cheap; the model load is what takes time,
    and that happens on the first query either way.
    """
    import subprocess
    import urllib.error
    import urllib.request

    def up():
        try:
            urllib.request.urlopen(OLLAMA, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            return False

    if up():
        return True
    logging.info("ollama is not running - starting it")
    # Launched from the app, PATH is launchd's minimal one and will not have
    # /usr/local/bin in it, so look for the binary rather than trusting PATH.
    import shutil
    exe = shutil.which("ollama") or next(
        (c for c in ("/usr/local/bin/ollama", "/opt/homebrew/bin/ollama") if Path(c).exists()),
        None)
    if exe:
        subprocess.Popen([exe, "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    else:
        subprocess.run(["open", "-a", "Ollama"], capture_output=True)

    for _ in range(wait):
        if up():
            logging.info("ollama is up")
            return True
        time.sleep(1)
    logging.warning("ollama did not come up - notes will have no summary")
    return False


def pending(root: Path):
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "session.json").exists():
            continue              # capture still running or died
        if (d / "capture.pid").exists():
            continue              # still recording
        if (d / ".processed").exists():
            continue
        yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--me", default="Bill", help="speaker label for the mic track")
    ap.add_argument("--bleed-ratio", type=float, default=0.5,
                    help="mic/far level below which matching speech is treated as bleed")
    ap.add_argument("--session", type=Path,
                    help="re-write the notes for one session folder, then exit")
    ap.add_argument("--once", action="store_true", help="process what is pending, then exit")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    if args.session:
        logging.info("loading models...")
        model, pipeline = load_models()
        ensure_ollama()
        process_session(args.session, model, pipeline, args.me, args.bleed_ratio)
        return

    args.root.mkdir(parents=True, exist_ok=True)
    todo = list(pending(args.root))
    if args.once and not todo:
        logging.info("nothing pending in %s", args.root)
        return

    logging.info("loading models...")
    model, pipeline = load_models()
    ensure_ollama()

    while True:
        for d in pending(args.root):
            logging.info("=== %s ===", d.name)
            try:
                process_session(d, model, pipeline, args.me, args.bleed_ratio)
            except Exception as e:
                logging.error("failed on %s: %s", d.name, e, exc_info=True)
                (d / ".failed").write_text(str(e))
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
