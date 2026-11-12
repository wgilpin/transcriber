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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from pyannote.audio import Audio, Pipeline
from pyannote.core import Annotation

import transcribe as T

DEFAULT_ROOT = Path.home() / "Documents" / "Recordings" / "meetingd"
UTTERANCE_GAP = 0.8      # seconds of silence that ends an utterance
BLEED_WINDOW = 2.0       # how far to look for the same words on the far track
BLEED_OVERLAP = 0.6      # token overlap that counts as "the same words"


# --- models -----------------------------------------------------------------
# Mirrors the loading in transcribe.main(). Duplicated deliberately: that file
# has uncommitted work in it and is not mine to refactor.
def load_models():
    if torch.cuda.is_available():
        pyannote_device, whisper_device = torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        pyannote_device, whisper_device = torch.device("mps"), "cpu"
    else:
        pyannote_device, whisper_device = torch.device("cpu"), "cpu"

    logging.info("whisper on %s, pyannote on %s", whisper_device, pyannote_device)
    model = whisper.load_model(T.WHISPER_MODEL, device=whisper_device)
    pipeline = Pipeline.from_pretrained(T.DIARIZATION_MODEL, use_auth_token=T.hf_api_key)
    if hasattr(pipeline, "segmentation") and hasattr(pipeline.segmentation, "hyperparameters"):
        pipeline.segmentation.hyperparameters = {"min_duration_on": 0.1, "onset": 0.7}
    pipeline.to(pyannote_device)
    return model, pipeline


def transcribe_words(model, path):
    result = model.transcribe(str(path), word_timestamps=True, fp16=False)
    return [
        {"text": w["word"], "start": w["start"], "end": w["end"]}
        for seg in result["segments"]
        for w in seg.get("words", [])
    ]


def diarize(pipeline, path):
    try:
        waveform, sr = Audio(sample_rate=16000, mono=True)(str(path))
        return pipeline({"waveform": waveform, "sample_rate": sr})
    except Exception as e:
        logging.error("diarization failed for %s: %s", path.name, e)
        return Annotation()


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


def level(env, hop, start, end):
    a, b = int(start / hop), max(int(end / hop), int(start / hop) + 1)
    seg = env[a:b]
    return float(seg.max()) if len(seg) else 0.0


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
    kept, dropped = [], 0
    for u in mic_utts:
        tok = tokens(u["text"])
        if not tok:
            kept.append(u)
            continue
        near = [f for f in far_utts
                if f["end"] > u["start"] - BLEED_WINDOW and f["start"] < u["end"] + BLEED_WINDOW]
        far_tok = set().union(*(tokens(f["text"]) for f in near)) if near else set()
        overlap = len(tok & far_tok) / len(tok)

        mic_level = level(mic_env, hop, u["start"], u["end"])
        far_level = level(far_env, hop, u["start"], u["end"])
        ratio = mic_level / far_level if far_level > 1e-6 else float("inf")

        if overlap >= BLEED_OVERLAP and ratio < ratio_threshold:
            dropped += 1
            continue
        kept.append(u)
    return kept, dropped


# --- one session ------------------------------------------------------------
def process_session(d: Path, model, pipeline, me: str, ratio_threshold: float):
    session = json.loads((d / "session.json").read_text())
    meeting = {}
    if (d / "meeting.json").exists():
        meeting = json.loads((d / "meeting.json").read_text())

    mic, far = d / "mic.wav", d / "far.wav"
    base = session.get("started")

    def offset(key):
        if not base or not session.get(key):
            return 0.0
        from datetime import datetime
        f = "%Y-%m-%dT%H:%M:%SZ"
        return (datetime.strptime(session[key], f) - datetime.strptime(base, f)).total_seconds()

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

    mic_utts = group_utterances(mic_words)
    far_utts = group_utterances(far_words)
    dropped = 0
    if mic_utts and far_utts and mic.exists() and far.exists():
        mic_env, hop = rms_envelope(mic)
        far_env, _ = rms_envelope(far)
        mic_utts, dropped = drop_bleed(mic_utts, far_utts, mic_env, far_env, hop, ratio_threshold)
        logging.info("dropped %d mic utterance(s) as speaker bleed", dropped)

    merged = [w for u in mic_utts for w in u["words"]] + far_words
    merged.sort(key=lambda w: w["start"])
    transcript = T.format_final_transcript(merged)

    summary = T.get_llm_summary_and_actions(transcript)
    title = meeting.get("title") or T.get_llm_title(summary or transcript)
    if title in ("Ad-hoc call", "") or T.sanitize_filename(title) != title:
        title = T.get_llm_title(summary or transcript)

    marks = []
    mf = d / "marks.jsonl"
    if mf.exists():
        marks = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]

    out = d / f"{T.sanitize_filename(title)}.md"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if meeting.get("meet_url"):
            f.write(f"*{meeting['meet_url']}*\n\n")
        f.write(f"{summary}\n\n")
        if marks:
            f.write("## Markers\n\n")
            for m in marks:
                f.write(f"- `{m.get('ts','')}` {m.get('mark','')}: {m.get('title','')}\n")
            f.write("\n")
        if dropped:
            f.write(f"*{dropped} mic utterance(s) removed as speaker bleed.*\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(transcript)
        f.write("\n")

    (d / ".processed").write_text(out.name)
    logging.info("wrote %s", out)
    return out


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
    ap.add_argument("--once", action="store_true", help="process what is pending, then exit")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    todo = list(pending(args.root))
    if args.once and not todo:
        logging.info("nothing pending in %s", args.root)
        return

    logging.info("loading models...")
    model, pipeline = load_models()

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
