import os
import sys
import re
import argparse
import logging
from datetime import timedelta
from dotenv import load_dotenv

import torch
import ollama
from huggingface_hub.errors import GatedRepoError
from parakeet_mlx import from_pretrained as load_parakeet
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.core.io import Audio

from colorama import Fore, Back, Style

# --- Configuration ---
# Parakeet TDT runs on the GPU through MLX, so unlike openai-whisper it needs no
# CPU fallback on this machine. Measured at ~105x realtime: a half-hour meeting
# transcribes in about 15 seconds.
ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
# A whole meeting at once would hold the full attention matrix in memory; these
# chunks keep it flat, and the overlap is what stitches words across the seams.
CHUNK_DURATION_S = 120.0
CHUNK_OVERLAP_S = 15.0
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
# Gated on Hugging Face: accept the terms with the account behind HF_API_TOKEN.
# pyannote.audio 4 resolves the old 3.1 name here too, so there is no way round it.
OLLAMA_MODEL = "gemma4:26b"
SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac"]
# Increased threshold for fixing UNKNOWN segments to handle longer pauses
CLEANUP_TIME_THRESHOLD_S = 1.5

# --- Environment Setup ---
load_dotenv()
hf_api_key = os.getenv("HF_API_TOKEN")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
if hf_api_key:
    logging.info("Hugging Face API token loaded.")
else:
    logging.warning(
        "Hugging Face API token not found. Diarization may fail if model needs auth."
    )


# --- Speech recognition ---


def load_asr():
    """Loads the ASR model. MLX uses the GPU directly, so there is no device to pick."""
    logging.info("Loading ASR model: %s", ASR_MODEL)
    return load_parakeet(ASR_MODEL)


def transcribe_words(asr_model, filepath):
    """Transcribes audio into word chunks of {"text", "start", "end"}.

    Parakeet emits sentencepiece subword tokens - " A", "w", "es", "ome", "." -
    in which a leading space marks the start of a word. Regrouping them on that
    boundary reproduces Whisper's word list exactly, leading space and trailing
    punctuation included, so everything downstream of here is unchanged.
    """
    result = asr_model.transcribe(
        str(filepath), chunk_duration=CHUNK_DURATION_S, overlap_duration=CHUNK_OVERLAP_S
    )
    words = []
    for sentence in result.sentences:
        for token in sentence.tokens:
            if words and not token.text.startswith(" "):
                words[-1]["text"] += token.text
                words[-1]["end"] = token.end
            else:
                words.append(
                    {"text": token.text, "start": token.start, "end": token.end}
                )
    return words


def load_diarizer(device):
    """Loads the diarization pipeline.

    There is no older model to fall back to: pyannote.audio 4 resolves the 3.1
    name to community-1 as well, so a gated repo fails either way. Say so
    plainly rather than burying a 403 traceback.
    """
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_api_key)
    except GatedRepoError:
        raise SystemExit(
            f"\n{DIARIZATION_MODEL} is gated on Hugging Face.\n"
            f"Accept the terms at https://huggingface.co/{DIARIZATION_MODEL}\n"
            "with the account that owns HF_API_TOKEN, then run this again.\n"
        )
    pipeline.to(device)
    return pipeline


def diarize(pipeline, filepath):
    """Runs diarization and returns a plain Annotation.

    pyannote.audio 4 wraps the result in a DiarizeOutput holding two versions:
    speaker_diarization, where overlapping speech belongs to everyone talking,
    and exclusive_speaker_diarization, where each instant has one speaker. We
    label individual words, which can only have one speaker, so we take the
    exclusive one - otherwise a word in an overlap is assigned to whichever
    track happens to come first.
    """
    try:
        waveform, sample_rate = Audio(sample_rate=16000, mono="downmix")(str(filepath))
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    except Exception as e:
        logging.error("Diarization failed for %s: %s", filepath, e, exc_info=True)
        return Annotation()
    return getattr(result, "exclusive_speaker_diarization", result)

# --- Helper Functions ---


def format_timestamp(seconds: float) -> str:
    """Converts seconds to a HH:MM:SS.mmm format."""
    return str(timedelta(seconds=seconds))


def sanitize_filename(name):
    """Removes invalid characters from a string to make it a valid filename."""
    name = name.replace("\n", " ").replace("**", "").strip()
    return re.sub(r'[\\/*?:"<>|]', "", name)


def apply_collar_fix(
    diarization: Annotation,
    collar_seconds: float = 0.5,
    unknown_duration_threshold_s: float = 1.5,
) -> Annotation:
    """Corrects short, 'sandwiched' UNKNOWN segments."""
    if not diarization.uri or len(diarization.labels()) == 0:
        return diarization

    corrected_diarization = diarization.copy()
    turns = list(diarization.itertracks(yield_label=True))
    if len(turns) < 3:
        return diarization

    for i in range(1, len(turns) - 1):
        prev_turn, _, prev_speaker = turns[i - 1]
        current_turn, _, current_speaker = turns[i]
        next_turn, _, next_speaker = turns[i + 1]

        is_short_unknown = (
            "UNKNOWN" in current_speaker
            and current_turn.duration < unknown_duration_threshold_s
        )
        speakers_match = prev_speaker == next_speaker
        is_within_collar = (current_turn.start - prev_turn.end) < collar_seconds and (
            next_turn.start - current_turn.end
        ) < collar_seconds

        if is_short_unknown and speakers_match and is_within_collar:
            logging.info(
                f"Collar Fix: Re-labeling segment [{current_turn.start:.2f}s - {current_turn.end:.2f}s] from {current_speaker} to {prev_speaker}."
            )
            corrected_diarization[current_turn] = prev_speaker

    return corrected_diarization.support()


def fix_leading_unknowns(combined_result, time_threshold=CLEANUP_TIME_THRESHOLD_S):
    """Merges an UNKNOWN word with the subsequent speaker's turn if they are close in time."""
    if len(combined_result) < 2:
        return combined_result

    for i in range(len(combined_result) - 1):
        current_item = combined_result[i]
        next_item = combined_result[i + 1]

        if (
            current_item["speaker"] == "UNKNOWN"
            and next_item["speaker"] != "UNKNOWN"
            and (next_item["start"] - current_item["end"]) < time_threshold
        ):
            current_item["speaker"] = next_item["speaker"]

    return combined_result


def fix_trailing_unknowns(combined_result, time_threshold=CLEANUP_TIME_THRESHOLD_S):
    """Merges an UNKNOWN word with the preceding speaker's turn if they are close in time."""
    if len(combined_result) < 2:
        return combined_result

    for i in range(len(combined_result) - 1, 0, -1):
        current_item = combined_result[i]
        prev_item = combined_result[i - 1]

        if (
            current_item["speaker"] == "UNKNOWN"
            and prev_item["speaker"] != "UNKNOWN"
            and (current_item["start"] - prev_item["end"]) < time_threshold
        ):
            current_item["speaker"] = prev_item["speaker"]

    return combined_result


def combine_transcription_and_diarization(transcription_chunks, diarization_result):
    """Merges Whisper's transcription with Pyannote's diarization."""
    full_transcript = []
    speaker_turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization_result.itertracks(yield_label=True)
    ]

    for chunk in transcription_chunks:
        word_start = chunk.get("start")
        word_end = chunk.get("end")

        if word_start is None or word_end is None:
            continue

        word_mid_time = word_start + (word_end - word_start) / 2
        speaker = "UNKNOWN"
        for turn in speaker_turns:
            if turn["start"] <= word_mid_time <= turn["end"]:
                speaker = turn["speaker"]
                break

        full_transcript.append(
            {
                "start": word_start,
                "end": word_end,
                "text": chunk.get("text", ""),
                "speaker": speaker,
            }
        )
    return full_transcript


def format_final_transcript(combined_result):
    """Formats the combined transcript into a readable, speaker-labeled string, merging consecutive utterances from the same speaker."""
    # This first block groups words into lines based on speaker changes.
    # It can sometimes create separate consecutive lines for the same speaker if there are long pauses.
    output_lines = []
    current_speaker = None
    current_line = ""
    for item in combined_result:
        if item["speaker"] != current_speaker:
            if current_line:
                output_lines.append(current_line)
            timestamp = format_timestamp(item["start"])
            current_speaker = item["speaker"]
            current_line = f"[{timestamp}] **{current_speaker}**:"
        current_line += item["text"]
    if current_line:
        output_lines.append(current_line)

    if not output_lines:
        return ""

    # --- Post-processing step to merge consecutive lines from the same speaker ---
    merged_lines = []
    speaker_regex = re.compile(r"\[.*?\] \*\*(.*?)\*\*:")

    for line in output_lines:
        match = speaker_regex.search(line)
        # If a line doesn't have a speaker tag, treat it as a continuation of the previous line.
        if not match:
            if merged_lines:
                merged_lines[-1] += f" {line.strip()}"
            else:
                merged_lines.append(line)
            continue

        current_speaker_name = match.group(1)
        text_content = line[match.end() :]

        # Check the last line in our merged list.
        if not merged_lines:
            merged_lines.append(line)
            continue

        last_line = merged_lines[-1]
        last_speaker_match = speaker_regex.search(last_line)

        # If the last line has a speaker, it's the same as the current one, AND the speaker is not UNKNOWN, then merge.
        if (
            last_speaker_match
            and last_speaker_match.group(1) == current_speaker_name
            and current_speaker_name != "UNKNOWN"
        ):
            # Append the text content of the current line to the end of the last line.
            merged_lines[-1] += text_content
        else:
            # Otherwise, this is a new speaker (or UNKNOWN), so add the new line as is.
            merged_lines.append(line)

    # Perform final cleanup on the now-merged lines.
    return "\n\n".join(
        line.replace(" :", ":").replace(" ,", ",").replace(" .", ".")
        for line in merged_lines
    )


def query_ollama(prompt):
    """Sends a prompt to Ollama and returns the response."""
    logging.info(f"Querying Ollama with model '{OLLAMA_MODEL}'...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for summarizing meeting transcripts.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Failed to query Ollama: {e}")
        return "Error: Could not retrieve response from Ollama."


def get_llm_title(source_text: str, attendees: list[str] = None):
    """Generates a descriptive filename title from a text (summary or transcript) using the LLM."""
    logging.info("Generating descriptive title with LLM...")

    possible_names_prompt = ""
    if attendees and len(attendees) >= 2:
        possible_names_prompt = f"""Start the filename with the names {attendees[0]} and {attendees[1]}. Do not mention these names again.
Example: Bill and Tony discussing expenses"""

    prompt = f"""Based on the following text, generate a single, short, descriptive phrase of 4-8 words suitable for a filename.
- Do NOT use any special characters like quotes or colons.
- Do NOT explain your thinking.
- Provide ONLY the title phrase itself.

Example: Initial Project Kick-off with the Analytics Team

{possible_names_prompt}

Here is the text:
---
{source_text}
---
"""
    title = query_ollama(prompt)
    clean_title = title.split("\n")[0].strip()
    return sanitize_filename(clean_title)


def get_llm_summary_and_actions(transcript_text):
    """
    Generates a summary and action items from a transcript in a single LLM call.
    """
    logging.info("Generating summary and action items with LLM...")
    prompt = f"""Please provide a summary and action points (if any) for the following meeting transcript.
Focus on the key decisions and outcomes.
The summary should be short and cover the main points. Do not discuss the tone or length of the meeting.
For the action items, extract all explicit action items and identify the assigned person if mentioned. Format as a markdown list. If no action items are found, state that clearly.
Your entire response MUST be in markdown format, starting with a '## Summary' section followed by an '## Action Items' section. For example:

## Summary

The meeting was a discussion between Bill and Dave on pipeline observability. They concluded more work was needed.

## Action Items

* Document the current status (Assigned to Bill).
* Set up a team meeting to discuss (Assigned to Dave).

Here is the transcript:
---
{transcript_text}
---
"""
    summary_and_actions = query_ollama(prompt)
    return summary_and_actions


# --- Time Marker Highlight Functions ---


def parse_tmk_timestamps(filepath: str) -> list[float]:
    """Parses a .tmk file and returns a list of timestamps in seconds."""
    if not os.path.exists(filepath):
        return []

    timestamps = []
    # Regex for formats like [00061:48.57]
    pattern = re.compile(r"\[(\d+):(\d{2}\.\d{2})\]")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Could not read tmk file {filepath}: {e}")
        return []

    for match in pattern.finditer(content):
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total_seconds = (minutes * 60) + seconds
        timestamps.append(total_seconds)

    logging.info(
        f"Found {len(timestamps)} time markers in {os.path.basename(filepath)}."
    )
    return timestamps


def get_formatted_text_in_window(
    combined_transcript: list, start_sec: float, end_sec: float
) -> str:
    """Extracts transcript text within a time window and formats it as 'SPEAKER: text' turns."""
    window_words = [
        item
        for item in combined_transcript
        if item["start"] < end_sec and item["end"] > start_sec
    ]

    if not window_words:
        return ""

    # Group words into conversational turns
    turns = []
    current_turn_speaker = window_words[0]["speaker"]
    current_turn_text = ""
    for word in window_words:
        if word["speaker"] != current_turn_speaker:
            if current_turn_text:
                turns.append(f"{current_turn_speaker}: {current_turn_text.strip()}")
            current_turn_speaker = word["speaker"]
            current_turn_text = ""
        current_turn_text += word["text"]

    if current_turn_text:
        turns.append(f"{current_turn_speaker}: {current_turn_text.strip()}")

    return "\n".join(turns)


def get_llm_topic(context_text: str) -> str:
    """Uses an LLM to find the main topic of a text block."""
    logging.info("Getting topic for highlight context...")
    prompt = f"""Analyze the following transcript excerpt and identify the primary topic or theme.
- Be concise. Your response should be a short phrase of 3-7 words.
- Do not add any preamble like "The topic is...".
- Example response: 'Quarterly budget review for engineering'

Here is the transcript excerpt:
---
{context_text}
---
"""
    topic = query_ollama(prompt)
    return topic.strip().replace('"', "")


def get_llm_highlight(topic: str, highlight_text: str) -> str:
    """Uses an LLM to summarize a text snippet with a specific topic in mind."""
    logging.info("Getting summary for highlight...")
    prompt = f"""The general topic of conversation is: "{topic}"

Based on this topic, please summarize the most important statement, question, or decision in the following short transcript snippet.
- The summary must be a single, complete sentence.
- If a specific person is central to the point, start the sentence with their name (e.g., "Bill argued that...").
- Do not add any explanation or preamble.

Here is the snippet:
---
{highlight_text}
---
"""
    highlight = query_ollama(prompt)
    return highlight.strip()


def generate_highlights_section(combined_transcript: list, audio_filepath: str) -> str:
    """Checks for a .tmk file and generates a markdown section for time-marked highlights."""
    tmk_filepath = os.path.splitext(audio_filepath)[0] + ".tmk"
    time_markers = parse_tmk_timestamps(tmk_filepath)
    if not time_markers:
        return ""

    logging.info("--- Generating Highlights from Time Markers ---")
    highlights = []
    for idx, tmk_seconds in enumerate(time_markers):
        logging.info(f"Processing marker {idx}/{len(time_markers)}...")
        # 1. Get 90-minute context
        context_start = max(0, tmk_seconds - 90 * 60)
        context_text = get_formatted_text_in_window(
            combined_transcript, context_start, tmk_seconds
        )
        if not context_text:
            logging.warning(f"No context found for marker at {tmk_seconds:.2f}s.")
            continue

        # 2. Get topic from context
        topic = get_llm_topic(context_text)

        # 3. Get 4-minute highlight snippet (3 before, 1 after)
        highlight_start = max(0, tmk_seconds - 3 * 60)
        highlight_end = tmk_seconds + 1 * 60
        highlight_text = get_formatted_text_in_window(
            combined_transcript, highlight_start, highlight_end
        )
        if not highlight_text:
            logging.warning(
                f"No highlight text found for marker at {tmk_seconds:.2f}s."
            )
            continue

        # 4. Get summary for the highlight snippet
        summary_line = get_llm_highlight(topic, highlight_text)

        # 5. Format and append to list
        tmk_minutes = int(tmk_seconds / 60)
        highlights.append(f"[{tmk_minutes}m] {summary_line}")

    if not highlights:
        return ""

    highlights_list = "\n".join(f"{i+1}. {h}" for i, h in enumerate(highlights))
    logging.info("--- Finished Generating Highlights ---")
    return f"## Highlights\n\n{highlights_list}"


# --- Main Processing Logic ---


def process_audio_file(
    filepath, asr_model, diarization_pipeline, interactive_mode=False
):
    """
    Processes an audio file to generate a transcript markdown file.
    """
    base_filepath = os.path.splitext(filepath)[0]
    logging.info("------------------------------")
    output_filename = f"{base_filepath}.md"

    logging.info("Starting audio processing for: %s", os.path.basename(filepath))

    # 1. Transcription
    logging.info("Step 1: Transcribing audio...")
    transcription_chunks = transcribe_words(asr_model, filepath)

    # 2. Diarization
    logging.info("Step 2: Performing speaker diarization...")
    diarization_fixed = apply_collar_fix(diarize(diarization_pipeline, filepath))

    # 3. Combine and Clean
    logging.info("Step 3: Combining transcription and diarization...")
    combined_transcript = combine_transcription_and_diarization(
        transcription_chunks, diarization_fixed
    )

    max_cleanup_loops = 5
    logging.info("Step 3a: Starting iterative cleanup of UNKNOWN segments...")
    for i in range(max_cleanup_loops):
        initial_state = [item.copy() for item in combined_transcript]
        combined_transcript = fix_leading_unknowns(combined_transcript)
        combined_transcript = fix_trailing_unknowns(combined_transcript)
        if combined_transcript == initial_state:
            logging.info(f"Transcript stabilized after {i + 1} cleanup loop(s).")
            break
    else:
        logging.warning("Cleanup loop reached max iterations without stabilizing.")

    # 4. Final Formatting
    formatted_transcript = format_final_transcript(combined_transcript)

    # 5. Generate Highlights (if .tmk file exists)
    highlights_section = generate_highlights_section(combined_transcript, filepath)

    # 6. LLM Summarization and Titling (Conditional)
    if not interactive_mode:
        summary_section = get_llm_summary_and_actions(formatted_transcript)
        title_source = summary_section if summary_section else formatted_transcript
        title = get_llm_title(title_source)
        output_filename = f"{base_filepath} {title}.md"
    else:
        # In interactive mode, we use placeholder content initially
        summary_section = "## Summary\n\n[Summary to be generated after speaker naming.]\n\n## Action Items\n\n[Action items to be generated after speaker naming.]"

    # 7. Write to File
    logging.info(f"Writing output to {os.path.basename(output_filename)}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Notes: {os.path.basename(filepath)}\n\n")
        f.write(f"{summary_section}\n\n")
        if highlights_section:
            f.write(f"{highlights_section}\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(formatted_transcript)
        f.write("\n")
    logging.info(
        f"Successfully created base transcript for {os.path.basename(filepath)}."
    )

    return output_filename


# --- Interactive Naming Logic ---


def parse_md_transcript(content):
    """Parses the markdown transcript to extract speaker turns."""
    turn_pattern = re.compile(r"\[(.*?)\] \*\*(SPEAKER_\d+|UNKNOWN)\*\*:(.*)")
    lines = content.split("\n")
    turns = []
    for line in lines:
        match = turn_pattern.match(line)
        if match:
            turns.append(
                {
                    "timestamp": match.group(1).split(".")[0],
                    "speaker": match.group(2),
                    "text": match.group(3).strip(),
                }
            )
    return turns


def generate_speaker_hint_from_turns(
    speaker_label, turns, start_turn=0, context_turns=1
):
    """Generates a conversational hint from a list of turns."""
    occurrence_index = -1
    for i in range(start_turn, len(turns)):
        if turns[i]["speaker"] == speaker_label:
            occurrence_index = i
            break

    if occurrence_index == -1:
        return None, -1

    start = max(0, occurrence_index - context_turns)
    end = min(len(turns), occurrence_index + context_turns + 1)

    hint_lines = []
    for i in range(start, end):
        turn = turns[i]
        speaker = turn["speaker"]
        text = turn["text"]
        display_speaker = (
            (Fore.CYAN + f"**{speaker}**")
            if speaker == speaker_label
            else (Fore.WHITE + speaker)
        )
        timestamp = Style.DIM + f"[{turn.get('timestamp', '?')}]" + Style.NORMAL
        hint_lines.append(f"    {timestamp} {display_speaker}: {text}")

    return "\n".join(hint_lines), occurrence_index


def _get_speaker_name(label: str, turns: list, assigned_names: set) -> str | None:
    """
    Interactively prompts the user to assign a name to a single speaker label.

    Returns:
        - A string with the chosen name.
        - None, if the user wants to skip just this speaker ('?').
        - The sentinel string "SKIP_ALL", if the user wants to skip all renaming for this file.
    """
    search_from_turn = 0
    while True:
        hint, found_at = generate_speaker_hint_from_turns(
            label, turns, start_turn=search_from_turn
        )
        if not hint:
            print(f"\nNo more new snippets for {label}. Restarting from the beginning.")
            search_from_turn = 0
            continue

        # --- Display context and prompt ---
        snippet_time = turns[found_at].get("timestamp", "?")
        print(f"\n--- Who is {label}? (Press Enter for next hint) ---")
        print(f"  Context (at {Fore.YELLOW + snippet_time + Fore.WHITE}):")
        print(hint)

        # Build a clear prompt for the user
        prompt_message = f"Enter name for {Fore.CYAN + label + Fore.WHITE}"
        sorted_names = sorted(list(assigned_names))
        if assigned_names:
            print("\n  Or choose from existing:")
            for i, name in enumerate(sorted_names):
                print(f"    {i+1}. {name}")
            prompt_message += ", choose a number,"
        
        prompt_message += f", use '?' to skip speaker, or 'skip' to skip all: "
        user_input = input(prompt_message).strip()
        user_input_lower = user_input.lower()

        # --- Process user input ---
        if user_input_lower == "skip":
            logging.info("User chose to skip all remaining speaker naming for this file.")
            return "SKIP_ALL"
        
        if user_input_lower == "?":
            logging.info(f"Skipping speaker {label} for now.")
            return None

        if not user_input:
            search_from_turn = found_at + 1
            continue

        try:
            choice_index = int(user_input) - 1
            if assigned_names and 0 <= choice_index < len(sorted_names):
                return sorted_names[choice_index]
            else:
                print(Fore.RED + "Invalid number. Please try again." + Fore.WHITE)
                continue
        except ValueError:
            return user_input # Not a number, so treat as a new name.

def _gather_all_speaker_names(turns: list, speaker_labels: list) -> dict | None:
    """
    Iterates through all speaker labels to build a map of labels to real names.
    Returns the name_map, or None if the user chose to skip all renaming for the file.
    """
    name_map = {}
    assigned_names = set()
    print("\n--- Assign Speaker Names ---")

    for label in speaker_labels:
        # Check if this label has already been assigned a name due to an earlier merge
        if label in name_map:
            continue

        result = _get_speaker_name(label, turns, assigned_names)

        if result == "SKIP_ALL":
            return None  # Propagate the "skip all" command

        if result:  # A name was provided (result is not None)
            name_map[label] = result
            assigned_names.add(result)
            
    return name_map

def _apply_names_to_transcript(content: str, name_map: dict) -> str:
    """
    Replaces all speaker labels in the transcript with their assigned names.
    """
    updated_content = content
    # Replace speaker labels like **SPEAKER_01** with their assigned names
    for old_label, new_name in name_map.items():
        updated_content = updated_content.replace(f"**{old_label}**", f"**{new_name}**")
    
    # After initial replacement, handle potential in-text occurrences if they exist (though unlikely with current diarization output)
    # This part might be redundant if speaker labels only appear as **SPEAKER_XX**, but kept for robustness.
    # The original regex for parts was potentially problematic, simplifying.
    for old_label, new_name in name_map.items():
        # This regex targets SPEAKER_XX or similar in plain text if it somehow appears
        # It's more aggressive and might need refinement if speaker names genuinely contain "SPEAKER_"
        # For now, stick to just replacing the bolded ones which is where they primarily appear.
        pass # Removed the problematic regex replacement

    return updated_content

def _regenerate_llm_content(named_transcript_text: str, name_map: dict) -> tuple[str, str, str]:
    """
    Generates new summary, attendees list, and title using the named transcript.
    """
    new_summary_section = get_llm_summary_and_actions(named_transcript_text)
    
    # Ensure attendees list is based on all values in name_map, plus any UNKNOWNs if they were not renamed
    all_speakers_in_transcript = set(re.findall(r"\*\*([A-Za-z0-9_]+)\*\*", named_transcript_text))
    unique_attendees = sorted(list(all_speakers_in_transcript - {"UNKNOWN"}))
    
    attendees_list_str = "\n".join(f"* {name}" for name in unique_attendees)
    attendees_section = f"## Attendees\n\n{attendees_list_str}"
    
    # Pass only explicitly named attendees (not UNKNOWN) to the title function
    unique_named_attendees_for_title = sorted(list(set(name_map.values()) - {"UNKNOWN"}))
    title_source = new_summary_section if new_summary_section else named_transcript_text
    new_title = get_llm_title(title_source, unique_named_attendees_for_title)
    
    return new_summary_section, attendees_section, new_title

def _assemble_final_markdown(original_content: str, named_transcript_text: str, attendees_section: str, summary_section: str, new_title: str) -> str:
    """
    Assembles the final markdown file from its constituent parts.
    """
    # Extract original sections
    title_match = re.search(r"(# Meeting Notes: .*?)\n\n", original_content, re.DOTALL)
    original_audio_filename = "Audio File"
    if title_match:
        original_title_line = title_match.group(1)
        audio_filename_match = re.search(r"# Meeting Notes: (.*?)$", original_title_line)
        if audio_filename_match:
            original_audio_filename = audio_filename_match.group(1)

    highlights_match = re.search(r"(## Highlights\n\n.*?)(\n\n---|\n\n##)", original_content, re.DOTALL)
    highlights_section = highlights_match.group(1).strip() if highlights_match else ""

    # New title line (important for the filename update)
    final_title_line = f"# Meeting Notes: {original_audio_filename}"

    final_content_parts = [final_title_line, "\n\n", attendees_section, "\n\n", summary_section, "\n\n"]
    if highlights_section:
        final_content_parts.extend([highlights_section, "\n\n"])
    final_content_parts.extend(["---\n\n", f"## Full Transcript\n\n{named_transcript_text}"])

    return "".join(final_content_parts)


def _save_and_cleanup(final_content: str, new_title: str, md_filepath: str, original_content: str):
    """
    Saves the new markdown file and removes the old one.
    """
    audio_filename_match = re.search(r"# Meeting Notes: (.*?)\n", original_content)
    if audio_filename_match:
        original_audio_name_with_ext = audio_filename_match.group(1).strip()
        original_audio_name_base = os.path.splitext(original_audio_name_with_ext)[0]
        original_dir = os.path.dirname(md_filepath)
        base_filepath = os.path.join(original_dir, original_audio_name_base)
    else:
        # Fallback if title line parsing fails
        base_filepath = os.path.splitext(os.path.splitext(md_filepath)[0])[0]

    # Construct the new filename, ensuring it's in the same directory as the original audio
    new_filepath = f"{base_filepath} {new_title}.md"

    with open(new_filepath, "w", encoding="utf-8") as f:
        f.write(final_content)

    if new_filepath != md_filepath:
        logging.info(f"Renaming output file to: {os.path.basename(new_filepath)}")
        try:
            os.remove(md_filepath)
        except OSError as e:
            logging.error(f"Error removing old file {md_filepath}: {e}")

    logging.info(f"Successfully updated speaker names and summaries in {os.path.basename(new_filepath)}.")

def interactive_renaming(md_filepath: str):
    """
    Main orchestrator for the interactive speaker renaming process.
    """
    logging.info(Fore.YELLOW + f"Interactive renaming for {os.path.basename(md_filepath)}..." + Fore.WHITE)

    # 1. Read and Parse File
    try:
        with open(md_filepath, "r", encoding="utf-8") as f:
            original_content = f.read()
    except FileNotFoundError:
        logging.error(f"Could not find file for interactive session: {md_filepath}")
        return

    transcript_section = original_content.split("## Full Transcript\n\n")[-1]
    turns = parse_md_transcript(transcript_section)
    
    # --- CHANGE IN LOGIC IS HERE ---
    # OLD BEHAVIOR: Sorts speakers alphabetically, e.g., [SPEAKER_00, SPEAKER_01]
    # speaker_labels = sorted(list(set(turn["speaker"] for turn in turns if "SPEAKER" in turn["speaker"])))

    # NEW BEHAVIOR: Gets speakers in order of their first appearance in the transcript.
    all_speakers_in_order = (turn["speaker"] for turn in turns if "SPEAKER" in turn["speaker"])
    speaker_labels = list(dict.fromkeys(all_speakers_in_order))
    # --- END OF CHANGE ---

    name_map = None 
    if speaker_labels:
        # 2. Interactively Get Speaker Names
        name_map = _gather_all_speaker_names(turns, speaker_labels)
    else:
        logging.info("No generic 'SPEAKER_XX' labels found to rename in this file.")
        # If no generic labels, we still want to regenerate summary, so we proceed with an empty map
        name_map = {}
        
    # If user selected SKIP_ALL, name_map will be None.
    # In this case, we proceed with an empty map, effectively renaming nothing,
    # but still regenerating the summary/title based on the original SPEAKER_XX labels.
    if name_map is None:
        logging.info("Skipping interactive naming. Regenerating summary with existing speaker labels.")
        name_map = {} # Treat as if no names were entered.

    # 3. Apply Names and Regenerate Content
    logging.info(f"Applying new names: {name_map if name_map else 'No new names to apply.'}")
    
    named_content_full = _apply_names_to_transcript(original_content, name_map)
    named_transcript_text = named_content_full.split("## Full Transcript\n\n")[-1]
    
    summary_section, attendees_section, new_title = _regenerate_llm_content(named_transcript_text, name_map)

    # 4. Assemble and Save Final Markdown
    final_content = _assemble_final_markdown(original_content, named_transcript_text, attendees_section, summary_section, new_title)
    _save_and_cleanup(final_content, new_title, md_filepath, original_content)
def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Transcribe, diarize, and summarize audio files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to an audio file or a folder (defaults to current directory).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After processing, or if an output file exists, prompt for speaker names.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        logging.error(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

    # --- Model Loading ---
    logging.info("Loading AI models...")
    if torch.cuda.is_available():
        pyannote_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        pyannote_device = torch.device("mps")
    else:
        pyannote_device = torch.device("cpu")

    logging.info(f"Pyannote will run on: {pyannote_device}")

    try:
        asr_model = load_asr()
        diarization_pipeline = load_diarizer(pyannote_device)
    except Exception as e:
        logging.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)
    logging.info("Models loaded successfully.")

    # --- File Processing ---
    files_to_process = []
    if os.path.isfile(args.path):
        if os.path.splitext(args.path)[1].lower() in SUPPORTED_EXTENSIONS:
            files_to_process.append(args.path)
        else:
            logging.error(
                f"Unsupported file type: {args.path}. Supported are: {SUPPORTED_EXTENSIONS}"
            )
            sys.exit(1)
    elif os.path.isdir(args.path):
        for filename in sorted(os.listdir(args.path)):
            file_path = os.path.join(args.path, filename)
            if (
                os.path.isfile(file_path)
                and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS
            ):
                files_to_process.append(file_path)

    is_single_file_run = os.path.isfile(args.path)

    for file_path in files_to_process:
        print("\n\n")
        try:
            directory, base_name_with_ext = os.path.split(file_path)
            base_name = os.path.splitext(base_name_with_ext)[0]
            directory = directory or "."

            existing_md_file = None
            for f in os.listdir(directory):
                # Match files that start with the base audio name but might have titles added.
                if f.startswith(base_name) and f.endswith(".md"):
                    existing_md_file = os.path.join(directory, f)
                    break

            if existing_md_file:
                with open(existing_md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                is_already_named = "## Attendees" in content

                # Always run interactive_renaming if --interactive is passed on a single file
                # This allows re-running naming even on already named files if desired.
                if args.interactive and (not is_already_named or is_single_file_run):
                    interactive_renaming(existing_md_file)

            else:
                output_md_path_from_process = process_audio_file(
                    file_path,
                    asr_model,
                    diarization_pipeline,
                    interactive_mode=args.interactive,
                )

                if args.interactive and output_md_path_from_process:
                    interactive_renaming(output_md_path_from_process)

        except Exception as e:
            logging.error(
                f"Failed to process {os.path.basename(file_path)}: {e}", exc_info=True
            )


if __name__ == "__main__":
    main()