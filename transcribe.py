import os
import sys
import re
import argparse
import logging
from datetime import timedelta
from dotenv import load_dotenv

import torch
import ollama
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.audio.core.io import Audio

# --- Configuration ---
WHISPER_MODEL = "base.en"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
OLLAMA_MODEL = "gemma3:27b"
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

# --- Helper Functions ---


def format_timestamp(seconds: float) -> str:
    """Converts seconds to a HH:MM:SS.mmm format."""
    return str(timedelta(seconds=seconds))


def sanitize_filename(name):
    """Removes invalid characters from a string to make it a valid filename."""
    name = name.replace('\n', ' ').replace('**', '').strip()
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
    """Formats the combined transcript into a readable, speaker-labeled string."""
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
    return "\n\n".join(
        line.replace(" :", ":").replace(" ,", ",").replace(" .", ".")
        for line in output_lines
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


def get_llm_title(source_text):
    """Generates a descriptive filename title from a text (summary or transcript) using the LLM."""
    logging.info("Generating descriptive title with LLM...")
    prompt = f"""Based on the following text, generate a single, short, descriptive phrase of 4-8 words suitable for a filename.
- Do NOT use any special characters like quotes or colons.
- Do NOT explain your thinking.
- Provide ONLY the title phrase itself.

Example: Initial Project Kick-off with the Analytics Team

Here is the text:
---
{source_text}
---
"""
    title = query_ollama(prompt)
    clean_title = title.split('\n')[0].strip()
    return sanitize_filename(clean_title)


def get_llm_summary_and_actions(transcript_text):
    """
    Generates a summary and action items from a transcript in a single LLM call.
    """
    logging.info("Generating summary and action items with LLM...")
    prompt = f"""Please provide a summary and action points (if any) for the following meeting transcript.
Focus on the key decisions and outcomes.
The summary should be short and cover the main points.
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


# --- Main Processing Logic ---

def process_audio_file(
    filepath, whisper_model, diarization_pipeline, interactive_mode=False
):
    """
    Processes an audio file to generate a transcript markdown file.
    """
    base_filepath = os.path.splitext(filepath)[0]
    logging.info("------------------------------")
    # In interactive mode, the initial output is temporary.
    output_filename = f"{base_filepath}.md"
    
    logging.info("Starting audio processing for: %s", os.path.basename(filepath))

    # 1. Transcription
    logging.info("Step 1: Transcribing audio...")
    transcription_result = whisper_model.transcribe(
        filepath, word_timestamps=True, fp16=False
    )
    transcription_chunks = []
    for segment in transcription_result["segments"]:
        for word in segment["words"]:
            transcription_chunks.append(
                {"text": word["word"], "start": word["start"], "end": word["end"]}
            )

    # 2. Diarization
    logging.info("Step 2: Performing speaker diarization...")
    try:
        audio_loader = Audio(sample_rate=16000, mono=True)
        waveform, sample_rate = audio_loader(filepath)
        diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    except Exception as e:
        logging.error(f"Error during diarization: {e}", exc_info=True)
        diarization_result = Annotation()

    diarization_fixed = apply_collar_fix(diarization_result)

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

    # 5. LLM Summarization (Conditional)
    if not interactive_mode:
        summary_section = get_llm_summary_and_actions(formatted_transcript)
        title = get_llm_title(summary_section)
        output_filename = f"{base_filepath} {title}.md"
    else:
        # In interactive mode, we use placeholder content initially
        summary_section = "## Summary\n\n[Summary to be generated after speaker naming.]\n\n## Action Items\n\n[Action items to be generated after speaker naming.]"

    # 6. Write to File
    logging.info(f"Writing output to {os.path.basename(output_filename)}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Notes: {os.path.basename(filepath)}\n\n")
        f.write(f"{summary_section}\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(formatted_transcript)
        f.write("\n")
    logging.info(f"Successfully created base transcript for {os.path.basename(filepath)}.")
    
    return output_filename


# --- Interactive Naming Logic ---

def parse_md_transcript(content):
    """Parses the markdown transcript to extract speaker turns."""
    turn_pattern = re.compile(r"\[.*?\] \*\*(SPEAKER_\d+|UNKNOWN)\*\*:(.*)")
    lines = content.split('\n')
    turns = []
    for line in lines:
        match = turn_pattern.match(line)
        if match:
            turns.append({"speaker": match.group(1), "text": match.group(2).strip()})
    return turns

def generate_speaker_hint_from_turns(speaker_label, turns, start_turn=0, context_turns=1):
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
        speaker = turn['speaker']
        text = turn['text']
        display_speaker = f"**{speaker}**" if speaker == speaker_label else speaker
        hint_lines.append(f"    {display_speaker}: {text}")
        
    return "\n".join(hint_lines), occurrence_index


def interactive_renaming(md_filepath):
    """Handles the interactive speaker renaming process, including re-running summaries."""
    logging.info(f"Starting interactive renaming for {os.path.basename(md_filepath)}...")
    
    with open(md_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    transcript_section = content.split("## Full Transcript\n\n")[-1]
    turns = parse_md_transcript(transcript_section)
    
    speaker_labels = sorted(list(set(turn["speaker"] for turn in turns if "SPEAKER" in turn["speaker"])))
    
    name_map = {}
    if not speaker_labels:
        logging.info("No generic speakers found to rename in this file.")
        return

    print("\n--- Assign Speaker Names ---")
    skip_all = False
    for label in speaker_labels:
        search_from_turn = 0
        while True:
            hint, found_at = generate_speaker_hint_from_turns(label, turns, start_turn=search_from_turn)
            if not hint:
                print(f"\nNo more occurrences of {label} found.")
                break

            print(f"\n--- Who is {label}? (Press Enter for next hint) ---")
            print("  Context:")
            print(hint)
            
            new_name = input(f"Enter name for {label} (or 'SKIP' for all, '?' for this one): ")

            if new_name.lower() == 'skip':
                skip_all = True
                break
            elif new_name == '?':
                logging.info(f"Skipping speaker {label} for now.")
                break 
            elif new_name:
                name_map[label] = new_name.strip()
                break
            else:
                search_from_turn = found_at + 1
        if skip_all:
            break
            
    if not name_map:
        logging.info("No names were assigned. File remains unchanged.")
        return

    logging.info(f"Applying new names: {name_map}")
    updated_content = content
    for old_label, new_name in name_map.items():
        updated_content = updated_content.replace(f"**{old_label}**", f"**{new_name}**")

    named_transcript_text = updated_content.split("## Full Transcript\n\n")[-1]
    
    new_summary_section = get_llm_summary_and_actions(named_transcript_text)

    # Replace old summary/actions section and add attendees
    final_content = re.sub(
        r"(# Meeting Notes: .*?\n\n)(.*?)(\n\n---)",
        f"\\1{new_summary_section}\\3",
        updated_content,
        flags=re.DOTALL
    )

    attendees_list = "\n".join(f"* {name}" for name in sorted(name_map.values()))
    attendees_section = f"## Attendees\n\n{attendees_list}\n\n"
    
    if "## Attendees" in final_content:
        final_content = re.sub(r"(## Attendees\n\n)(.*?)(\n\n---)", f"\\1{attendees_list}\\3", final_content, flags=re.DOTALL)
    else:
        final_content = final_content.replace("\n\n---", f"\n\n{attendees_section}---")

    new_title = get_llm_title(new_summary_section)
    base_filepath = os.path.splitext(md_filepath)[0].split(' ')[0] 
    new_filepath = f"{base_filepath} {new_title}.md"

    with open(new_filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    if new_filepath != md_filepath:
        logging.info(f"Renaming output file to: {os.path.basename(new_filepath)}")
        os.remove(md_filepath)
    
    logging.info(f"Successfully updated speaker names and summaries in {os.path.basename(new_filepath)}.")


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

    whisper_device = "cpu"
    logging.info(f"Pyannote will run on: {pyannote_device}")
    logging.info(f"Whisper will run on: {whisper_device}")

    try:
        whisper_model = whisper.load_model(WHISPER_MODEL, device=whisper_device)
        hyperparameters = {"min_duration_on": 0.1, "onset": 0.7}
        diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=hf_api_key
        )
        diarization_pipeline.segmentation.hyperparameters = hyperparameters
        diarization_pipeline.to(pyannote_device)
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
            logging.error(f"Unsupported file type: {args.path}. Supported are: {SUPPORTED_EXTENSIONS}")
            sys.exit(1)
    elif os.path.isdir(args.path):
        for filename in sorted(os.listdir(args.path)):
            file_path = os.path.join(args.path, filename)
            if (os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS):
                files_to_process.append(file_path)

    is_single_file_run = os.path.isfile(args.path)

    for file_path in files_to_process:
        print("\n\n")
        logging.info(f"Processing file: {os.path.basename(file_path)}")
        try:
            directory, base_name = os.path.split(os.path.splitext(file_path)[0])
            directory = directory or '.'
            
            existing_md_file = None
            for f in os.listdir(directory):
                if f.startswith(base_name) and f.endswith('.md'):
                    existing_md_file = os.path.join(directory, f)
                    break
            
            if existing_md_file:
                with open(existing_md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                is_already_named = "## Attendees" in content

                if args.interactive and (not is_already_named or is_single_file_run):
                    interactive_renaming(existing_md_file)
                else:
                    logging.info(f"Skipping '{os.path.basename(file_path)}'; already processed and named.")
            
            else:
                output_md_path_from_process = process_audio_file(
                    file_path, whisper_model, diarization_pipeline, interactive_mode=args.interactive
                )
                
                if args.interactive and output_md_path_from_process:
                    interactive_renaming(output_md_path_from_process)

        except Exception as e:
            logging.error(f"Failed to process {os.path.basename(file_path)}: {e}", exc_info=True)


if __name__ == "__main__":
    main()