import os
import sys
import argparse
import torch
import ollama
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import whisper
from datetime import timedelta
import logging

# --- Configuration ---
WHISPER_MODEL = "base.en"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
OLLAMA_MODEL = "gemma2:27b"
SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac"]

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv("HF_API_TOKEN")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
if hf_api_key:
    logging.info("Hugging Face API token loaded.")
else:
    logging.warning("Hugging Face API token not found. Diarization may fail if model needs auth.")

# --- Helper Functions ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds to a HH:MM:SS.mmm format."""
    return str(timedelta(seconds=seconds))

def generate_speaker_hint(speaker_label, transcript_data, max_words=25):
    """
    Generates a text snippet from the first turn of a given speaker to use as a hint.
    """
    hint_words = []
    # Find the first instance of the speaker and start collecting their words
    for i, item in enumerate(transcript_data):
        if item['speaker'] == speaker_label:
            # We found the first word, now collect the following words from the same speaker
            for j in range(0,3):
                follow_up_item = transcript_data[j]
                hint_words.append(follow_up_item['text'].strip())
            break # Exit the outer loop once the first turn is processed
    
    return " \n".join(hint_words)

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

        is_short_unknown = ("UNKNOWN" in current_speaker and current_turn.duration < unknown_duration_threshold_s)
        speakers_match = prev_speaker == next_speaker
        is_within_collar = ((current_turn.start - prev_turn.end) < collar_seconds and (next_turn.start - current_turn.end) < collar_seconds)

        if is_short_unknown and speakers_match and is_within_collar:
            logging.info(f"Collar Fix: Re-labeling segment [{current_turn.start:.2f}s - {current_turn.end:.2f}s] from {current_speaker} to {prev_speaker}.")
            corrected_diarization[current_turn] = prev_speaker

    return corrected_diarization.support()

def fix_leading_unknowns(combined_result, time_threshold=0.5):
    """Merges an UNKNOWN word with the subsequent speaker's turn if they are close in time."""
    if len(combined_result) < 2:
        return combined_result

    for i in range(len(combined_result) - 1):
        current_item = combined_result[i]
        next_item = combined_result[i + 1]

        if (current_item["speaker"] == "UNKNOWN" and
            next_item["speaker"] != "UNKNOWN" and
            (next_item["start"] - current_item["end"]) < time_threshold):
            
            logging.info(f"Leading Fix: Merging UNKNOWN '{current_item['text'].strip()}' into {next_item['speaker']}.")
            current_item["speaker"] = next_item["speaker"]
    
    return combined_result

def fix_trailing_unknowns(combined_result, time_threshold=0.5):
    """Merges an UNKNOWN word with the preceding speaker's turn if they are close in time."""
    if len(combined_result) < 2:
        return combined_result

    for i in range(len(combined_result) - 1, 0, -1):
        current_item = combined_result[i]
        prev_item = combined_result[i - 1]

        if (current_item["speaker"] == "UNKNOWN" and
            prev_item["speaker"] != "UNKNOWN" and
            (current_item["start"] - prev_item["end"]) < time_threshold):
            
            logging.info(f"Trailing Fix: Merging UNKNOWN '{current_item['text'].strip()}' into {prev_item['speaker']}.")
            current_item["speaker"] = prev_item["speaker"]

    return combined_result

def combine_transcription_and_diarization(transcription_chunks, diarization_result):
    """Merges Whisper's transcription with Pyannote's diarization."""
    full_transcript = []
    speaker_turns = [{"start": turn.start, "end": turn.end, "speaker": speaker}
                     for turn, _, speaker in diarization_result.itertracks(yield_label=True)]

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

        full_transcript.append({
            "start": word_start, "end": word_end,
            "text": chunk.get("text", ""), "speaker": speaker
        })
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
    return "\n".join(line.replace(" :", ":").replace(" ,", ",").replace(" .", ".") for line in output_lines)

def query_ollama(transcript_text, prompt_template):
    """Sends a prompt with the transcript to Ollama and returns the response."""
    logging.info(f"Querying Ollama with model '{OLLAMA_MODEL}'...")
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": "You are an expert assistant for summarizing meeting transcripts."},
            {"role": "user", "content": prompt_template.format(transcript=transcript_text)}
        ])
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Failed to query Ollama: {e}")
        return "Error: Could not retrieve response from Ollama."

# --- Main Processing Logic ---

def process_audio_file(filepath, whisper_model, diarization_pipeline, interactive=False):
    """Main processing function for a single audio file."""
    output_filename = os.path.splitext(filepath)[0] + ".md"
    logging.info(f"Starting processing for: {os.path.basename(filepath)}")

    # 1. Transcription
    logging.info("Step 1: Transcribing audio...")
    transcription_result = whisper_model.transcribe(filepath, word_timestamps=True, fp16=False)
    transcription_chunks = []
    for segment in transcription_result["segments"]:
        for word in segment["words"]:
            transcription_chunks.append({"text": word["word"], "start": word["start"], "end": word["end"]})

    # 2. Diarization
    logging.info("Step 2: Performing speaker diarization...")
    diarization_result = diarization_pipeline(filepath)
    diarization_fixed = apply_collar_fix(diarization_result)

    # 3. Combine and Clean
    logging.info("Step 3: Combining transcription and diarization...")
    combined_transcript = combine_transcription_and_diarization(transcription_chunks, diarization_fixed)

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

    # 4. Interactive Speaker Naming (if flag is set)
    if interactive:
        name_map = {}
        speaker_labels = sorted(list(set(item['speaker'] for item in combined_transcript if item['speaker'] != "UNKNOWN")))
        if speaker_labels:
            print("\n--- Assign Speaker Names ---")
            for label in speaker_labels:
                # Generate a longer, more helpful hint for identifying the speaker
                hint_snippet = generate_speaker_hint(label, combined_transcript, max_words=25)
                
                print(f"\nWho is {label}?")
                print(f"  Hint: \"{hint_snippet}...\"")
                new_name = input(f"Enter name for {label}: ")

                if new_name:
                    name_map[label] = new_name.strip()

        if name_map:
            for item in combined_transcript:
                if item['speaker'] in name_map:
                    item['speaker'] = name_map[item['speaker']]
            logging.info(f"Applied new speaker names: {name_map}")

    # 5. Final Formatting
    formatted_transcript = format_final_transcript(combined_transcript)

    # 6. LLM Summarization
    logging.info("Step 6: Generating summary and action items...")
    summary_prompt = "Please provide a concise, professional summary of the following meeting transcript. Focus on the key decisions and outcomes."
    action_items_prompt = "Based on the transcript, extract all explicit action items. For each, identify the assigned person if mentioned. Format as a markdown list."
    summary = query_ollama(formatted_transcript, summary_prompt)
    action_items = query_ollama(formatted_transcript, action_items_prompt)

    # 7. Write to File
    logging.info(f"Step 7: Writing output to {os.path.basename(output_filename)}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Notes: {os.path.basename(filepath)}\n\n")
        f.write(f"## Summary\n\n{summary}\n\n")
        f.write(f"## Action Items\n\n{action_items}\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(formatted_transcript)
        f.write("\n")
    logging.info(f"Successfully processed {os.path.basename(filepath)}.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Transcribe, diarize, and summarize audio files.")
    parser.add_argument("path", nargs='?', default=".", help="Path to an audio file or a folder (defaults to current directory).")
    parser.add_argument("--interactive", action="store_true", help="Prompt for speaker names interactively during processing.")
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
        diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_api_key)
        diarization_pipeline.segmentation.hyperparameters = hyperparameters
        diarization_pipeline.to(pyannote_device)
    except Exception as e:
        logging.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)
    logging.info("Models loaded successfully.")

    # --- File Processing ---
    if os.path.isfile(args.path):
        if os.path.splitext(args.path)[1].lower() in SUPPORTED_EXTENSIONS:
            process_audio_file(args.path, whisper_model, diarization_pipeline, args.interactive)
        else:
            logging.error(f"Unsupported file type: {args.path}. Supported are: {SUPPORTED_EXTENSIONS}")
            sys.exit(1)
    elif os.path.isdir(args.path):
        logging.info(f"Processing all supported audio files in folder: {args.path}")
        for filename in sorted(os.listdir(args.path)):
            file_path = os.path.join(args.path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                output_md_path = os.path.splitext(file_path)[0] + ".md"
                if os.path.exists(output_md_path):
                    logging.info(f"Skipping '{filename}': Output file already exists.")
                    continue
                try:
                    process_audio_file(file_path, whisper_model, diarization_pipeline, args.interactive)
                except Exception as e:
                    logging.error(f"Failed to process {filename}: {e}", exc_info=True)
            else:
                logging.warning(f"Skipping non-supported file or sub-directory: {filename}")

if __name__ == "__main__":
    main()