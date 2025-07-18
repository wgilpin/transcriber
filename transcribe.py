import os
import sys
import argparse
import torch
import ollama
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import whisper # MODIFIED: Import the original OpenAI Whisper library
from datetime import timedelta
import logging

# --- Configuration ---
WHISPER_MODEL = "base.en" # Using '.en' model for English-only is often faster and more accurate
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
OLLAMA_MODEL = "gemma2:27b" # Updated to a more recent model
SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac"]

from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv("HF_API_TOKEN")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
if hf_api_key:
    logging.info("Hugging Face API token loaded.")
else:
    logging.warning("Hugging Face API token not found. Diarization may fail if model needs auth.")


def format_timestamp(seconds):
    """Converts seconds to a HH:MM:SS.mmm format."""
    return str(timedelta(seconds=seconds))

def apply_collar_fix(diarization: Annotation, collar_seconds: float = 0.5, unknown_duration_s: float = 1.5):
    """Applies a 'collar fix' to short, UNKNOWN speaker segments."""
    if not diarization.uri:
        return diarization
    corrected_diarization = Annotation(uri=diarization.uri)
    turns = list(diarization.itertracks(yield_label=True))
    if len(turns) < 3:
        for turn, track, speaker in turns:
            corrected_diarization[turn, track] = speaker
        return corrected_diarization

    modified_turns = list(turns)
    for i in range(1, len(turns) - 1):
        current_turn, _, current_speaker = turns[i]
        prev_turn, _, prev_speaker = turns[i-1]
        next_turn, _, next_speaker = turns[i+1]
        is_short_unknown = "UNKNOWN" in current_speaker and current_turn.duration < unknown_duration_s
        speakers_match = prev_speaker == next_speaker
        is_within_collar = (current_turn.start - prev_turn.end) < collar_seconds and \
                           (next_turn.start - current_turn.end) < collar_seconds
        if is_short_unknown and speakers_match and is_within_collar:
            modified_turns[i] = (current_turn, modified_turns[i][1], prev_speaker)
            logging.info(f"Applying collar fix: Re-labeling short UNKNOWN segment at {format_timestamp(current_turn.start)} to '{prev_speaker}'.")
    for turn, track, speaker in modified_turns:
        corrected_diarization[turn, track] = speaker
    return corrected_diarization.support()

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

        # Use .get() with a default to avoid errors if 'text' key is missing
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

def process_audio_file(filepath, whisper_model, diarization_pipeline):
    """Main processing function for a single audio file."""
    output_filename = os.path.splitext(filepath)[0] + ".md"
    logging.info(f"Starting processing for: {os.path.basename(filepath)}")

    # 1. Transcription (MODIFIED: Using original OpenAI Whisper on CPU)
    logging.info("Step 1: Transcribing audio with OpenAI Whisper on CPU...")
    # The 'word_timestamps' option is crucial for speaker alignment
    transcription_result = whisper_model.transcribe(filepath, word_timestamps=True, fp16=False)

    # NEW: Adapt whisper's output to the format our script expects
    # The original script expected {'text': ' word', 'timestamp': (start, end)}
    # We will create that from the new output format.
    transcription_chunks = []
    for segment in transcription_result["segments"]:
        for word in segment["words"]:
            transcription_chunks.append({
                "text": word["word"],
                "start": word["start"],
                "end": word["end"]
            })

    # 2. Diarization (on GPU if available)
    logging.info("Step 2: Performing speaker diarization with Pyannote...")
    diarization_result = diarization_pipeline(filepath)
    diarization_fixed = apply_collar_fix(diarization_result)

    # 3. Combine and Format
    logging.info("Step 3: Combining transcription and diarization...")
    combined_transcript = combine_transcription_and_diarization(transcription_chunks, diarization_fixed)
    formatted_transcript = format_final_transcript(combined_transcript)

    # 4. LLM Summarization
    logging.info("Step 4: Generating summary and action items with Ollama...")
    summary_prompt = "Please provide a concise, professional summary of the following meeting transcript. Focus on the key decisions and outcomes."
    action_items_prompt = "Based on the transcript, extract all explicit action items. For each, identify the assigned person if mentioned. Format as a markdown list."
    summary = query_ollama(formatted_transcript, summary_prompt)
    action_items = query_ollama(formatted_transcript, action_items_prompt)

    # 5. Write to File
    logging.info(f"Step 5: Writing output to {os.path.basename(output_filename)}")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"# Meeting Notes: {os.path.basename(filepath)}\n\n")
        f.write(f"## Summary\n\n{summary}\n\n")
        f.write(f"## Action Items\n\n{action_items}\n\n")
        f.write("---\n\n## Full Transcript\n\n")
        f.write(formatted_transcript)
        f.write("\n")
    logging.info(f"Successfully processed {os.path.basename(filepath)}.")


def main():
    parser = argparse.ArgumentParser(description="Transcribe, diarize, and summarize audio files using Whisper and Pyannote.")
    parser.add_argument("path", nargs='?', default=".", help="Path to an audio file or a folder containing audio files (defaults to current directory).")
    args = parser.parse_args()
    input_path = args.path

    if not os.path.exists(input_path):
        logging.error(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    # --- MODIFIED: Multi-device loading strategy ---
    logging.info("Loading AI models...")

    # Determine device for Pyannote (GPU if available, else CPU)
    # Using torch.cuda.is_available() for broader GPU support (NVIDIA)
    if torch.cuda.is_available():
        pyannote_device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon
        pyannote_device = torch.device("mps")
    else:
        pyannote_device = torch.device("cpu")
    
    # Whisper will always run on CPU as requested
    whisper_device = "cpu"
    
    logging.info(f"Pyannote will run on: {pyannote_device}")
    logging.info(f"Whisper will run on: {whisper_device}")

    try:
        # Load OpenAI Whisper model onto the CPU
        # fp16=False is recommended for CPU-only operation.
        whisper_model = whisper.load_model(WHISPER_MODEL, device=whisper_device)

        # Load Pyannote and move to its designated device
        diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_api_key)
        diarization_pipeline.to(pyannote_device)

    except Exception as e:
        logging.error(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Models loaded successfully.")

    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in SUPPORTED_EXTENSIONS:
            process_audio_file(input_path, whisper_model, diarization_pipeline)
        else:
            logging.error(f"Unsupported file type: {input_path}. Supported are: {SUPPORTED_EXTENSIONS}")
            sys.exit(1)
    elif os.path.isdir(input_path):
        logging.info(f"Processing all supported audio files in folder: {input_path}")
        for filename in sorted(os.listdir(input_path)): # sorted for deterministic order
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                # Check if output already exists
                output_md_path = os.path.splitext(file_path)[0] + ".md"
                if os.path.exists(output_md_path):
                    logging.info(f"Skipping '{filename}': Output file already exists.")
                    continue
                try:
                    process_audio_file(file_path, whisper_model, diarization_pipeline)
                except Exception as e:
                    logging.error(f"Failed to process {filename}: {e}", exc_info=True)
            else:
                logging.warning(f"Skipping non-supported file or sub-directory: {filename}")

if __name__ == "__main__":
    main()