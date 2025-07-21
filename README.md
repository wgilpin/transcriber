# AI-Powered Meeting Transcription & Summarization

This Python script automates the process of transcribing audio recordings of meetings, identifying different speakers, and generating concise summaries with actionable insights using a local Large Language Model (LLM).

It's designed for anyone who needs to process meeting audio efficiently, turning long recordings into structured, easy-to-read notes.

## Features

- **Accurate Transcription**: Utilizes OpenAI's **Whisper** model for high-quality, timestamped transcription.
- **Speaker Diarization**: Employs **`pyannote.audio`** to distinguish between different speakers in the recording and label their dialogue.
- **LLM-Powered Insights**: Connects to a local **Ollama** instance to:

  - Generate a concise summary of the meeting.
  - Extract key decisions and action items.
  - Create a descriptive title for the notes file based on its content.

- **Interactive Speaker Naming**: An optional `--interactive` mode allows you to replace generic labels (e.g., `SPEAKER_01`) with actual names, with the script providing conversational context to help you identify who is who.
- **Intelligent Formatting**:

  - Merges consecutive lines from the same speaker to create clean, readable dialogue blocks.
  - Carefully handles `UNKNOWN` speakers to avoid incorrect merging.
  - Cleans up short, unassigned speech segments between known speakers.

- **Batch Processing**: Can process a single audio file or an entire directory of them in one go.

## Requirements

### Software

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- An LLM pulled via Ollama (the script is configured for `gemma3:27b` but can be changed).

  ```bash
  ollama pull gemma3:27b
  ```

- [PyTorch](https://pytorch.org/get-started/locally/)

### Python Packages

You can install the required packages using pip. It's recommended to do this in a virtual environment.

```bash
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install openai-whisper
pip install pyannote.audio
pip install ollama
pip install python-dotenv
```

_Note: The `pyannote.audio` library may require you to accept user conditions on the Hugging Face Hub for the models it uses._

### API Keys

- **Hugging Face**: The `pyannote/speaker-diarization-3.1` model requires authentication. You will need a Hugging Face account and an API token with "read" access.

## Setup

1. **Clone or Download**: Place the `transcribe.py` script in your desired project directory.

2. **Install Dependencies**: Install the required Python packages as listed in the **Requirements** section above.

3. **Set Up Environment File**: Create a file named `.env` in the same directory as the script. This file will securely store your Hugging Face API token. Add your token to it like this:

  ```
  HF_API_TOKEN="hf_YourHuggingFaceApiTokenGoesHere"
  ```

4. **Ensure Ollama is Running**: Start the Ollama application or run `ollama serve` in your terminal to make the LLM available for the script.

## Usage

The script is run from the command line.

### Basic Command

```bash
python transcribe.py [path_to_audio_or_folder] [options]
```

### Arguments & Options

- `path` (required): The path to a single audio file (`.mp3`, `.wav`, `.m4a`) or a directory containing audio files.
- `--interactive` (optional): Enables interactive mode after processing. This will prompt you to assign names to the detected speakers.

### Examples

**1\. Process a Single Audio File**

This will transcribe, diarize, and generate a complete markdown file with a summary and title.

```bash
python transcribe.py "/path/to/your/meeting.mp3"
```

**2\. Process a Folder of Audio Files**

The script will iterate through all supported audio files in the specified directory.

```bash
python transcribe.py "./project_meetings/"
```

**3\. Process a File with Interactive Speaker Naming**

Use the `--interactive` flag to get a prompt to name speakers after the initial processing is complete.

```bash
python transcribe.py meeting.wav --interactive
```

The script will first create a base markdown file. Then, it will show you snippets of conversation for each speaker (`SPEAKER_00`, `SPEAKER_01`, etc.) and ask you to provide a name. Once done, it will update the transcript, regenerate the summary and action items with the correct names, and rename the file.

**4\. Re-run Naming on an Already Processed File**

If you have an output `.md` file and want to rename the speakers, just run the script again with the `--interactive` flag on the original audio file. The script will detect the existing markdown file and initiate the renaming process.

```bash
# If meeting.md already exists...
python transcribe.py meeting.wav --interactive
```

## How It Works: The Processing Pipeline

1. **Transcription**: The audio is first transcribed by **Whisper** to generate text with word-level timestamps.
2. **Diarization**: **`pyannote.audio`** analyzes the audio to determine _who_ spoke and _when_, creating a timeline of speaker segments.
3. **Combination & Cleanup**: The script merges the transcription with the diarization data, assigning a speaker label to each word. It then runs cleanup routines to fix short `UNKNOWN` segments that occur between two turns from the same speaker.
4. **Formatting**: The raw data is formatted into a readable transcript, merging consecutive dialogue from the same speaker into a single block.
5. **LLM Enhancement**:

  - The full transcript is sent to **Ollama** to generate a summary and a list of action items.
  - This summary is then sent back to Ollama to generate a descriptive filename.

6. **Final Output**: A single markdown (`.md`) file is created, containing the summary, action items, attendee list (if named), and the full transcript.

## Configuration

The following models can be configured at the top of the `transcribe.py` script:

- `WHISPER_MODEL`: The Whisper model size (e.g., `base.en`, `medium.en`).
- `DIARIZATION_MODEL`: The Pyannote diarization model from Hugging Face.
- `OLLAMA_MODEL`: The model name to be used with your local Ollama instance.
