# YouTube Video Transcription Tool

This project is a command-line tool designed to download YouTube videos, extract audio, perform speaker diarization, and transcribe the audio using Whisper. The final output is a text file containing the transcribed text with speaker annotations.

## Features

- Download YouTube videos using video ID or URL.
- Extract audio from the downloaded video.
- Perform speaker diarization to identify different speakers in the audio.
- Transcribe audio segments using Whisper.
- Merge transcripts with speaker annotations into a final text file.

## Requirements

- Python 3.6 or higher
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://ffmpeg.org/)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [whisper](https://github.com/openai/whisper)
- [rich](https://github.com/Textualize/rich)
- [inflect](https://github.com/jaraco/inflect)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `ffmpeg` is installed and available in your system's PATH.

## Usage

Run the script with the following options:

- To process a YouTube video using its URL:
  ```bash
  python main.py -u <video-url>
  ```

- To process a YouTube video using its ID:
  ```bash
  python main.py -i <video-id>
  ```

- Additional options:
  - `-d`, `--force-download`: Force download the video even if it exists.
  - `-e`, `--force-extraction`: Force audio extraction even if it exists.

## Environment Setup for Diarization

- Set up Hugging Face authentication for `pyannote.audio` by setting the `HUGGINGFACE_TOKEN` environment variable.

## Output

The final transcript will be saved as `final_transcript.txt` in the current directory.

## License

This project is licensed under the MIT License.
