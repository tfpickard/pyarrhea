#!/usr/bin/env python
import os
import subprocess
from yt_dlp import YoutubeDL
from rich import print
# Step 1: Download the YouTube video
def download_video(id, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    if os.path.exists(os.path.join(output_dir, f"{id}.mp4")):
        print("Video already downloaded.")
        return os.path.join(output_dir, f"{video_id}.mp4")
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'mp4',
        'verbose': True
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

# Step 2: Extract audio using ffmpeg
def extract_audio(video_file, audio_file=""):
    print("Extracting audio from {video_file}...")
    if audio_file == "":
        audio_file = os.path.splitext(video_file)[0] + ".wav"
    command = ['ffmpeg', '-y', '-i', video_file, '-ac', '1', '-ar', '16000', audio_file]
    print(f"Running command: {' '.join(command)}")
    ret = subprocess.run(command, check=True)
    print(f"Audio extracted to {audio_file} {ret.returncode}")
    return audio_file

# Step 3: Perform speaker diarization (using pyannote.audio)
# Note: pyannote requires some setup (e.g., Hugging Face authentication) and usually a pretrained model.
# The following is pseudocode outlining the process.
def diarize_audio(audio_file):
    # This pseudocode assumes you have set up pyannote with a pretrained model
    from pyannote.audio import Pipeline
    print("Performing speaker diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
    diarization = pipeline(audio_file)
    # diarization now contains segments with speaker labels and timestamps.
    # You might iterate over diarization like:
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
        print(f"[{turn.start:.2f} - {turn.end:.2f}] {speaker}")
    return segments

# Step 4: Transcribe audio segments with Whisper
def transcribe_audio_segments(audio_file, segments):
    import whisper
    model = whisper.load_model("base")
    transcripts = []
    for seg in segments:
        # Extract the segment from the audio file (using ffmpeg again or a library like pydub)
        seg_audio = f"temp_{seg['speaker']}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
        duration = seg['end'] - seg['start']
        command = [
            'ffmpeg', '-y', '-i', audio_file,
            '-ss', str(seg['start']),
            '-t', str(duration),
            seg_audio
        ]
        subprocess.run(command, check=True)
        
        # Transcribe using Whisper
        result = model.transcribe(seg_audio)
        transcripts.append({
            'start': seg['start'],
            'end': seg['end'],
            'speaker': seg['speaker'],
            'text': result['text'].strip()
        })
        os.remove(seg_audio)  # Clean up the temporary file
    return transcripts

# Step 5: Merge transcripts with speaker annotations
def merge_transcripts(transcripts):
    # Sort segments by start time
    transcripts.sort(key=lambda x: x['start'])
    merged_text = ""
    for seg in transcripts:
        merged_text += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"
    return merged_text

# Example usage:
if __name__ == "__main__":
    video_id = "nfAqTSjMBJk"
    x = "https://youtu.be/nfAqTSjMBJk"
    video_file = download_video(video_id)
    print(video_file)
    audio_file = extract_audio(video_file)
    
    segments = diarize_audio(audio_file)
    transcripts = transcribe_audio_segments(audio_file, segments)
    final_transcript = merge_transcripts(transcripts)
    
    with open("final_transcript.txt", "w", encoding="utf-8") as f:
        f.write(final_transcript)
    
    print("Transcription complete! Check final_transcript.txt")

