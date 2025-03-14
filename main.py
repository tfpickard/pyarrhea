#!/usr/bin/env python
import time
import os
import argparse
import sys
import subprocess
from yt_dlp import YoutubeDL
from rich import print
from rich.panel import Panel
import inflect
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
import requests
import threading
import simpleaudio as sa

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process YouTube video for transcription.")
    parser.add_argument('-i', '--video-id', type=str, help='YouTube video ID')
    parser.add_argument('-u', '--video-url', type=str, help='YouTube video URL')
    parser.add_argument('-d', '--force-download', action='store_true', help='Force download the video even if it exists')
    parser.add_argument('-e', '--force-extraction', action='store_true', help='Force audio extraction even if it exists')
    return parser.parse_args()

def download_video(id, output_dir="downloads", force=False):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={id}"
    if os.path.exists(os.path.join(output_dir, f"{id}.mp4")) and not force:
        print("Video already downloaded.")
        return os.path.join(output_dir, f"{id}.mp4")
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'format': 'mp4',
        'verbose': False
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

# Step 2: Extract audio using ffmpeg
def extract_audio(video_file, audio_file="", force=False):
    print(f"Extracting audio from {video_file}...")
    if audio_file == "":
        audio_file = os.path.splitext(video_file)[0] + ".wav"
    if os.path.exists(audio_file) and not force:
        print(f"Audio file {audio_file} already exists.")
        return audio_file
    command = ['ffmpeg', '-y', '-i', video_file, '-ac', '1', '-ar', '16000', audio_file]
    print(f"Running command: {' '.join(command)}")
    ret = subprocess.run(command, check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Audio extracted to {audio_file} {ret.returncode}")
    return audio_file


# Step 3: Perform speaker diarization (using pyannote.audio)
# Note: pyannote requires some setup (e.g., Hugging Face authentication) and usually a pretrained model.
# The following is pseudocode outlining the process.
def diarize_audio(audio_file):
    # This pseudocode assumes you have set up pyannote with a pretrained model
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch
    print("Performing speaker diarization...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]).to(device)

    with ProgressHook() as progress_hook:
        # progress_hook.set_description("Diarizing audio")
        # progress_hook.set_total(100)
        # progress_hook.set_progresfs(0)
        pipeline.setup_hook(file=audio_file,     hook=progress_hook)
        start_time = time.time()
        diarization = pipeline(audio_file, hook=progress_hook)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # progress_hook.set_progress(100)
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
    speakers = dict()
    inflector = inflect.engine()
    i = 0
    style = {
        "title": {
                "k":["[bold sky_blue2]", "[/bold sky_blue2]"],
                "v":["[bold light_coral]", "[/bold light_coral]"]
        },
        "time": {
            "k":["[italic royal_blue1]", "[/italic royal_blue1]"],
            "v":["[bold light_green1]", "[bold /light_green1]"]
        },
        "speaker": {
            "k":["[italic dodger_blue1]", "[/italic dodger_blue1]"],
            "v":["[bold sea_green1]", "[bold /sea_green1]"]
        },
    }
    num = {
        "plum": ["[bold plum]", "[/bold plum]"],
        "salmon": ["[bold light_salmon1]", "[/bold light_salmon1]"],
        "orange": ["[bold orange_red1]", "[/bold orange_red1]"],
        "sg": ["[bold spring_green4]", "[/bold spring_green4]"],
    }
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
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Function to play audio
        # def play_audio(file_path):
        #     wave_obj = sa.WaveObject.from_wave_file(file_path)
        #     play_obj = wave_obj.play()
        #     play_obj.wait_done()
        #
        # # Start a thread to play the audio
        # audio_thread = threading.Thread(target=play_audio, args=(seg_audio,))
        # audio_thread.start()
        result = model.transcribe(seg_audio)

        transcripts.append({
            'start': seg['start'],
            'end': seg['end'],
            'speaker': seg['speaker'],
            'text': result['text'].strip()
        })
        if seg['speaker'] not in speakers:
            speakers[seg['speaker']] = [0,len(speakers)+1]
            print(f"New speaker number {len(speakers)+1}: {seg['speaker']}")
        speakers[seg['speaker']][0] = speakers.get(seg['speaker'], [0,0])[0] + 1
        c = speakers[seg['speaker']][0]
        cc = inflector.ordinal(c)
        s = speakers[seg['speaker']][1]
        ss = inflector.ordinal(s) 
        n = len(speakers)

        dur = seg['end'] - seg['start']
        # print(transcripts[-1])
        stk = style['title']['k']
        stv = style['title']['v']
        sdk = style['time']['k'] 
        sdv = style['time']['v']
        ssk = style['speaker']['k']
        ssv = style['speaker']['v']
        ttext = f"{stk[0]}Transcription{stk[1]} {stv[0]}{i}: {stv[1]}"
        ttext += f"{num['plum'][0]}({dur:.2f}s{num['plum'][1]})\n"
        text = f"{ssk[0]}Speaker{ssk[1]} "
        text += f"({num['plum'][0]}{ss}{num['plum'][1]} of "
        text += f"{num['sg'][0]}{n}{num['sg'][1]}: "
        text += f"{ssv[0]}{seg['speaker']},{ssv[1]} "
        text += f"{num['orange'][0]}{cc} {num['orange'][1]}"
        text += f"{ssv[0]}voice detection so far{ssv[1]}\n"
        text += f"{sdk[0]}Duration:{sdk[1]} "
        text += f"{sdv[0]}{seg['start']:.2f} - {seg['end']:.2f}{sdv[1]} "
        text += f"[bold dark_magenta]Transcript:[/bold dark_magenta] "
        t = result['text'].strip()
        text += f"[italic honeydew2]{t}[/italic honeydew2]\n"
        # sys.stdout(text)
        print(Panel(text, title=ttext))
        # audio_thread.join()  # Wait for the audio to finish playing
        os.remove(seg_audio)  # Clean up the temporary file
        i += 1
    return (transcripts, speakers)

# Step 5: Merge transcripts with speaker annotations
def merge_transcripts(transcripts):
    # Sort segments by start time
    try:
        transcripts.sort(key=lambda x: x['start'])
    except Exception as e:
        print(transcripts)
        print(f"transcripts type {type(transcripts)}")
        print(f"Error sorting transcripts: {e}")
    merged_text = ""
    for seg in transcripts:
        merged_text += f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker']}: {seg['text']}\n"
    return merged_text

def get_id_from_url(url):
    if "shorts" in url:
        id = url.split("/")[-1]
    else:
        id = url.split("v=")[-1].split("&")[0]
    return id
def send_to_chatgpt_4o(transcript, num_speakers):
    prompt = (
        f"The following transcript contains {num_speakers} speakers. "
        "Please identify the speakers based on any names or hints in the text. "
        "If identities cannot be inferred, suggest roles such as interviewer/interviewee, father/son, etc.\n\n"
        f"{transcript}"
    )
    response = client.completions.create(model="gpt-4o",
    prompt=prompt,
    max_tokens=500,
    temperature=0.5)
    if response:
        return response.choices[0].text.strip()
    else:
        print("Failed to get a response from GPT-4o.")
        return None

def main():
    args = parse_arguments()
    if args.video_url:
        video_id = get_id_from_url(args.video_url)
        print(video_id)
    elif args.video_id:
        video_id = args.video_id
    else:
        print("Please provide a video URL or ID.")
        sys.exit(1)

    video_file = download_video(video_id, force=args.force_download)
    print(video_file)
    audio_file = extract_audio(video_file, force=args.force_extraction)

    segments = diarize_audio(audio_file)
    transcripts, speakers = transcribe_audio_segments(audio_file, segments)
    final_transcript = merge_transcripts(transcripts)

    with open("final_transcript.txt", "w", encoding="utf-8") as f:
        f.write(final_transcript)

    print("Transcription complete! Check final_transcript.txt")

    # Send the final transcript to ChatGPT 4o
    num_speakers = len(transcripts[1])
    chatgpt_response = send_to_chatgpt_4o(final_transcript, num_speakers)
    if chatgpt_response:
        print(Panel(chatgpt_response, title="[bold magenta]GPT-4o Speaker Identification[/bold magenta]", 
                    border_style="bold green", expand=False))

if __name__ == "__main__":
    main()
