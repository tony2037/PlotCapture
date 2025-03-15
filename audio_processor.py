import subprocess
import whisper
import os

def generate_transcript(video_path, subtitle_file=None, config=None):
    """Generate transcript from video audio or subtitle file."""
    if subtitle_file and os.path.exists(subtitle_file):
        with open(subtitle_file, 'r') as f:
            return f.read()
    
    # Extract audio
    subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-acodec', 'mp3', config["audio_file"], '-y'], check=True)
    
    # Transcribe with Whisper
    model = whisper.load_model("medium")
    result = model.transcribe(config["audio_file"])
    
    # Format transcript with timestamps
    segments = result['segments']
    formatted_transcript = ""
    
    for segment in segments:
        start_time = format_timestamp(segment['start'])
        text = segment['text'].strip()
        formatted_transcript += f"[{start_time}] {text} "
    
    return formatted_transcript.strip()

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
