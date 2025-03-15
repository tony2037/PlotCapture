import subprocess
import whisper

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
    return result['text']
