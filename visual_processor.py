import os
import subprocess
from transformers import pipeline

def extract_frames(video_path, config):
    """Extract frames from video at specified interval."""
    os.makedirs(config["frame_dir"], exist_ok=True)
    fps = 1 / config["frame_interval"]
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{config["frame_dir"]}frame_%04d.jpg', '-y'],
        check=True
    )

def generate_captions(config):
    """Generate captions for extracted frames using BLIP."""
    # Load BLIP model
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Process frames
    frame_files = sorted([f for f in os.listdir(config["frame_dir"]) if f.endswith('.jpg')])
    captions = {}
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(config["frame_dir"], frame_file)
        caption = captioner(frame_path)[0]['generated_text']
        seconds = i * config["frame_interval"]
        timestamp = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        captions[timestamp] = caption
    return captions
