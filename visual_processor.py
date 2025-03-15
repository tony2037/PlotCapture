import os
import subprocess
from transformers import pipeline

def extract_frames(video_path, config):
    """Extract frames from video at specified interval."""
    # Get video title from path (removing extension)
    video_title = os.path.splitext(os.path.basename(video_path))[0]

    # Create specific directory for this video
    video_frame_dir = os.path.join(config["frame_dir"], video_title)

    # Remove directory if it exists and create a new empty one
    if os.path.exists(video_frame_dir):
        import shutil
        shutil.rmtree(video_frame_dir)
    os.makedirs(video_frame_dir)

    fps = 1 / config["frame_interval"]
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', 
         f'{video_frame_dir}/frame_%04d.jpg', '-y'],
        check=True
    )
    return video_frame_dir

def generate_captions(frames_path, frame_interval):
    """Generate captions for extracted frames using BLIP."""
    # Load BLIP model
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Process frames
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    captions = {}
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_path, frame_file)
        caption = captioner(frame_path)[0]['generated_text']
        seconds = i * frame_interval
        timestamp = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
        captions[timestamp] = caption
    return captions
