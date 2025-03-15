import os
import json
from video_reader import read_videos
from audio_processor import generate_transcript
from visual_processor import extract_frames, generate_captions
from plot_synthesizer import synthesize_plot

# Load config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

def process_video(video_path):
    # Use filename (without extension) as title
    title = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nProcessing video: {title}")

    # Generate transcript
    transcript = generate_transcript(video_path, config=CONFIG)
    print("Transcript:\n", transcript)

    # Extract frames and generate captions
    extract_frames(video_path, CONFIG)
    captions = generate_captions(CONFIG)
    print("Frame Captions:\n", captions)

    # Synthesize plot
    plot = synthesize_plot(title, transcript, captions, CONFIG)
    print("\nGenerated Plot Description:\n", plot)

def main():
    # Ensure video directory exists
    if not os.path.exists(CONFIG["video_dir"]):
        os.makedirs(CONFIG["video_dir"])
        print(f"Created empty video directory: {CONFIG['video_dir']}. Please add .mp4 or .avi files.")
        return

    # Read all videos
    video_paths = read_videos(CONFIG)
    if not video_paths:
        print(f"No .mp4 or .avi files found in {CONFIG['video_dir']}")
        return

    # Process each video
    for video_path in video_paths:
        process_video(video_path)

if __name__ == "__main__":
    main()
