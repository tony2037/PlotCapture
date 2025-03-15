import os

def read_videos(config):
    """Read all .mp4 and .avi files from the video directory."""
    video_dir = config["video_dir"]
    supported_formats = config["supported_formats"]
    
    if not os.path.exists(video_dir):
        return []
    
    video_files = [
        os.path.join(video_dir, f) 
        for f in os.listdir(video_dir) 
        if os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    return sorted(video_files)  # Sort for consistent order
