import torch
from plot_generators import plot_t5_impl as plot_llm

def synthesize_plot(title, transcript, captions, config):
    """Synthesize a plot description using FLAN-T5-large."""
    # Determine device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Format captions
    captions_str = ", ".join([f"At {k}: {v}" for k, v in captions.items()])
    
    # Create prompt
    prompt = f"""
    Based on the following transcript, frame descriptions with timestamps, and title.
    First, translate everything to English, then create a detailed plot description that captures the sequence of events and key details in a coherent narrative:
    - Title: "{title}"
    - Transcript: "{transcript}"
    - Frame Descriptions: {captions_str}
    """

    return plot_llm(prompt, device)
