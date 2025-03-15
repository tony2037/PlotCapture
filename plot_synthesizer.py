import torch
from transformers import pipeline

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

    # Load FLAN-T5-large
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=0 if device == "cuda" else -1  # 0 for GPU, -1 for CPU
    )
    
    # Generate plot (increase max_length for better output)
    response = llm(prompt, max_length=1000, truncation=True)
    return response[0]['generated_text'].strip()
