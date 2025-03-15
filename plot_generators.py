import torch
from transformers import pipeline

def plot_t5_impl(prompt, device):
    """Implementation of plot generation using FLAN-T5-large model."""
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