"""
model_runner.py
Centralized model & tokenizer loader for AI4Privacy modes.

This keeps heavy-weight objects in one place so Observe, Protect,
and Enable can share them without duplicating GPU/CPU memory.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_NAME = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"

# Decide device once at import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    """
    Load the pre‑trained transformer model and tokenizer.

    Returns
    -------
    model : PreTrainedModel
        Hugging Face model for token classification.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    device : torch.device
        The device (GPU or CPU) on which the model is located.
    """
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer, device
