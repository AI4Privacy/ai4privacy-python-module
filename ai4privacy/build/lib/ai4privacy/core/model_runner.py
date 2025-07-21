"""
model_runner.py
Centralized model & tokenizer loader for AI4Privacy modes.

This keeps heavy-weight objects in one place so Observe, Protect,
and other modes can share them without duplicating GPU/CPU memory.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# --- Model Names ---
MODEL_ENGLISH = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"
MODEL_MULTILINGUAL = "ai4privacy/llama-ai4privacy-multilingual-anonymiser-openpii"
MODEL_CATEGORICAL = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Caching and Disclaimer ---
_model_cache = {}
_disclaimer_shown = False

def get_model_and_tokenizer(multilingual=False, classify_pii=False):
    """
    Load the appropriate pre-trained model and tokenizer based on flags.
    Models are cached in memory after first load.
    """
    global _disclaimer_shown
    
    # Determine which model to use
    if classify_pii:
        model_key = "categorical"
        model_name = MODEL_CATEGORICAL
    elif multilingual:
        model_key = "multilingual"
        model_name = MODEL_MULTILINGUAL
    else:
        model_key = "english"
        model_name = MODEL_ENGLISH

    # Return the cached model if available
    if model_key in _model_cache:
        return _model_cache[model_key]

    # Show disclaimer only on the very first model load
    if not _disclaimer_shown:
        disclaimer = """
====================================================================================
[ai4privacy] Disclaimer 📢:
AI4Privacy is trained on the world's largest open-source privacy dataset. 
For production use, please evaluate results carefully. For assistance, contact
us at our website https://ai4privacy.com or email support@ai4privacy.com.
====================================================================================
"""
        print(disclaimer)
        _disclaimer_shown = True

    # Load the model
    print(f"\n[ai4privacy] Loading model: {model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Cache the loaded model and tokenizer
    _model_cache[model_key] = (model, tokenizer, device)
    
    return _model_cache[model_key]