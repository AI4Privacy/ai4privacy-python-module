"""
observe.py
Implements the **Observe** mode: non‑intrusive analytics about privacy
content present in raw text, without persisting or returning any masked text.
"""

from collections import Counter, defaultdict
from ai4privacy.inference.model_runner import load_model_and_tokenizer
import torch

model, tokenizer, device = load_model_and_tokenizer()

def _get_labels(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
    preds = logits.argmax(dim=-1).cpu()
    return preds, inputs["input_ids"]

def report_statistics(texts):
    """
    Analyse a list of texts and return privacy‑label statistics.

    Parameters
    ----------
    texts : List[str]

    Returns
    -------
    dict
        {
          'num_entries': int,
          'label_counts': {label: count},
          'label_ratio': {label: count/total_tokens},
          'total_tokens': int
        }
    """
    preds, input_ids = _get_labels(texts)
    counts = Counter()
    total_tokens = 0
    id2label = model.config.id2label
    for pred_row, ids_row in zip(preds, input_ids):
        for pred, tok_id in zip(pred_row, ids_row):
            tok = tokenizer.convert_ids_to_tokens(tok_id.item())
            if tok.startswith("▁") or tok.startswith("Ġ") or not tok:  # typical sub‑word markers
                pass
            counts[id2label[pred.item()]] += 1
            total_tokens += 1
    # Remove 'O' from privacy stats if desired
    label_counts = {k: v for k, v in counts.items() if k != "O"}
    ratios = {k: v/total_tokens for k, v in label_counts.items()}
    return {
        "num_entries": len(texts),
        "total_tokens": total_tokens,
        "label_counts": label_counts,
        "label_ratio": ratios
    }
