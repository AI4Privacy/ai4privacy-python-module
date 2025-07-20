"""
semantic_layer.py
This module contains the core PII detection and processing logic.
"""

import torch
from .model_runner import get_model_and_tokenizer

def _get_token_predictions(texts, model, tokenizer, device):
    """Predict token-level privacy labels for a list of texts."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)

    results = []
    for i in range(len(texts)):
        tok_preds = []
        for j, (start, end) in enumerate(offsets[i]):
            if start == end: continue
            token_id = inputs["input_ids"][i, j].cpu().item()
            tok_preds.append({
                "word": tokenizer.convert_ids_to_tokens(token_id),
                "start": start.item(),
                "end": end.item(),
                "predicted_label": model.config.id2label[pred_ids[i, j].cpu().item()],
                "predicted_score": probs[i, j, pred_ids[i, j]].cpu().item()
            })
        results.append(tok_preds)
    return results

def _aggregate_entities(token_predictions, model, score_threshold=0.5):
    """Aggregate token predictions into entity spans using BIO scheme."""
    spans = []
    current = None
    for tok in token_predictions:
        label = tok["predicted_label"]
        if label == "O":
            if current:
                current["activation"] = sum(current["scores"]) / len(current["scores"])
                if current["activation"] >= score_threshold:
                    spans.append(current)
                current = None
            continue
        tag, ent_type = label.split("-", 1)
        if tag == "B":
            if current:
                current["activation"] = sum(current["scores"]) / len(current["scores"])
                if current["activation"] >= score_threshold:
                    spans.append(current)
            current = {"label": ent_type, "start": tok["start"], "end": tok["end"], "scores": [tok["predicted_score"]]}
        elif tag == "I" and current and current["label"] == ent_type:
            current["end"] = tok["end"]
            current["scores"].append(tok["predicted_score"])
    if current:
        current["activation"] = sum(current["scores"]) / len(current["scores"])
        if current["activation"] >= score_threshold:
            spans.append(current)
    return spans

def _enhance_spans(text, spans):
    """Post-process raw spans (merge overlaps, heuristics, etc.)."""
    if not spans: return []
    spans = sorted(spans, key=lambda x: x["start"])
    merged = []
    if spans:
        merged.append(spans[0])
        for cur in spans[1:]:
            last = merged[-1]
            if cur["label"] == last["label"] and cur["start"] <= last["end"] + 5:
                last["end"] = max(last["end"], cur["end"])
                last["activation"] = max(last["activation"], cur["activation"])
            else:
                merged.append(cur)
    return merged

def analyze(texts, score_threshold=0.5, batch_size=32, multilingual=False, classify_pii=False):
    """Analyzes a list of texts to find and return all PII entities."""
    model, tokenizer, device = get_model_and_tokenizer(
        multilingual=multilingual,
        classify_pii=classify_pii
    )
    
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        chunk_preds = _get_token_predictions(chunk, model, tokenizer, device)
        for txt, preds in zip(chunk, chunk_preds):
            spans = _aggregate_entities(preds, model, score_threshold)
            spans = _enhance_spans(txt, spans)
            results.append(spans)
    return results