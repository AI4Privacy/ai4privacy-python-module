"""
protect.py
Implements the **Protect** mode: irreversible anonymisation through
state‑of‑the‑art token‑classification and masking.

The core logic is adapted from the original `mask.py`, with the
heavy lifting (model & tokenizer) delegated to
`ai4privacy.inference.model_runner`.
"""

import re
import torch
from ai4privacy.inference.model_runner import load_model_and_tokenizer

# Load shared model & tokenizer
model, tokenizer, device = load_model_and_tokenizer()

# --------------------------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------------------------
def _get_token_predictions(texts):
    """
    Predict token‑level privacy labels for a list of texts.

    Parameters
    ----------
    texts : List[str]
        Batch of input strings.

    Returns
    -------
    List[List[dict]]
        Token‑prediction dicts for each text.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True
    )
    offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)

    results = []
    for i, text in enumerate(texts):
        tok_preds = []
        for j, (start, end) in enumerate(offsets[i]):
            if start == end:  # special token
                continue
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

def _aggregate_entities(token_predictions, score_threshold=0.5):
    spans = []
    current = None
    for tok in token_predictions:
        label = tok["predicted_label"]
        if label == "O":
            if current:
                current["activation"] = sum(current["scores"])/len(current["scores"])
                if current["activation"] >= score_threshold:
                    spans.append(current)
                current = None
            continue
        tag, ent_type = label.split("-", 1)
        if tag == "B":
            if current:
                current["activation"] = sum(current["scores"])/len(current["scores"])
                if current["activation"] >= score_threshold:
                    spans.append(current)
            current = {"label": ent_type, "start": tok["start"], "end": tok["end"], "scores": [tok["predicted_score"]]}
        elif tag == "I":
            if current and current["label"] == ent_type:
                current["end"] = tok["end"]
                current["scores"].append(tok["predicted_score"])
            else:  # mis‑aligned I: treat as B
                if current:
                    current["activation"] = sum(current["scores"])/len(current["scores"])
                    if current["activation"] >= score_threshold:
                        spans.append(current)
                current = {"label": ent_type, "start": tok["start"], "end": tok["end"], "scores": [tok["predicted_score"]]}
    if current:
        current["activation"] = sum(current["scores"])/len(current["scores"])
        if current["activation"] >= score_threshold:
            spans.append(current)
    return spans

def _enhance_spans(text, spans):
    """
    Post‑process raw spans (merge overlaps, trim, email heuristics, etc.)
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x["start"])
    enhanced = []
    i = 0
    while i < len(spans):
        s = spans[i]
        # Heuristic expansion for EMAIL
        if s["label"] == "EMAIL":
            start = s["start"]
            while start > 0 and not text[start-1].isspace() and text[start-1] not in "!?:;,.":
                start -= 1
            end = s["end"]
            while end < len(text) and not text[end].isspace() and text[end] not in "!?:;,.":
                end += 1
            value = text[start:end]
            if "@" in value and "." in value.split("@")[-1]:
                enhanced.append({"label":"EMAIL","start":start,"end":end,"activation":s["activation"],"scores":[s["activation"]]})
                i += 1
                while i < len(spans) and spans[i]["start"] < end:
                    i += 1
                continue
        enhanced.append(s)
        i += 1
    # Merge overlaps/same‑label close spans
    merged = [enhanced[0]]
    for cur in enhanced[1:]:
        last = merged[-1]
        if cur["label"] == last["label"] and cur["start"] <= last["end"] + 5:
            last["end"] = max(last["end"], cur["end"])
            last["activation"] = max(last["activation"], cur["activation"])
        else:
            merged.append(cur)
    return merged

def _mask_text(text, spans):
    spans = sorted(spans, key=lambda x: x["start"])
    masked, replacements, last_idx = "", [], 0
    for span in spans:
        eff_start = span["start"]
        while eff_start < span["end"] and text[eff_start].isspace():
            eff_start += 1
        if eff_start >= span["end"]:
            continue
        value = text[eff_start:span["end"]]
        placeholder = f"[PII_{len(replacements)+1}]"

        masked += text[last_idx:eff_start] + placeholder
        replacements.append({
            "label": span["label"],
            "start": eff_start,
            "end": span["end"],
            "value": value,
            "label_index": len(replacements)+1,
            "activation": span["activation"]
        })
        last_idx = span["end"]
    masked += text[last_idx:]
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked, replacements

# --------------------------------------------------------------------
# PUBLIC API
# --------------------------------------------------------------------
def mask(text, verbose=False, score_threshold=0.5):
    """
    Anonymise a single text string.

    Parameters
    ----------
    text : str
        Input text.
    verbose : bool, default False
        If True, returns dict with metadata; otherwise masked text only.
    score_threshold : float, default 0.5
        Confidence threshold for entity acceptance.

    Returns
    -------
    str or dict
        Masked text or detailed dict if verbose=True.
    """
    t_preds = _get_token_predictions([text])[0]
    spans = _aggregate_entities(t_preds, score_threshold)
    spans = _enhance_spans(text, spans)
    masked, repl = _mask_text(text, spans)
    if verbose:
        return {"original_text": text, "masked_text": masked, "replacements": repl}
    return masked

def batch(texts, verbose=False, score_threshold=0.5, batch_size=32):
    """
    Mask a list of texts.

    Returns list of masked texts or metadata dicts, matching input order.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        chunk_preds = _get_token_predictions(chunk)
        for txt, preds in zip(chunk, chunk_preds):
            spans = _aggregate_entities(preds, score_threshold)
            spans = _enhance_spans(txt, spans)
            masked, repl = _mask_text(txt, spans)
            if verbose:
                results.append({"original_text": txt, "masked_text": masked, "replacements": repl})
            else:
                results.append(masked)
    return results

# Alias for backward compatibility
get_token_predictions = _get_token_predictions
