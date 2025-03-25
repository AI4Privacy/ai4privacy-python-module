import re
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load model and tokenizer globally, move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#MODEL_NAME = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
MODEL_NAME = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_token_predictions(texts):
    """
    Tokenize a list of texts and get model predictions for each token in the batch.
    Runs on GPU if available.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offsets = inputs["offset_mapping"]
    input_ids = inputs["input_ids"].to(device)
    model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}
    with torch.no_grad():
        outputs = model(**model_inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(probs, dim=-1)
    
    token_predictions = []
    for i in range(len(texts)):
        text_offsets = offsets[i]
        text_input_ids = input_ids[i].cpu()  # Move back to CPU for token conversion
        text_predicted_ids = predicted_ids[i].cpu()
        text_probs = probs[i].cpu()
        text_token_predictions = []
        for j, (start, end) in enumerate(text_offsets):
            if start == end:  # Skip special tokens
                continue
            token = tokenizer.convert_ids_to_tokens(text_input_ids[j].item())
            pred_label = model.config.id2label[text_predicted_ids[j].item()]
            pred_score = text_probs[j, text_predicted_ids[j]].item()
            text_token_predictions.append({
                "word": token,
                "start": start.item(),
                "end": end.item(),
                "predicted_label": pred_label,
                "predicted_score": pred_score
            })
        token_predictions.append(text_token_predictions)
    return token_predictions

def aggregate_entities(token_predictions, score_threshold=0.5):
    spans = []
    current_span = None
    for token in token_predictions:
        label = token["predicted_label"]
        if label == "O":
            if current_span:
                current_span["activation"] = sum(current_span["scores"]) / len(current_span["scores"])
                if current_span["activation"] >= score_threshold:
                    spans.append(current_span)
                current_span = None
            continue
        if label.startswith("B-"):
            if current_span:
                current_span["activation"] = sum(current_span["scores"]) / len(current_span["scores"])
                if current_span["activation"] >= score_threshold:
                    spans.append(current_span)
            entity_type = label[2:]
            current_span = {
                "label": entity_type,
                "start": token["start"],
                "end": token["end"],
                "scores": [token["predicted_score"]]
            }
        elif label.startswith("I-"):
            entity_type = label[2:]
            if current_span and current_span["label"] == entity_type:
                current_span["end"] = token["end"]
                current_span["scores"].append(token["predicted_score"])
            else:
                if current_span:
                    current_span["activation"] = sum(current_span["scores"]) / len(current_span["scores"])
                    if current_span["activation"] >= score_threshold:
                        spans.append(current_span)
                current_span = {
                    "label": entity_type,
                    "start": token["start"],
                    "end": token["end"],
                    "scores": [token["predicted_score"]]
                }
    if current_span:
        current_span["activation"] = sum(current_span["scores"]) / len(current_span["scores"])
        if current_span["activation"] >= score_threshold:
            spans.append(current_span)
    return spans

def enhance_spans(text, spans):
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    enhanced = []
    i = 0
    while i < len(sorted_spans):
        span = sorted_spans[i]
        if span["label"] == "EMAIL":
            start = span["start"]
            while start > 0 and not text[start-1].isspace() and text[start-1] not in "!?,;:":
                start -= 1
            end = span["end"]
            while end < len(text) and not text[end].isspace() and text[end] not in "!?,;:":
                end += 1
            value = text[start:end]
            if "@" in value and "." in value[value.index("@"):]:
                enhanced.append({
                    "label": "EMAIL",
                    "start": start,
                    "end": end,
                    "activation": span["activation"],
                    "scores": [span["activation"]]
                })
                i += 1
                while i < len(sorted_spans) and sorted_spans[i]["start"] < end:
                    i += 1
                continue
        enhanced.append(span)
        i += 1
    
    if not enhanced:
        return []
    merged = [enhanced[0]]
    for current in enhanced[1:]:
        last = merged[-1]
        if (current["label"] == last["label"] and 
            current["start"] <= last["end"] + 5):
            last["end"] = max(last["end"], current["end"])
            last["activation"] = max(last["activation"], current["activation"])
        else:
            merged.append(current)
    return merged

def mask_text(text, spans):
    sorted_spans = sorted(spans, key=lambda x: x["start"])
    masked_text = ""
    replacements = []
    last_idx = 0
    for span in sorted_spans:
        effective_start = span["start"]
        while effective_start < span["end"] and text[effective_start].isspace():
            effective_start += 1
        if effective_start >= span["end"]:
            continue
        value = text[effective_start:span["end"]].strip()
        placeholder = f"[{value}]"
        masked_text += text[last_idx:effective_start] + placeholder
        replacements.append({
            "label": span["label"],
            "start": effective_start,
            "end": span["end"],
            "value": value,
            "label_index": len(replacements) + 1,
            "activation": span["activation"]
        })
        last_idx = span["end"]
    masked_text += text[last_idx:]
    masked_text = re.sub(r"\s+", " ", masked_text).strip()
    return masked_text, replacements

def mask(text, verbose=False, score_threshold=0.5):
    token_predictions = get_token_predictions([text])[0]
    spans = aggregate_entities(token_predictions, score_threshold)
    enhanced_spans = enhance_spans(text, spans)
    masked_text, replacements = mask_text(text, enhanced_spans)
    if verbose:
        return {
            "original_text": text,
            "masked_text": masked_text,
            "replacements": replacements
        }
    return masked_text

def batch(texts, verbose=False, score_threshold=0.5, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_token_predictions = get_token_predictions(batch_texts)
        for text, token_predictions in zip(batch_texts, batch_token_predictions):
            spans = aggregate_entities(token_predictions, score_threshold)
            enhanced_spans = enhance_spans(text, spans)
            masked_text, replacements = mask_text(text, enhanced_spans)
            if verbose:
                results.append({
                    "original_text": text,
                    "masked_text": masked_text,
                    "replacements": replacements
                })
            else:
                results.append(masked_text)
    return results