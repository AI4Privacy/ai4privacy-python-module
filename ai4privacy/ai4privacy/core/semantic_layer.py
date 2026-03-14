"""
semantic_layer.py — v2 production post-processing pipeline

PII detection with 9-step post-processing:
1. Sliding window (512/256 stride) for long texts
2. Smart anti-O bias (concentration-based, not naive)
3. B/I disagreement fix
4. I-means-continue BIO decoding
5. Left-expand to word boundary
6. Short-label trim (1 word max)
7. GENDER -> SEX label merge
8. Right-trim trailing punctuation
9. Overlap dedup
"""

import torch
from collections import Counter
from .model_runner import get_model_and_tokenizer

# ── Constants ────────────────────────────────────────────────────────
PUNCT_TIGHT = set('.,;:!?()[]{}"\'/\\@#$%^&*+=<>|~`')
SHORT_LABELS = {"SEX", "CURRENCY", "BLOODTYPE", "TITLE", "ORDINALDIRECTION", "AGE", "PIN"}
LABEL_MERGES = {"GENDER": "SEX"}


# ── 1. Sliding window inference ──────────────────────────────────────

def _get_sliding_window_logits(text, model, tokenizer, device, window=512, stride=256):
    encoding = tokenizer(text, return_tensors="pt", truncation=False,
                         return_offsets_mapping=True)
    all_input_ids = encoding["input_ids"][0]
    all_offsets = encoding["offset_mapping"][0]
    seq_len = len(all_input_ids)

    if seq_len <= window:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=window, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        return logits, offsets

    logit_sums = torch.zeros(seq_len, model.config.num_labels)
    logit_counts = torch.zeros(seq_len)

    for start_idx in range(0, seq_len, stride):
        end_idx = min(start_idx + window, seq_len)
        chunk_ids = all_input_ids[start_idx:end_idx].unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids).to(device)
        with torch.no_grad():
            chunk_logits = model(
                input_ids=chunk_ids, attention_mask=chunk_mask
            ).logits[0].cpu()
        for i, pos in enumerate(range(start_idx, end_idx)):
            logit_sums[pos] += chunk_logits[i]
            logit_counts[pos] += 1
        if end_idx >= seq_len:
            break

    avg_logits = logit_sums / logit_counts.unsqueeze(-1).clamp(min=1)
    return avg_logits, all_offsets


# ── 2. Smart anti-O bias ────────────────────────────────────────────

def _apply_anti_o_bias(logits, id2label, min_concentration=0.4, min_prob=0.001):
    """Override O only when top non-O label dominates non-O mass.

    concentration = top_non_O_prob / total_non_O_prob
    Override when concentration >= 0.4 AND prob >= 0.1%.
    """
    o_idx = next(k for k, v in id2label.items() if v == "O")
    result = logits.clone()
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1)

    for i in range(len(pred_ids)):
        if pred_ids[i].item() != o_idx:
            continue
        o_prob = probs[i, o_idx].item()
        non_o_probs = probs[i].clone()
        non_o_probs[o_idx] = 0
        top_non_o_prob = non_o_probs.max().item()
        non_o_total = 1.0 - o_prob

        if non_o_total < 1e-6:
            continue
        concentration = top_non_o_prob / non_o_total

        if concentration >= min_concentration and top_non_o_prob >= min_prob:
            result[i, o_idx] = float('-inf')

    return result


# ── 3. B/I disagreement fix ─────────────────────────────────────────

def _fix_bi_disagreement(logits, offsets, id2label):
    """When I-tags outvote the B-tag (>= 60%), flip B to match."""
    pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
    fixed = list(pred_ids)

    i = 0
    while i < len(pred_ids):
        s, e = offsets[i]
        if s == e:
            i += 1
            continue
        tag = id2label.get(pred_ids[i], "O")
        if not tag.startswith("B-"):
            i += 1
            continue

        b_label = tag[2:]
        j = i + 1
        i_labels = []
        while j < len(pred_ids):
            sj, ej = offsets[j]
            if sj == ej:
                j += 1
                continue
            jtag = id2label.get(pred_ids[j], "O")
            if jtag.startswith("I-"):
                i_labels.append(jtag[2:])
                j += 1
            else:
                break

        if len(i_labels) >= 2:
            counts = Counter(i_labels)
            dominant_label, dominant_count = counts.most_common(1)[0]
            if dominant_count >= len(i_labels) * 0.6 and dominant_label != b_label:
                b_target = f"B-{dominant_label}"
                b_id = next((k for k, v in id2label.items() if v == b_target), None)
                if b_id is not None:
                    fixed[i] = b_id

        i = j if j > i else i + 1

    return torch.tensor(fixed)


# ── 4. Decode: I-means-continue ──────────────────────────────────────

def _decode_bio(logits, offsets, id2label):
    """BIO decode with I-means-continue: any I- tag continues current span."""
    fixed_pred_ids = _fix_bi_disagreement(logits, offsets, id2label)
    pred_ids = fixed_pred_ids.numpy()
    probs = torch.softmax(logits, dim=-1)
    spans = []
    current_label = None
    current_start = current_end = None
    current_score = 0.0

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        tag = id2label.get(int(pred_ids[i]), "O")
        score = probs[i, int(pred_ids[i])].item()

        if tag.startswith("B-"):
            if current_label:
                spans.append((current_start, current_end, current_label, current_score))
            current_label = tag[2:]
            current_start, current_end = start.item(), end.item()
            current_score = score
        elif tag.startswith("I-") and current_label:
            current_end = end.item()
            current_score = max(current_score, score)
        else:
            if current_label:
                spans.append((current_start, current_end, current_label, current_score))
            current_label = None

    if current_label:
        spans.append((current_start, current_end, current_label, current_score))
    return spans


# ── 5-9. Post-processing ────────────────────────────────────────────

def _postprocess(spans, text):
    result = []
    for start, end, label, score in spans:
        # Trim leading whitespace (LLaMA Ġ tokenizer includes spaces in offsets)
        while start < end and text[start].isspace():
            start += 1
        if start >= end:
            continue
        # 5. Left expand to word boundary
        while start > 0 and not text[start - 1].isspace() and text[start - 1] not in PUNCT_TIGHT:
            start -= 1
        # 6. Short label trim to 1 word
        if label in SHORT_LABELS:
            stop_chars = PUNCT_TIGHT - {'+', '-'} if label == "BLOODTYPE" else PUNCT_TIGHT
            word_end = start
            while word_end < len(text) and not text[word_end].isspace() and text[word_end] not in stop_chars:
                word_end += 1
            end = word_end
        # 7. Label merge
        label = LABEL_MERGES.get(label, label)
        result.append((start, end, label, score))

    # 8. Right-trim trailing punctuation (skip BLOODTYPE)
    result2 = []
    for start, end, label, score in result:
        if label != "BLOODTYPE":
            while end > start and (text[end - 1] in PUNCT_TIGHT or text[end - 1].isspace()):
                end -= 1
        if end > start:
            result2.append((start, end, label, score))
    result = result2

    # 9. Dedup overlapping spans
    deduped = []
    for s in sorted(result):
        if deduped and s[0] < deduped[-1][1] and s[2] == deduped[-1][2]:
            deduped[-1] = (deduped[-1][0], max(deduped[-1][1], s[1]), s[2], max(deduped[-1][3], s[3]))
        elif deduped and s[0] < deduped[-1][1]:
            pass  # overlapping different label — keep first
        else:
            deduped.append(s)
    return deduped


# ── Legacy API (v1 compatibility) ───────────────────────────────────

def _get_token_predictions(texts, model, tokenizer, device):
    """Legacy: per-token predictions with naive anti-O."""
    o_label_id = model.config.label2id.get('O')

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True
    )
    offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    results = []
    for i in range(len(texts)):
        tok_preds = []
        for j, (start, end) in enumerate(offsets[i]):
            if start == end:
                continue

            token_probs = probs[i, j]
            top_pred_id = token_probs.argmax().item()

            if o_label_id is not None and top_pred_id == o_label_id:
                token_probs[o_label_id] = -1
                final_pred_id = token_probs.argmax().item()
            else:
                final_pred_id = top_pred_id

            predicted_score = probs[i, j, final_pred_id].item()
            predicted_label = model.config.id2label[final_pred_id]

            token_id = inputs["input_ids"][i, j].cpu().item()
            tok_preds.append({
                "word": tokenizer.convert_ids_to_tokens(token_id),
                "start": start.item(), "end": end.item(),
                "predicted_label": predicted_label, "predicted_score": predicted_score
            })
        results.append(tok_preds)
    return results


# ── Public API ───────────────────────────────────────────────────────

def analyze(texts, score_threshold=0.001, batch_size=32, multilingual=False,
            classify_pii=False, developer_verbose=False):
    """Analyze texts to find PII using the v2 production pipeline."""
    model, tokenizer, device = get_model_and_tokenizer(
        multilingual=multilingual, classify_pii=classify_pii
    )
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    # Check if model uses BIO labels (has B- and I- prefixed labels)
    has_bio = any(v.startswith("B-") for v in id2label.values())

    analysis_results = []
    for text in texts:
        if has_bio:
            # v2 pipeline: sliding window + smart anti-O + full PP
            logits, offsets = _get_sliding_window_logits(text, model, tokenizer, device)
            logits = _apply_anti_o_bias(logits, id2label)
            raw_spans = _decode_bio(logits, offsets, id2label)
            spans = _postprocess(raw_spans, text)

            span_dicts = []
            for start, end, label, score in spans:
                if score >= score_threshold:
                    span_dicts.append({
                        'label': label,
                        'start': start,
                        'end': end,
                        'activation': score,
                        'value': text[start:end]
                    })
        else:
            # Legacy: non-BIO model (binary PII/O), use word grouping
            token_preds = _get_token_predictions([text], model, tokenizer, device)[0]
            span_dicts = _group_and_filter_entities(token_preds, score_threshold)

        result_item = {'spans': span_dicts}
        if developer_verbose:
            if has_bio:
                result_item['developer_details'] = {
                    'pipeline': 'v2_production',
                    'steps': ['sliding_window', 'anti_o_bias', 'bi_fix',
                              'i_continues', 'left_expand', 'short_trim',
                              'label_merge', 'right_trim', 'dedup'],
                }
            else:
                token_preds = _get_token_predictions([text], model, tokenizer, device)[0]
                result_item['developer_details'] = token_preds
        analysis_results.append(result_item)

    return analysis_results


def _group_and_filter_entities(token_predictions, score_threshold=0.01):
    """Legacy: groups tokens into words by Ġ boundary for non-BIO models."""
    if not token_predictions:
        return []

    word_groups = []
    current_group = []

    for token in token_predictions:
        is_new_word = token['word'].startswith('Ġ') or not current_group

        if is_new_word and current_group:
            word_groups.append(current_group)
            current_group = [token]
        else:
            current_group.append(token)

    if current_group:
        word_groups.append(current_group)

    pii_spans = []
    for group in word_groups:
        if not group:
            continue

        max_score = -1
        best_label = "O"
        for token in group:
            if token['predicted_score'] > max_score:
                max_score = token['predicted_score']
                best_label = token['predicted_label']

        if best_label != "O" and max_score >= score_threshold:
            clean_label = best_label.split('-', 1)[-1]

            pii_spans.append({
                'label': clean_label,
                'start': group[0]['start'],
                'end': group[-1]['end'],
                'activation': max_score,
                'value': "".join([t['word'] for t in group]).replace('Ġ', ' ').strip()
            })

    return pii_spans
