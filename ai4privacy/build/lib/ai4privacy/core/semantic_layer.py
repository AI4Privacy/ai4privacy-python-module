"""
semantic_layer.py
This module contains the core PII detection and processing logic.
"""

import torch
from .model_runner import get_model_and_tokenizer

def _get_token_predictions(texts, model, tokenizer, device):
    """
    Predicts token-level privacy labels. The logic now finds the highest-scoring
    non-"O" label to overcome the model's bias towards "O".
    """
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
            if start == end: continue
            
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

def _group_and_filter_entities(token_predictions, score_threshold=0.01):
    """
    Groups tokens into words based on whitespace and identifies groups as PII
    if any token in the group exceeds the score threshold.
    This replaces the old B-I tag aggregation logic.
    """
    if not token_predictions:
        return []

    word_groups = []
    current_group = []
    
    # Group tokens into words. A new word starts with the 'Ġ' character (a space).
    for token in token_predictions:
        is_new_word = token['word'].startswith('Ġ') or not current_group
        
        if is_new_word and current_group:
            word_groups.append(current_group)
            current_group = [token]
        else:
            current_group.append(token)
    
    if current_group:
        word_groups.append(current_group)

    # Process each word group to determine if it constitutes PII
    pii_spans = []
    for group in word_groups:
        if not group: continue

        # Find the highest score and corresponding label within the group
        max_score = -1
        best_label = "O"
        for token in group:
            if token['predicted_score'] > max_score:
                max_score = token['predicted_score']
                best_label = token['predicted_label']
        
        # If the best label is PII and meets the threshold, create a single span for the whole group
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

def analyze(texts, score_threshold=0.001, batch_size=32, multilingual=False, classify_pii=False, developer_verbose=False):
    """Analyzes texts to find PII using the new word-grouping logic."""
    model, tokenizer, device = get_model_and_tokenizer(
        multilingual=multilingual, classify_pii=classify_pii
    )
    
    analysis_results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        chunk_token_preds = _get_token_predictions(chunk, model, tokenizer, device)
        for txt, token_preds in zip(chunk, chunk_token_preds):
            # Use the new, robust entity grouping function
            spans = _group_and_filter_entities(token_preds, score_threshold)
            
            result_item = {'spans': spans}
            if developer_verbose:
                result_item['developer_details'] = token_preds
            analysis_results.append(result_item)
            
    return analysis_results