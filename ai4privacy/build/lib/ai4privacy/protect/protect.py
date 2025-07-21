"""
protect.py
Implements the **Protect** mode: irreversible anonymisation.
"""
from ..core.semantic_layer import analyze as analyze_for_pii

def _mask_text(text, spans):
    if not spans:
        return text, []
    spans = sorted(spans, key=lambda x: x["start"])
    masked_text = ""
    replacements = []
    last_end = 0
    for i, span in enumerate(spans):
        effective_start = span['start']
        while effective_start < span['end'] and text[effective_start].isspace():
            effective_start += 1
        if effective_start >= span['end']:
            masked_text += text[last_end:span['end']]
            last_end = span['end']
            continue
        masked_text += text[last_end:effective_start]
        placeholder = f"[PII_{i+1}]"
        masked_text += placeholder
        replacements.append({
            "label": span["label"], "start": effective_start, "end": span["end"],
            "value": text[effective_start:span['end']], "label_index": i + 1,
            "activation": span["activation"]
        })
        last_end = span['end']
    masked_text += text[last_end:]
    return masked_text, replacements

def protect(text, verbose=False, score_threshold=0.01, multilingual=False, classify_pii=False, developer_verbose=False):
    """
    Finds and masks PII in a single text string.
    """
    analysis_result = analyze_for_pii(
        [text], score_threshold, 1, multilingual, classify_pii, developer_verbose
    )[0]
    
    spans = analysis_result['spans']
    masked, repl = _mask_text(text, spans)

    if not verbose and not developer_verbose:
        return masked

    return_dict = {
        "original_text": text,
        "masked_text": masked,
        "replacements": repl
    }
    if developer_verbose:
        return_dict['developer_details'] = analysis_result['developer_details']
    
    return return_dict

def batch_protect(texts, verbose=False, score_threshold=0.01, batch_size=32, multilingual=False, classify_pii=False, developer_verbose=False):
    """Finds and masks PII in a list of texts."""
    all_analysis_results = analyze_for_pii(
        texts, score_threshold, batch_size, multilingual, classify_pii, developer_verbose
    )
    
    final_results = []
    for i, text in enumerate(texts):
        analysis_result = all_analysis_results[i]
        spans = analysis_result['spans']
        masked, repl = _mask_text(text, spans)

        if not verbose and not developer_verbose:
            final_results.append(masked)
        else:
            return_dict = {
                "original_text": text,
                "masked_text": masked,
                "replacements": repl
            }
            if developer_verbose:
                return_dict['developer_details'] = analysis_result['developer_details']
            final_results.append(return_dict)
            
    return final_results