"""
protect.py
Implements the **Protect** mode: irreversible anonymisation.
"""
import re
from ..core.semantic_layer import analyze as analyze_for_pii

def _mask_text(text, spans):
    # This function remains unchanged.
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

def protect(text, verbose=False, score_threshold=0.5, multilingual=False, classify_pii=False):
    """
    Finds and masks PII in a single text string.

    Parameters
    ----------
    text : str
        The input text.
    verbose : bool, default False
        If True, returns a dict with details, otherwise the masked text.
    score_threshold : float, default 0.5
        Confidence threshold for entity acceptance.
    multilingual : bool, default False
        Set to True to use the multilingual anonymiser model.
    classify_pii : bool, default False
        Set to True to use the multilingual categorical model.
        This overrides the `multilingual` flag.

    Returns
    -------
    str or dict
        Masked text, or a details dictionary if verbose=True.
    """
    spans = analyze_for_pii(
        [text],
        score_threshold=score_threshold,
        multilingual=multilingual,
        classify_pii=classify_pii
    )[0]
    masked, repl = _mask_text(text, spans)
    if verbose:
        return {"original_text": text, "masked_text": masked, "replacements": repl}
    return masked

def batch_protect(texts, verbose=False, score_threshold=0.01, batch_size=32, multilingual=False, classify_pii=False):
    """Finds and masks PII in a list of texts."""
    all_spans = analyze_for_pii(
        texts,
        score_threshold=score_threshold,
        batch_size=batch_size,
        multilingual=multilingual,
        classify_pii=classify_pii
    )
    results = []
    for i, text in enumerate(texts):
        spans = all_spans[i]
        masked, repl = _mask_text(text, spans)
        if verbose:
            results.append({"original_text": text, "masked_text": masked, "replacements": repl})
        else:
            results.append(masked)
    return results