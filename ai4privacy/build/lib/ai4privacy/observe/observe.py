"""
observe.py
Implements the **Observe** mode: non-intrusive analytics about PII.
"""
from collections import Counter
from ..core.semantic_layer import analyze as analyze_for_pii

def observe(texts, score_threshold=0.01, batch_size=32, multilingual=False, classify_pii=False):
    """
    Analyzes text(s) and returns statistics about detected PII entities.

    Parameters
    ----------
    texts : str or List[str]
        A single text string or a list of texts to analyze.
    score_threshold : float, default 0.5
        Confidence threshold for entity acceptance.
    batch_size : int, default 32
        The batch size for model inference if a list is provided.
    multilingual : bool, default False
        Set to True to use the multilingual anonymiser model.
    classify_pii : bool, default False
        Set to True to use the multilingual categorical model.
        This overrides the `multilingual` flag.

    Returns
    -------
    dict
        A dictionary containing statistics about the found PII.
    """
    is_single_string = isinstance(texts, str)
    if is_single_string:
        texts = [texts]

    all_spans = analyze_for_pii(
        texts,
        score_threshold=score_threshold,
        batch_size=batch_size,
        multilingual=multilingual,
        classify_pii=classify_pii
    )
    
    label_counts = Counter()
    texts_with_pii = 0
    
    for spans in all_spans:
        if spans:
            texts_with_pii += 1
        for span in spans:
            label_counts[span["label"]] += 1
    
    total_entities = sum(label_counts.values())

    return {
        "num_texts_processed": len(texts),
        "num_texts_with_pii": texts_with_pii,
        "pii_entity_counts": dict(label_counts),
        "total_pii_entities_found": total_entities,
    }