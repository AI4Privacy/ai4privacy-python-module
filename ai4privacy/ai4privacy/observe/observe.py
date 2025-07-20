"""
observe.py
Implements the **Observe** mode: non-intrusive analytics about PII.
"""
from collections import Counter
from ..core.semantic_layer import analyze as analyze_for_pii

def observe(texts, score_threshold=0.01, batch_size=32, multilingual=False, classify_pii=False, developer_verbose=False):
    """
    Analyzes text(s) and returns statistics and the detected PII entities.
    """
    is_single_string = isinstance(texts, str)
    if is_single_string:
        texts = [texts]

    all_analysis_results = analyze_for_pii(
        texts, score_threshold, batch_size, multilingual, classify_pii, developer_verbose
    )
    
    label_counts = Counter()
    texts_with_pii = 0
    
    for result in all_analysis_results:
        spans = result['spans']
        if spans:
            texts_with_pii += 1
        for span in spans:
            label_counts[span["label"]] += 1
    
    total_entities = sum(label_counts.values())

    return_dict = {
        "num_texts_processed": len(texts),
        "num_texts_with_pii": texts_with_pii,
        "pii_entity_counts": dict(label_counts),
        "total_pii_entities_found": total_entities,
    }

    # Add the privacy mask (the list of found PII entities) to the output
    privacy_masks = [res['spans'] for res in all_analysis_results]
    return_dict['privacy_mask'] = privacy_masks[0] if is_single_string else privacy_masks

    if developer_verbose:
        dev_details = [res.get('developer_details', []) for res in all_analysis_results]
        return_dict['developer_details'] = dev_details[0] if is_single_string else dev_details
        
    return return_dict