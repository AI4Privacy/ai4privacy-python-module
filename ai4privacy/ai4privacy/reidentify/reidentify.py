"""
Reidentify -- restore original text from masked output.

Works with the replacements list returned by protect(verbose=True).
"""

import re


def reidentify(masked_text, replacements):
    """
    Restore original values in masked text using the replacements list.

    Parameters
    ----------
    masked_text : str
        Text containing [PII_1], [PII_2], ... placeholders.
    replacements : list[dict]
        The ``replacements`` list from ``protect(verbose=True)``.
        Each dict must have at least ``label_index`` (int) and ``value`` (str).

    Returns
    -------
    str
        Text with all [PII_N] placeholders replaced by their original values.
    """
    if not replacements:
        return masked_text

    text = masked_text

    # Process in reverse order (highest index first) so that replacing
    # a placeholder doesn't shift the positions of earlier ones.
    sorted_replacements = sorted(
        replacements,
        key=lambda r: r.get("label_index", 0),
        reverse=True,
    )

    for r in sorted_replacements:
        idx = r.get("label_index")
        value = r.get("value", "")
        if idx is None:
            continue
        placeholder = f"[PII_{idx}]"
        text = text.replace(placeholder, value)

    return text


def batch_reidentify(results):
    """
    Restore original text for a batch of protect(verbose=True) results.

    Parameters
    ----------
    results : list[dict]
        Each dict should have ``masked_text`` (str) and ``replacements`` (list).

    Returns
    -------
    list[str]
        List of restored original texts.
    """
    restored = []
    for result in results:
        masked = result.get("masked_text", "")
        replacements = result.get("replacements", [])
        restored.append(reidentify(masked, replacements))
    return restored
