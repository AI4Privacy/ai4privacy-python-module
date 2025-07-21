"""
mask.py - Deprecated Functions for Backward Compatibility.

This module provides the old `mask` and `batch` functions. These are now
aliases for the new `protect` and `batch_protect` functions and will be
removed in a future version.
"""
import warnings
from .protect.protect import protect, batch_protect

def mask(*args, **kwargs):
    """
    DEPRECATED: This function is an alias for protect().
    It will be removed in a future version.
    """
    warnings.warn(
        "'mask' will be deprecated in the next version. Please use 'protect()' for masking or 'observe()' for analysis.",
        DeprecationWarning,
        stacklevel=2
    )
    return protect(*args, **kwargs)

def batch(*args, **kwargs):
    """
    DEPRECATED: This function is an alias for batch_protect().
    It will be removed in a future version.
    """
    warnings.warn(
        "'batch' will be deprecated in the next version. Please use 'batch_protect()' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return batch_protect(*args, **kwargs)