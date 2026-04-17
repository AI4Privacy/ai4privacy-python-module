"""
AI4Privacy – human-centered privacy masking library.
"""

from .observe.observe import observe
from .protect.protect import protect, batch_protect
from .reidentify.reidentify import reidentify, batch_reidentify

__version__ = "0.5.1"

__all__ = [
    "observe",
    "protect",
    "batch_protect",
    "reidentify",
    "batch_reidentify",
    "mask",
    "batch",
]