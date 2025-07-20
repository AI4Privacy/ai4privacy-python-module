"""
AI4Privacy – human-centered privacy masking library.
"""

from .observe.observe import observe
from .protect.protect import protect, batch_protect

__version__ = "0.3.0"

__all__ = [
    "observe",
    "protect",
    "batch_protect",
]