"""
AI4Privacy – human‑centered privacy masking library.

Exposes high‑level functions for Observe and Protect modes.
Enable mode will be added in a future release.
"""

from .observe.observe import report_statistics
from .protect.protect import mask, batch

__version__ = "0.2.0"

__all__ = [
    "report_statistics",
    "mask",
    "batch",
]
