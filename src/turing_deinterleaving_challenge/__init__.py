"""
turing-deinterleaving-challenge: A set of utilities to support the turing's deinterleaving challenge
"""

from __future__ import annotations

from importlib.metadata import version

from .data import PulseTrain, download_dataset, DeinterleavingChallengeDataset

__all__ = (
    "PulseTrain",
    "__version__",
    "download_dataset",
    "DeinterleavingChallengeDataset",
)
__version__ = version(__name__)
