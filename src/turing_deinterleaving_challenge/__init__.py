"""
turing-deinterleaving-challenge: A set of utilities to support the turing's deinterleaving challenge
"""

from __future__ import annotations

from importlib.metadata import version

from .data import DeinterleavingChallengeDataset, PulseTrain, download_dataset
from .models import Deinterleaver, evaluate_model_on_dataset
from .visualisation import (
    plot_data,
    plot_pdws,
    plot_pulse_train,
    plot_true_vs_predicted_features,
    scatter_features,
)

__all__ = (
    "Deinterleaver",
    "DeinterleavingChallengeDataset",
    "PulseTrain",
    "__version__",
    "download_dataset",
    "evaluate_model_on_dataset",
    "plot_data",
    "plot_pdws",
    "plot_pulse_train",
    "plot_true_vs_predicted_features",
    "scatter_features",
)
__version__ = version(__name__)
