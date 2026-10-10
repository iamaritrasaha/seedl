"""SeeDL: simple, deep-learning-aware visualizations."""

from ._training import Tracker
from .api import (
    activations,
    attention,
    compare,
    compare_predictions,
    convolution,
    dataset,
    embeddings,
    explain,
    gradients,
    learn,
    loss_landscape,
    predictions,
    see,
    track,
    watch,
)
from .exceptions import MissingDependencyError, SeeDLError, UnsupportedObjectError
from .view import Visualization

__all__ = [
    "MissingDependencyError",
    "SeeDLError",
    "Tracker",
    "UnsupportedObjectError",
    "Visualization",
    "activations",
    "attention",
    "compare",
    "compare_predictions",
    "convolution",
    "dataset",
    "embeddings",
    "explain",
    "gradients",
    "learn",
    "loss_landscape",
    "predictions",
    "see",
    "track",
    "watch",
]

__version__ = "0.1.0"
