"""SeeDL: simple, deep-learning-aware visualizations."""

from ._recording import TrainingSnapshot
from ._training import Tracker
from .api import (
    activations,
    attention,
    bloch,
    circuit,
    compare,
    compare_predictions,
    convolution,
    dataset,
    density_matrix,
    embeddings,
    entanglement,
    explain,
    feature_maps,
    features,
    gradients,
    learn,
    loss_landscape,
    predictions,
    see,
    statevector,
    track,
    vit_attention,
    watch,
)
from .exceptions import MissingDependencyError, SeeDLError, UnsupportedObjectError
from .view import Visualization

__all__ = [
    "MissingDependencyError",
    "SeeDLError",
    "Tracker",
    "TrainingSnapshot",
    "UnsupportedObjectError",
    "Visualization",
    "activations",
    "attention",
    "bloch",
    "circuit",
    "compare",
    "compare_predictions",
    "convolution",
    "dataset",
    "density_matrix",
    "embeddings",
    "entanglement",
    "explain",
    "feature_maps",
    "features",
    "gradients",
    "learn",
    "loss_landscape",
    "predictions",
    "see",
    "statevector",
    "track",
    "vit_attention",
    "watch",
]

__version__ = "0.3.0"
