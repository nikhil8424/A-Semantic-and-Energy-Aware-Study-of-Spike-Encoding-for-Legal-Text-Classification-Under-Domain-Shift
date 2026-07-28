"""
Base class for spike encoders.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseSpikeEncoder(ABC):
    """
    Abstract base class for all spike encoding methods.
    All encoders receive dense continuous-valued embeddings and return
    binary spike trains of shape (time_steps, n_features).
    """

    def __init__(self, time_steps: int = 50, **kwargs):
        self.time_steps = time_steps
        self.name = "base"
        self._params: dict = {"time_steps": time_steps}

    @abstractmethod
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a batch of embeddings into spike trains.

        Args:
            embeddings: ndarray of shape (batch_size, embedding_dim)

        Returns:
            spike_trains: ndarray of shape (batch_size, time_steps, embedding_dim)
                         dtype float32, values in {0, 1}
        """
        ...

    def encode_single(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a single embedding vector."""
        return self.encode(embedding[np.newaxis])[0]

    def get_params(self) -> dict:
        """Return encoder hyperparameters."""
        return dict(self._params)

    def firing_rate(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Compute average firing rate per feature dimension.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            rates: (batch, features) — mean spikes per time step
        """
        return spike_trains.mean(axis=1)

    def spike_count(self, spike_trains: np.ndarray) -> np.ndarray:
        """
        Count total spikes per sample.

        Args:
            spike_trains: (batch, time_steps, features)

        Returns:
            counts: (batch,)
        """
        return spike_trains.sum(axis=(1, 2))

    def sparsity(self, spike_trains: np.ndarray) -> float:
        """
        Fraction of zero entries in spike trains (higher = more sparse).
        """
        return float(1.0 - spike_trains.mean())

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1] along each sample."""
        mins = x.min(axis=-1, keepdims=True)
        maxs = x.max(axis=-1, keepdims=True)
        denom = maxs - mins
        denom = np.where(denom == 0, 1.0, denom)
        return (x - mins) / denom
