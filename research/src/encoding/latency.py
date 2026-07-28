"""
Latency Coding.

High-activation features fire early; low-activation features fire late
or not at all. Encodes information in the *timing* of the first spike.

Reference:
    Thorpe, S., Delorme, A., & Van Rullen, R. (2001). Spike-based
    strategies for rapid processing. Neural Networks.
"""

import numpy as np
from .base import BaseSpikeEncoder


class LatencyEncoder(BaseSpikeEncoder):
    """
    Latency (Time-to-First-Spike) Coding.

    The first spike time t_i for feature i is:
        t_i = T · exp(-τ · v_i)   for v_i ∈ [0, 1]

    A high value fires at t ≈ 0; a near-zero value fires near t ≈ T.
    Values below a threshold produce no spike.
    """

    def __init__(
        self,
        time_steps: int = 50,
        tau: float = 5.0,
        normalize: bool = True,
        threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "latency"
        self.tau = tau
        self.normalize = normalize
        self.threshold = threshold
        self._params.update({"tau": tau, "normalize": normalize, "threshold": threshold})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features)  — at most one spike per feature
        """
        x = self._normalize(embeddings.astype(np.float32))

        batch, features = x.shape
        spikes = np.zeros((batch, self.time_steps, features), dtype=np.float32)

        # Compute first spike time: t = T * exp(-tau * v)
        # High v → small t (early fire); low v → large t (late fire)
        spike_times = (self.time_steps * np.exp(-self.tau * x)).astype(int)
        spike_times = np.clip(spike_times, 0, self.time_steps - 1)

        # Mask features below threshold (they produce no spike)
        no_spike_mask = x < self.threshold

        # Vectorized: use advanced indexing to set spikes
        valid_mask = ~no_spike_mask
        batch_indices, feature_indices = np.where(valid_mask)
        time_indices = spike_times[batch_indices, feature_indices]
        spikes[batch_indices, time_indices, feature_indices] = 1.0

        return spikes
