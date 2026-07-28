"""
Temporal Contrast Coding.

Converts embedding values into sequences where spikes mark
significant changes (transitions) across quantization levels.

Reference:
    Auge, D. et al. (2021). A survey of encoding techniques for
    signal processing in spiking neural networks. Neural Processing Letters.
"""

import numpy as np
from .base import BaseSpikeEncoder


class TemporalEncoder(BaseSpikeEncoder):
    """
    Temporal Contrast (Step-Forward) Coding.

    The embedding is quantized into n_levels discrete bins.
    At each time step the encoder advances through the quantization
    ladder; a spike is emitted whenever the current level changes
    relative to the previous step.

    For static embeddings (no temporal dimension), we simulate a
    temporal sequence by sweeping threshold levels over time steps:
        s_{t,i} = 1  if  ⌊v_i · n_levels⌋ == t   (for t < n_levels)
    i.e. feature i fires exactly once at the time step corresponding
    to its quantization bin.
    """

    def __init__(
        self,
        time_steps: int = 50,
        n_levels: int = 10,
        **kwargs,
    ):
        super().__init__(time_steps=time_steps, **kwargs)
        self.name = "temporal"
        self.n_levels = n_levels
        self._params.update({"n_levels": n_levels})

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: (batch, features)

        Returns:
            spikes: (batch, time_steps, features)
        """
        x = self._normalize(embeddings.astype(np.float32))  # [0, 1]

        batch, features = x.shape
        spikes = np.zeros((batch, self.time_steps, features), dtype=np.float32)

        # Map value → bin index in [0, n_levels-1]
        bins = np.floor(x * self.n_levels).astype(int)
        bins = np.clip(bins, 0, self.n_levels - 1)

        # Map bin index → time step (stretch n_levels onto time_steps)
        time_indices = (bins * self.time_steps // self.n_levels).astype(int)
        time_indices = np.clip(time_indices, 0, self.time_steps - 1)

        for b in range(batch):
            for f in range(features):
                spikes[b, time_indices[b, f], f] = 1.0

        return spikes
