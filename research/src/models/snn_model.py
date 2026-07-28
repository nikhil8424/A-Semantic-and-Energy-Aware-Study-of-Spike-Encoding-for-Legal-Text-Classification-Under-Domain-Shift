"""
Spiking Neural Network Classifier.
Built with snntorch and PyTorch.
Accepts spike-encoded inputs and produces class predictions.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False
    logger = logging.getLogger(__name__)
    logger.warning("snntorch not available. SNN will use a PyTorch fallback.")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# snntorch SNN model
# ─────────────────────────────────────────────────────────────────────

class SNNTorchModel(nn.Module):
    """
    Fully-connected SNN using Leaky Integrate-and-Fire neurons (snntorch).
    Processes spike trains over `time_steps` and uses rate-coded output.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        dropout: float = 0.3,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers

        # Build fully-connected layers
        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        in_dim = input_size
        for _ in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(in_dim, hidden_size))
            self.lif_layers.append(
                snn.Leaky(beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid())
            )
            in_dim = hidden_size

        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (time_steps, batch, input_size) spike trains

        Returns:
            spike_out: (time_steps, batch, num_classes)
            mem_out:   (batch, num_classes)  — membrane potential at last step
        """
        time_steps = x.shape[0]

        # Initialize membrane potentials
        mems = [lif.init_leaky() for lif in self.lif_layers]
        mem_out = self.lif_out.init_leaky()

        spike_rec = []
        for t in range(time_steps):
            xt = x[t]  # (batch, input_size)
            spk = xt
            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_layers)):
                cur = fc(self.dropout(spk))
                spk, mems[i] = lif(cur, mems[i])

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spike_rec.append(spk_out)

        spike_out = torch.stack(spike_rec, dim=0)  # (T, batch, num_classes)
        return spike_out, mem_out


# ─────────────────────────────────────────────────────────────────────
# Pure-PyTorch fallback (no snntorch required)
# ─────────────────────────────────────────────────────────────────────

class SimpleSNNFallback(nn.Module):
    """
    Simplified SNN fallback without snntorch.
    Integrates spike trains across time, then applies a linear classifier.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: (time_steps, batch, features) → integrate over time
        rate = x.mean(dim=0)  # (batch, features)
        return self.net(rate), None


# ─────────────────────────────────────────────────────────────────────
# High-level classifier wrapper
# ─────────────────────────────────────────────────────────────────────

class SNNClassifier:
    """
    High-level SNN classifier wrapper.
    Accepts spike trains (numpy arrays) and returns classification metrics.
    """

    def __init__(self, config: dict, encoding_name: str = "poisson_rate"):
        self.config = config
        self.encoding_name = encoding_name
        snn_cfg = config.get("snn", {})
        arch = snn_cfg.get("architecture", {})
        self.hidden_size = arch.get("hidden_size", 256)
        self.num_hidden_layers = arch.get("num_hidden_layers", 2)
        self.dropout = arch.get("dropout", 0.3)
        neuron_cfg = snn_cfg.get("neuron", {})
        self.beta = neuron_cfg.get("beta", 0.9)
        self.threshold = neuron_cfg.get("threshold", 1.0)
        train_cfg = snn_cfg.get("training", {})
        self.lr = train_cfg.get("learning_rate", 1e-3)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.time_steps = config.get("encoding", {}).get("time_steps", 50)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.num_classes: int = 2

    def build_model(self, input_size: int, num_classes: int):
        """Instantiate the SNN architecture."""
        self.num_classes = num_classes
        if HAS_SNNTORCH:
            self.model = SNNTorchModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                beta=self.beta,
                threshold=self.threshold,
                dropout=self.dropout,
                num_hidden_layers=self.num_hidden_layers,
            ).to(self.device)
        else:
            logger.warning("Using SimpleSNNFallback (snntorch not installed)")
            self.model = SimpleSNNFallback(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                dropout=self.dropout,
            ).to(self.device)

    def train(
        self,
        spike_trains: np.ndarray,
        labels: np.ndarray,
        val_spikes: np.ndarray,
        val_labels: np.ndarray,
    ) -> dict:
        """
        Train the SNN classifier.

        Args:
            spike_trains: (n_train, time_steps, features)
            labels:       (n_train,)
            val_spikes:   (n_val, time_steps, features)
            val_labels:   (n_val,)

        Returns:
            dict with training history and final validation metrics
        """
        n_train, T, features = spike_trains.shape
        num_classes = int(labels.max()) + 1
        self.build_model(features, num_classes)

        # Tensors: transpose to (time_steps, batch, features) for SNN
        X_train = torch.tensor(spike_trains, dtype=torch.float32)  # (N, T, F)
        y_train = torch.tensor(labels, dtype=torch.long)
        X_val = torch.tensor(val_spikes, dtype=torch.float32)
        y_val = torch.tensor(val_labels, dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        history = []
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, correct = 0.0, 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)  # (batch, T, F)
                y_batch = y_batch.to(self.device)

                # Transpose: (T, batch, F)
                x_t = X_batch.permute(1, 0, 2)
                spike_out, mem_out = self.model(x_t)

                if HAS_SNNTORCH:
                    # Rate-coded output: sum spikes over time
                    logits = spike_out.sum(dim=0)  # (batch, num_classes)
                else:
                    logits = spike_out

                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(1) == y_batch).sum().item()

            train_acc = correct / len(train_ds)
            val_metrics = self.evaluate(X_val, y_val)
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                "train_acc": train_acc,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_stats)
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} "
                f"loss={epoch_stats['train_loss']:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_metrics.get('accuracy', 0):.4f}"
            )

        return {
            "history": history,
            "encoding": self.encoding_name,
            "model": "snn",
            "final_val": self.evaluate(X_val, y_val),
        }

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Run inference and return metrics."""
        self.model.eval()
        all_preds = []
        loader = DataLoader(TensorDataset(X, y), batch_size=64)

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                x_t = X_batch.permute(1, 0, 2)
                spike_out, _ = self.model(x_t)
                if HAS_SNNTORCH:
                    logits = spike_out.sum(dim=0)
                else:
                    logits = spike_out
                all_preds.extend(logits.argmax(1).cpu().numpy())

        y_np = y.numpy()
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(y_np, all_preds)),
            "f1_macro": float(f1_score(y_np, all_preds, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_np, all_preds, average="micro", zero_division=0)),
        }

    def count_synaptic_operations(
        self, spike_trains: np.ndarray
    ) -> dict:
        """
        Count Synaptic Operations (SOPs) for energy analysis.
        SOPs = sum of spikes × fan-out (number of downstream connections).
        """
        n, T, features = spike_trains.shape
        snn_cfg = self.config.get("snn", {})
        arch = snn_cfg.get("architecture", {})
        hidden = arch.get("hidden_size", 256)
        n_layers = arch.get("num_hidden_layers", 2)

        total_spikes = float(spike_trains.sum())
        avg_spikes_per_sample = total_spikes / n

        # Layer sizes: input → hidden → ... → hidden → output
        layer_sizes = [features] + [hidden] * n_layers + [self.num_classes]
        total_sops_per_sample = 0
        
        # Compute actual input firing rate (spikes per time step per feature)
        input_firing_rate = avg_spikes_per_sample / (spike_trains.shape[1] * spike_trains.shape[2])
        
        for i in range(len(layer_sizes) - 1):
            # SOPs = spikes_at_layer_i × fan_out
            # For input layer: spikes = avg_spikes_per_sample
            # For hidden layers: estimate based on input firing rate
            spikes_at_layer = avg_spikes_per_sample if i == 0 else avg_spikes_per_sample * input_firing_rate
            total_sops_per_sample += spikes_at_layer * layer_sizes[i + 1]

        return {
            "total_spikes": total_spikes,
            "avg_spikes_per_sample": avg_spikes_per_sample,
            "avg_sops_per_sample": float(total_sops_per_sample),
            "sparsity": float(1.0 - spike_trains.mean()),
        }
