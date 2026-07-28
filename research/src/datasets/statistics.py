"""
Dataset Statistics for Legal NLP Research Framework.
Computes and reports statistical properties of legal datasets.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DatasetStatistics:
    """
    Computes comprehensive statistics for a legal NLP dataset.
    Supports multi-class and multi-label tasks.
    """

    def __init__(self, data: dict, info: dict = None):
        """
        Args:
            data: dict with split names → list of {"text": str, "label": ...}
            info: optional metadata dict from DATASET_REGISTRY
        """
        self.data = data
        self.info = info or {}
        self._stats: Optional[dict] = None

    def compute_all(self) -> dict:
        """Compute all statistics and return as a dict."""
        stats = {
            "dataset_info": self.info,
            "splits": {},
            "global": {},
        }

        all_texts = []
        all_labels = []

        for split_name, rows in self.data.items():
            texts = [r["text"] for r in rows]
            labels = [r["label"] for r in rows]
            all_texts.extend(texts)
            all_labels.extend(labels)
            stats["splits"][split_name] = self._split_stats(texts, labels, split_name)

        # Global stats
        stats["global"] = self._split_stats(all_texts, all_labels, "all")
        self._stats = stats
        return stats

    def print_summary(self):
        """Print a nicely formatted statistics summary."""
        if self._stats is None:
            self.compute_all()

        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(
            f"\n[bold cyan]Dataset: {self.info.get('description', 'Unknown')}[/bold cyan]"
        )

        # Split sizes
        table = Table(title="Split Sizes", show_header=True)
        table.add_column("Split")
        table.add_column("Samples", justify="right")
        table.add_column("Classes / Labels", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Max Tokens", justify="right")

        for split_name, sp in self._stats["splits"].items():
            table.add_row(
                split_name,
                str(sp["n_samples"]),
                str(sp.get("n_classes", "?")),
                f"{sp['token_stats']['mean']:.0f}",
                str(sp["token_stats"]["max"]),
            )
        console.print(table)

        # Label distribution for train split
        if "train" in self._stats["splits"]:
            train_stats = self._stats["splits"]["train"]
            dist = train_stats.get("label_distribution", {})
            if dist:
                label_table = Table(title="Label Distribution (train)", show_header=True)
                label_table.add_column("Label")
                label_table.add_column("Count", justify="right")
                label_table.add_column("Fraction", justify="right")
                n = train_stats["n_samples"]
                for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1])[:20]:
                    label_table.add_row(str(lbl), str(cnt), f"{cnt/n:.3f}")
                console.print(label_table)

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _split_stats(self, texts: list, labels: list, split_name: str) -> dict:
        lengths_words = [len(t.split()) for t in texts]
        lengths_chars = [len(t) for t in texts]

        # Label analysis
        is_multilabel = labels and isinstance(labels[0], (list, tuple))
        if is_multilabel:
            flat = [lbl for row in labels for lbl in row]
            n_classes = len(set(flat))
            label_dist = dict(Counter(flat))
            avg_labels = float(np.mean([len(row) for row in labels]))
        else:
            n_classes = len(set(str(l) for l in labels))
            label_dist = dict(Counter(str(l) for l in labels))
            avg_labels = 1.0

        return {
            "split": split_name,
            "n_samples": len(texts),
            "n_classes": n_classes,
            "is_multilabel": is_multilabel,
            "avg_labels_per_sample": avg_labels,
            "label_distribution": label_dist,
            "class_imbalance_ratio": self._imbalance_ratio(label_dist),
            "word_stats": {
                "mean": float(np.mean(lengths_words)) if lengths_words else 0,
                "median": float(np.median(lengths_words)) if lengths_words else 0,
                "max": int(np.max(lengths_words)) if lengths_words else 0,
                "min": int(np.min(lengths_words)) if lengths_words else 0,
                "std": float(np.std(lengths_words)) if lengths_words else 0,
            },
            "char_stats": {
                "mean": float(np.mean(lengths_chars)) if lengths_chars else 0,
                "max": int(np.max(lengths_chars)) if lengths_chars else 0,
            },
            # Token stats use word count as approximation (≈ 1.3× tokens)
            "token_stats": {
                "mean": float(np.mean(lengths_words) * 1.3) if lengths_words else 0,
                "median": float(np.median(lengths_words) * 1.3) if lengths_words else 0,
                "max": int(np.max(lengths_words) * 1.3) if lengths_words else 0,
                "pct_over_512": float(np.mean([l * 1.3 > 512 for l in lengths_words]))
                if lengths_words
                else 0,
            },
        }

    @staticmethod
    def _imbalance_ratio(dist: dict) -> float:
        if not dist:
            return 0.0
        counts = list(dist.values())
        if max(counts) == 0:
            return 0.0
        return float(max(counts) / max(1, min(counts)))
