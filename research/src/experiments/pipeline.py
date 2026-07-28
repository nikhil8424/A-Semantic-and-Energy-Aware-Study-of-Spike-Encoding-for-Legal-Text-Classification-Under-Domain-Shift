"""
Experiment Pipeline for Legal NLP Research Framework.
Orchestrates the full end-to-end experimental workflow:
  Dataset → Embeddings → Spike Encoding → SNN Training →
  Evaluation → Visualization → Report
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """
    Full experiment pipeline for spike encoding vs transformer comparison.
    Each stage is independently runnable and results are cached.
    """

    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config.get("storage", {}).get("results", "storage/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        dataset_key: str = "case_hold",
        transformer_key: str = "legal_bert",
        encodings: Optional[list[str]] = None,
        skip_stages: Optional[list[str]] = None,
    ) -> dict:
        """
        Run the complete experiment pipeline.

        Args:
            dataset_key:     which dataset to use
            transformer_key: which transformer model to use
            encodings:       list of encoding names (None = all enabled in config)
            skip_stages:     list of stage names to skip

        Returns:
            dict with all experiment results
        """
        skip = set(skip_stages or [])
        if encodings is None:
            enc_cfg = self.config.get("encoding", {}).get("methods", {})
            encodings = [k for k, v in enc_cfg.items() if v.get("enabled", True)]

        logger.info("=" * 60)
        logger.info("SPIKE-LEGAL-NLP EXPERIMENT PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Dataset:     {dataset_key}")
        logger.info(f"Transformer: {transformer_key}")
        logger.info(f"Encodings:   {encodings}")
        logger.info("=" * 60)

        t0 = time.time()

        # ── Stage 1: Load dataset ─────────────────────────────────────
        if "dataset" not in skip:
            logger.info("\n[Stage 1/8] Loading dataset…")
            data, ds_info = self._stage_load_dataset(dataset_key)
            self._results["dataset_key"] = dataset_key
            self._results["dataset_info"] = ds_info
        else:
            logger.info("[Stage 1/8] Skipped (dataset)")
            data = self._load_cached_data(dataset_key)
            ds_info = {}

        # ── Stage 2: Compute embeddings ───────────────────────────────
        if "embeddings" not in skip:
            logger.info("\n[Stage 2/8] Extracting transformer embeddings…")
            embeddings = self._stage_embeddings(data, transformer_key, dataset_key)
            self._results["embeddings_shape"] = {
                k: v.shape for k, v in embeddings.items()
            }
        else:
            logger.info("[Stage 2/8] Skipped (embeddings)")
            embeddings = self._load_cached_embeddings(transformer_key, dataset_key)

        # ── Stage 3: Spike encoding ───────────────────────────────────
        if "encoding" not in skip:
            logger.info("\n[Stage 3/8] Generating spike trains…")
            spike_trains_all = self._stage_spike_encoding(embeddings, encodings)
        else:
            logger.info("[Stage 3/8] Skipped (encoding)")
            spike_trains_all = {}

        # ── Stage 4: Transformer baseline ────────────────────────────
        if "transformer_eval" not in skip:
            logger.info("\n[Stage 4/8] Evaluating transformer baseline…")
            transformer_result = self._stage_transformer_eval(
                data, embeddings, transformer_key, dataset_key
            )
            self._results["transformer"] = transformer_result
        else:
            logger.info("[Stage 4/8] Skipped (transformer_eval)")
            transformer_result = {}

        # ── Stage 5: SNN training ─────────────────────────────────────
        if "snn" not in skip:
            logger.info("\n[Stage 5/8] Training SNN classifiers…")
            snn_results = self._stage_snn_training(data, spike_trains_all, embeddings)
            self._results["snn"] = snn_results
        else:
            logger.info("[Stage 5/8] Skipped (snn)")
            snn_results = {}

        # Merge classification results
        clf_results = {}
        if transformer_result:
            clf_results[f"transformer_{transformer_key}"] = transformer_result
        for enc_name, res in snn_results.items():
            clf_results[f"snn_{enc_name}"] = res.get("final_val", res)

        # ── Stage 6: Semantic preservation ────────────────────────────
        if "semantic" not in skip and spike_trains_all and "train" in embeddings:
            logger.info("\n[Stage 6/8] Semantic preservation analysis…")
            semantic_results = self._stage_semantic(embeddings, spike_trains_all)
            self._results["semantic"] = semantic_results
        else:
            logger.info("[Stage 6/8] Skipped (semantic)")
            semantic_results = {}

        # ── Stage 7: Energy analysis ──────────────────────────────────
        if "energy" not in skip and spike_trains_all:
            logger.info("\n[Stage 7/8] Energy analysis…")
            energy_results = self._stage_energy(
                data, embeddings, spike_trains_all, transformer_key, snn_results
            )
            self._results["energy"] = energy_results
        else:
            logger.info("[Stage 7/8] Skipped (energy)")
            energy_results = {}

        # ── Stage 8: Visualization + Report ───────────────────────────
        logger.info("\n[Stage 8/8] Generating visualizations and report…")
        figure_paths = []
        if spike_trains_all:
            figure_paths.extend(
                self._stage_visualizations(
                    data, embeddings, spike_trains_all,
                    clf_results, semantic_results, energy_results,
                )
            )

        report_path = self._stage_report(
            dataset_key, transformer_key,
            clf_results, semantic_results, energy_results,
            None, figure_paths,
        )
        self._results["report"] = report_path

        elapsed = time.time() - t0
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(f"Report: {report_path}")
        logger.info(f"{'='*60}")

        # Save full results JSON
        out_path = self.results_dir / f"results_{dataset_key}_{transformer_key}.json"
        with open(out_path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)
        logger.info(f"Full results saved → {out_path}")

        return self._results

    # ─────────────────────────────────────────────────────────────────
    # Stage Implementations
    # ─────────────────────────────────────────────────────────────────

    def _stage_load_dataset(self, dataset_key: str) -> tuple:
        from ..datasets import DatasetManager, DatasetStatistics
        from ..datasets.manager import DATASET_REGISTRY

        dm = DatasetManager(self.config)
        data = dm.load(dataset_key)
        info = DATASET_REGISTRY.get(dataset_key, {})
        stats = DatasetStatistics(data, info)
        stats_dict = stats.compute_all()
        stats.print_summary()

        # Save stats
        stats_path = self.results_dir / f"dataset_stats_{dataset_key}.json"
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2, default=str)

        return data, stats_dict

    def _stage_embeddings(self, data: dict, transformer_key: str, dataset_key: str) -> dict:
        from ..models import TransformerBaseline

        model = TransformerBaseline(self.config, transformer_key)
        embeddings = {}
        for split_name, rows in data.items():
            if not rows:
                continue
            logger.info(f"  Extracting {split_name} embeddings ({len(rows)} samples)…")
            emb = model.get_embeddings(
                rows,
                batch_size=32,
                cache_key=f"{dataset_key}_{split_name}",
            )
            embeddings[split_name] = emb
        return embeddings

    def _stage_spike_encoding(self, embeddings: dict, encodings: list[str]) -> dict:
        from .. import encoding as enc_module

        time_steps = self.config.get("encoding", {}).get("time_steps", 50)
        enc_cfg = self.config.get("encoding", {}).get("methods", {})
        spike_trains_all = {}

        for enc_name in encodings:
            if enc_name not in enc_module.ENCODERS:
                logger.warning(f"Unknown encoder '{enc_name}', skipping")
                continue
            enc_params = enc_cfg.get(enc_name, {})
            encoder = enc_module.ENCODERS[enc_name](time_steps=time_steps, **enc_params)
            spike_trains_all[enc_name] = {}
            for split_name, emb in embeddings.items():
                logger.info(f"  [{enc_name}] Encoding {split_name} split ({len(emb)} samples)…")
                spk = encoder.encode(emb)
                spike_trains_all[enc_name][split_name] = spk
                logger.info(
                    f"    -> shape={spk.shape}, sparsity={encoder.sparsity(spk):.2%}"
                )
        return spike_trains_all

    def _stage_transformer_eval(
        self, data: dict, embeddings: dict, transformer_key: str, dataset_key: str
    ) -> dict:
        from ..models import TransformerBaseline
        from ..evaluation import ClassificationMetrics

        model = TransformerBaseline(self.config, transformer_key)
        train_rows = data.get("train", [])
        val_rows = data.get("validation", data.get("test", []))

        if not train_rows or not val_rows:
            logger.warning("Insufficient data for transformer eval")
            return {}

        result = model.train_linear_probe(
            train_rows, val_rows, cache_key=f"{dataset_key}"
        )
        return result

    def _stage_snn_training(
        self, data: dict, spike_trains_all: dict, embeddings: dict
    ) -> dict:
        from ..models import SNNClassifier

        snn_results = {}
        train_rows = data.get("train", [])
        val_rows = data.get("validation", data.get("test", []))

        # Encode labels as integers
        all_labels = [r["label"] for r in train_rows + val_rows]
        if isinstance(all_labels[0], list):
            # Multi-label: convert to int via argmax of label list
            all_labels_flat = [tuple(sorted(l)) for l in all_labels]
            unique = sorted(set(all_labels_flat))
            label_map = {l: i for i, l in enumerate(unique)}
            y_train = np.array([label_map[tuple(sorted(r["label"]))] for r in train_rows])
            y_val = np.array([label_map.get(tuple(sorted(r["label"])), 0) for r in val_rows])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_all = le.fit_transform([str(r["label"]) for r in train_rows + val_rows])
            y_train = y_all[: len(train_rows)]
            y_val = y_all[len(train_rows) :]

        for enc_name, splits in spike_trains_all.items():
            spk_train = splits.get("train")
            spk_val = splits.get("validation", splits.get("test"))

            if spk_train is None or spk_val is None:
                continue

            # Match label count with available samples
            n_train = min(len(y_train), len(spk_train))
            n_val = min(len(y_val), len(spk_val))

            logger.info(f"  Training SNN [{enc_name}] n_train={n_train}…")
            clf = SNNClassifier(self.config, encoding_name=enc_name)
            result = clf.train(
                spk_train[:n_train],
                y_train[:n_train],
                spk_val[:n_val],
                y_val[:n_val],
            )
            snn_results[enc_name] = result

        return snn_results

    def _stage_semantic(self, embeddings: dict, spike_trains_all: dict) -> dict:
        from ..evaluation import SemanticPreservation

        sem = SemanticPreservation(self.config)
        original_emb = embeddings.get("train") if "train" in embeddings else embeddings.get("test")
        if original_emb is None:
            return {}

        spike_by_enc = {
            enc_name: splits.get("train", splits.get("test"))
            for enc_name, splits in spike_trains_all.items()
            if splits.get("train") is not None or splits.get("test") is not None
        }
        return sem.compare_encodings(original_emb, spike_by_enc)

    def _stage_energy(
        self, data, embeddings, spike_trains_all, transformer_key, snn_results
    ) -> dict:
        from ..models import TransformerBaseline, SNNClassifier
        from ..evaluation import EnergyAnalyzer

        analyzer = EnergyAnalyzer(self.config)
        model = TransformerBaseline(self.config, transformer_key)
        model.load(mode="embedding")

        max_len = self.config.get("preprocessing", {}).get("max_length", 512)
        n_mac = model.count_mac_operations(seq_length=max_len)
        n_params = sum(p.numel() for p in model.model.parameters())
        transformer_energy = analyzer.estimate_transformer_energy(n_mac, n_params, max_len)

        # SNN params estimate
        snn_hidden = self.config.get("snn", {}).get("architecture", {}).get("hidden_size", 256)
        first_emb = list(embeddings.values())[0] if embeddings else None
        input_dim = first_emb.shape[1] if first_emb is not None else 768
        n_layers = self.config.get("snn", {}).get("architecture", {}).get("num_hidden_layers", 2)
        snn_params = input_dim * snn_hidden + (n_layers - 1) * snn_hidden ** 2

        # Collect SOPs
        sop_counts = {}
        spk_by_enc = {}
        for enc_name, splits in spike_trains_all.items():
            spk = splits.get("train") if "train" in splits else splits.get("test")
            if spk is None:
                continue
            spk_by_enc[enc_name] = spk
            clf = SNNClassifier(self.config, enc_name)
            train_labels = [r["label"] for r in data.get("train", [])]
            # Handle multi-label lists
            if train_labels and isinstance(train_labels[0], list):
                flat_labels = [item for sublist in train_labels for item in sublist]
                max_label = max(flat_labels) if flat_labels else 1
            else:
                max_label = max(train_labels) if train_labels else 1
            clf.num_classes = max(2, int(max_label) + 1)
            sop_info = clf.count_synaptic_operations(spk[:200])
            sop_counts[enc_name] = sop_info["avg_sops_per_sample"]

        return analyzer.analyze_all_encodings(
            spk_by_enc, sop_counts, transformer_energy, snn_params, include_memory=True
        )

    def _stage_visualizations(
        self, data, embeddings, spike_trains_all, clf_results, semantic_results, energy_results
    ) -> list[str]:
        from ..visualization import ResearchPlotter

        plotter = ResearchPlotter(self.config)
        paths = []

        # Classification comparison
        if clf_results:
            p = plotter.plot_classification_comparison(clf_results)
            paths.append(p)

        # Spike raster plots
        spk_for_raster = {
            enc: splits.get("train", splits.get("test"))
            for enc, splits in spike_trains_all.items()
            if splits.get("train") is not None or splits.get("test") is not None
        }
        if spk_for_raster:
            p = plotter.plot_encoding_comparison_rasters(spk_for_raster)
            paths.append(p)
            p = plotter.plot_firing_rates(spk_for_raster)
            paths.append(p)

        # Semantic preservation
        if semantic_results:
            p = plotter.plot_semantic_preservation(semantic_results)
            paths.append(p)

        # Embedding scatter
        if embeddings and spk_for_raster:
            emb = embeddings.get("train") if "train" in embeddings else list(embeddings.values())[0]
            train_rows = data.get("train", [])
            if train_rows and len(train_rows) > 0:
                raw_labels = [r["label"] for r in train_rows]
                if isinstance(raw_labels[0], list):
                    labels = np.array([len(l) for l in raw_labels])
                else:
                    labels = np.array(raw_labels)
                labels = labels[:len(emb)]
                p = plotter.plot_embedding_scatter(emb, spk_for_raster, labels)
                paths.append(p)

        # Energy analysis
        if energy_results:
            p = plotter.plot_energy_comparison(energy_results)
            paths.append(p)
            if semantic_results:
                p = plotter.plot_energy_sparsity_tradeoff(energy_results, semantic_results)
                paths.append(p)

        # Summary dashboard
        if clf_results and semantic_results and energy_results:
            p = plotter.plot_summary_dashboard(clf_results, semantic_results, energy_results)
            paths.append(p)

        return [p for p in paths if p]

    def _stage_report(
        self, dataset_key, transformer_key, clf_results, semantic_results,
        energy_results, shift_results, figure_paths
    ) -> str:
        from ..reporting import ReportGenerator

        gen = ReportGenerator(self.config)
        title = (
            "A Semantic- and Energy-Aware Study of Spike Encoding "
            "for Legal Text Classification Under Domain Shift"
        )
        return gen.generate(
            title=title,
            experiment_config=self.config,
            classification_results=clf_results,
            semantic_results=semantic_results,
            energy_results=energy_results,
            domain_shift_results=shift_results if shift_results else None,
            figure_paths=figure_paths,
            filename=f"report_{dataset_key}_{transformer_key}",
        )

    # ─────────────────────────────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_cached_data(self, dataset_key: str) -> dict:
        from ..datasets import DatasetManager
        dm = DatasetManager(self.config)
        return dm.load(dataset_key)

    def _load_cached_embeddings(self, transformer_key: str, dataset_key: str) -> dict:
        import pickle
        emb_dir = Path(self.config.get("storage", {}).get("embeddings_cache", "storage/embeddings"))
        result = {}
        for split in ["train", "validation", "test"]:
            p = emb_dir / f"{transformer_key}_{dataset_key}_{split}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    result[split] = pickle.load(f)
        return result
