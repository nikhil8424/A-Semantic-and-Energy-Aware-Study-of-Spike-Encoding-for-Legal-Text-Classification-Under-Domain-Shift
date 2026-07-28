"""
Domain Shift Evaluation.
Measures how models trained on a source legal domain generalize
to target legal domains.
Addresses Research Question 3 and Hypothesis H4.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class DomainShiftEvaluator:
    """
    Evaluates model robustness under domain shift by testing models
    trained on a source dataset on different target datasets.
    """

    def __init__(self, config: dict):
        self.config = config
        ds_cfg = config.get("evaluation", {}).get("domain_shift", {})
        self.source_dataset = ds_cfg.get("source_dataset", "case_hold")
        self.target_datasets = ds_cfg.get("target_datasets", ["ecthr_a", "ecthr_b"])

    def evaluate_transfer(
        self,
        model,
        source_name: str,
        target_name: str,
        source_embeddings: np.ndarray,
        source_labels: np.ndarray,
        target_embeddings: np.ndarray,
        target_labels: np.ndarray,
        model_type: str = "transformer",
    ) -> dict:
        """
        Evaluate zero-shot transfer from source to target domain.
        Trains on source, evaluates on target.

        Returns:
            dict with source performance, target performance, and drop metrics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        logger.info(f"Domain shift: {source_name} → {target_name} [{model_type}]")

        scaler = StandardScaler()
        X_src = scaler.fit_transform(source_embeddings)

        le = LabelEncoder()
        y_src = le.fit_transform([str(l) for l in source_labels])

        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_src, y_src)

        # Source eval
        y_src_pred = clf.predict(X_src)
        src_acc = float(accuracy_score(y_src, y_src_pred))
        src_f1 = float(f1_score(y_src, y_src_pred, average="macro", zero_division=0))

        # Target eval: align label space
        X_tgt = scaler.transform(target_embeddings)
        y_tgt_raw = [str(l) for l in target_labels]

        # Handle unseen labels
        known = set(le.classes_)
        y_tgt_mapped = [y if y in known else le.classes_[0] for y in y_tgt_raw]
        y_tgt = le.transform(y_tgt_mapped)

        y_tgt_pred = clf.predict(X_tgt)
        tgt_acc = float(accuracy_score(y_tgt, y_tgt_pred))
        tgt_f1 = float(f1_score(y_tgt, y_tgt_pred, average="macro", zero_division=0))

        # H-score (Harmonic mean of source and target performance)
        h_score = (
            2 * src_acc * tgt_acc / (src_acc + tgt_acc)
            if (src_acc + tgt_acc) > 0
            else 0.0
        )

        result = {
            "source": source_name,
            "target": target_name,
            "model_type": model_type,
            "source_accuracy": src_acc,
            "source_f1_macro": src_f1,
            "target_accuracy": tgt_acc,
            "target_f1_macro": tgt_f1,
            "accuracy_drop": float(src_acc - tgt_acc),
            "f1_drop": float(src_f1 - tgt_f1),
            "relative_drop_pct": float((src_acc - tgt_acc) / max(src_acc, 1e-9) * 100),
            "h_score": h_score,
        }
        logger.info(
            f"  Source acc={src_acc:.4f}, Target acc={tgt_acc:.4f}, "
            f"Drop={result['accuracy_drop']:.4f}"
        )
        return result

    def compare_models_on_domain_shift(
        self,
        results: list[dict],
    ) -> dict:
        """
        Aggregate domain shift results across models and compute rankings.

        Args:
            results: list of result dicts from evaluate_transfer()

        Returns:
            summary dict
        """
        by_model = {}
        for r in results:
            key = f"{r['model_type']}"
            if key not in by_model:
                by_model[key] = []
            by_model[key].append(r)

        summary = {}
        for model_key, model_results in by_model.items():
            avg_drop = np.mean([r["accuracy_drop"] for r in model_results])
            avg_target = np.mean([r["target_accuracy"] for r in model_results])
            avg_h = np.mean([r["h_score"] for r in model_results])
            summary[model_key] = {
                "avg_accuracy_drop": float(avg_drop),
                "avg_target_accuracy": float(avg_target),
                "avg_h_score": float(avg_h),
                "n_transfers": len(model_results),
                "per_transfer": model_results,
            }

        # Rank by robustness (lowest accuracy drop)
        ranked = sorted(summary.items(), key=lambda x: x[1]["avg_accuracy_drop"])
        summary["_ranking_by_robustness"] = [k for k, _ in ranked]

        return summary

    def analyze_embedding_shift(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_name: str,
        target_name: str,
    ) -> dict:
        """
        Measure distributional shift between source and target embeddings
        using Maximum Mean Discrepancy (MMD) and centroid distance.
        """
        from sklearn.metrics.pairwise import rbf_kernel

        n = min(200, len(source_embeddings), len(target_embeddings))
        src = source_embeddings[:n]
        tgt = target_embeddings[:n]

        # Centroid shift
        src_centroid = src.mean(axis=0)
        tgt_centroid = tgt.mean(axis=0)
        centroid_dist = float(np.linalg.norm(src_centroid - tgt_centroid))

        # Cosine distance between centroids
        cos_sim = float(
            np.dot(src_centroid, tgt_centroid)
            / (np.linalg.norm(src_centroid) * np.linalg.norm(tgt_centroid) + 1e-9)
        )

        # Simplified MMD (Gaussian kernel)
        try:
            gamma = 1.0 / src.shape[1]
            K_ss = rbf_kernel(src, src, gamma=gamma).mean()
            K_tt = rbf_kernel(tgt, tgt, gamma=gamma).mean()
            K_st = rbf_kernel(src, tgt, gamma=gamma).mean()
            mmd = float(K_ss + K_tt - 2 * K_st)
        except Exception:
            mmd = None

        # Variance ratio
        src_var = float(src.var(axis=0).mean())
        tgt_var = float(tgt.var(axis=0).mean())

        return {
            "source": source_name,
            "target": target_name,
            "centroid_distance": centroid_dist,
            "centroid_cosine_similarity": cos_sim,
            "mmd": mmd,
            "source_variance": src_var,
            "target_variance": tgt_var,
            "variance_ratio": float(tgt_var / max(src_var, 1e-9)),
        }
