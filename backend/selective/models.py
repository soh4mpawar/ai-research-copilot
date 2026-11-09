"""
Selective Answering Models & Baselines.

Design & Protocol:
1. 2x2 Factorial Logistic Design:
   - M1_Relevance_Raw: Relevance features, raw passages.
   - M2_Relevance_Dedup: Relevance features, deduplicated passages.
   - M3_Full_Raw: Relevance + paper/section structural features, raw passages.
   - M4_Full_Dedup: Relevance + paper/section structural features, deduplicated passages.
2. Identical Preprocessing & Tuning:
   - StandardScaler fit strictly on training features.
   - L2 regularization hyperparameter C selected by validation log-loss with deterministic tie-breaking
     (preferring smaller C / stronger regularization on tie).
   - Outputs treated as estimated ranking scores, not established calibrated probabilities.
3. Deployable Operating Points:
   - Decision thresholds selected on validation data targeting coverage levels 0.50, 0.75, and 0.90.
   - Score ties respected; validation and test deviations reported. Never adjusted on test labels.
4. Historical Gate Baseline (B2):
   - Preserves original heuristic 4-condition decisions, including missing-ID handling.
   - Evaluated as a discrete binary decision baseline (single coverage-risk point).
"""

from typing import Sequence, Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from backend.selective.context import FinalContextRecord
from backend.selective.features import extract_features_from_record


class AlwaysAcceptBaseline:
    """Baseline B0: Unconditional acceptance (Coverage = 1.0)."""

    def predict_score(self, record: FinalContextRecord) -> float:
        return 1.0

    def decide(self, record: FinalContextRecord, threshold: float = 0.5) -> int:
        return 1


class ValidationTunedMaxThreshold:
    """Baseline B1: Validation-tuned single relevance score threshold on top passage."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.operating_thresholds: Dict[float, float] = {0.50: 0.5, 0.75: 0.5, 0.90: 0.5}

    def predict_score(self, record: FinalContextRecord) -> float:
        if not record.passages:
            return 0.0
        return float(max(p.score for p in record.passages))

    def decide(self, record: FinalContextRecord, threshold: Optional[float] = None) -> int:
        th = self.threshold if threshold is None else threshold
        return int(self.predict_score(record) >= th)

    def fit_validation_thresholds(
        self,
        val_records: Sequence[FinalContextRecord],
        val_labels: Sequence[int],
        target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
    ) -> Dict[float, Dict[str, Any]]:
        """Fit operating thresholds on validation records targeting coverage levels."""
        if not val_records:
            return {}

        val_scores = np.array([self.predict_score(r) for r in val_records])
        n_val = len(val_scores)
        sorted_scores = np.sort(val_scores)[::-1]

        res = {}
        for cov in target_coverages:
            k = max(1, min(n_val, int(round(cov * n_val))))
            th = float(sorted_scores[k - 1])
            achieved = float(np.mean(val_scores >= th))
            self.operating_thresholds[cov] = th
            res[cov] = {
                "threshold": round(th, 4),
                "target_coverage": cov,
                "val_achieved_coverage": round(achieved, 4),
                "val_deviation": round(abs(achieved - cov), 4)
            }
        return res


class FrozenHistoricalGateBaseline:
    """
    Baseline B2: Frozen historical 4-condition heuristic gate.
    Evaluated as a discrete binary decision rule {0, 1}.
    Preserves historical condition logic and missing paper ID handling.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        secondary_threshold: float = 0.15,
        deep_focus_threshold: float = 0.85,
        min_deep_focus_candidates: int = 3
    ):
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.deep_focus_threshold = deep_focus_threshold
        self.min_deep_focus_candidates = min_deep_focus_candidates

    def decide(self, record: FinalContextRecord) -> int:
        """Evaluates frozen 4-condition gate on the common final context."""
        if not record.passages:
            return 0

        scores = [p.score for p in record.passages]
        top_score = max(scores) if scores else 0.0

        # Group scores by paper ID (missing paper IDs treated as distinct unknown sources)
        paper_scores: Dict[str, List[float]] = {}
        for idx, p in enumerate(record.passages):
            pid = p.paper_id if p.paper_id else f"__unknown_{idx}__"
            if pid not in paper_scores:
                paper_scores[pid] = []
            paper_scores[pid].append(p.score)

        # Condition A: Same-Paper Reinforcement
        same_paper_passed = False
        for pid, sc_list in paper_scores.items():
            if not pid.startswith("__unknown_") and len(sc_list) >= 2:
                sorted_sc = sorted(sc_list, reverse=True)
                if sorted_sc[0] >= self.threshold and sorted_sc[1] >= self.secondary_threshold:
                    same_paper_passed = True
                    break

        # Condition B: Same-Paper Deep Topic Focus (Subsumed by Condition C)
        deep_focus_passed = False
        if not same_paper_passed:
            for pid, sc_list in paper_scores.items():
                if len(sc_list) >= self.min_deep_focus_candidates and max(sc_list) >= self.deep_focus_threshold:
                    deep_focus_passed = True
                    break

        # Condition C: High-Confidence Single-Chunk Override
        high_confidence_override = False
        if not same_paper_passed and not deep_focus_passed:
            if top_score >= self.deep_focus_threshold:
                high_confidence_override = True

        # Condition D: Multi-Paper Cross-Source Consensus
        multi_paper_passed = False
        if not same_paper_passed and not deep_focus_passed and not high_confidence_override:
            qualifying = [pid for pid, scs in paper_scores.items() if max(scs) >= self.threshold]
            if len(qualifying) >= 2:
                multi_paper_passed = True

        return int(same_paper_passed or deep_focus_passed or high_confidence_override or multi_paper_passed)


class LogisticGatingModel:
    """
    L2-Regularized Logistic Regression Gating Model for the 2x2 Experimental Design.

    Cells:
    - M1_Relevance_Raw: feature_mode="relevance_only", deduplicate=False
    - M2_Relevance_Dedup: feature_mode="relevance_only", deduplicate=True
    - M3_Full_Raw: feature_mode="combined", deduplicate=False
    - M4_Full_Dedup: feature_mode="combined", deduplicate=True
    """

    def __init__(
        self,
        name: str = "LogisticGatingModel",
        feature_mode: str = "combined",
        deduplicate: bool = True,
        C: float = 1.0,
        decision_threshold: float = 0.5
    ):
        self.name = name
        self.feature_mode = feature_mode
        self.deduplicate = deduplicate
        self.C = C
        self.decision_threshold = decision_threshold
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty="l2", C=self.C, solver="liblinear", random_state=42)
        self.is_fitted = False
        self.operating_thresholds: Dict[float, float] = {0.50: 0.5, 0.75: 0.5, 0.90: 0.5}

    def extract_features(self, records: Sequence[FinalContextRecord]) -> np.ndarray:
        feats = [
            extract_features_from_record(r, feature_mode=self.feature_mode, deduplicate=self.deduplicate)
            for r in records
        ]
        return np.array(feats, dtype=np.float64)

    def fit(
        self,
        train_records: Sequence[FinalContextRecord],
        train_labels: Sequence[int],
        val_records: Optional[Sequence[FinalContextRecord]] = None,
        val_labels: Optional[Sequence[int]] = None,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0)
    ):
        """
        Fit scaler and model strictly on training data.
        Select C by validation log-loss with deterministic tie-breaking (smaller C chosen on tie).
        """
        X_train_raw = self.extract_features(train_records)
        y_train = np.array(train_labels, dtype=int)

        self.scaler.fit(X_train_raw)
        X_train_scaled = self.scaler.transform(X_train_raw)

        if val_records is not None and val_labels is not None and len(val_records) > 0:
            X_val_raw = self.extract_features(val_records)
            X_val_scaled = self.scaler.transform(X_val_raw)
            y_val = np.array(val_labels, dtype=int)

            best_c = self.C
            best_val_loss = float("inf")
            for c_cand in sorted(c_grid):  # sorted ascending enforces deterministic smaller-C tie-breaking
                clf = LogisticRegression(penalty="l2", C=c_cand, solver="liblinear", random_state=42)
                clf.fit(X_train_scaled, y_train)
                # Compute log loss on validation split
                val_probs = clf.predict_proba(X_val_scaled)
                # If only 1 class in validation, fall back to negative accuracy
                try:
                    loss = float(log_loss(y_val, val_probs, labels=[0, 1]))
                except Exception:
                    loss = float(-np.mean(clf.predict(X_val_scaled) == y_val))

                if loss < best_val_loss - 1e-6:
                    best_val_loss = loss
                    best_c = c_cand
            self.C = best_c

        self.model = LogisticRegression(penalty="l2", C=self.C, solver="liblinear", random_state=42)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

    def predict_score(self, record: FinalContextRecord) -> float:
        """Predict continuous estimated ranking score P(Y=1 | x)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_score.")
        X_raw = extract_features_from_record(record, feature_mode=self.feature_mode, deduplicate=self.deduplicate)
        X_scaled = self.scaler.transform(X_raw.reshape(1, -1))
        # Continuous score for class 1
        return float(self.model.predict_proba(X_scaled)[0, 1])

    def decide(self, record: FinalContextRecord, threshold: Optional[float] = None) -> int:
        th = self.decision_threshold if threshold is None else threshold
        return int(self.predict_score(record) >= th)

    def fit_validation_thresholds(
        self,
        val_records: Sequence[FinalContextRecord],
        target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
    ) -> Dict[float, Dict[str, Any]]:
        """Fit operating thresholds on validation data targeting coverage levels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before selecting validation thresholds.")
        if not val_records:
            return {}

        val_scores = np.array([self.predict_score(r) for r in val_records])
        n_val = len(val_scores)
        sorted_scores = np.sort(val_scores)[::-1]

        res = {}
        for cov in target_coverages:
            k = max(1, min(n_val, int(round(cov * n_val))))
            th = float(sorted_scores[k - 1])
            achieved = float(np.mean(val_scores >= th))
            self.operating_thresholds[cov] = th
            res[cov] = {
                "threshold": round(th, 4),
                "target_coverage": cov,
                "val_achieved_coverage": round(achieved, 4),
                "val_deviation": round(abs(achieved - cov), 4)
            }
        return res


class LengthDiagnosticBaseline:
    """
    Diagnostic baseline using only context length features (passage count, total token count).
    Fit on training data and tuned on validation data.
    Used to detect whether construction shortcuts (e.g. passage count or length imbalances)
    can trivially predict sufficiency.
    """

    def __init__(self, name: str = "B_Diag_Length"):
        self.name = name
        self.scaler = StandardScaler()
        self.model = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", random_state=42)
        self.is_fitted = False
        self.operating_thresholds: Dict[float, float] = {0.50: 0.5, 0.75: 0.5, 0.90: 0.5}

    def extract_features(self, records: Sequence[FinalContextRecord]) -> np.ndarray:
        feats = []
        for r in records:
            p_count = float(len(r.passages))
            t_count = float(sum(p.token_count for p in r.passages))
            feats.append([p_count, t_count])
        return np.array(feats, dtype=np.float64)

    def fit(
        self,
        train_records: Sequence[FinalContextRecord],
        train_labels: Sequence[int],
        val_records: Optional[Sequence[FinalContextRecord]] = None,
        val_labels: Optional[Sequence[int]] = None
    ):
        X_train_raw = self.extract_features(train_records)
        y_train = np.array(train_labels, dtype=int)
        self.scaler.fit(X_train_raw)
        X_train_scaled = self.scaler.transform(X_train_raw)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

    def predict_score(self, record: FinalContextRecord) -> float:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_score.")
        X_raw = np.array([[float(len(record.passages)), float(sum(p.token_count for p in record.passages))]], dtype=np.float64)
        X_scaled = self.scaler.transform(X_raw)
        return float(self.model.predict_proba(X_scaled)[0, 1])

    def decide(self, record: FinalContextRecord, threshold: Optional[float] = None) -> int:
        th = 0.5 if threshold is None else threshold
        return int(self.predict_score(record) >= th)

    def fit_validation_thresholds(
        self,
        val_records: Sequence[FinalContextRecord],
        target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
    ) -> Dict[float, Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before selecting validation thresholds.")
        if not val_records:
            return {}

        val_scores = np.array([self.predict_score(r) for r in val_records])
        n_val = len(val_scores)
        sorted_scores = np.sort(val_scores)[::-1]

        res = {}
        for cov in target_coverages:
            k = max(1, min(n_val, int(round(cov * n_val))))
            th = float(sorted_scores[k - 1])
            achieved = float(np.mean(val_scores >= th))
            self.operating_thresholds[cov] = th
            res[cov] = {
                "threshold": round(th, 4),
                "target_coverage": cov,
                "val_achieved_coverage": round(achieved, 4),
                "val_deviation": round(abs(achieved - cov), 4)
            }
        return res

