"""Optional feature-evaluation analyzers (research tooling, separate from the leak-safe core).

Each analyzer maps a feature column (and a target) to one float. Spearman IC and the two-sample
KS distance are pure NumPy/Polars; mutual information and the tree R^2 import scikit-learn lazily.
TreeR2 scores on a held-out tail split rather than in-sample.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import polars as pl


class FeatureAnalyzer(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, feature: np.ndarray, target: np.ndarray) -> float:
        ...


def _finite_pair(feature, target):
    f = np.asarray(feature, float); t = np.asarray(target, float)
    m = np.isfinite(f) & np.isfinite(t)
    return f[m], t[m]


class SpearmanIC(FeatureAnalyzer):
    def __init__(self):
        super().__init__("spearman_ic")

    def calculate(self, feature, target) -> float:
        f, t = _finite_pair(feature, target)
        if len(f) < 10:
            return float("nan")
        rf = pl.Series(f).rank(method="average").to_numpy().astype(float)
        rt = pl.Series(t).rank(method="average").to_numpy().astype(float)
        rf -= rf.mean(); rt -= rt.mean()
        d = np.sqrt((rf ** 2).sum() * (rt ** 2).sum())
        return float((rf * rt).sum() / d) if d else float("nan")


class KSSeparation(FeatureAnalyzer):
    """Two-sample KS distance between the feature on target==1 vs target==0 (binary target).
    Larger = the feature separates the two classes more."""
    def __init__(self):
        super().__init__("ks_separation")

    def calculate(self, feature, target) -> float:
        f, t = _finite_pair(feature, target)
        a, b = np.sort(f[t > 0.5]), np.sort(f[t <= 0.5])
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        allv = np.concatenate([a, b])
        ca = np.searchsorted(a, allv, side="right") / len(a)
        cb = np.searchsorted(b, allv, side="right") / len(b)
        return float(np.max(np.abs(ca - cb)))


class MutualInfo(FeatureAnalyzer):
    """Nonparametric mutual information (catches nonlinear dependence). Needs scikit-learn."""
    def __init__(self, seed: int = 42):
        super().__init__("mutual_info"); self.seed = seed

    def calculate(self, feature, target) -> float:
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError as e:
            raise ImportError("MutualInfo needs scikit-learn (pip install scikit-learn)") from e
        f, t = _finite_pair(feature, target)
        return float(mutual_info_regression(f.reshape(-1, 1), t, random_state=self.seed)[0])


class TreeR2(FeatureAnalyzer):
    """Out-of-sample R^2 of a shallow random forest feature->target, scored on a held-out tail
    split (not in-sample). Needs scikit-learn."""
    def __init__(self, test_frac: float = 0.3, seed: int = 42):
        super().__init__("tree_r2"); self.test_frac = test_frac; self.seed = seed

    def calculate(self, feature, target) -> float:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score
        except ImportError as e:
            raise ImportError("TreeR2 needs scikit-learn (pip install scikit-learn)") from e
        f, t = _finite_pair(feature, target)
        if len(f) < 50:
            return float("nan")
        cut = int(len(f) * (1 - self.test_frac))
        m = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=self.seed)
        m.fit(f[:cut].reshape(-1, 1), t[:cut])
        return float(r2_score(t[cut:], m.predict(f[cut:].reshape(-1, 1))))


def analyze_features(matrix: pl.DataFrame, target: np.ndarray, analyzers, feature_cols=None) -> pl.DataFrame:
    """Apply each analyzer to each feature column; return a tidy DataFrame (feature x analyzer)."""
    cols = feature_cols or [c for c in matrix.columns if c != "timestamp"]
    rows = []
    for c in cols:
        fv = matrix[c].to_numpy()
        rows.append({"feature": c, **{a.name: a.calculate(fv, target) for a in analyzers}})
    return pl.DataFrame(rows)
