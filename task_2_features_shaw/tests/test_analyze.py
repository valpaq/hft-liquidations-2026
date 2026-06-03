"""Tests for the optional analyzer suite (features/analyze.py)."""
import numpy as np
import polars as pl
import pytest

from features.analyze import SpearmanIC, KSSeparation, analyze_features

rng = np.random.default_rng(0)


def test_spearman_ic_sign_and_monotone():
    x = np.linspace(0, 1, 500)
    assert SpearmanIC().calculate(x, 2 * x + 1) == pytest.approx(1.0)
    assert SpearmanIC().calculate(x, -x) == pytest.approx(-1.0)
    assert abs(SpearmanIC().calculate(x, rng.standard_normal(500))) < 0.2


def test_spearman_ic_ignores_nan_and_short():
    assert np.isnan(SpearmanIC().calculate(np.array([1.0, np.nan]), np.array([1.0, 2.0])))


def test_ks_separation_detects_shift():
    n = 1000
    feat = np.concatenate([rng.standard_normal(n), rng.standard_normal(n) + 5])
    target = np.concatenate([np.zeros(n), np.ones(n)])
    assert KSSeparation().calculate(feat, target) > 0.9
    same = rng.standard_normal(2 * n)
    assert KSSeparation().calculate(same, target) < 0.2


def test_analyze_features_tidy_frame():
    m = pl.DataFrame({"a": np.linspace(0, 1, 200), "b": rng.standard_normal(200)})
    target = np.linspace(0, 1, 200)
    out = analyze_features(m, target, [SpearmanIC()])
    assert out.columns == ["feature", "spearman_ic"]
    assert out.filter(pl.col("feature") == "a")["spearman_ic"][0] == pytest.approx(1.0)
