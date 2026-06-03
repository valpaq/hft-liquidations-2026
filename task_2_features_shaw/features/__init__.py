"""Leak-safe liquidation feature-engineering infrastructure (task D).

    from features import FeatureContext, FeatureSet, DEFAULT_FEATURES, load_frames
    trades, bbo, lb, lby = load_frames("btc", trades_sample=2_000_000)
    ctx = FeatureContext.from_frames(trades, bbo, lb, lby)
    matrix = FeatureSet(DEFAULT_FEATURES).run(ctx)
    report = FeatureSet(DEFAULT_FEATURES).validate(ctx)
"""
from .causal import causal_window_sum, causal_count, asof_backward
from .context import FeatureContext
from .base import Feature
from . import library
from .library import DEFAULT_FEATURES
from .pipeline import FeatureSet
from .validate import (check_finite, check_alignment, check_no_lookahead,
                       check_no_lookahead_tracked, validate_feature, validate_feature_set)
from .io import load_frames, sample_trades

__all__ = ["FeatureContext", "Feature", "FeatureSet", "DEFAULT_FEATURES", "library",
           "causal_window_sum", "causal_count", "asof_backward",
           "check_finite", "check_alignment", "check_no_lookahead", "check_no_lookahead_tracked",
           "validate_feature", "validate_feature_set", "load_frames", "sample_trades"]
