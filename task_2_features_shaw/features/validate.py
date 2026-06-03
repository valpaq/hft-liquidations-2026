"""Feature validators (task D): no inf, optional no-NaN, alignment, and no-forward-looking.
`check_no_lookahead` recomputes each probe trade's feature from only the data available by t
(Exchange 2 availability = raw + 200 ms); a causal feature is unchanged, a future-peeker changes."""
from __future__ import annotations
import numpy as np
import polars as pl

from .context import FeatureContext


def check_finite(name: str, values: np.ndarray, allow_nan: bool = False) -> dict:
    v = np.asarray(values, dtype=np.float64)
    n_nan, n_inf = int(np.isnan(v).sum()), int(np.isinf(v).sum())
    ok = n_inf == 0 and (allow_nan or n_nan == 0)
    return {"name": name, "check": "finite", "ok": ok, "n_nan": n_nan, "n_inf": n_inf}


def check_alignment(name: str, values: np.ndarray, n_trades: int) -> dict:
    ok = len(values) == n_trades
    return {"name": name, "check": "alignment", "ok": ok, "len": len(values), "n_trades": n_trades}


def check_no_lookahead(feature, ctx: FeatureContext, n_probe: int = 256, seed: int = 0,
                       tol: float = 1e-9, exhaustive_below: int = 5000) -> dict:
    """Recompute the feature for probe trades on a context truncated to data available by t
    (ctx.truncated_single) and require the value to be unchanged. Probes every trade when
    n_trades <= exhaustive_below, else a random sample of n_probe. This is a check, not a proof:
    it assumes the feature is a pure function of the context (a feature that caches state across
    calls can defeat the differential test), and on large data a leak confined to a few rows can
    slip through the sample (raise n_probe)."""
    full = feature.calculate(ctx)
    n = ctx.n_trades
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= max(n_probe, exhaustive_below) else rng.choice(n, n_probe, replace=False)

    violations, max_diff = 0, 0.0
    for i in idx:
        got = feature.calculate(ctx.truncated_single(int(i)))[0]
        ref = full[int(i)]
        if np.isnan(got) and np.isnan(ref):
            continue
        if np.isnan(got) or np.isnan(ref):
            violations += 1
            continue
        d = abs(got - ref)
        if d > tol:
            violations += 1
            max_diff = max(max_diff, d)
    return {"name": feature.name, "check": "no_lookahead", "ok": violations == 0,
            "n_violations": violations, "n_probe": int(len(idx)), "max_abs_diff": float(max_diff)}


def check_no_lookahead_tracked(feature, ctx: FeatureContext) -> dict:
    """Exhaustive (every trade, O(n)) leak check: run the feature with accessor tracking and
    require the max source timestamp it read to be <= the trade time. Complements the differential
    ``check_no_lookahead``, which also catches reads that bypass the public methods (event at t)."""
    mu = ctx.max_used_ts(feature)
    viol = int((mu > ctx.trade_ts).sum())
    return {"name": feature.name, "check": "no_lookahead_tracked", "ok": viol == 0,
            "n_violations": viol}


def validate_feature(feature, ctx: FeatureContext, allow_nan: bool = False, **kw) -> dict:
    v = feature.calculate(ctx)
    fin = check_finite(feature.name, v, allow_nan=allow_nan)
    al = check_alignment(feature.name, v, ctx.n_trades)
    look = check_no_lookahead(feature, ctx, **kw) if al["ok"] else {"ok": False, "n_violations": 0}
    tracked = check_no_lookahead_tracked(feature, ctx) if al["ok"] else {"ok": False, "n_violations": 0}
    return {"feature": feature.name, "lookback_us": feature.lookback_us,
            "finite_ok": fin["ok"], "n_nan": fin["n_nan"], "n_inf": fin["n_inf"],
            "aligned": al["ok"], "lookahead_ok": look["ok"],
            "lookahead_violations": look["n_violations"], "tracked_ok": tracked["ok"],
            "ok": fin["ok"] and al["ok"] and look["ok"] and tracked["ok"]}


def validate_feature_set(features, ctx: FeatureContext,
                         allow_nan: bool = False, **kw) -> pl.DataFrame:
    return pl.DataFrame([validate_feature(f, ctx, allow_nan=allow_nan, **kw) for f in features])
