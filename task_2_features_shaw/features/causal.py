"""Causal window primitives. The window is [t-W, t): the upper bound uses searchsorted
side='left', so an event at the trade's own timestamp t is not counted.
"""
from __future__ import annotations
import numpy as np


def causal_window_sum(src_ts: np.ndarray, prefix: np.ndarray,
                      q: np.ndarray, w_us: int) -> np.ndarray:
    """Sum a time-sorted value over [q-w, q) for each query time q. `prefix` is its cumulative
    sum with a leading 0 (length len(src_ts)+1)."""
    hi = np.searchsorted(src_ts, q, side="left")
    lo = np.searchsorted(src_ts, q - w_us, side="left")
    return prefix[hi] - prefix[lo]


def causal_count(src_ts: np.ndarray, q: np.ndarray, w_us: int) -> np.ndarray:
    """Count events in [q-w, q) for each query time q."""
    hi = np.searchsorted(src_ts, q, side="left")
    lo = np.searchsorted(src_ts, q - w_us, side="left")
    return (hi - lo).astype(np.float64)


def asof_backward(src_ts: np.ndarray, src_val: np.ndarray, q: np.ndarray,
                  exclude_beyond: bool = True) -> np.ndarray:
    """Last src_val with src_ts <= q (forward fill); NaN before the first observation.
    exclude_beyond also returns NaN past the last observation, for future look-ups like markout
    labels; set it False to read the current/past quote at t, which always exists."""
    if len(src_ts) == 0:
        return np.full(len(q), np.nan)
    i = np.searchsorted(src_ts, q, side="right") - 1
    out = np.where(i >= 0, src_val[np.clip(i, 0, len(src_val) - 1)], np.nan).astype(np.float64)
    out[q < src_ts[0]] = np.nan
    if exclude_beyond:
        out[q > src_ts[-1]] = np.nan
    return out
