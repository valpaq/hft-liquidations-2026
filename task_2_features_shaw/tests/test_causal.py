"""Causal primitive tests, known-answer synthetic data."""
import numpy as np
import pytest

from features.causal import causal_window_sum, causal_count, asof_backward

S = 1_000_000


def _prefix(values):
    return np.concatenate([[0.0], np.cumsum(values)])


def test_window_sum_excludes_event_at_t():
    ts = np.array([0, 5, 10]) * S
    prefix = _prefix([100.0, 200.0, 400.0])
    out = causal_window_sum(ts, prefix, np.array([10 * S]), 30 * S)
    assert out[0] == 300.0


def test_window_sum_includes_lower_bound_and_past():
    ts = np.array([0, 5, 10]) * S
    prefix = _prefix([100.0, 200.0, 400.0])
    assert causal_window_sum(ts, prefix, np.array([11 * S]), 30 * S)[0] == 700.0
    assert causal_window_sum(ts, prefix, np.array([6 * S]), 3 * S)[0] == 200.0


def test_window_sum_empty_window_is_zero():
    ts = np.array([0, 5, 10]) * S
    prefix = _prefix([100.0, 200.0, 400.0])
    assert causal_window_sum(ts, prefix, np.array([3 * S]), 2 * S)[0] == 0.0


def test_window_count_excludes_event_at_t():
    ts = np.array([0, 5, 10]) * S
    assert causal_count(ts, np.array([10 * S]), 30 * S)[0] == 2.0
    assert causal_count(ts, np.array([11 * S]), 30 * S)[0] == 3.0


def test_asof_backward_forward_fills():
    ts = np.array([0, 10, 20]) * S
    val = np.array([1.0, 2.0, 3.0])
    assert asof_backward(ts, val, np.array([15 * S]))[0] == 2.0
    assert asof_backward(ts, val, np.array([10 * S]))[0] == 2.0
    assert asof_backward(ts, val, np.array([20 * S]))[0] == 3.0


def test_asof_backward_nan_outside_coverage():
    ts = np.array([0, 10, 20]) * S
    val = np.array([1.0, 2.0, 3.0])
    assert np.isnan(asof_backward(ts, val, np.array([25 * S]))[0])
    assert np.isnan(asof_backward(ts, val, np.array([-5 * S]))[0])


def test_primitives_are_vectorised_and_aligned():
    ts = np.array([0, 5, 10]) * S
    prefix = _prefix([100.0, 200.0, 400.0])
    q = np.array([6, 11, 3]) * S
    out = causal_window_sum(ts, prefix, q, 30 * S)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, [300.0, 700.0, 100.0])
