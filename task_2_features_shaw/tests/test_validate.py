"""Validator tests, including the no-forward-looking check."""
import numpy as np
import polars as pl
import pytest

from features.base import Feature
from features.context import FeatureContext
from features import library as lib
from features.validate import (check_finite, check_alignment, check_no_lookahead,
                               check_no_lookahead_tracked, validate_feature, validate_feature_set)

S = 1_000_000


def _frames():
    trades = pl.DataFrame({"timestamp": [50 * S, 100 * S, 150 * S], "ticker": ["perp:btcusdt"] * 3,
                           "side": ["buy", "sell", "buy"], "price": [100.0, 100.0, 100.0],
                           "amount": [1.0, 1.0, 1.0]})
    bbo = pl.DataFrame({"timestamp": [t * S for t in [10, 40, 55, 90, 105, 140, 155, 200]],
                        "ticker": ["perp:btcusdt"] * 8,
                        "bid_price": [100.0, 100.0, 101.0, 100.0, 102.0, 100.0, 103.0, 104.0],
                        "bid_amount": [1.0] * 8,
                        "ask_price": [101.0, 101.0, 102.0, 101.0, 103.0, 101.0, 104.0, 105.0],
                        "ask_amount": [1.0] * 8})
    liq_exchange_1 = pl.DataFrame({"timestamp": [t * S for t in [45, 95, 145]], "ticker": ["perp:btcusdt"] * 3,
                                "side": ["sell", "buy", "sell"], "price": [100.0] * 3, "amount": [10.0] * 3})
    liq_exchange_2 = pl.DataFrame({"timestamp": [t * S for t in [48, 98]], "ticker": ["btcusdt"] * 2,
                              "side": ["buy", "sell"], "price": [100.0] * 2, "amount": [5.0] * 2})
    return trades, bbo, liq_exchange_1, liq_exchange_2


class LeakyForward(Feature):
    """Deliberately reads a FUTURE quote (mid at t+30s) -> must be flagged."""
    name = "leaky_forward"
    lookback_us = 0
    def calculate(self, ctx):
        return ctx.asof("mid", offset_us=30 * S)


class LeakyInclusiveAtT(Feature):
    """Bypasses the public window and includes the liquidation AT exactly t (side='right'),
    violating the [t-W, t) contract -> must be flagged."""
    name = "leaky_inclusive_at_t"
    lookback_us = 30 * S
    def calculate(self, ctx):
        st = ctx._streams["liq_combined"]
        hi = np.searchsorted(st.ts, ctx.trade_ts, side="right")
        lo = np.searchsorted(st.ts, ctx.trade_ts - self.lookback_us, side="left")
        return st.prefix["notional"][hi] - st.prefix["notional"][lo]


def test_check_finite_clean():
    assert check_finite("x", np.array([1.0, 2.0, 3.0]))["ok"]


def test_check_finite_rejects_inf_and_nan():
    assert not check_finite("x", np.array([1.0, np.inf]))["ok"]
    assert not check_finite("x", np.array([1.0, np.nan]))["ok"]
    assert check_finite("x", np.array([1.0, np.nan]), allow_nan=True)["ok"]


def test_check_alignment():
    assert check_alignment("x", np.arange(3), n_trades=3)["ok"]
    assert not check_alignment("x", np.arange(2), n_trades=3)["ok"]


def test_no_lookahead_passes_for_causal_feature():
    ctx = FeatureContext.from_frames(*_frames())
    rep = check_no_lookahead(lib.LiqNotional(30), ctx)
    assert rep["ok"], rep
    assert rep["n_violations"] == 0


def test_no_lookahead_passes_for_bbo_at_t_feature():
    ctx = FeatureContext.from_frames(*_frames())
    assert check_no_lookahead(lib.BBOSpread(), ctx)["ok"]


def test_no_lookahead_catches_future_peeking_feature():
    ctx = FeatureContext.from_frames(*_frames())
    rep = check_no_lookahead(LeakyForward(), ctx)
    assert not rep["ok"]
    assert rep["n_violations"] > 0


def test_no_lookahead_catches_event_at_exactly_t():
    trades = pl.DataFrame({"timestamp": [100 * S], "ticker": ["perp:btcusdt"], "side": ["buy"],
                           "price": [100.0], "amount": [1.0]})
    bbo = pl.DataFrame({"timestamp": [90 * S], "ticker": ["perp:btcusdt"], "bid_price": [100.0],
                        "bid_amount": [1.0], "ask_price": [101.0], "ask_amount": [1.0]})
    liq_exchange_1 = pl.DataFrame({"timestamp": [100 * S], "ticker": ["perp:btcusdt"], "side": ["sell"],
                               "price": [100.0], "amount": [10.0]})
    liq_exchange_2 = pl.DataFrame({"timestamp": [], "ticker": [], "side": [], "price": [], "amount": []},
                             schema={"timestamp": pl.Int64, "ticker": pl.Utf8, "side": pl.Utf8,
                                     "price": pl.Float64, "amount": pl.Float64})
    ctx = FeatureContext.from_frames(trades, bbo, liq_exchange_1, liq_exchange_2)
    rep = check_no_lookahead(LeakyInclusiveAtT(), ctx)
    assert not rep["ok"], "feature that includes the liquidation at exactly t must be flagged"


def test_validate_feature_set_all_pass():
    ctx = FeatureContext.from_frames(*_frames())
    report = validate_feature_set(lib.DEFAULT_FEATURES, ctx)
    assert isinstance(report, pl.DataFrame)
    assert report.height == len(lib.DEFAULT_FEATURES)
    assert report["ok"].all(), report


def test_validate_feature_reports_misalignment_without_crashing():
    ctx = FeatureContext.from_frames(*_frames())
    class ShortFeature(Feature):
        name = "short"
        lookback_us = 0
        def calculate(self, ctx):
            return np.array([1.0])
    rep = validate_feature(ShortFeature(), ctx)
    assert rep["aligned"] is False and rep["ok"] is False


def test_tracked_lookahead_is_exhaustive():
    ctx = FeatureContext.from_frames(*_frames())
    assert check_no_lookahead_tracked(lib.LiqNotional(30), ctx)["ok"]
    assert check_no_lookahead_tracked(lib.BBOSpread(), ctx)["ok"]
    rep = check_no_lookahead_tracked(LeakyForward(), ctx)
    assert not rep["ok"]
    assert rep["n_violations"] == ctx.n_trades
