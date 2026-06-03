"""FeatureContext tests, synthetic Polars frames, known answers."""
import numpy as np
import polars as pl
import pytest

from features.context import FeatureContext

S = 1_000_000


def _frames():
    liq_exchange_1 = pl.DataFrame({"timestamp": [10 * S], "ticker": ["perp:btcusdt"],
                                "side": ["sell"], "price": [100.0], "amount": [10.0]})
    liq_exchange_2 = pl.DataFrame({"timestamp": [10 * S], "ticker": ["btcusdt"],
                              "side": ["buy"], "price": [100.0], "amount": [5.0]})
    trades = pl.DataFrame({"timestamp": [9 * S, 10 * S + 100_000, 10 * S + 300_000],
                           "ticker": ["perp:btcusdt"] * 3, "side": ["buy", "sell", "buy"],
                           "price": [100.0, 100.0, 100.0], "amount": [1.0, 1.0, 1.0]})
    bbo = pl.DataFrame({"timestamp": [8 * S, 9 * S, 11 * S],
                        "ticker": ["perp:btcusdt"] * 3,
                        "bid_price": [99.0, 100.0, 100.0], "bid_amount": [3.0, 1.0, 1.0],
                        "ask_price": [101.0, 102.0, 102.0], "ask_amount": [1.0, 1.0, 1.0]})
    return trades, bbo, liq_exchange_1, liq_exchange_2


def test_exchange_2_shifted_200ms_before_combine():
    ctx = FeatureContext.from_frames(*_frames())
    out = ctx.window_sum("liq_combined", "notional", 30 * S)
    np.testing.assert_allclose(out, [0.0, 1000.0, 1500.0])


def test_exchange_2_only_stream_respects_shift():
    ctx = FeatureContext.from_frames(*_frames())
    out = ctx.window_sum("liq_exchange_2", "notional", 30 * S)
    np.testing.assert_allclose(out, [0.0, 0.0, 500.0])


def test_window_count():
    ctx = FeatureContext.from_frames(*_frames())
    np.testing.assert_allclose(ctx.window_count("liq_combined", 30 * S), [0.0, 1.0, 2.0])


def test_side_imbalance_combined():
    ctx = FeatureContext.from_frames(*_frames())
    imb = ctx.window_side_imbalance("liq_combined", 30 * S)
    assert imb[0] == 0.0
    np.testing.assert_allclose(imb[2], (500.0 - 1000.0) / 1500.0)


def test_asof_mid_forward_filled():
    ctx = FeatureContext.from_frames(*_frames())
    mid = ctx.asof("mid")
    np.testing.assert_allclose(mid, [101.0, 101.0, 101.0])


def test_time_since_last_seconds():
    ctx = FeatureContext.from_frames(*_frames())
    tsl = ctx.time_since_last("liq_combined")
    np.testing.assert_allclose(tsl[2], 0.1)


def test_trade_arrays_and_maker_side():
    ctx = FeatureContext.from_frames(*_frames())
    assert ctx.n_trades == 3
    np.testing.assert_array_equal(ctx.trade_s, [1.0, -1.0, 1.0])


def test_truncated_single_keeps_only_data_available_by_t():
    ctx = FeatureContext.from_frames(*_frames())
    tr = ctx.truncated_single(2)
    assert tr.n_trades == 1
    assert tr.window_sum("liq_combined", "notional", 30 * S)[0] == 1500.0
    np.testing.assert_allclose(tr.asof("mid")[0], 101.0)


def test_truncated_single_handles_empty_streams():
    ctx = FeatureContext.from_frames(*_frames())
    tr = ctx.truncated_single(0)
    assert tr.window_sum("liq_combined", "notional", 30 * S)[0] == 0.0
    assert tr.window_count("liq_combined", 30 * S)[0] == 0.0
    assert tr.time_since_last("liq_combined")[0] >= 1e8


def test_max_used_ts_handles_empty_bbo():
    from features import library as lib
    trades = pl.DataFrame({"timestamp": [50 * S, 100 * S], "ticker": ["x"] * 2,
                           "side": ["buy", "sell"], "price": [100.0] * 2, "amount": [1.0] * 2})
    empty_bbo = pl.DataFrame(schema={"timestamp": pl.Int64, "ticker": pl.Utf8,
                                     "bid_price": pl.Float64, "bid_amount": pl.Float64,
                                     "ask_price": pl.Float64, "ask_amount": pl.Float64})
    lb = pl.DataFrame({"timestamp": [45 * S], "ticker": ["x"], "side": ["sell"],
                       "price": [100.0], "amount": [10.0]})
    lby = pl.DataFrame(schema={"timestamp": pl.Int64, "ticker": pl.Utf8, "side": pl.Utf8,
                               "price": pl.Float64, "amount": pl.Float64})
    ctx = FeatureContext.from_frames(trades, empty_bbo, lb, lby)
    mu = ctx.max_used_ts(lib.BBOSpread())
    assert len(mu) == 2
    assert np.all(mu <= ctx.trade_ts)
