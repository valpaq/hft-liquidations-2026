"""Feature base contract + feature-library value tests."""
import numpy as np
import polars as pl
import pytest

from features.base import Feature
from features.context import FeatureContext
from features import library as lib

S = 1_000_000


def _ctx():
    liq_exchange_1 = pl.DataFrame({"timestamp": [10 * S], "ticker": ["perp:btcusdt"],
                                "side": ["sell"], "price": [100.0], "amount": [10.0]})
    liq_exchange_2 = pl.DataFrame({"timestamp": [10 * S], "ticker": ["btcusdt"],
                              "side": ["buy"], "price": [100.0], "amount": [5.0]})
    trades = pl.DataFrame({"timestamp": [9 * S, 10 * S + 100_000, 10 * S + 300_000],
                           "ticker": ["perp:btcusdt"] * 3, "side": ["buy", "sell", "buy"],
                           "price": [100.0, 100.0, 100.0], "amount": [1.0, 1.0, 1.0]})
    bbo = pl.DataFrame({"timestamp": [3 * S, 8 * S, 9 * S, 11 * S], "ticker": ["perp:btcusdt"] * 4,
                        "bid_price": [100.0, 99.0, 100.0, 100.0], "bid_amount": [1.0, 3.0, 1.0, 1.0],
                        "ask_price": [100.0, 101.0, 102.0, 102.0], "ask_amount": [1.0, 1.0, 1.0, 1.0]})
    return FeatureContext.from_frames(trades, bbo, liq_exchange_1, liq_exchange_2)


def test_feature_is_abstract():
    with pytest.raises(TypeError):
        Feature()


def test_custom_feature_subclass_works():
    class Const(Feature):
        name = "const"
        def calculate(self, ctx):
            return np.full(ctx.n_trades, 7.0)
    out = Const().calculate(_ctx())
    np.testing.assert_array_equal(out, [7.0, 7.0, 7.0])


def test_default_features_count_and_unique_names():
    feats = lib.DEFAULT_FEATURES
    assert len(feats) == 10
    names = [f.name for f in feats]
    assert len(set(names)) == 10
    assert all(f.lookback_us >= 0 for f in feats)


def test_liq_notional_name_and_lookback():
    f = lib.LiqNotional(30)
    assert f.name == "liq_notional_30s"
    assert f.lookback_us == 30 * S


def test_liq_notional_30s_values():
    np.testing.assert_allclose(lib.LiqNotional(30).calculate(_ctx()), [0.0, 1000.0, 1500.0])


def test_exchange_2_liq_notional_respects_shift():
    np.testing.assert_allclose(lib.Exchange2LiqNotional(30).calculate(_ctx()), [0.0, 0.0, 500.0])


def test_side_imbalance_values():
    out = lib.LiqSideImbalance(30).calculate(_ctx())
    np.testing.assert_allclose(out[2], (500.0 - 1000.0) / 1500.0)


def test_event_count_values():
    np.testing.assert_allclose(lib.LiqEventCount(30).calculate(_ctx()), [0.0, 1.0, 2.0])


def test_velocity_values():
    out = lib.LiqVelocity(30, 120).calculate(_ctx())
    np.testing.assert_allclose(out[2], 1500.0 / 1501.0)
    assert out[0] == 0.0


def test_time_since_liq_values():
    out = lib.TimeSinceLiq().calculate(_ctx())
    np.testing.assert_allclose(out[2], 0.1)
    assert out[0] >= 1e8


def test_bbo_spread_values():
    out = lib.BBOSpread().calculate(_ctx())
    np.testing.assert_allclose(out, [198.01980198] * 3)


def test_mid_return_5s_values():
    out = lib.MidReturn(5).calculate(_ctx())
    np.testing.assert_allclose(out, [100.0, 100.0, 100.0])


def test_all_features_return_aligned_finite_arrays():
    ctx = _ctx()
    for f in lib.DEFAULT_FEATURES:
        v = f.calculate(ctx)
        assert v.shape == (ctx.n_trades,), f.name
        assert np.isfinite(v).all(), f.name


def test_liq_strength_signed_log():
    out = lib.LiqStrength(30).calculate(_ctx())
    np.testing.assert_allclose(out[2], -np.log1p(500.0))
    assert out[0] == 0.0
