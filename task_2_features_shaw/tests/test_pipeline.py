"""FeatureSet pipeline tests: in-memory run + streaming equivalence."""
import numpy as np
import polars as pl
import pytest

from features.context import FeatureContext
from features import library as lib
from features.pipeline import FeatureSet

S = 1_000_000


def _frames():
    trades = pl.DataFrame({"timestamp": [t * S for t in [50, 80, 100, 130, 150]],
                           "ticker": ["perp:btcusdt"] * 5, "side": ["buy", "sell", "buy", "sell", "buy"],
                           "price": [100.0] * 5, "amount": [1.0, 2.0, 1.0, 3.0, 1.0]})
    bbo = pl.DataFrame({"timestamp": [t * S for t in [10, 40, 55, 90, 105, 140, 155, 200]],
                        "ticker": ["perp:btcusdt"] * 8,
                        "bid_price": [100.0, 100.0, 101.0, 100.0, 102.0, 100.0, 103.0, 104.0], "bid_amount": [1.0] * 8,
                        "ask_price": [101.0, 101.0, 102.0, 101.0, 103.0, 101.0, 104.0, 105.0], "ask_amount": [2.0] * 8})
    liq_exchange_1 = pl.DataFrame({"timestamp": [t * S for t in [45, 95, 145]], "ticker": ["perp:btcusdt"] * 3,
                                "side": ["sell", "buy", "sell"], "price": [100.0] * 3, "amount": [10.0] * 3})
    liq_exchange_2 = pl.DataFrame({"timestamp": [t * S for t in [48, 98]], "ticker": ["btcusdt"] * 2,
                              "side": ["buy", "sell"], "price": [100.0] * 2, "amount": [5.0] * 2})
    return trades, bbo, liq_exchange_1, liq_exchange_2


def test_run_returns_feature_matrix():
    ctx = FeatureContext.from_frames(*_frames())
    fs = FeatureSet(lib.DEFAULT_FEATURES)
    m = fs.run(ctx)
    assert m.height == ctx.n_trades
    assert m.columns[0] == "timestamp"
    for f in lib.DEFAULT_FEATURES:
        assert f.name in m.columns
        np.testing.assert_allclose(m[f.name].to_numpy(), f.calculate(ctx), equal_nan=True)


def test_validate_wrapper_reports_all_pass():
    fs = FeatureSet(lib.DEFAULT_FEATURES)
    rep = fs.validate(FeatureContext.from_frames(*_frames()))
    assert rep["ok"].all()


def test_streaming_matches_in_memory(tmp_path):
    trades, bbo, lb, lby = _frames()
    p = {}
    for name, df in [("trades", trades), ("bbo", bbo), ("lb", lb), ("lby", lby)]:
        p[name] = str(tmp_path / f"{name}.parquet"); df.write_parquet(p[name])

    fs = FeatureSet(lib.DEFAULT_FEATURES)
    ref = fs.run(FeatureContext.from_frames(trades, bbo, lb, lby))
    streamed = fs.run_streaming(p["trades"], p["bbo"], p["lb"], p["lby"], batch_rows=2)

    assert streamed.height == ref.height
    for c in ref.columns:
        np.testing.assert_allclose(streamed[c].to_numpy(), ref[c].to_numpy(), equal_nan=True,
                                   err_msg=f"mismatch in {c}")

    op = str(tmp_path / "feats.parquet")
    assert fs.run_streaming(p["trades"], p["bbo"], p["lb"], p["lby"], batch_rows=2, out_path=op) == op
    on_disk = pl.read_parquet(op)
    assert on_disk.height == ref.height
    for c in ref.columns:
        np.testing.assert_allclose(on_disk[c].to_numpy(), ref[c].to_numpy(), equal_nan=True)
