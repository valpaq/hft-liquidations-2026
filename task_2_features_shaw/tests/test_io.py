"""IO tests, focus on the reproducible reservoir sample (the only non-trivial logic)."""
import polars as pl
import pytest

from features.io import sample_trades


def test_sample_is_correct_size_and_reproducible(tmp_path):
    path = str(tmp_path / "trades.parquet")
    pl.DataFrame({"timestamp": list(range(1000)), "ticker": ["x"] * 1000,
                  "side": ["buy", "sell"] * 500, "price": [100.0] * 1000,
                  "amount": [1.0] * 1000}).write_parquet(path)

    a = sample_trades(path, n=100, seed=7)
    b = sample_trades(path, n=100, seed=7)
    assert a.height == 100
    assert a.sort("timestamp")["timestamp"].to_list() == b.sort("timestamp")["timestamp"].to_list()
    assert set(a.columns) == {"timestamp", "ticker", "side", "price", "amount"}


def test_sample_larger_than_table_returns_all(tmp_path):
    path = str(tmp_path / "trades.parquet")
    pl.DataFrame({"timestamp": [1, 2, 3], "ticker": ["x"] * 3, "side": ["buy"] * 3,
                  "price": [1.0] * 3, "amount": [1.0] * 3}).write_parquet(path)
    assert sample_trades(path, n=100, seed=1).height == 3
