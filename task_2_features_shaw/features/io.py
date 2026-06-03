"""Data loading helpers for the feature pipeline.

Liquidation/BBO tables are loaded whole (small enough); trades are huge, so `sample_trades`
draws a reproducible reservoir sample for the example run via DuckDB (no full load).
"""
from __future__ import annotations
from pathlib import Path
import polars as pl


def _data_root() -> str:
    """Find `liquidation_task/data` by walking up from this file (works from any CWD or
    submission subfolder)."""
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        cand = p / "liquidation_task" / "data"
        if cand.is_dir():
            return str(cand)
    return "liquidation_task/data"


DATA = _data_root()


def trades_path(sym): return f"{DATA}/exchange_1_trades/perp_{sym}usdt.parquet"
def bbo_path(sym):    return f"{DATA}/exchange_1_booktickers/perp_{sym}usdt.parquet"
def binliq_path(sym): return f"{DATA}/exchange_1_liquidations/perp_{sym}usdt.parquet"
def bybliq_path(sym): return f"{DATA}/exchange_2_liquidations/{sym}usdt.parquet"


def sample_trades(path: str, n: int, seed: int = 42) -> pl.DataFrame:
    """Reproducible reservoir sample of `n` trades (returns all if the table is smaller)."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'; SET threads=4;")
    return con.execute(
        f"SELECT * FROM read_parquet('{path}') USING SAMPLE {n} ROWS (reservoir, {seed})"
    ).pl()


def load_frames(sym: str, trades_sample: int | None = None, seed: int = 42):
    """Return (trades, bbo, liq_exchange_1, liq_exchange_2) for a symbol. If trades_sample is given,
    trades are a reproducible reservoir sample of that size; otherwise the whole (large) trades
    table is loaded, in which case prefer the streaming pipeline."""
    trades = (sample_trades(trades_path(sym), trades_sample, seed)
              if trades_sample else pl.read_parquet(trades_path(sym)))
    bbo = pl.read_parquet(bbo_path(sym),
                          columns=["timestamp", "bid_price", "bid_amount", "ask_price", "ask_amount"])
    return (trades, bbo, pl.read_parquet(binliq_path(sym)), pl.read_parquet(bybliq_path(sym)))
