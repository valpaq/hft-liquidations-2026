"""FeatureSet runs features into a matrix and validates them. ``run(ctx)`` works in memory
(sample); ``run_streaming`` prepares liquidations+BBO once and streams trades in batches that
reuse them, so a windowed feature always sees the full past stream."""
from __future__ import annotations
import polars as pl

from .context import FeatureContext
from .validate import validate_feature_set


class FeatureSet:
    def __init__(self, features):
        self.features = list(features)

    def run(self, ctx: FeatureContext) -> pl.DataFrame:
        cols = {"timestamp": ctx.trade_ts}
        for f in self.features:
            cols[f.name] = f.calculate(ctx)
        return pl.DataFrame(cols)

    def validate(self, ctx: FeatureContext, **kw) -> pl.DataFrame:
        return validate_feature_set(self.features, ctx, **kw)

    def run_streaming(self, trades_path, bbo_path, liq_exchange_1_path, liq_exchange_2_path,
                      batch_rows: int = 20_000_000, out_path: str | None = None):
        """Full-data path: liquidations and BBO are prepared once, then trades stream in batches.
        With out_path each batch is written as it is computed, so memory stays bounded (the full
        feature matrix for 400-700M trades would not fit in RAM); returns the path. Without
        out_path the batches are concatenated and returned, for small in-memory use.
        Assumes the trades file is sorted by timestamp, and that features read only the
        liquidation/BBO streams and the trade's own fields (there is no trades-as-history stream)."""
        import pyarrow.parquet as pq
        base = FeatureContext.from_frames(pl.read_parquet(trades_path, n_rows=0),
                                          pl.read_parquet(bbo_path),
                                          pl.read_parquet(liq_exchange_1_path),
                                          pl.read_parquet(liq_exchange_2_path))
        pf = pq.ParquetFile(trades_path)
        writer, parts = None, []
        for batch in pf.iter_batches(batch_size=batch_rows,
                                     columns=["timestamp", "side", "price", "amount"]):
            res = self.run(base.replace_trades(pl.from_arrow(batch)))
            if out_path:
                tbl = res.to_arrow()
                writer = writer or pq.ParquetWriter(out_path, tbl.schema)
                writer.write_table(tbl)
            else:
                parts.append(res)
        if out_path:
            if writer is None:
                self.run(base).write_parquet(out_path)
            else:
                writer.close()
            return out_path
        return pl.concat(parts) if parts else self.run(base)
