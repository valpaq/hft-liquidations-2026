"""FeatureContext: the Polars boundary. from_frames shifts Exchange 2 +200 ms, sorts the liquidation
streams, and precomputes prefix sums and the BBO series. Features reach data only through its
causal window methods, not the raw frames."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import polars as pl

from .causal import causal_window_sum, causal_count, asof_backward

EXCHANGE_2_LAG_US = 200_000


@dataclass
class _Stream:
    ts: np.ndarray
    prefix: dict[str, np.ndarray]


@dataclass
class FeatureContext:
    trade_ts: np.ndarray
    trade_price: np.ndarray
    trade_s: np.ndarray
    trade_w: np.ndarray
    _streams: dict[str, _Stream] = field(default_factory=dict)
    _bbo_ts: np.ndarray = field(default=None)
    _bbo: dict[str, np.ndarray] = field(default_factory=dict)
    _track: bool = False
    _max_used: np.ndarray = field(default=None)

    @staticmethod
    def _make_stream(df: pl.DataFrame) -> _Stream:
        df = df.sort("timestamp")
        ts = df["timestamp"].to_numpy()
        notional = (df["price"] * df["amount"]).to_numpy()
        is_buy = (df["side"] == "buy").to_numpy()
        buy = np.where(is_buy, notional, 0.0)
        sell = np.where(~is_buy, notional, 0.0)
        pref = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
                {"notional": notional, "buy_notional": buy, "sell_notional": sell}.items()}
        return _Stream(ts=ts, prefix=pref)

    @staticmethod
    def _trade_arrays(trades: pl.DataFrame):
        trades = trades.sort("timestamp")
        return (trades["timestamp"].to_numpy(),
                trades["price"].to_numpy(),
                np.where((trades["side"] == "buy").to_numpy(), 1.0, -1.0),
                np.minimum((trades["price"] * trades["amount"]).to_numpy(), 100_000.0))

    @classmethod
    def from_frames(cls, trades: pl.DataFrame, bbo: pl.DataFrame,
                    liq_exchange_1: pl.DataFrame, liq_exchange_2: pl.DataFrame,
                    exchange_2_lag_us: int = EXCHANGE_2_LAG_US) -> "FeatureContext":
        t, price, s, w = cls._trade_arrays(trades)

        exchange_2_shift = liq_exchange_2.with_columns((pl.col("timestamp") + exchange_2_lag_us).alias("timestamp"))
        combined = pl.concat([liq_exchange_1.select("timestamp", "side", "price", "amount"),
                              exchange_2_shift.select("timestamp", "side", "price", "amount")])
        streams = {
            "liq_exchange_1": cls._make_stream(liq_exchange_1),
            "liq_exchange_2": cls._make_stream(exchange_2_shift),
            "liq_combined": cls._make_stream(combined),
        }

        if not bbo["timestamp"].is_sorted():
            bbo = bbo.sort("timestamp")
        bts = bbo["timestamp"].to_numpy()
        mid = ((bbo["bid_price"] + bbo["ask_price"]) / 2).to_numpy()
        spread = ((bbo["ask_price"] - bbo["bid_price"]) / mid * 1e4)
        imb = ((bbo["bid_amount"] - bbo["ask_amount"]) / (bbo["bid_amount"] + bbo["ask_amount"])).to_numpy()
        bbo_series = {"mid": mid, "spread_bps": np.asarray(spread), "imbalance": imb}

        return cls(trade_ts=t, trade_price=price, trade_s=s, trade_w=w,
                   _streams=streams, _bbo_ts=bts, _bbo=bbo_series)

    def truncated_single(self, i: int) -> "FeatureContext":
        """1-trade context for trade ``i`` keeping only data a causal feature may use at t.
        Event streams are sliced **strictly before t** (side='left'), so a feature that
        sneaks in an event at exactly t differs from the full context and is flagged; BBO
        keeps `<= t` (the quote at t is observable). Used by the no-lookahead validator."""
        t = int(self.trade_ts[i])
        streams = {}
        for name, st in self._streams.items():
            k = int(np.searchsorted(st.ts, t, side="left"))
            streams[name] = _Stream(ts=st.ts[:k], prefix={v: p[:k + 1] for v, p in st.prefix.items()})
        kb = int(np.searchsorted(self._bbo_ts, t, side="right"))
        bbo = {key: arr[:kb] for key, arr in self._bbo.items()}
        return FeatureContext(
            trade_ts=self.trade_ts[i:i + 1], trade_price=self.trade_price[i:i + 1],
            trade_s=self.trade_s[i:i + 1], trade_w=self.trade_w[i:i + 1],
            _streams=streams, _bbo_ts=self._bbo_ts[:kb], _bbo=bbo)

    def replace_trades(self, trades: pl.DataFrame) -> "FeatureContext":
        """New context for a different trade batch, sharing the (heavy) prepared streams and
        BBO arrays. Used by streaming so liquidations/BBO are prepared once, not per batch."""
        t, price, s, w = self._trade_arrays(trades)
        return FeatureContext(trade_ts=t, trade_price=price, trade_s=s, trade_w=w,
                              _streams=self._streams, _bbo_ts=self._bbo_ts, _bbo=self._bbo)

    @property
    def n_trades(self) -> int:
        return len(self.trade_ts)

    def _note(self, used_ts: np.ndarray) -> None:
        if self._track:
            np.maximum(self._max_used, np.where(np.isnan(used_ts), -np.inf, used_ts),
                       out=self._max_used)

    def _window_last_ts(self, src_ts: np.ndarray, lookback_us: int) -> np.ndarray:
        hi = np.searchsorted(src_ts, self.trade_ts, side="left")
        lo = np.searchsorted(src_ts, self.trade_ts - lookback_us, side="left")
        out = np.full(self.n_trades, np.nan)
        m = hi > lo
        out[m] = src_ts[hi[m] - 1]
        return out

    def window_sum(self, stream: str, value: str, lookback_us: int) -> np.ndarray:
        st = self._streams[stream]
        if self._track:
            self._note(self._window_last_ts(st.ts, lookback_us))
        return causal_window_sum(st.ts, st.prefix[value], self.trade_ts, lookback_us)

    def window_count(self, stream: str, lookback_us: int) -> np.ndarray:
        st = self._streams[stream]
        if self._track:
            self._note(self._window_last_ts(st.ts, lookback_us))
        return causal_count(st.ts, self.trade_ts, lookback_us)

    def window_side_imbalance(self, stream: str, lookback_us: int) -> np.ndarray:
        buy = self.window_sum(stream, "buy_notional", lookback_us)
        sell = self.window_sum(stream, "sell_notional", lookback_us)
        total = buy + sell
        out = np.zeros_like(total)
        np.divide(buy - sell, total, out=out, where=total > 0)
        return out

    def time_since_last(self, stream: str, default_s: float = 1e9) -> np.ndarray:
        ts = self._streams[stream].ts
        if ts.size == 0:
            return np.full(self.n_trades, default_s)
        i = np.searchsorted(ts, self.trade_ts, side="left") - 1
        last = np.where(i >= 0, ts[np.clip(i, 0, len(ts) - 1)], np.nan)
        if self._track:
            self._note(last)
        out = (self.trade_ts - last) / 1e6
        return np.where(np.isnan(out), default_s, out)

    def asof(self, series: str, offset_us: int = 0) -> np.ndarray:
        q = self.trade_ts + offset_us
        if self._track:
            if self._bbo_ts.size == 0:
                self._note(np.full(self.n_trades, np.nan))
            else:
                j = np.searchsorted(self._bbo_ts, q, side="right") - 1
                self._note(np.where(j >= 0, self._bbo_ts[np.clip(j, 0, len(self._bbo_ts) - 1)], np.nan))
        return asof_backward(self._bbo_ts, self._bbo[series], q, exclude_beyond=offset_us > 0)

    def max_used_ts(self, feature) -> np.ndarray:
        """Run ``feature`` with accessor tracking on and return, per trade, the latest source
        timestamp it read (``-inf`` where it read nothing). A causal feature keeps this <= the
        trade time, so the validator's ``max_used_ts > trade_ts`` test never fires on it."""
        self._max_used = np.full(self.n_trades, -np.inf)
        self._track = True
        try:
            feature.calculate(self)
        finally:
            self._track = False
        return self._max_used.copy()
