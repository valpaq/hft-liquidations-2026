"""Concrete example features. Each is a thin, causal composition of FeatureContext methods.

Adding a feature = one small class; the pipeline and validators pick it up automatically.
"""
from __future__ import annotations
import numpy as np

from .base import Feature
from .context import FeatureContext

S = 1_000_000


class LiqNotional(Feature):
    inputs = ("liq_combined",)

    def __init__(self, window_s: int):
        self.window_s = window_s
        self.name = f"liq_notional_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.window_sum("liq_combined", "notional", self.lookback_us)


class LiqEventCount(Feature):
    inputs = ("liq_combined",)

    def __init__(self, window_s: int):
        self.name = f"liq_event_count_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.window_count("liq_combined", self.lookback_us)


class LiqSideImbalance(Feature):
    inputs = ("liq_combined",)

    def __init__(self, window_s: int):
        self.name = f"liq_side_imbalance_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.window_side_imbalance("liq_combined", self.lookback_us)


class LiqStrength(Feature):
    """Signed log-strength of the net liquidation flow: sign(buy-sell)*log1p(|buy-sell|).
    The log compresses the heavy notional tail. Not in DEFAULT_FEATURES; an available extra."""
    inputs = ("liq_combined",)

    def __init__(self, window_s: int = 30):
        self.name = f"liq_strength_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        buy = ctx.window_sum("liq_combined", "buy_notional", self.lookback_us)
        sell = ctx.window_sum("liq_combined", "sell_notional", self.lookback_us)
        delta = buy - sell
        return np.sign(delta) * np.log1p(np.abs(delta))


class Exchange2LiqNotional(Feature):
    inputs = ("liq_exchange_2",)

    def __init__(self, window_s: int):
        self.name = f"exchange_2_liq_notional_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.window_sum("liq_exchange_2", "notional", self.lookback_us)


class LiqVelocity(Feature):
    inputs = ("liq_combined",)

    def __init__(self, short_s: int = 30, long_s: int = 120):
        self.short_us, self.long_us = short_s * S, long_s * S
        self.name = "liq_velocity"
        self.lookback_us = long_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        short = ctx.window_sum("liq_combined", "notional", self.short_us)
        long = ctx.window_sum("liq_combined", "notional", self.long_us)
        return short / (long + 1.0)


class TimeSinceLiq(Feature):
    inputs = ("liq_combined",)
    name = "time_since_liq_s"
    lookback_us = 0

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.time_since_last("liq_combined")


class BBOSpread(Feature):
    inputs = ("bbo",)
    name = "bbo_spread_bps"

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.asof("spread_bps")


class BBOImbalance(Feature):
    inputs = ("bbo",)
    name = "bbo_imbalance"

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        return ctx.asof("imbalance")


class MidReturn(Feature):
    inputs = ("bbo",)

    def __init__(self, window_s: int = 5):
        self.window_us = window_s * S
        self.name = f"mid_ret_{window_s}s"
        self.lookback_us = window_s * S

    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        now = ctx.asof("mid", 0)
        past = ctx.asof("mid", -self.window_us)
        return (now - past) / past * 1e4


DEFAULT_FEATURES = [
    LiqNotional(30), LiqNotional(120), LiqEventCount(30), LiqSideImbalance(30),
    Exchange2LiqNotional(30), LiqVelocity(30, 120), TimeSinceLiq(),
    BBOSpread(), BBOImbalance(), MidReturn(5),
]
