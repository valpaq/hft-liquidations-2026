"""Feature base class: the common calculate(...) interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .context import FeatureContext


class Feature(ABC):
    """One feature value per trade, computed causally from a FeatureContext. Subclasses set
    ``name``, ``lookback_us`` (causal horizon) and ``inputs``, and implement
    ``calculate(ctx) -> np.ndarray`` of length ``ctx.n_trades``."""
    name: str = ""
    lookback_us: int = 0
    inputs: tuple[str, ...] = ()

    @abstractmethod
    def calculate(self, ctx: FeatureContext) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, lookback_us={self.lookback_us})"
