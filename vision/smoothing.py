"""
vision/smoothing.py — Cursor Smoothing Algorithms

Provides the ExponentialMovingAverage (EMA) filter used to reduce
high-frequency jitter in the hand-cursor signal.

For stroke/tremor rehabilitation:
  - Lower alpha → more smoothing (good for severe tremor, less responsive)
  - Higher alpha → less smoothing (more responsive, shows true tremor)

No project-level imports — pure math only.
"""

from __future__ import annotations
from typing import Optional, Tuple


class ExponentialMovingAverage:
    """
    Two-dimensional Exponential Moving Average filter.

    The smoothed value is:
        x_smooth[t] = α * x_raw[t] + (1 - α) * x_smooth[t-1]

    Args:
        alpha: Smoothing factor in (0, 1].
               0.22 → strong smoothing (Easy / severe tremor)
               0.30 → moderate smoothing (Medium)
               0.40 → light smoothing   (Hard / assessment mode)
    """

    def __init__(self, alpha: float = 0.30) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self.alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Feed a new raw sample and return the smoothed (x, y).

        On the very first call the filter is seeded with the raw value
        (no lag on first detection).
        """
        if self._x is None:
            self._x = x
            self._y = y
        else:
            α = self.alpha
            self._x = α * x + (1 - α) * self._x
            self._y = α * y + (1 - α) * self._y
        return self._x, self._y

    def reset(self) -> None:
        """Clear filter state.  Call on game restart to avoid position lag."""
        self._x = None
        self._y = None

    @property
    def is_initialised(self) -> bool:
        return self._x is not None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        """Current smoothed value, or None if no samples have been fed."""
        if self._x is None:
            return None
        return self._x, self._y
