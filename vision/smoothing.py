"""
vision/smoothing.py — Coordinate Smoothing Algorithms & Low-Pass Filters

Provides robust, low-latency coordinate smoothing filters for hand/cursor tracking:
  1. OneEuroFilter2D: Speed-adaptive low-pass filter (Casiez et al., CHI 2012)
     - High smoothing at low speeds -> eliminates camera jitter & tremors
     - Low smoothing at high speeds -> zero perceptible latency during fast gestures
  2. ExponentialMovingAverage (EMA): Fixed-rate 2D low-pass filter
     - Configurable alpha factor

Both filters strictly preserve and expose:
  - raw_position:      (rx, ry) — latest raw, unfiltered coordinate sample
  - smoothed_position: (sx, sy) — latest filtered coordinate sample

No project-level imports — pure math and standard library only.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple


def _alpha(cutoff: float, dt: float) -> float:
    """Compute the low-pass filter alpha factor from cutoff frequency (Hz) and delta time (s)."""
    if cutoff <= 0.0 or dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class LowPassFilter1D:
    """Single-channel first-order low-pass filter."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha: float = alpha
        self._s: Optional[float] = None

    def filter(self, val: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = max(0.0, min(1.0, alpha))
        if self._s is None:
            self._s = float(val)
        else:
            self._s = self.alpha * float(val) + (1.0 - self.alpha) * self._s
        return self._s

    def reset(self) -> None:
        self._s = None

    @property
    def value(self) -> Optional[float]:
        return self._s


class OneEuroFilter1D:
    """
    Single-channel One-Euro Filter (speed-adaptive low-pass filter).

    Args:
        min_cutoff: Minimum cutoff frequency in Hz (calms jitter at rest).
        beta:       Speed coefficient (reduces lag during fast movement).
        d_cutoff:   Cutoff frequency for the derivative filter in Hz.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta:       float = 0.007,
        d_cutoff:   float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)

        self._x_filter = LowPassFilter1D()
        self._dx_filter = LowPassFilter1D()
        self._last_raw: Optional[float] = None

    def filter(self, val: float, dt: float = 1.0 / 30.0) -> float:
        if dt <= 0.0:
            dt = 1.0 / 30.0

        # Estimate derivative (speed)
        if self._last_raw is None:
            edx = 0.0
        else:
            dx = (val - self._last_raw) / dt
            alpha_d = _alpha(self.d_cutoff, dt)
            edx = self._dx_filter.filter(dx, alpha_d)

        self._last_raw = float(val)

        # Dynamic cutoff based on movement speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = _alpha(cutoff, dt)
        return self._x_filter.filter(val, alpha)

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_raw = None


class OneEuroFilter2D:
    """
    Two-dimensional speed-adaptive One-Euro Filter for (X, Y) coordinate streams.

    Combines strong jitter suppression when stationary with immediate responsiveness
    during rapid intentional motion.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta:       float = 0.007,
        d_cutoff:   float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta       = float(beta)
        self.d_cutoff   = float(d_cutoff)

        self._fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff)

        self._raw_pos:      Optional[Tuple[float, float]] = None
        self._smoothed_pos: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float, dt: float = 1.0 / 30.0) -> Tuple[float, float]:
        """
        Feed a new raw coordinate sample and return the smoothed (X, Y).
        """
        rx = float(x)
        ry = float(y)
        self._raw_pos = (rx, ry)

        sx = self._fx.filter(rx, dt)
        sy = self._fy.filter(ry, dt)
        self._smoothed_pos = (sx, sy)
        return sx, sy

    def reset(self) -> None:
        """Clear filter history (call on game restart or hand re-entry)."""
        self._fx.reset()
        self._fy.reset()
        self._raw_pos = None
        self._smoothed_pos = None

    @property
    def raw_position(self) -> Optional[Tuple[float, float]]:
        """Latest unfiltered coordinate sample."""
        return self._raw_pos

    @property
    def smoothed_position(self) -> Optional[Tuple[float, float]]:
        """Latest smoothed coordinate sample."""
        return self._smoothed_pos

    @property
    def is_initialised(self) -> bool:
        return self._smoothed_pos is not None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        """Alias for smoothed_position."""
        return self._smoothed_pos


class ExponentialMovingAverage:
    """
    Two-dimensional Exponential Moving Average filter.

    Formula:
        x_smooth[t] = α * x_raw[t] + (1 - α) * x_smooth[t-1]
    """

    def __init__(self, alpha: float = 0.30) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self.alpha = float(alpha)
        self._raw_pos:      Optional[Tuple[float, float]] = None
        self._smoothed_pos: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Feed a new raw sample and return the smoothed (x, y).
        dt is accepted for interface compatibility with OneEuroFilter2D.
        """
        rx = float(x)
        ry = float(y)
        self._raw_pos = (rx, ry)

        if self._smoothed_pos is None:
            self._smoothed_pos = (rx, ry)
        else:
            α = self.alpha
            sx = α * rx + (1.0 - α) * self._smoothed_pos[0]
            sy = α * ry + (1.0 - α) * self._smoothed_pos[1]
            self._smoothed_pos = (sx, sy)

        return self._smoothed_pos

    def reset(self) -> None:
        """Clear filter state. Call on game restart to avoid position lag."""
        self._raw_pos = None
        self._smoothed_pos = None

    @property
    def raw_position(self) -> Optional[Tuple[float, float]]:
        """Latest unfiltered coordinate sample."""
        return self._raw_pos

    @property
    def smoothed_position(self) -> Optional[Tuple[float, float]]:
        """Latest smoothed coordinate sample."""
        return self._smoothed_pos

    @property
    def is_initialised(self) -> bool:
        return self._smoothed_pos is not None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        """Current smoothed value, or None if no samples have been fed."""
        return self._smoothed_pos


def create_smoother(
    algorithm: Optional[str] = None,
    alpha: float = 0.30,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> OneEuroFilter2D | ExponentialMovingAverage:
    """
    Factory helper to instantiate the configured coordinate smoother.
    """
    algo = (algorithm or "ONE_EURO").upper()
    if algo == "EMA":
        return ExponentialMovingAverage(alpha=alpha)
    return OneEuroFilter2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
