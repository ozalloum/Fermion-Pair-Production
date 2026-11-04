"""Analytic, spatially homogeneous electric-field backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


class ElectricPulse(Protocol):
    """Minimal interface expected by the mode solvers."""

    def field(self, t: float | np.ndarray) -> float | np.ndarray: ...

    def potential(self, t: float | np.ndarray) -> float | np.ndarray: ...

    def time_window(self, tail_factor: float = 12.0) -> tuple[float, float]: ...


@dataclass(frozen=True)
class SauterPulse:
    r"""A pulse ``E(t)=E0 sech²((t-center)/tau)`` in temporal gauge.

    The potential convention is ``E=-dA/dt`` and
    ``A(t)=-E0*tau*tanh((t-center)/tau)``.  ``E0`` is nonnegative; the
    charge sign can be absorbed into the momentum reflection symmetry.
    Natural units are used throughout.
    """

    amplitude: float
    duration: float
    center: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be nonnegative")
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if not np.isfinite(self.center):
            raise ValueError("center must be finite")

    def field(self, t: float | np.ndarray) -> float | np.ndarray:
        x = (np.asarray(t) - self.center) / self.duration
        z = np.exp(-2.0 * np.abs(x))
        ans = self.amplitude * (4.0 * z / (1.0 + z) ** 2)
        return float(ans) if np.ndim(ans) == 0 else ans

    def potential(self, t: float | np.ndarray) -> float | np.ndarray:
        x = (np.asarray(t) - self.center) / self.duration
        ans = -self.amplitude * self.duration * np.tanh(x)
        return float(ans) if np.ndim(ans) == 0 else ans

    def resolution_scale(self) -> float:
        """Shortest field-feature scale used for the automatic step cap."""
        return self.duration

    def time_window(self, tail_factor: float = 12.0) -> tuple[float, float]:
        if not np.isfinite(tail_factor) or tail_factor <= 0.0:
            raise ValueError("tail_factor must be positive")
        half_width = tail_factor * self.duration
        return self.center - half_width, self.center + half_width


@dataclass(frozen=True)
class PulseTrain:
    """A finite sum of Sauter pulses sharing the same spatially uniform axis."""

    pulses: tuple[SauterPulse, ...]

    def __init__(self, pulses: Iterable[SauterPulse]):
        object.__setattr__(self, "pulses", tuple(pulses))
        if not self.pulses:
            raise ValueError("a pulse train must contain at least one pulse")

    def field(self, t: float | np.ndarray) -> float | np.ndarray:
        ans = sum(p.field(t) for p in self.pulses)
        return float(ans) if np.ndim(ans) == 0 else ans

    def potential(self, t: float | np.ndarray) -> float | np.ndarray:
        ans = sum(p.potential(t) for p in self.pulses)
        return float(ans) if np.ndim(ans) == 0 else ans

    def resolution_scale(self) -> float:
        active = [p.duration for p in self.pulses if p.amplitude > 0]
        return min(active or [p.duration for p in self.pulses])

    def time_window(self, tail_factor: float = 12.0) -> tuple[float, float]:
        windows = [p.time_window(tail_factor) for p in self.pulses]
        return min(w[0] for w in windows), max(w[1] for w in windows)

