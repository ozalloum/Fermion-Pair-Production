"""PairPulse: reproducible mode solvers for pulsed-field pair creation."""

from .pulses import ElectricPulse, PulseTrain, SauterPulse
from .solvers import ModeResult, integrated_density, sauter_exact_occupation, sauter_log_occupation
from .solvers import solve_dirac_mode, solve_qke_mode

__all__ = [
    "ElectricPulse", "ModeResult", "PulseTrain", "SauterPulse",
    "integrated_density", "sauter_exact_occupation", "sauter_log_occupation", "solve_dirac_mode",
    "solve_qke_mode",
]

