"""Independent mode-evolution formulations for fermion pair production."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, hypot, log, log1p, pi, sqrt
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from .pulses import ElectricPulse, SauterPulse


@dataclass(frozen=True)
class ModeResult:
    occupation: float
    diagnostic: float
    nfev: int
    time_span: tuple[float, float]


def _validate_mode(momentum: float, mass: float, charge: float) -> None:
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be positive")
    if not np.isfinite(charge) or charge <= 0.0:
        raise ValueError("charge magnitude must be positive")
    if not np.isfinite(momentum):
        raise ValueError("momentum must be finite")


def _instantaneous_kinematics(t: float, p: float, m: float,
                              q: float, pulse: ElectricPulse) -> tuple[float, float]:
    kinetic_p = p - q * float(pulse.potential(t))
    omega = hypot(m, kinetic_p)
    return kinetic_p, omega


def _time_window(pulse, tail_factor):
    t0, tf = map(float, pulse.time_window(tail_factor))
    if not np.isfinite([t0, tf]).all() or t0 >= tf:
        raise ValueError("time_window must return finite, strictly increasing endpoints")
    return t0, tf


def _maximum_step(t0: float, tf: float, p: float, m: float, q: float,
                  pulse: ElectricPulse, coefficient: float = 0.18,
                  max_step: float | None = None) -> float:
    # The phase scale alone cannot resolve a narrow source inside a wide window.
    samples = np.linspace(t0, tf, 257)
    potentials = np.array([float(pulse.potential(float(t))) for t in samples])
    if not np.isfinite(potentials).all():
        raise ValueError("pulse potential must be finite on the integration window")
    pi_max = np.max(np.abs(p - q * potentials))
    cap = coefficient / hypot(m, pi_max)
    feature = getattr(pulse, "resolution_scale", None)
    if feature is not None:
        scale = float(feature())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("resolution_scale() must be finite and positive")
        cap = min(cap, scale / 8.0)
    elif max_step is None:
        warnings.warn("Custom pulse has no resolution_scale(); supply max_step "
                      "to resolve its shortest field feature. The phase cap alone "
                      "can miss a pulse.", UserWarning, stacklevel=3)
    if max_step is not None:
        cap = min(cap, max_step)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("parameters exceed the supported finite step-size range")
    return cap


def _validate_controls(tail_factor, rtol, atol, max_step_coefficient):
    for name, value in (("tail_factor", tail_factor), ("rtol", rtol),
                        ("atol", atol), ("max_step_coefficient", max_step_coefficient)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def solve_dirac_mode(
    momentum: float,
    pulse: ElectricPulse,
    *,
    mass: float = 1.0,
    charge: float = 1.0,
    tail_factor: float = 12.0,
    rtol: float = 2e-10,
    atol: float = 2e-12,
    max_step_coefficient: float = 0.18,
    max_step: float | None = None,
) -> ModeResult:
    r"""Evolve the two-component Dirac mode and project onto the out state.

    The Hamiltonian is ``H(t)=pi(t) sigma_1 + m sigma_3``, with
    ``pi=p-q A(t)``.  Initial and final instants are finite, placed
    ``tail_factor`` pulse widths beyond the outer pulse centers.
    """
    _validate_mode(momentum, mass, charge)
    _validate_controls(tail_factor, rtol, atol, max_step_coefficient)
    if max_step is not None and (not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("max_step must be finite and positive, or None")
    t0, tf = _time_window(pulse, tail_factor)
    pi0, _ = _instantaneous_kinematics(t0, momentum, mass, charge, pulse)
    pif, _ = _instantaneous_kinematics(tf, momentum, mass, charge, pulse)

    def eigenvectors(kinetic_p: float) -> tuple[np.ndarray, np.ndarray]:
        theta = np.arctan2(kinetic_p, mass)
        positive = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=complex)
        negative = np.array([-np.sin(theta / 2.0), np.cos(theta / 2.0)], dtype=complex)
        return positive, negative

    _, initial = eigenvectors(pi0)
    positive_out, _ = eigenvectors(pif)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        kinetic_p = momentum - charge * float(pulse.potential(t))
        return -1j * np.array([
            mass * y[0] + kinetic_p * y[1],
            kinetic_p * y[0] - mass * y[1],
        ])

    sol = solve_ivp(rhs, (t0, tf), initial, method="DOP853", rtol=rtol,
                    atol=atol, max_step=_maximum_step(t0, tf, momentum, mass, charge, pulse, max_step_coefficient, max_step))
    if not sol.success:
        raise RuntimeError(f"Dirac mode integration failed: {sol.message}")
    state = sol.y[:, -1]
    if not np.isfinite(state).all():
        raise RuntimeError("Dirac integration returned a nonfinite state")
    occupation = float(abs(np.vdot(positive_out, state)) ** 2)
    norm_error = float(abs(np.vdot(state, state).real - 1.0))
    return ModeResult(occupation, norm_error, int(sol.nfev), (float(t0), float(tf)))


def solve_qke_mode(
    momentum: float,
    pulse: ElectricPulse,
    *,
    mass: float = 1.0,
    charge: float = 1.0,
    tail_factor: float = 12.0,
    rtol: float = 2e-10,
    atol: float = 2e-12,
    max_step_coefficient: float = 0.18,
    max_step: float | None = None,
) -> ModeResult:
    r"""Integrate the non-Markovian quantum-kinetic mode equations.

    For ``(f,u,v)`` the equations are

    ``f_dot = W u/2``, ``u_dot = W(1-2f)-2 omega v``, and
    ``v_dot = 2 omega u``, where ``W=q E m / omega²``.  The spinor
    representation and this system are integrated independently.
    """
    _validate_mode(momentum, mass, charge)
    _validate_controls(tail_factor, rtol, atol, max_step_coefficient)
    if max_step is not None and (not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("max_step must be finite and positive, or None")
    t0, tf = _time_window(pulse, tail_factor)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        kinetic_p, omega = _instantaneous_kinematics(t, momentum, mass, charge, pulse)
        electric = float(pulse.field(t))
        coupling = charge * electric * mass / (omega * omega)
        f, u, v = y
        return np.array([
            0.5 * coupling * u,
            coupling * (1.0 - 2.0 * f) - 2.0 * omega * v,
            2.0 * omega * u,
        ])

    sol = solve_ivp(rhs, (t0, tf), np.zeros(3), method="DOP853", rtol=rtol,
                    atol=atol, max_step=_maximum_step(t0, tf, momentum, mass, charge, pulse, max_step_coefficient, max_step))
    if not sol.success:
        raise RuntimeError(f"QKE mode integration failed: {sol.message}")
    f, u, v = sol.y[:, -1]
    if not np.isfinite([f, u, v]).all():
        raise RuntimeError("QKE integration returned a nonfinite state")
    invariant = (1.0 - 2.0 * f) ** 2 + u * u + v * v
    return ModeResult(float(f), float(abs(invariant - 1.0)), int(sol.nfev),
                      (float(t0), float(tf)))


def _log_one_minus_exp_minus_two_x(log_x: float) -> float:
    """log(1-exp(-2*x)) given log(x), without small-x cancellation."""
    if log_x < -350.0:
        return log(2.0) + log_x
    if log_x > 5.0:
        return 0.0  # omitted term is smaller than 1e-128
    return log(-float(np.expm1(-2.0 * exp(log_x))))


def sauter_log_occupation(momentum: float, pulse: SauterPulse, *,
                         mass: float = 1.0, charge: float = 1.0) -> float:
    r"""Natural log of the exact one-channel asymptotic Sauter occupation.

    Energy gaps are rationalized; the dominant exponential and small sinh
    factors are evaluated separately. -inf denotes zero field or a decay
    exponent beyond the floating-point range. Unrepresentable kinematics
    raise ValueError rather than being silently converted to zero.
    """
    _validate_mode(momentum, mass, charge)
    if not isinstance(pulse, SauterPulse):
        raise TypeError("the exact benchmark requires a single SauterPulse")
    if pulse.amplitude == 0.0:
        return -np.inf
    r, tau = abs(momentum), pulse.duration
    log_a = log(charge) + log(pulse.amplitude) + log(tau)
    a = charge * pulse.amplitude * tau
    if not np.isfinite(a):
        if log_a > log(np.finfo(float).max):
            raise ValueError("Sauter impulse exceeds the supported float64 range")
        a = exp(log_a)
    if a == 0.0:
        a = exp(log_a)  # underflow here is harmless; log_a retains the scale
    plus, minus = r + a, abs(r - a)
    wp, wm = hypot(mass, plus), hypot(mass, minus)
    total = wp + wm
    denominators = (wp + plus, wm + minus)
    if not np.isfinite([plus, wp, wm, total, *denominators]).all():
        raise ValueError("Sauter kinematics exceed the supported float64 range")
    # total - 2*max(r,a) = m^2/(wp+r+a) + m^2/(wm+abs(r-a)).
    log_delta = float(np.logaddexp(2*log(mass)-log(denominators[0]),
                                  2*log(mass)-log(denominators[1])))
    def log_gap(offset):
        return float(np.logaddexp(log(2.0)+log(offset) if offset > 0 else -np.inf,
                                 log_delta))
    log_gap_p = log_gap(max(a-r, 0.0))  # total - 2*r
    log_gap_a = log_gap(max(r-a, 0.0))  # total - 2*a
    scale = log(pi) + log(tau)
    log_decay = scale + log_gap_a
    if log_decay > log(np.finfo(float).max):
        return -np.inf
    leading = -exp(log_decay)
    log_small = scale + log_a + log_gap_p - log(total)
    log_large = scale + log_a + log1p(2.0*(r/total))
    value = fsum([leading,
        _log_one_minus_exp_minus_two_x(log_small),
        _log_one_minus_exp_minus_two_x(log_large),
        -_log_one_minus_exp_minus_two_x(scale+log(wp)),
        -_log_one_minus_exp_minus_two_x(scale+log(wm))])
    if not np.isfinite(value) or value > 64*np.finfo(float).eps:
        raise ArithmeticError("invalid Sauter log occupation; use higher precision")
    return min(value, 0.0)  # only a sub-roundoff positive residual is rounded down


def sauter_exact_occupation(momentum: float, pulse: SauterPulse, *,
                            mass: float = 1.0, charge: float = 1.0) -> float:
    """Exact asymptotic occupation; use sauter_log_occupation for tiny yields."""
    return exp(sauter_log_occupation(momentum, pulse, mass=mass, charge=charge))


def integrated_density(momentum: np.ndarray, occupation: np.ndarray, *,
                       check_bounds: bool = False, bounds_tolerance: float = 1e-10) -> float:
    """Integrate a one-dimensional momentum spectrum with measure ``dp/(2 pi)``."""
    p = np.asarray(momentum, dtype=float)
    f = np.asarray(occupation, dtype=float)
    if p.ndim != 1 or f.shape != p.shape or p.size < 2:
        raise ValueError("momentum and occupation must be equal-length 1D arrays")
    if not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError("momentum and occupation must contain only finite values")
    if not np.isfinite(bounds_tolerance) or bounds_tolerance < 0:
        raise ValueError("bounds_tolerance must be finite and nonnegative")
    if check_bounds and np.any((f < -bounds_tolerance) | (f > 1+bounds_tolerance)):
        raise ValueError("occupation lies outside the fermionic bounds")
    order = np.argsort(p)
    p_sorted, f_sorted = p[order], f[order]
    if np.any(np.diff(p_sorted) <= 0):
        raise ValueError("momentum nodes must be unique")
    integral = np.sum(0.5 * (f_sorted[1:] + f_sorted[:-1]) * np.diff(p_sorted))
    return float(integral / (2.0 * pi))
