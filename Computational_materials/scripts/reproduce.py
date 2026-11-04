#!/usr/bin/env python3
"""Recreate the figures and CSV tables used in the PairPulse manuscript."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from pairpulse import (PulseTrain, SauterPulse, integrated_density,
                       sauter_exact_occupation, solve_dirac_mode,
                       solve_qke_mode)


def save_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def spectrum(pulse, momenta: np.ndarray, *, exact: bool = False):
    dirac, qke, exact_values, inv, norm, nfev = [], [], [], [], [], []
    for p in momenta:
        d = solve_dirac_mode(float(p), pulse)
        k = solve_qke_mode(float(p), pulse)
        dirac.append(d.occupation)
        qke.append(k.occupation)
        inv.append(k.diagnostic)
        norm.append(d.diagnostic)
        nfev.append(d.nfev + k.nfev)
        exact_values.append(sauter_exact_occupation(float(p), pulse)
                            if exact and isinstance(pulse, SauterPulse) else np.nan)
    return tuple(np.asarray(v, dtype=float) for v in
                 (dirac, qke, exact_values, inv, norm, nfev))


# Compact axis and legend typography at the manuscript inclusion width.
SPECTRUM_FONT_SIZE = 20 * (0.94 / 10.8) / (0.78 / 6.3) * 0.80
LEGEND_FONT_SIZE = 20 * (0.94 / 10.8) / (0.78 / 6.3) * 0.70

def _style_spectrum_axes(ax) -> None:
    """Use reduced labels and ticks with compact legends for submission."""
    ax.xaxis.label.set_size(SPECTRUM_FONT_SIZE)
    ax.yaxis.label.set_size(SPECTRUM_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=SPECTRUM_FONT_SIZE)
    ax.xaxis.get_offset_text().set_size(SPECTRUM_FONT_SIZE)
    ax.yaxis.get_offset_text().set_size(SPECTRUM_FONT_SIZE)
    # Freeze a padded layout so PDF and PNG backends use the same positions.
    fig = ax.figure
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    fig.canvas.draw()
    fig.canvas.draw()
    fig.set_layout_engine("none")


def plot_spectra(output: Path, p_single, d, k, exact, inv,
                 p_double, dd, dk) -> None:
    """Draw Figures 1--3 from supplied arrays without recomputing the data."""
    # Figure 1: cross-validation against an analytic result.
    fig, ax = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    ax.semilogy(p_single, exact, color="black", lw=2.0, label="exact Sauter result")
    ax.semilogy(p_single, d, "o", ms=3.5, mfc="none", label="Dirac modes")
    ax.semilogy(p_single, k, "x", ms=4.2, label="quantum-kinetic modes")
    ax.set(xlabel=r"canonical momentum $p/m$", ylabel=r"occupation $f_p$")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), borderaxespad=0,
              handlelength=1.5, labelspacing=0.15, handletextpad=0.6, columnspacing=1.2, ncol=3)
    _style_spectrum_axes(ax)
    fig.savefig(output / "figure1_sauter_validation.pdf")
    fig.savefig(output / "figure1_sauter_validation.png", dpi=300)
    plt.close(fig)

    # Figure 2: spectra for a pulse train; no exact formula is assumed.
    fig, ax = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    ax.plot(p_double, dd, "o-", ms=3.2, lw=1.1, label="Dirac modes")
    ax.plot(p_double, dk, "x--", ms=3.6, lw=1.0, label="quantum-kinetic modes")
    ax.set(xlabel=r"canonical momentum $p/m$", ylabel=r"occupation $f_p$")
    ax.set_yticks(np.arange(0, 0.014, 0.002))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), borderaxespad=0,
              handlelength=1.5, labelspacing=0.15, handletextpad=0.6, columnspacing=1.2, ncol=2)
    _style_spectrum_axes(ax)
    fig.savefig(output / "figure2_double_pulse.pdf")
    fig.savefig(output / "figure2_double_pulse.png", dpi=300)
    plt.close(fig)

    # Figure 3: the QKE Bloch-vector invariant, sampled over the spectrum.
    fig, ax = plt.subplots(figsize=(6.3, 4.6), constrained_layout=True)
    ax.semilogy(p_single, np.maximum(inv, 1e-18), "o-", ms=3.4)
    ax.set(xlabel=r"canonical momentum $p/m$", ylabel="absolute invariant error")
    _style_spectrum_axes(ax)
    fig.savefig(output / "figure3_qke_invariant.pdf")
    fig.savefig(output / "figure3_qke_invariant.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--points", type=int, default=31)
    args = parser.parse_args()
    if args.points < 9:
        parser.error("--points must be at least 9")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # Exact single-pulse benchmark.
    single = SauterPulse(amplitude=0.3, duration=2.0)
    p_single = np.linspace(-1.5, 1.5, args.points)
    d, k, exact, inv, norm, nfev = spectrum(single, p_single, exact=True)
    save_csv(args.output / "sauter_spectrum.csv",
             ["momentum", "dirac", "qke", "exact", "qke_invariant_error", "dirac_norm_error", "nfev_total"],
             list(zip(p_single, d, k, exact, inv, norm, nfev)))
    abs_d = np.abs(d - exact)
    abs_k = np.abs(k - exact)
    resolved = exact > 1e-12
    rel_d = abs_d[resolved] / exact[resolved]
    rel_k = abs_k[resolved] / exact[resolved]

    # Coherent double-pulse illustration: same-polarity pulses are separated in time.
    double = PulseTrain((SauterPulse(0.30, 1.1, -3.5),
                         SauterPulse(0.30, 1.1, 3.5)))
    p_double = np.linspace(-1.8, 1.8, args.points)
    dd, dk, _, dinv, dnorm, dnfev = spectrum(double, p_double)
    save_csv(args.output / "double_pulse_spectrum.csv",
             ["momentum", "dirac", "qke", "qke_invariant_error", "dirac_norm_error", "nfev_total"],
             list(zip(p_double, dd, dk, dinv, dnorm, dnfev)))

    yields = [
        ("single_sauter_dirac", integrated_density(p_single, d)),
        ("single_sauter_qke", integrated_density(p_single, k)),
        ("single_sauter_exact", integrated_density(p_single, exact)),
        ("double_sauter_dirac", integrated_density(p_double, dd)),
        ("double_sauter_qke", integrated_density(p_double, dk)),
    ]
    save_csv(args.output / "integrated_yields.csv", ["case", "density_per_length"], yields)

    # Tail convergence at p=0, where the exact answer is available.
    tail_rows = []
    for factor in (6.0, 8.0, 10.0, 12.0, 14.0):
        e = sauter_exact_occupation(0.0, single)
        td = solve_dirac_mode(0.0, single, tail_factor=factor)
        tq = solve_qke_mode(0.0, single, tail_factor=factor)
        tail_rows.append((factor, e, td.occupation, tq.occupation,
                          abs(td.occupation-e), abs(tq.occupation-e),
                          td.diagnostic, tq.diagnostic, td.nfev, tq.nfev))
    save_csv(args.output / "tail_convergence.csv",
             ["tail_factor", "exact", "dirac", "qke", "dirac_abs_error", "qke_abs_error",
              "dirac_norm_error", "qke_invariant_error", "dirac_nfev", "qke_nfev"],
             tail_rows)

    plot_spectra(args.output, p_single, d, k, exact, inv, p_double, dd, dk)

    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": plt.matplotlib.__version__,
        "platform": platform.platform(), "points_per_spectrum": args.points,
        "elapsed_seconds": time.perf_counter() - started,
        "max_absolute_error_dirac_exact": float(np.max(abs_d)),
        "max_absolute_error_qke_exact": float(np.max(abs_k)),
        "max_relative_error_dirac_exact_resolved_modes": float(np.max(rel_d)),
        "max_relative_error_qke_exact_resolved_modes": float(np.max(rel_k)),
        "max_qke_invariant_error_single": float(np.max(inv)),
        "max_dirac_norm_error_single": float(np.max(norm)),
        "max_qke_invariant_error_double": float(np.max(dinv)),
        "max_dirac_norm_error_double": float(np.max(dnorm)),
        "sum_nfev_single_spectrum": int(np.sum(nfev)),
        "sum_nfev_double_spectrum": int(np.sum(dnfev)),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
