#!/usr/bin/env python3
"""Generate the package/workflow diagrams and the finite-window error map.

All graphics use Matplotlib and are saved as vector PDF and raster PNG. The
error-map CSV records every mode calculation behind the heat map.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LogNorm

from pairpulse import SauterPulse, sauter_exact_occupation, solve_dirac_mode, solve_qke_mode


INK = "#18324b"
BLUE = "#dceaf5"
TEAL = "#dcefe9"
GOLD = "#f8edcf"
PALE = "#f4f6f8"
MUTED = "#556575"


def _box(ax, xy, width, height, title, body, color=BLUE, *, title_size=11,
         body_size=9, mono=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor=INK, linewidth=1.05,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.035, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + 0.025, y + height - 0.085, body, transform=ax.transAxes,
            ha="left", va="top", fontsize=body_size, color=INK,
            family="monospace" if mono else "sans-serif", linespacing=1.35)


def _arrow(ax, start, end, *, color=INK, rad=0.0, lw=1.5):
    arrow = FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=10, linewidth=lw, color=color,
        shrinkA=0, shrinkB=0, capstyle="round", joinstyle="round",
        connectionstyle=f"arc3,rad={rad}", clip_on=False, zorder=3,
    )
    ax.add_patch(arrow)


def make_structure_figure(output: Path, project_root: Path) -> None:
    """Draw the public package tree as artifact groups tied to real paths."""
    required = (
        "src/pairpulse/pulses.py", "src/pairpulse/solvers.py",
        "tests/test_pairpulse.py", "notebooks/PairPulse_reproduction.ipynb",
        "scripts/reproduce.py", "scripts/make_reproducibility_figures.py",
        "REPRODUCIBILITY.txt", "README.md", "pyproject.toml",
        "LICENSE", "CHECKSUMS.sha256",
    )
    missing = [rel for rel in required if not (project_root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Package structure has missing files: {missing}")

    # Fixed-width vector artwork: larger box text remains larger when embedded.
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    ax.text(0.5, 0.98, "Reproducibility materials",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.5, 0.925,
            "Mode solvers, validation, numerical records, and figures",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color=MUTED)

    root_patch = FancyBboxPatch((0.31, 0.775), 0.38, 0.115,
        boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=GOLD,
        edgecolor=INK, linewidth=1.3, transform=ax.transAxes)
    ax.add_patch(root_patch)
    ax.text(0.5, 0.87, "Computational_materials/", fontsize=10.5, weight="bold",
            color=INK, ha="center", va="top", transform=ax.transAxes)
    ax.text(0.5, 0.82, "MIT-licensed source archive", fontsize=10,
            color=INK, ha="center", va="top", transform=ax.transAxes)

    groups = (
        (0.015, BLUE, "Source and\nverification",
         "src/pairpulse/\n  pulses.py: E(t), A(t)\n  solvers.py: Dirac, QKE\n  Exact Sauter benchmark\n\n"
         "tests/test_pairpulse.py\nnotebooks/\n  PairPulse_reproduction.ipynb\n\n"
         "README, pyproject, LICENSE"),
        (0.350, TEAL, "Scripts and\noutputs",
         "scripts/reproduce.py\n  Spectra, yields, diagnostics\n\n"
         "scripts/\nmake_reproducibility_figures.py\n  Diagrams and error map\n\n"
         "results/\n  Figures: PDF and PNG\nGenerated: CSV, JSON"),
        (0.685, GOLD, "Reproduction and\nchecks",
         "REPRODUCIBILITY.txt\n  Execution notes\n\n"
         "scripts/validate_study.py\n  Refinements and sweeps\n\n"
         "scripts/verify_checksums.py\nCHECKSUMS.sha256"),
    )
    for x, color, title, body in groups:
        patch = FancyBboxPatch((x, 0.165), 0.30, 0.515,
            boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=color,
            edgecolor=INK, linewidth=1.3, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + 0.010, 0.65, title, ha="left", va="top", fontsize=12.5,
                fontweight="bold", color=INK, transform=ax.transAxes,
                linespacing=1.15)
        ax.text(x + 0.010, 0.555, body, ha="left", va="top", fontsize=10.5,
                fontstretch="condensed", color=INK,
                transform=ax.transAxes, linespacing=1.45)
    ax.plot([0.5, 0.5], [0.765, 0.73], color=INK, lw=1.3, transform=ax.transAxes)
    ax.plot([0.165, 0.835], [0.73, 0.73], color=INK, lw=1.3, transform=ax.transAxes)
    for x in (0.165, 0.5, 0.835):
        _arrow(ax, (x, 0.73), (x, 0.69))
    ax.text(0.5, 0.11, "Reproduce: Python scripts → datasets and figures → checksums",
            ha="center", va="center", fontsize=10.5, color=INK,
            fontweight="semibold", transform=ax.transAxes)
    ax.text(0.5, 0.05, "Included scripts generate numerical datasets and figures.",
            ha="center", va="center", fontsize=10, color=MUTED, transform=ax.transAxes)
    fig.savefig(output / "reproducibility_structure.pdf")
    fig.savefig(output / "reproducibility_structure.png", dpi=300)
    plt.close(fig)


def make_workflow_figure(output: Path) -> None:
    """Draw the supplied six-stage portrait workflow from editable primitives."""
    fig, ax = plt.subplots(figsize=(7.8, 10.0))
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    ax.text(0.5, 0.975, "Computational workflow", ha="center", va="top",
            fontsize=19, weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.927, "Shared field conventions; separate state equations",
            ha="center", va="top", fontsize=12, color=MUTED, transform=ax.transAxes)
    boxes = [
        (0.03, 0.67, "1  Set parameters", "Pulse parameters and units\nMomentum grid p/m\nTail extent L/τ\nSolver tolerances", GOLD),
        (0.03, 0.39, "2  Define the field", "Sauter pulse or pulse train\nE(t), A(t), finite window", BLUE),
        (0.03, 0.11, "3  Solve each mode", "Dirac spinor: DOP853\nQKE state (f,u,v): DOP853\nSeparate integrations", TEAL),
        (0.56, 0.67, "4  Validate each run", "Exact Sauter benchmark\nFinal norm and invariant\nDirac–QKE agreement\nTail-window scan", BLUE),
        (0.56, 0.39, "5  Save datasets", "Spectrum and yield CSV\nTail and p–L/τ map CSV\nRun metadata JSON", TEAL),
        (0.56, 0.11, "6  Plot and verify", "Python/Matplotlib figures\nPDF and PNG outputs\nPackage checksums", GOLD),
    ]
    for x, y, title, body, color in boxes:
        _box(ax, (x,y), 0.41, 0.22, title, body, color=color,
             title_size=13.5, body_size=12)
    for x in (0.235, 0.765):
        _arrow(ax, (x, 0.657), (x, 0.625))
        _arrow(ax, (x, 0.377), (x, 0.345))
    ax.plot([0.453, 0.5, 0.5], [0.22, 0.22, 0.78], color=INK,
            lw=1.5, transform=ax.transAxes)
    _arrow(ax, (0.5, 0.78), (0.547, 0.78))
    ax.text(0.5, 0.064, "reproduce.py · make_reproducibility_figures.py",
            ha="center", va="center", fontsize=10, color=MUTED,
            transform=ax.transAxes)
    ax.text(0.5, 0.032, "redraw_figures.py · verify_checksums.py",
            ha="center", va="center", fontsize=10, color=MUTED,
            transform=ax.transAxes)
    fig.savefig(output / "computational_workflow.pdf")
    fig.savefig(output / "computational_workflow.png", dpi=300)
    plt.close(fig)


def make_window_error_map(output: Path, *, points: int,
                          tail_factors: tuple[float, ...]) -> dict[str, float]:
    """Compute and plot absolute finite-window errors for the exact benchmark."""
    pulse = SauterPulse(amplitude=0.3, duration=2.0)
    momenta = np.linspace(-1.5, 1.5, points)
    rows: list[tuple[float, ...]] = []
    error_dirac = np.empty((len(tail_factors), points))
    error_qke = np.empty_like(error_dirac)
    for i, factor in enumerate(tail_factors):
        for j, momentum in enumerate(momenta):
            exact = sauter_exact_occupation(float(momentum), pulse)
            d = solve_dirac_mode(float(momentum), pulse, tail_factor=factor)
            k = solve_qke_mode(float(momentum), pulse, tail_factor=factor)
            error_dirac[i, j] = abs(d.occupation - exact)
            error_qke[i, j] = abs(k.occupation - exact)
            rows.append((momentum, factor, exact, d.occupation, k.occupation,
                         error_dirac[i, j], error_qke[i, j],
                         d.diagnostic, k.diagnostic, d.nfev, k.nfev))

    csv_path = output / "sauter_window_error_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("momentum", "tail_extent_over_tau", "exact_asymptotic",
                         "dirac_finite_window", "qke_finite_window",
                         "dirac_absolute_error", "qke_absolute_error",
                         "dirac_norm_error", "qke_invariant_error",
                         "dirac_nfev", "qke_nfev"))
        writer.writerows(rows)

    return plot_window_error_map(output, momenta, tail_factors, error_dirac, error_qke)


def plot_window_error_map(output: Path, momenta, tail_factors,
                          error_dirac, error_qke) -> dict:
    """Draw Figure 4 from supplied errors without new integrations."""
    # A floor is needed only to display roundoff-level values on logarithmic axes.
    floor = 1e-16
    ceiling = max(float(error_dirac.max()), float(error_qke.max()))
    norm = LogNorm(vmin=floor, vmax=ceiling)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, err, title in zip(axes, (error_dirac, error_qke),
                              ("(a)", "(b)")):
        mesh = ax.pcolormesh(momenta, tail_factors, np.maximum(err, floor),
                             shading="nearest", cmap="magma_r", norm=norm)
        ax.set_title(title, fontsize=22, color=INK, pad=8)
        ax.set_xlabel(r"canonical momentum $p/m$", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        ax.set_xticks((-1.5, 0.0, 1.5))
        ax.set_yticks(tail_factors)
        ax.grid(which="major", color="white", linewidth=0.35, alpha=0.45)
    axes[0].set_ylabel(r"tail extent $L/\tau$", fontsize=20)
    colorbar = fig.colorbar(mesh, ax=axes, pad=0.025, shrink=0.96)
    colorbar.set_label(r"absolute error $|f_p(L)-f_p^{\rm exact}|$", rotation=90, fontsize=20)
    colorbar.ax.tick_params(labelsize=20)
    colorbar.set_ticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6])
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12)
    fig.canvas.draw()
    fig.canvas.draw()
    fig.set_layout_engine("none")
    fig.savefig(output / "figure4_sauter_window_error_map.pdf")
    fig.savefig(output / "figure4_sauter_window_error_map.png", dpi=300)
    plt.close(fig)

    return {
        "map_points": int(len(momenta)),
        "map_tail_factors": [float(x) for x in tail_factors],
        "map_max_absolute_error_dirac": float(error_dirac.max()),
        "map_max_absolute_error_qke": float(error_qke.max()),
        "map_min_absolute_error_dirac": float(error_dirac.min()),
        "map_min_absolute_error_qke": float(error_qke.min()),
        "map_plot_floor": floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--tail-factors", type=float, nargs="+",
                        default=(4.0, 6.0, 8.0, 10.0, 12.0, 14.0))
    args = parser.parse_args()
    if args.points < 9:
        parser.error("--points must be at least 9")
    if any(x <= 0 for x in args.tail_factors):
        parser.error("all --tail-factors must be positive")
    if len(set(args.tail_factors)) != len(args.tail_factors):
        parser.error("--tail-factors must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    make_structure_figure(args.output, project_root)
    make_workflow_figure(args.output)
    stats = make_window_error_map(args.output, points=args.points,
                                  tail_factors=tuple(args.tail_factors))
    print(stats)


if __name__ == "__main__":
    main()
