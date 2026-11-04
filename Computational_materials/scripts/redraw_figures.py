#!/usr/bin/env python3
"""Redraw four scientific figures and two supplementary diagrams from archived CSVs, without new integrations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reproduce import plot_spectra
from make_reproducibility_figures import make_structure_figure, make_workflow_figure, plot_window_error_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    single = np.genfromtxt(args.data / 'sauter_spectrum.csv', delimiter=',', names=True)
    double = np.genfromtxt(args.data / 'double_pulse_spectrum.csv', delimiter=',', names=True)
    plot_spectra(args.output, single['momentum'], single['dirac'], single['qke'],
                 single['exact'], single['qke_invariant_error'],
                 double['momentum'], double['dirac'], double['qke'])
    make_structure_figure(args.output, Path(__file__).resolve().parents[1])
    make_workflow_figure(args.output)
    window = np.genfromtxt(args.data / 'sauter_window_error_map.csv', delimiter=',', names=True)
    momenta = np.unique(window['momentum'])
    tails = np.unique(window['tail_extent_over_tau'])
    errors = np.full((2, len(tails), len(momenta)), np.nan)
    seen = set()
    for row in window:
        i = int(np.searchsorted(tails, row['tail_extent_over_tau']))
        j = int(np.searchsorted(momenta, row['momentum']))
        if (i, j) in seen:
            raise ValueError('Duplicate momentum/tail coordinate in error-map data')
        seen.add((i, j))
        errors[:, i, j] = row['dirac_absolute_error'], row['qke_absolute_error']
    if not np.all(np.isfinite(errors)):
        raise ValueError('Incomplete or non-finite error-map data')
    plot_window_error_map(args.output, momenta, tails, *errors)
    print('Redrew four scientific figures and two supplementary diagrams from archived data.')


if __name__ == '__main__':
    main()
