#!/usr/bin/env python3
"""Plot the Bit* Dubins experiment separately from the main comparison.

Reads a results directory produced by ``run_bit_star_dubins_experiment.py``
(default: ``comparison_results_paper/bit_star_dubins``) and generates:

1. an aggregate performance figure (feasibility, path length and CPU time per
   obstacle-density block), and
2. trajectory + curvature figures for one or more scenarios (by default
   ``prog09_seed8``).

The trajectory/curvature figures reuse ``plot_comparison.plot_scenario`` so
they keep exactly the same layout and feasibility annotations as the main
comparison, but only the Bit* Dubins path is drawn.

Examples
--------
    # Summary + trajectory for the default scenario (prog09_seed8)
    python plot_bit_star_dubins.py

    # Same, but for a run generated with --pool-seed 33
    python plot_bit_star_dubins.py --pool-seed 33

    # Summary + trajectory for one specific scenario, as PDF
    python plot_bit_star_dubins.py --scenario prog09_seed4 --ext pdf

    # Summary + trajectories for all 100 scenarios
    python plot_bit_star_dubins.py --all --ext pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_comparison as pc

RESULTS_DIR = Path('comparison_results_paper/bit_star_dubins')
PLANNER = 'bit_star_dubins'
DEFAULT_SCENARIO = 'prog09_seed8'

# Distinct style for the standalone Bit* figures.
pc.PLANNER_STYLE[PLANNER] = dict(
    color='#1E88E5', ls='-', lw=2.5, marker='o', label='Bit* Dubins')
pc.DUBINS_PLANNERS.add(PLANNER)


def _group_of(label):
    return label[:6]


def plot_summary(df, out_dir, ext='pdf'):
    """Aggregate per-block metrics for Bit* Dubins."""
    df = df.copy()
    df['group'] = df['scenario'].apply(_group_of)
    df['time_s'] = df['cpu_time'].fillna(df['elapsed'])
    df['feasible'] = df['feasible'].astype(bool)
    df['length_feas'] = df['length'].where(df['feasible'])

    agg = df.groupby('group').agg(
        n_obs=('n_obs', 'first'),
        feas=('feasible', 'mean'),
        len_mean=('length_feas', 'mean'),
        len_std=('length_feas', 'std'),
        time_mean=('time_s', 'mean'),
        time_std=('time_s', 'std'),
    ).sort_index()

    groups = agg.index.tolist()
    x = np.arange(len(groups))
    labels = [f'{g}\n{int(n)} obs' for g, n in zip(groups, agg['n_obs'])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.bar(x, agg['feas'] * 100, color='#2E7D32', alpha=0.85)
    ax.set_ylim(0, 105)
    ax.set_title('Feasibility rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feasible runs (%)', fontsize=11)
    for xi, v in zip(x, agg['feas'] * 100):
        ax.text(xi, v + 2, f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    ax.bar(x, agg['len_mean'], yerr=agg['len_std'].fillna(0),
           capsize=3, color='#1565C0', alpha=0.85)
    ax.set_title('Path length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Length (m)', fontsize=11)
    for xi, v in zip(x, agg['len_mean']):
        if np.isfinite(v):
            ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', va='bottom',
                    fontsize=8)

    ax = axes[2]
    ax.bar(x, agg['time_mean'], yerr=agg['time_std'].fillna(0),
           capsize=3, color='#E65100', alpha=0.85)
    ax.set_title('Computation cost', fontsize=12, fontweight='bold')
    ax.set_ylabel('CPU time (s)', fontsize=11)
    for xi, v in zip(x, agg['time_mean']):
        ax.text(xi, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    len_feas = df.loc[df['feasible'], 'length']
    len_txt = f'{len_feas.mean():.2f} m' if len(len_feas) else '--'
    fig.suptitle(
        f'Bit* Dubins — {len(df)} scenarios | '
        f'feasible {df["feasible"].mean() * 100:.0f}% | '
        f'length (feasible) {len_txt} | '
        f'CPU {df["time_s"].mean():.3f} s',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = out_dir / f'{PLANNER}_summary.{ext}'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')
    return out


def select_scenarios(df, args):
    all_scenarios = sorted(df['scenario'].unique())

    if args.scenario:
        known = set(all_scenarios)
        selected = [s for s in args.scenario if s in known]
        for s in args.scenario:
            if s not in known:
                print(f'  Warning: scenario not found: {s}')
        return selected

    selected = all_scenarios
    if args.prog:
        selected = [s for s in selected if args.prog in s]
    if args.seed is not None:
        selected = [
            s for s in selected
            if int(df.loc[df['scenario'] == s, 'seed'].iloc[0]) == args.seed
        ]

    if args.all or args.prog or args.seed is not None:
        return selected

    if DEFAULT_SCENARIO in all_scenarios:
        return [DEFAULT_SCENARIO]
    return selected[:1]


def main():
    parser = argparse.ArgumentParser(
        description='Plot the Bit* Dubins results separately.')
    parser.add_argument('--results-dir', type=Path, default=None,
                        help='Bit* results directory (default: '
                             'comparison_results_paper/bit_star_dubins, or '
                             'comparison_results_paper/bit_star_dubins_pool{N} '
                             'with --pool-seed)')
    parser.add_argument('--pool-seed', type=int, default=None,
                        help='shortcut for '
                             'comparison_results_paper/bit_star_dubins_pool{N} '
                             '(the directory created by '
                             'run_bit_star_dubins_experiment.py --pool-seed N)')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='output directory (default: <results-dir>/plots)')
    parser.add_argument('--scenario', action='append', default=None,
                        help='scenario label to plot (repeatable)')
    parser.add_argument('--prog', default=None,
                        help='plot scenarios whose label contains PROG')
    parser.add_argument('--seed', type=int, default=None,
                        help='plot scenarios with this seed')
    parser.add_argument('--all', action='store_true',
                        help='plot all scenarios')
    parser.add_argument('--ext', default='pdf', choices=['png', 'pdf'],
                        help='figure format (default: %(default)s)')
    parser.add_argument('--no-summary', action='store_true',
                        help='skip the aggregate summary figure')
    parser.add_argument('--no-trajectory', action='store_true',
                        help='skip the trajectory/curvature figures')
    args = parser.parse_args()

    if args.results_dir is None:
        if args.pool_seed is not None:
            args.results_dir = Path(
                f'comparison_results_paper/bit_star_dubins_pool'
                f'{args.pool_seed}')
        else:
            args.results_dir = RESULTS_DIR

    if not args.results_dir.is_dir():
        hint = ''
        if args.pool_seed is not None:
            hint = (f'\nRun first: python run_bit_star_dubins_experiment.py '
                    f'--pool-seed {args.pool_seed}')
        raise SystemExit(
            f'results directory not found: {args.results_dir}{hint}')

    out_dir = args.out_dir or (args.results_dir / 'plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pc.read_df_safe(args.results_dir)
    df = pc.ensure_feasible(df)
    paths_store = pc._load_pickle_safe(args.results_dir / 'paths.pkl')
    obstacles_store = pc._load_pickle_safe(args.results_dir / 'obstacles.pkl')
    configs = pc._load_pickle_safe(args.results_dir / 'config.pkl')

    if not args.no_summary:
        plot_summary(df, out_dir, ext=args.ext)

    if args.no_trajectory:
        return

    scenarios = select_scenarios(df, args)
    if not scenarios:
        print('  No scenarios selected.')
        return

    print(f'Plotting {len(scenarios)} scenario(s) ...')
    for sc in scenarios:
        df_sub = df[df['scenario'] == sc]
        pc.plot_scenario(
            sc, df_sub, paths_store,
            obstacles_store.get(sc, np.array([])),
            save_dir=out_dir, planners=[PLANNER],
            sc_config=configs.get(sc), ext=args.ext,
        )


if __name__ == '__main__':
    main()
