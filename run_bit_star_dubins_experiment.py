#!/usr/bin/env python3
"""Run Bit* Dubins on the scenarios of a comparison experiment.

By default the scenario set (obstacles, start/goal poses, curvature bound, etc.)
is read from an existing results directory (the ``progressive`` experiment), so
the script evaluates *exactly* the same 100 scenarios
(``prog00..prog09`` x ``seed0..seed9``) without touching the main results.

Alternatively, pass ``--pool-seed N`` to *generate* a fresh ``progressive``
scenario set with a different obstacle layout.  Note that a ``seed`` alone does
**not** change the obstacles: in the ``progressive`` design all seeds of a given
``progNN`` share the same layout, and the positions are fixed by ``pool_seed``.
Use ``--max-obs`` / ``--center-size`` to change the number/size of obstacles and
``--seeds`` to pick the planner seeds.

Outputs are written to a separate directory (default:
``comparison_results_paper/bit_star_dubins`` for the stored scenarios, or
``comparison_results_paper/bit_star_dubins_pool{N}`` when ``--pool-seed`` is
given) in the same format as the main experiment: ``summary.pkl``,
``summary.csv``, ``paths.pkl``, ``config.pkl`` and ``obstacles.pkl``.

Examples
--------
    # Full run on the same 100 scenarios used by the main comparison
    python run_bit_star_dubins_experiment.py -j 4

    # Different obstacle positions: same design, new obstacle pool
    python run_bit_star_dubins_experiment.py --pool-seed 33 -j 4
    python run_bit_star_dubins_experiment.py --pool-seed 33 --seeds 0 1 2 -j 4

    # More/fewer obstacles or a different central obstacle size
    python run_bit_star_dubins_experiment.py --pool-seed 33 --max-obs 12
    python run_bit_star_dubins_experiment.py --pool-seed 33 --center-size 0.4

    # Quick smoke test on the first 2 scenarios
    python run_bit_star_dubins_experiment.py --limit 2 -j 1
"""

import argparse
import re
import pickle
from pathlib import Path

import pandas as pd

from compare_rrt_de2d_nurbs import (
    SAFE_N_JOBS,
    run_bit_star_dubins,
    run_comparison,
    save_results,
    scenarios_progressive,
)


DEFAULT_SOURCE = Path('comparison_results_paper/progressive')
DEFAULT_OUTPUT = Path('comparison_results_paper/bit_star_dubins')
PLANNER_NAME = 'bit_star_dubins'
DEFAULT_SEEDS = list(range(10))


def _scenario_sort_key(label):
    m = re.match(r'^prog(\d+)_seed(\d+)$', label)
    if m:
        return (0, int(m.group(1)), int(m.group(2)))
    m = re.search(r'_seed(\d+)$', label)
    return (1, label, int(m.group(1)) if m else -1)


def load_scenarios(source_dir):
    """Rebuild the scenario descriptors from a previous results directory."""
    configs_path = source_dir / 'config.pkl'
    obstacles_path = source_dir / 'obstacles.pkl'
    if not configs_path.exists():
        raise FileNotFoundError(f'missing {configs_path}')
    if not obstacles_path.exists():
        raise FileNotFoundError(f'missing {obstacles_path}')

    with open(configs_path, 'rb') as f:
        configs = pickle.load(f)
    with open(obstacles_path, 'rb') as f:
        obstacles = pickle.load(f)

    scenarios = []
    for label in sorted(configs, key=_scenario_sort_key):
        cfg = configs[label]
        m = re.search(r'_seed(\d+)$', label)
        seed = int(m.group(1)) if m else 0
        scenarios.append(dict(
            seed=seed,
            obs_list=list(obstacles.get(label, [])),
            label=label,
            start=(cfg['start_x'], cfg['start_y']),
            goal=(cfg['goal_x'], cfg['goal_y']),
            th_start=cfg.get('th_start', 0.0),
            th_goal=cfg.get('th_goal', 0.0),
            radius=cfg.get('radius', 0.073),
            kappa_max=cfg.get('kappa_max', 1.0 / 0.73),
            nsampling=cfg.get('nsampling', 120),
        ))
    return scenarios


def summarize(df):
    """Compact console summary of the run."""
    feasible = df['feasible'].astype(bool)
    times = df['cpu_time'].fillna(df['elapsed'])
    print('\n=== Bit* Dubins summary ===')
    print(f'  scenarios         : {df["scenario"].nunique()}')
    print(f'  planner            : {PLANNER_NAME}')
    print(f'  success rate       : {df["success"].mean():.2%}')
    print(f'  collision-free     : {df["collision_free"].mean():.2%}')
    print(f'  feasible rate      : {feasible.mean():.2%}')
    if feasible.any():
        print(f'  length (feasible)  : '
              f'{df.loc[feasible, "length"].mean():.3f} m '
              f'(median {df.loc[feasible, "length"].median():.3f} m)')
    print(f'  CPU time           : {times.mean():.3f} s '
          f'(median {times.median():.3f} s)')


def main():
    parser = argparse.ArgumentParser(
        description='Run Bit* Dubins on the scenarios of a comparison '
                    'experiment, either stored (--source-dir) or generated '
                    'with a new obstacle pool (--pool-seed).')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE,
                        help='results directory containing config.pkl and '
                             'obstacles.pkl (default: %(default)s); ignored '
                             'when --pool-seed is given')
    parser.add_argument('--pool-seed', type=int, default=None,
                        help='generate a new progressive scenario set with '
                             'this obstacle-pool seed (changes the obstacle '
                             'positions); output defaults to '
                             'comparison_results_paper/bit_star_dubins_pool{N}')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='planner seeds to run (default: 0..9); with '
                             '--source-dir it filters the loaded scenarios')
    parser.add_argument('--max-obs', type=int, default=9,
                        help='maximum number of obstacles for --pool-seed '
                             '(default: %(default)s -> prog00..prog09)')
    parser.add_argument('--center-size', type=float, default=0.3,
                        help='radius of the fixed central obstacle for '
                             '--pool-seed (default: %(default)s)')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='where to write the Bit* results (default: '
                             'comparison_results_paper/bit_star_dubins, or '
                             '..._pool{N} with --pool-seed)')
    parser.add_argument('-j', '--n-jobs', type=int, default=SAFE_N_JOBS,
                        help=f'parallel workers (default: {SAFE_N_JOBS})')
    parser.add_argument('--limit', type=int, default=None,
                        help='run only the first N scenarios (smoke test)')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='disable checkpointing/resume')
    parser.add_argument('--fresh', action='store_true',
                        help='ignore an existing checkpoint and start over')
    args = parser.parse_args()

    if args.pool_seed is not None:
        seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS
        scenarios = scenarios_progressive(
            seeds, max_obs=args.max_obs, pool_seed=args.pool_seed,
            center_size=args.center_size)
        origin = (f'generated with pool_seed={args.pool_seed}, '
                  f'max_obs={args.max_obs}, center_size={args.center_size}, '
                  f'seeds={seeds}')
    else:
        if not args.source_dir.exists():
            raise SystemExit(f'source directory not found: {args.source_dir}')
        scenarios = load_scenarios(args.source_dir)
        origin = f'loaded from {args.source_dir}'
        if args.seeds is not None:
            keep = set(args.seeds)
            scenarios = [sc for sc in scenarios if sc['seed'] in keep]

    if args.out_dir is None:
        if args.pool_seed is not None:
            args.out_dir = Path(
                f'comparison_results_paper/bit_star_dubins_pool'
                f'{args.pool_seed}')
        else:
            args.out_dir = DEFAULT_OUTPUT

    if args.limit is not None:
        scenarios = scenarios[:args.limit]

    print(f'{len(scenarios)} scenario(s) {origin}')
    print(f'Output directory: {args.out_dir}')

    df, paths_store, obstacles_store = run_comparison(
        scenarios,
        planners=[(PLANNER_NAME, run_bit_star_dubins)],
        output_dir=str(args.out_dir),
        verbose=True,
        n_jobs=args.n_jobs,
        checkpoint=not args.no_checkpoint,
        resume=not args.fresh,
    )

    summarize(df)
    save_results(df, paths_store, obstacles_store, output_dir=str(args.out_dir))


if __name__ == '__main__':
    main()
