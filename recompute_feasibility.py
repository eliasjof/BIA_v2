"""Recompute the post-hoc feasibility columns of an existing comparison run.

The feasibility rule in ``compare_rrt_de2d_nurbs.py`` was changed to judge the
curvature constraint with the *Menger* maximum curvature of the sampled path
(plus a small relative tolerance, ``KAPPA_REL_TOL``) instead of a
finite-difference estimator that produced spurious curvature spikes on Dubins
paths (repeated vertices at straight/arc joints and non-uniform spacing).  Any
``summary.pkl`` produced before that change stores wrong ``cv`` / ``feasible``
values for the curvature-constrained planners.

This script rebuilds those two columns from the saved ``paths.pkl``,
``obstacles.pkl`` and ``config.pkl`` using the *same* rules and functions as a
fresh run (``compare_rrt_de2d_nurbs.posthoc_metrics`` / ``is_feasible``), so the
expensive planners do not need to be executed again.

Usage:
    python recompute_feasibility.py comparison_results_paper/progressive
    python recompute_feasibility.py comparison_results_paper/progressive --dry-run
"""
import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import compare_rrt_de2d_nurbs as C


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def recompute(exp_dir, dry_run=False):
    exp = Path(exp_dir)
    summary = pd.read_pickle(exp / 'summary.pkl')
    paths = _load(exp / 'paths.pkl')
    obstacles = _load(exp / 'obstacles.pkl')
    configs = _load(exp / 'config.pkl')

    old_feas = summary['feasible'].copy()
    old_cv = summary['cv'].copy()

    new_cv = np.empty(len(summary))
    new_feas = np.empty(len(summary), dtype=bool)

    for i, (scenario, planner) in enumerate(
            zip(summary['scenario'], summary['planner'])):
        cfg = configs[scenario]
        pts = paths.get(f'{scenario}_{planner}')
        obs = obstacles.get(scenario)
        cf = bool(summary['collision_free'].iloc[i])

        kappa_viol, kappa_respected, ws_viol, obs_viol = C.posthoc_metrics(
            pts, cfg['kappa_max'], obs, cfg['radius'],
            (cfg['xmin'], cfg['xmax'], cfg['ymin'], cfg['ymax']))

        cv_val = kappa_viol + ws_viol + obs_viol
        if np.isnan(cv_val):
            cv_val = np.inf
        if cv_val <= 1e-5:
            cv_val = 0.0
        new_cv[i] = cv_val
        new_feas[i] = C.is_feasible(cf, kappa_respected, ws_viol,
                                    obs_viol, cv_val)

    summary['cv'] = new_cv
    summary['feasible'] = new_feas

    print(f'{exp}')
    print(f'  records        : {len(summary)}')
    print(f'  feasible before: {int(old_feas.sum())} '
          f'({old_feas.mean() * 100:.1f}%)')
    print(f'  feasible after : {int(new_feas.sum())} '
          f'({new_feas.mean() * 100:.1f}%)')
    changed = int((old_feas.values != new_feas).sum())
    print(f'  changed rows   : {changed}')

    rates = (summary.assign(_old=old_feas.values)
             .groupby('planner')
             .agg(old_rate=('_old', 'mean'), new_rate=('feasible', 'mean'))
             .sort_values('new_rate', ascending=False))
    with pd.option_context('display.float_format', lambda v: f'{v:.2f}'):
        print(rates.to_string())

    if dry_run:
        print('  (dry-run: nothing written)')
        return

    backup = exp / 'summary_pre_cvfix.pkl'
    if not backup.is_file():
        shutil.copy2(exp / 'summary.pkl', backup)
        print(f'  backup saved to {backup.name}')
    else:
        print(f'  backup already exists ({backup.name}), kept')

    summary.to_pickle(exp / 'summary.pkl')
    summary.to_csv(exp / 'summary.csv', index=False)
    print(f'  updated {exp / "summary.pkl"} and summary.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('exp_dir', help='experiment dir containing summary.pkl')
    ap.add_argument('--dry-run', action='store_true',
                    help='only report the change, do not write')
    args = ap.parse_args()
    recompute(args.exp_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
