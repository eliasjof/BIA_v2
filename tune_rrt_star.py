"""Parameter search for the non-Dubins RRT* variants.

Sweeps ``connect_circle_dist`` (ccdist), ``expand_dis`` (steer step) and
``max_iter`` for the ``rrt_star`` and ``rrt_star_smooth`` variants, ranks each
configuration per scenario block (feasible first, then shortest length), and
picks the best configuration of each variant with a Friedman test plus Nemenyi
critical-difference / paired-Wilcoxon post-hoc.

The winning configuration of each variant is the one to be used later when
comparing against the DE2D baseline.

Usage:
    python tune_rrt_star.py                    # full grid, n_jobs=4
    python tune_rrt_star.py -j 8 --seeds 0,1,2,3
    python tune_rrt_star.py --analyze-only     # reuse cached all_records.csv
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, studentized_range
from statsmodels.stats.multitest import multipletests

import compare_rrt_de2d_nurbs as C


# ──────────────────────────────────────────────
# Search grid
# ──────────────────────────────────────────────
GRID_CCDIST = [1.0, 2.0, 3.0]
GRID_EXPAND = [0.1, 0.2, 0.3, 0.4]
GRID_MAXITER = [500, 1000, 2000, 4000, 8000]
VARIANTS = ['rrt_star', 'rrt_star_smooth']

# scenario pool matching the main comparison experiment
PROG_POOL_SEED = 35
PROG_CENTER_SIZE = 0.3
PROG_MAX_OBS = 9
OBS_LEVELS = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]


# ──────────────────────────────────────────────
# Runner factories
# ──────────────────────────────────────────────
def make_raw_runner(ccdist, expand, max_iter):
    def _fn(config, seed=None):
        return C.run_rrt(
            config, planner_type='rrt_star',
            extra_kw={'connect_circle_dist': ccdist,
                      'expand_dis': expand,
                      'max_iter': max_iter},
            seed=seed)
    return _fn


def make_smooth_runner(ccdist, expand, max_iter):
    def _fn(config, seed=None):
        return C._run_rrt_star_smooth_kw(
            config,
            C._smooth_kw(connect_circle_dist=ccdist, expand_dis=expand,
                         max_iter=max_iter),
            seed)
    return _fn


def config_name(variant, ccdist, expand, max_iter):
    return f'{variant}__cc{ccdist}__ed{expand}__mi{max_iter}'


def parse_config_name(name):
    variant, cc, ed, mi = name.split('__')
    return dict(variant=variant,
                connect_circle_dist=float(cc[2:]),
                expand_dis=float(ed[2:]),
                max_iter=int(mi[2:]))


# ──────────────────────────────────────────────
# Scenarios / task list
# ──────────────────────────────────────────────
def build_scenarios(seeds, reps=1):
    """Build the scenario list.

    The obstacle layout only depends on ``pool_seed`` (the pool is sliced),
    *not* on the scenario seed: the seed only controls the planner's RNG.
    So running the same scenario ``reps`` times = reusing the same obstacle
    layout with ``reps`` distinct planner seeds.  ``reps`` copies of every
    scenario are generated, each with a distinct seed and labelled
    ``..._repXX`` so that every run is an independent observation (its own
    ranking block).
    """
    all_scs = C.scenarios_progressive(
        seeds, max_obs=PROG_MAX_OBS, pool_seed=PROG_POOL_SEED,
        center_size=PROG_CENTER_SIZE)
    prefixes = tuple(f'prog{k:02d}' for k in OBS_LEVELS)
    base = [s for s in all_scs if s['label'].startswith(prefixes)]
    if reps <= 1:
        return base
    out = []
    for s in base:
        for r in range(reps):
            out.append(dict(
                seed=s['seed'] * 1000 + r,
                obs_list=s['obs_list'],
                label=f"{s['label']}_rep{r:02d}",
            ))
    return out


def build_task_list(scenarios, variants):
    tasks = []
    for variant in variants:
        factory = (make_raw_runner if variant == 'rrt_star'
                   else make_smooth_runner)
        for cc, ed, mi in itertools.product(GRID_CCDIST, GRID_EXPAND,
                                            GRID_MAXITER):
            pname = config_name(variant, cc, ed, mi)
            fn = factory(cc, ed, mi)
            for sc in scenarios:
                tasks.append((sc, pname, fn))
    return tasks


# ──────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────
def run_grid(scenarios, variants, n_jobs, verbose=True):
    tasks = build_task_list(scenarios, variants)
    if verbose:
        print(f'{len(tasks)} tasks  ({len(variants)} variants x '
              f'{len(GRID_CCDIST)*len(GRID_EXPAND)*len(GRID_MAXITER)} configs '
              f'x {len(scenarios)} scenarios)')
    if n_jobs == 1:
        results = [C._run_single_task(*t) for t in tasks]
    else:
        if n_jobs > 4:
            print(f'  [WARNING] n_jobs={n_jobs}: memory contention inflates '
                  f'timings (~2x at n_jobs=21). Prefer -j 4.')
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(C._run_single_task)(*t) for t in tasks)
    records = [r[0] for r in results]
    df = pd.DataFrame(records)
    df = df.rename(columns={'scenario': 'block', 'planner': 'config'})
    return df


# ──────────────────────────────────────────────
# Statistical analysis
# ──────────────────────────────────────────────
def _rank_blocks(df):
    """Per scenario block, rank configs lexicographically by:
    feasibility (feasible first), then length, then cv, then cpu time.

    Distinct ranks are kept (no forcing of infeasible configs to a shared
    worst rank) so that infeasible configurations are still ordered among
    themselves by the secondary criteria; this keeps the Friedman test
    well-defined."""
    rows = []
    for block, g in df.groupby('block'):
        g = g.sort_values(['feasible', 'length', 'cv', 'cpu_time'],
                          ascending=[False, True, True, True],
                          na_position='last').copy()
        g['rank'] = range(1, len(g) + 1)
        rows.append(g[['block', 'config', 'rank']])
    return pd.concat(rows, ignore_index=True)


def nemenyi_cd(k, n, alpha=0.05):
    """Critical difference for the Nemenyi post-hoc test (alpha one-sided)."""
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def analyze_variant(df_v, variant, out_dir, alpha=0.05):
    configs = sorted(df_v['config'].unique())
    rank_df = _rank_blocks(df_v)
    piv = rank_df.pivot(index='block', columns='config', values='rank')[configs]
    k, n = len(configs), piv.shape[0]

    if k >= 3:
        fried_stat, fried_p = friedmanchisquare(
            *[piv[c].values for c in configs])
        cd = nemenyi_cd(k, n, alpha)
    else:
        fried_stat, fried_p = np.nan, np.nan
        cd = np.inf
    mean_rank = piv.mean()
    std_rank = piv.std()
    best = mean_rank.idxmin()
    equivalent = sorted(mean_rank[mean_rank <= mean_rank[best] + cd].index,
                        key=lambda c: mean_rank[c])

    # pairwise Wilcoxon (one-sided: best has smaller rank) + Holm
    pvals = {}
    best_ranks = piv[best].values
    for c in configs:
        if c == best:
            continue
        try:
            _, p = wilcoxon(best_ranks, piv[c].values, alternative='less')
        except ValueError:
            p = 1.0
        pvals[c] = p
    if pvals:
        names = list(pvals)
        p_raw = [pvals[c] for c in names]
        reject, p_adj, *_ = multipletests(p_raw, alpha=alpha, method='holm')
        wil = pd.DataFrame({'config': names, 'p_raw': p_raw, 'p_holm': p_adj,
                            'better_than_others': reject})
    else:
        wil = pd.DataFrame(columns=['config', 'p_raw', 'p_holm',
                                    'better_than_others'])

    # aggregate metrics
    fea_len = df_v[df_v['feasible']].groupby('config')['length'].mean()
    agg = df_v.groupby('config').agg(
        feasible_rate=('feasible', 'mean'),
        success_rate=('success', 'mean'),
        mean_length=('length', 'mean'),
        mean_cv=('cv', 'mean'),
        mean_cpu=('cpu_time', 'mean'),
        mean_wall=('wall_time', 'mean'),
    )
    agg['mean_length_feasible'] = fea_len
    agg['mean_rank'] = mean_rank
    agg['std_rank'] = std_rank
    agg['within_cd_of_best'] = [c in equivalent for c in agg.index]
    agg = agg.sort_values('mean_rank')

    # recommended pick: among statistically-equivalent configs, prefer
    # highest feasibility, then shortest length, then lower curvature, then
    # fastest (mirrors the ranking used above).
    eq_agg = agg[agg['within_cd_of_best']].copy()
    eq_agg = eq_agg.sort_values(
        ['feasible_rate', 'mean_length', 'mean_cv', 'mean_cpu'],
        ascending=[False, True, True, True], na_position='last')
    recommended = eq_agg.index[0] if len(eq_agg) else best

    summary = agg.reset_index()
    summary.to_csv(out_dir / f'summary_{variant}.csv', index=False)
    wil.to_csv(out_dir / f'wilcoxon_{variant}.csv', index=False)

    _plot_mean_rank(agg, cd, variant,
                    out_dir / f'mean_rank_{variant}.pdf')

    report = dict(
        variant=variant,
        n_blocks=n,
        n_configs=k,
        friedman_statistic=float(fried_stat),
        friedman_pvalue=float(fried_p),
        critical_difference=float(cd),
        best_by_rank=dict(name=best, **parse_config_name(best)),
        recommended=dict(name=recommended, **parse_config_name(recommended)),
        equivalent_set=equivalent,
    )
    return summary, report


def _plot_mean_rank(agg, cd, variant, path):
    s = agg.sort_values('mean_rank')
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.20 * len(s) + 1)))
    ax.barh(y, s['mean_rank'], xerr=s['std_rank'], color='#4CA3AF',
            error_kw=dict(ecolor='gray', lw=0.7))
    if np.isfinite(cd):
        ax.axvline(s['mean_rank'].min() + cd, color='crimson', ls='--',
                   lw=1.2, label=f'CD (Nemenyi, {cd:.2f})')
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('mean rank (lower is better)')
    ax.set_title(f'Mean rank per configuration - {variant}')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_list(s, cast):
    return [cast(x) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--reps', type=int, default=1,
                   help='independent planner runs per scenario (default 1)')
    p.add_argument('--n-jobs', '-j', type=int, default=4)
    p.add_argument('--out', default='rrt_tuning')
    p.add_argument('--ccdist', default=None,
                   help='csv override, e.g. "1.5,2.5"')
    p.add_argument('--expand', default=None,
                   help='csv override, e.g. "0.2,0.5"')
    p.add_argument('--max-iter', default=None,
                   help='csv override, e.g. "1000,3000"')
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--force', action='store_true',
                   help='recompute even if all_records.csv exists')
    args = p.parse_args()

    global GRID_CCDIST, GRID_EXPAND, GRID_MAXITER
    if args.ccdist:
        GRID_CCDIST = parse_list(args.ccdist, float)
    if args.expand:
        GRID_EXPAND = parse_list(args.expand, float)
    if args.max_iter:
        GRID_MAXITER = parse_list(args.max_iter, int)

    seeds = parse_list(args.seeds, int)
    variants = [v for v in args.variants.split(',') if v.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_csv = out_dir / 'all_records.csv'

    if args.analyze_only or (records_csv.exists() and not args.force):
        if not records_csv.exists():
            print(f'{records_csv} not found; run without --analyze-only.')
            return
        print(f'Loading cached records from {records_csv}')
        df = pd.read_csv(records_csv)
    else:
        scenarios = build_scenarios(seeds, args.reps)
        print(f'{len(scenarios)} scenario blocks '
              f'({len(seeds)} seeds x obs levels {OBS_LEVELS} '
              f'x {args.reps} rep(s))')
        print(f'grid: ccdist={GRID_CCDIST}  expand={GRID_EXPAND}  '
              f'max_iter={GRID_MAXITER}')
        t0 = time.perf_counter()
        df = run_grid(scenarios, variants, args.n_jobs)
        df.to_csv(records_csv, index=False)
        print(f'ran {len(df)} tasks in {time.perf_counter()-t0:.1f}s -> '
              f'{records_csv}')

    reports = {}
    for variant in variants:
        df_v = df[df['config'].str.startswith(variant + '__')]
        if df_v.empty:
            continue
        print(f'\n{"="*66}\n=== {variant} ===')
        summary, report = analyze_variant(df_v, variant, out_dir)
        reports[variant] = report
        cols = ['config', 'mean_rank', 'feasible_rate', 'success_rate',
                'mean_length', 'mean_length_feasible', 'mean_cv', 'mean_cpu',
                'within_cd_of_best']
        print(summary[cols].head(12).to_string(index=False))
        print(f'\nFriedman chi2={report["friedman_statistic"]:.2f} '
              f'p={report["friedman_pvalue"]:.3g}  '
              f'CD={report["critical_difference"]:.2f}')
        print(f'best by rank  : {report["best_by_rank"]["name"]}')
        print(f'recommended   : {report["recommended"]["name"]}')
        print(f'equivalent set: {len(report["equivalent_set"])} configs '
              f'within CD')

    with open(out_dir / 'best_params.json', 'w') as f:
        json.dump(reports, f, indent=2)
    print(f'\nWrote {out_dir}/best_params.json')


if __name__ == '__main__':
    main()
