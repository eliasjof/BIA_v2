import pickle
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_ROOT = Path(__file__).parent / 'comparison_results'


def read_df_safe(exp_dir):
    """Try pickle, fall back to CSV (handles pandas version mismatches)."""
    pkl = exp_dir / 'summary.pkl'
    csv = exp_dir / 'summary.csv'
    if pkl.is_file():
        try:
            return pd.read_pickle(pkl)
        except Exception:
            print(f'  Warning: pickle version mismatch, falling back to CSV')
    if csv.is_file():
        df = pd.read_csv(csv)
        for col in ('success', 'collision_free', 'feasible'):
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    raise FileNotFoundError(f'No summary.pkl or summary.csv in {exp_dir}')


def _load_pickle_safe(path):
    if not path.is_file():
        return {}
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'  Warning: could not load {path.name}: {e}')
        return {}


def load_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    df = read_df_safe(exp_dir)
    paths = _load_pickle_safe(exp_dir / 'paths.pkl')
    obs = _load_pickle_safe(exp_dir / 'obstacles.pkl')
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(obs, dict):
        obs = {}
    return df, paths, obs


def print_experiment_analysis(df, name):
    if 'feasible' not in df.columns:
        df = df.copy()
        df['feasible'] = df['collision_free']
        df['cv'] = np.where(df['collision_free'], 0.0, np.nan)

    print(f'  RANKING (lower avg_rank = better)')
    rank_rows = []
    for scenario, grp in df.groupby('scenario', sort=False):
        grp = grp.sort_values(
            ['feasible', 'length', 'elapsed'],
            ascending=[False, True, True],
            na_position='last',
        )
        for rank, (_, row) in enumerate(grp.iterrows(), 1):
            rank_rows.append(dict(scenario=scenario, planner=row['planner'], rank=rank))
    rank_df = pd.DataFrame(rank_rows)
    ranking = (
        rank_df.groupby('planner')['rank']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .round(3)
        .sort_values('mean')
    )
    ranking.columns = ['avg_rank', 'std_rank', 'best', 'worst', 'count']
    print(ranking.to_string())
    print()

    print(f'  AGGREGATED METRICS')
    fea = df[df['feasible']]
    rows = []
    for planner, g in df.groupby('planner'):
        feas_mask = g['feasible']
        feasible_len = g.loc[feas_mask, 'length']
        rows.append(dict(
            planner=planner,
            success_rate=g['success'].mean(),
            feasible_rate=feas_mask.mean(),
            avg_length=g['length'].mean(),
            avg_length_fea=feasible_len.mean() if len(feasible_len) else np.nan,
            avg_cv=g['cv'].mean(),
            avg_time=g['elapsed'].mean(),
        ))
    agg = pd.DataFrame(rows).set_index('planner').round(3)
    agg = agg.sort_values(['feasible_rate', 'avg_length'], ascending=[False, True])
    print(agg.to_string())
    print('-' * 70)
    print()

    return df


def main():
    results_dir = RESULTS_ROOT
    if not results_dir.is_dir():
        print(f'Directory not found: {results_dir}')
        return

    exps = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / 'summary.pkl').is_file()
    )

    if not exps:
        print(f'No experiments found under {results_dir}/')
        print('Each experiment subdirectory must contain summary.pkl or summary.csv')
        return

    print(f'Found {len(exps)} experiment(s) in {results_dir}/\n')

    all_dfs = []
    for exp_dir in exps:
        try:
            df, paths, obs = load_experiment(exp_dir)
        except Exception as e:
            print(f'Error loading {exp_dir.name}: {e}')
            continue
        print(f'=== {exp_dir.name} ===')
        print_experiment_analysis(df, exp_dir.name)
        all_dfs.append(df)

    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        print('=== ALL COMBINED ===')
        print_experiment_analysis(combined, 'ALL COMBINED')
        out = results_dir / 'combined_summary.csv'
        combined.to_csv(out, index=False)
        print(f'Combined CSV saved to {out}  ({len(combined)} records)')


if __name__ == '__main__':
    main()
