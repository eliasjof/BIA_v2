import pickle, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker


PLANNER_STYLE = {
    'rrt_star':          dict(color='#2196F3', ls='-x',  lw=1.5, label='RRT*'),
    'rrt_star_smooth':   dict(color='#00BCD4', ls='-', lw=2.0, label='RRT* + B-spline'),
    'rrt_star_dubins':   dict(color='#F44336', ls='-',  lw=1.5, label='RRT* Dubins'),
    'rrt_dubins_smooth': dict(color='#FF9800', ls='-.', lw=2.0, label='RRT* Dubins + B-spline'),
    'modified_dubins_rrt_star': dict(color="#00FF62", ls='--', lw=2.0, label='Dubins-RRT*'),
    'modified_dubins_rrt_star_ccpoa': dict(color="#0A4922", ls='--', lw=2.0, label='Modified Dubins-RRT* + CCPOA'),
    'de2d_nurbs':        dict(color='#4CAF50', ls='-',  lw=2.5, label='DE2D_NURBS'),
    'pso2d_nurbs':       dict(color='#795548', ls=':',  lw=2.5, label='PSO2D_NURBS'),
    'rrt_star_asv':      dict(color='#9C27B0', ls='-d',  lw=2.0, label='RRT* ASV'),
    'rrt_star_ccdist_1_5': dict(color="#3F51B5", ls='--', lw=1.5, label='RRT* CCDIST 1.5'),
    'rrt_star_ccdist_2_5': dict(color="#673AB7", ls='--', lw=1.5, label='RRT* CCDIST 2.5'),
    'rrt_star_ccdist_3_0': dict(color="#585086", ls='dashdot', lw=1.5, label='RRT* CCDIST 3.0'),
    'rrt_star_smooth_ccdist_1_5': dict(color="#009688", ls='-.', lw=1.5, label='RRT* + B-spline CCDIST 1.5'),
    'rrt_star_smooth_ccdist_2_5': dict(color="#4CAF50", ls='-.', lw=1.5, label='RRT* + B-spline CCDIST 2.5'),
    'rrt_star_smooth_ccdist_3_0': dict(color="#8F0D63", ls='-.', lw=1.5, label='RRT* + B-spline CCDIST 3.0'),
    'rrt_star_expand_0_2': dict(color="#D32F2F", ls='--', lw=1.5, label='RRT* step 0.2'),
    'rrt_star_expand_0_5': dict(color="#F57C00", ls='--', lw=1.5, label='RRT* step 0.5'),
    'rrt_star_smooth_expand_0_2': dict(color="#AD1457", ls='-.', lw=1.5, label='RRT* + B-spline step 0.2'),
    'rrt_star_smooth_expand_0_5': dict(color="#6A1B9A", ls='-.', lw=1.5, label='RRT* + B-spline step 0.5'),
    'rrt_star_tuned': dict(color="#C62828", ls='--', lw=1.8, label='RRT* (tuned)'),
    'rrt_star_smooth_tuned': dict(color="#00695C", ls='-.', lw=1.8, label='RRT* + B-spline (tuned)'),
    'rrt_star_dubins_ccdist_3_5': dict(color="#E91E63", ls='dotted', lw=1.5, label='RRT* Dubins CCDIST 3.5'),
    'rrt_star_dubins_ccdist_6_0': dict(color="#9C27B0", ls='dotted', lw=1.5, label='RRT* Dubins CCDIST 6.0'),
}


def prog_group_label(g):
    m = re.match(r'prog(\d+)', g)
    if m:
        n = int(m.group(1))
        return f'scen-{n}obs'
    return g


RESULTS_ROOT = Path(__file__).parent / 'comparison_results_paper'

SCENARIO_PREFIX = 'prog'

# When True, grid-swept variants (e.g. `rrt_star_cc2.0_ed0.3_mi2000`) are
# reduced to a single representative row per variant, namely the configuration
# with the best mean rank over all blocks, relabelled to the canonical name
# (`rrt_star`, `rrt_star_smooth`, ...).  This makes the comparison use the
# *best* combination found in the exhaustive search instead of a single
# hand-picked configuration.
SELECT_BEST_GRID_CONFIG = True

SELECTED_PLANNERS = [
    'rrt_star',
    'rrt_star_smooth',
    'rrt_star_dubins',
    'rrt_dubins_smooth',
    'modified_dubins_rrt_star',
    'modified_dubins_rrt_star_ccpoa',
    'de2d_nurbs',
    'pso2d_nurbs',
    'rrt_star_asv',
]

# Base variant -> regex matching every configuration produced by the sweep.
# Order matters: more specific prefixes must come first.
GRID_VARIANT_PATTERNS = [
    ('rrt_star_smooth',
     r'^rrt_star_smooth$|^rrt_star_smooth_cc[0-9.]+_ed[0-9.]+_mi[0-9]+$'),
    ('rrt_star_dubins', r'^rrt_star_dubins_cc[0-9.]+_yaw\w+$'),
    ('rrt_dubins_smooth', r'^rrt_dubins_smooth_cc[0-9.]+_yaw\w+$'),
    ('modified_dubins_rrt_star_ccpoa',
     r'^modified_dubins_rrt_star_ccpoa_eta[0-9.]+_safe[0-9.]+$'),
    ('modified_dubins_rrt_star',
     r'^modified_dubins_rrt_star_eta[0-9.]+_safe[0-9.]+$'),
    ('rrt_star_asv', r'^rrt_star_asv_ca[0-9.]+_cb[0-9.]+$'),
    ('rrt_star', r'^rrt_star$|^rrt_star_cc[0-9.]+_ed[0-9.]+_mi[0-9]+$'),
]


def read_df_safe(exp_dir):
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


CURV_FEAS_THRESHOLD = 1e-5

DUBINS_PLANNERS = {
    'rrt_star_dubins', 'rrt_dubins_smooth',
    'modified_dubins_rrt_star', 'modified_dubins_rrt_star_ccpoa',
    'rrt_star_asv',
}

NO_CURV_PLANNERS = {'rrt_star'}


def ensure_feasible(df):
    df = df.copy()
    if 'collision_free' not in df.columns:
        df['collision_free'] = True
    has_feas = 'feasible' in df.columns
    feas_list = []
    for _, row in df.iterrows():
        # Prefer the authoritative per-run flag (computed with the strict
        # curvature rule in the runner / recompute_feasibility).  Only fall
        # back to a cv-based recomputation when it is missing, otherwise a
        # path just above the 5 % tolerance but with a rounded cv=0 would be
        # wrongly counted as feasible.
        if has_feas and not pd.isna(row.get('feasible', np.nan)):
            feas_list.append(bool(row['feasible']))
            continue
        col_free = row.get('collision_free', True)
        if isinstance(col_free, float):
            col_free = bool(col_free) if not np.isnan(col_free) else True
        if not col_free:
            feas_list.append(False)
            continue

        cv = row.get('cv')
        if cv is not None and not (isinstance(cv, float) and np.isnan(cv)):
            feas_list.append(float(cv) <= CURV_FEAS_THRESHOLD)
        else:
            feas_list.append(col_free)
    df['feasible'] = feas_list
    if 'cv' in df.columns:
        df['cv'] = df['cv'].apply(lambda x: 0.0 if (isinstance(x, (int, float)) and x <= CURV_FEAS_THRESHOLD) else x)
    else:
        df['cv'] = np.where(df['feasible'], 0.0, np.nan)
    df['success'] = df['success'].fillna(False).astype(bool)
    return df


def filter_planners(df):
    if SELECTED_PLANNERS:
        mask = df['planner'].isin(SELECTED_PLANNERS)
        n_removed = (~mask).sum()
        if n_removed:
            print(f'  (filtrados {n_removed} registros de planejadores nao selecionados)')
        return df[mask].copy()
    return df


def extract_prog_group(scenario):
    m = re.match(r'(prog\d+)', scenario)
    if m:
        return m.group(1)
    return scenario


def rank_within_seed(grp):
    n = len(grp)
    grp = grp.sort_values(
        ['feasible', 'length', 'time_s'],
        ascending=[False, True, True],
        na_position='last',
    )
    grp = grp.copy()
    grp['rank'] = range(1, n + 1)
    grp.loc[~grp['feasible'], 'rank'] = n
    return grp


def _variant_of(planner):
    """Return the swept base-variant name of a planner, or None if not swept."""
    for variant, pat in GRID_VARIANT_PATTERNS:
        if re.match(pat, planner):
            return variant
    return None


def _block_mean_rank(df, group_cols=('prog_group', 'seed')):
    """Mean of the per-block (group, seed) ranks of every planner in `df`."""
    rows = []
    for _, block in df.groupby(list(group_cols)):
        block = rank_within_seed(block)
        for _, row in block.iterrows():
            rows.append({'planner': row['planner'], 'rank': row['rank']})
    if not rows:
        return pd.Series(dtype=float)
    rdf = pd.DataFrame(rows)
    return rdf.groupby('planner')['rank'].mean()


# Ordering used only to *select* the representative configuration of a swept
# variant.  Mirrors `tune_rrt_star._rank_blocks`: feasibility first, then
# length, then curvature violation `cv`, then time.  Unlike
# `rank_within_seed`, infeasible configurations are NOT collapsed to a shared
# worst rank, so distinct sequential ranks keep the ordering well defined even
# when no configuration of a variant is feasible.
_SELECT_ORDER = ['feasible', 'length', 'cv', 'time_s']
_SELECT_ASC = [False, True, True, True]

# Relative tolerance used to declare two configurations' length (and curvature)
# *tied* when picking the representative.  Without it, a microscopically shorter
# but far slower configuration would win the tie-break on length (e.g. length
# 3.306259 vs 3.306336, a 0.002% difference, over a 60x runtime).  Differences
# below this tolerance are treated as equal and the fastest config is chosen.
_SELECT_QUAL_TOL = 0.01

# Swept variants (`..._mi{N}`) are only allowed to use a budget above 500
# iterations; the 500-iteration configuration is excluded from the
# representative selection.  The `cc = 3.0` rewiring radius is likewise excluded
# so that the compared configurations stay in the cheaper `cc` regimes.
_MIN_SWEEP_MAX_ITER = 1000
_EXCLUDED_SWEEP_CC = {3.0}

def _nemenyi_cd(k, n, alpha=0.05):
    """Critical difference of the Nemenyi post-hoc test (same as tune script)."""
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def _variant_mean_rank(sub):
    """Mean per-block distinct rank of each config (see `_SELECT_ORDER`)."""
    rows = []
    cols = [c for c in _SELECT_ORDER if c in sub.columns]
    asc = [_SELECT_ASC[_SELECT_ORDER.index(c)] for c in cols]
    for _, block in sub.groupby(['prog_group', 'seed']):
        block = block.sort_values(cols, ascending=asc, na_position='last')
        for rank, (_, row) in enumerate(block.iterrows(), 1):
            rows.append({'planner': row['planner'], 'rank': rank})
    if not rows:
        return pd.Series(dtype=float)
    rdf = pd.DataFrame(rows)
    return rdf.groupby('planner')['rank'].mean()


def select_best_grid_configs(df, verbose=True):
    """Collapse every swept variant to its best configuration.

    For each variant (`rrt_star`, `rrt_star_smooth`, ...) configurations are
    ranked per block (`prog_group`, `seed`) by feasibility, length, curvature
    and time, and the configuration with the best mean rank wins.  The primary
    criterion is the **feasible rate**: only configurations attaining the
    highest feasible rate of the variant are eligible.  Among those, the pick is
    made within the Nemenyi critical difference of the best rank; length and
    curvature differences below a relative tolerance are treated as ties, and
    the fastest of the tied configurations is chosen (this avoids a
    pathological, very slow configuration that is statistically
    indistinguishable from the winner).  Swept variants are restricted to
    budgets above 500 iterations (`_MIN_SWEEP_MAX_ITER`) and to a rewiring
    radius other than `cc = 3.0` (`_EXCLUDED_SWEEP_CC`).
    The winner is relabelled to the canonical variant name and the other
    configurations are dropped.  Non-swept planners (e.g. `de2d_nurbs`,
    `pso2d_nurbs`) are kept untouched.
    """
    df = df.copy()
    df['_variant'] = df['planner'].apply(_variant_of)

    # Discard the 500-iteration budget and the `cc = 3.0` radius from the swept
    # variants (keep `max_iter > 500`, `connect_circle_dist != 3.0`); non-swept
    # planners have neither suffix and are kept.
    _mi = pd.to_numeric(
        df['planner'].str.extract(r'_mi(\d+)$', expand=False), errors='coerce')
    _cc = pd.to_numeric(
        df['planner'].str.extract(r'_cc([0-9.]+)_ed[0-9.]+_mi\d+$',
                                  expand=False), errors='coerce')
    _keep = (_mi.isna() | (_mi >= _MIN_SWEEP_MAX_ITER)) & (
        ~_cc.isin(_EXCLUDED_SWEEP_CC))
    df = df[_keep].copy()

    keep = df['_variant'].isna().copy()
    picked = {}
    for variant, gsub in df[df['_variant'].notna()].groupby('_variant'):
        mean_rank = _variant_mean_rank(gsub)
        if mean_rank.empty:
            continue
        best = mean_rank.idxmin()

        k = int(mean_rank.shape[0])
        n = int(gsub.groupby(['prog_group', 'seed']).ngroups)
        cd = _nemenyi_cd(k, n) if k >= 2 and n > 0 else np.inf
        equivalent = mean_rank[mean_rank <= mean_rank.loc[best] + cd].index

        agg = gsub.groupby('planner').agg(
            feasible_rate=('feasible', 'mean'),
            mean_length=('length', 'mean'),
            mean_cv=('cv', 'mean'),
            mean_time=('time_s', 'mean'),
        )
        # Feasibility is the primary criterion: only configurations attaining
        # the highest feasible rate of the variant are eligible.  Among those,
        # prefer the statistically-equivalent ones (Nemenyi CD w.r.t. the best
        # rank), then those whose mean length and curvature are within
        # `_SELECT_QUAL_TOL` of the best (a relative tie), and finally pick the
        # fastest.  This prevents a microscopic (noise-level) length advantage
        # from selecting a far slower configuration.
        cand = agg[agg['feasible_rate'] >= agg['feasible_rate'].max() - 1e-9]
        cd_ok = cand.index.isin(equivalent)
        if cd_ok.any():
            cand = cand[cd_ok]
        if len(cand):
            len_min = cand['mean_length'].min()
            if np.isfinite(len_min) and len_min > 0:
                cand = cand[cand['mean_length']
                            <= len_min * (1.0 + _SELECT_QUAL_TOL)]
            cv_min = cand['mean_cv'].min()
            if np.isfinite(cv_min) and cv_min > 0:
                cand = cand[cand['mean_cv']
                            <= cv_min * (1.0 + _SELECT_QUAL_TOL)]
        chosen = cand['mean_time'].idxmin() if len(cand) else best

        picked[variant] = (chosen, float(mean_rank.loc[chosen]),
                           float(mean_rank.loc[best]), float(cd))
        keep |= (df['planner'] == chosen)

    reduced = df[keep].drop(columns=['_variant'])
    rename = {best: variant for variant, (best, *_rest) in picked.items()}
    reduced['planner'] = reduced['planner'].replace(rename)

    if verbose and picked:
        print('  [best-config] melhor configuracao por variante '
              '(rank medio / CD Nemenyi):')
        for variant, (chosen, mr, best_mr, cd) in picked.items():
            print(f'    {variant:35s} <- {chosen:48s} '
                  f'(avg_rank {mr:.3f}, CD {cd:.2f})')
    return reduced, picked


def print_prog_ranking_table(df):
    groups = sorted(df['prog_group'].unique())
    for g in groups:
        sub = df[df['prog_group'] == g]
        print(f'  === {g} (media do rank em {sub["seed"].nunique()} sementes) ===')
        rank_rows = []
        for _, seed_grp in sub.groupby('seed'):
            seed_grp = rank_within_seed(seed_grp)
            for _, row in seed_grp.iterrows():
                rank_rows.append({
                    'planner': row['planner'],
                    'rank': row['rank'],
                    'seed': row['seed'],
                })
        rank_df = pd.DataFrame(rank_rows)
        ranking = (
            rank_df.groupby('planner')['rank']
            .agg(['mean', 'std', 'min', 'max', 'count'])
            .round(3)
            .sort_values('mean')
        )
        ranking.columns = ['avg_rank', 'std_rank', 'best', 'worst', 'count']
        ranking['avg_rank'] = ranking['avg_rank'].map('{:.3f}'.format)
        print(ranking.to_string())
        print()


def print_prog_placement_table(df):
    groups = sorted(df['prog_group'].unique())
    for g in groups:
        sub = df[df['prog_group'] == g]
        print(f'  === {g} (contagem de colocacoes em {sub["seed"].nunique()} sementes) ===')
        placement_rows = []
        for _, seed_grp in sub.groupby('seed'):
            seed_grp = rank_within_seed(seed_grp)
            for _, row in seed_grp.iterrows():
                placement_rows.append({
                    'planner': row['planner'],
                    'rank': row['rank'],
                })
        place_df = pd.DataFrame(placement_rows)
        cross = (
            place_df.groupby(['planner', 'rank'])
            .size()
            .unstack(fill_value=0)
        )
        if cross.empty:
            print('  (sem dados)')
            print()
            continue
        col_order = sorted(cross.columns)
        cross = cross[col_order]
        cross = cross.reindex(
            sorted(cross.index,
                   key=lambda p: sum((col_order.index(r) if r in col_order else 99) * cross.loc[p, r]
                                     for r in col_order) if col_order else 0)
        )
        print(cross.to_string())
        print()


def print_overall_ranking(df):
    print('  === RANKING TOTAL (considerando todos os cenarios e sementes) ===')
    rank_rows = []
    for _, seed_grp in df.groupby(['prog_group', 'seed']):
        seed_grp = rank_within_seed(seed_grp)
        for _, row in seed_grp.iterrows():
            rank_rows.append({
                'planner': row['planner'],
                'rank': row['rank'],
            })
    rank_df = pd.DataFrame(rank_rows)
    ranking = (
        rank_df.groupby('planner')['rank']
        .agg(['mean', 'std', 'min', 'max', 'count'])
        .round(3)
        .sort_values('mean')
    )
    ranking.columns = ['avg_rank', 'std_rank', 'best', 'worst', 'count']
    ranking['avg_rank'] = ranking['avg_rank'].map('{:.3f}'.format)
    print(ranking.to_string())
    print()


def print_placement_overall(df):
    print('  === CONTAGEM TOTAL DE COLOCACOES ===')
    place_rows = []
    for _, seed_grp in df.groupby(['prog_group', 'seed']):
        seed_grp = rank_within_seed(seed_grp)
        for _, row in seed_grp.iterrows():
            place_rows.append({
                'planner': row['planner'],
                'rank': row['rank'],
            })
    place_df = pd.DataFrame(place_rows)
    cross = (
        place_df.groupby(['planner', 'rank'])
        .size()
        .unstack(fill_value=0)
    )
    col_order = sorted(cross.columns)
    cross = cross[col_order]
    cross = cross.reindex(
        sorted(cross.index,
               key=lambda p: sum((col_order.index(r) if r in col_order else 99) * cross.loc[p, r]
                                 for r in col_order) if col_order else 0)
    )
    print(cross.to_string())
    print()


def print_time_stats(df):
    print('  === TEMPO MEDIO POR ALGORITMO (CPU time) ===')
    rows = []
    for planner, g in df.groupby('planner'):
        elapsed = g['time_s'].dropna()
        rows.append(dict(
            planner=planner,
            count=len(elapsed),
            mean_time=f'{elapsed.mean():.2f}s' if len(elapsed) else 'nan',
            std_time=f'{elapsed.std():.2f}s' if len(elapsed) > 1 else 'nan',
            min_time=f'{elapsed.min():.2f}s' if len(elapsed) else 'nan',
            max_time=f'{elapsed.max():.2f}s' if len(elapsed) else 'nan',
            total_time=f'{elapsed.sum():.2f}s' if len(elapsed) else 'nan',
        ))
    tbl = pd.DataFrame(rows).set_index('planner')
    tbl = tbl.sort_values('mean_time')
    print(tbl.to_string())
    print()


def print_radar_chart(df):
    plot_dir = RESULTS_ROOT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    rank_rows = []
    for _, seed_grp in df.groupby(['prog_group', 'seed']):
        seed_grp = rank_within_seed(seed_grp)
        for _, row in seed_grp.iterrows():
            rank_rows.append({
                'prog_group': row['prog_group'],
                'planner': row['planner'],
                'rank': row['rank'],
            })
    rank_df = pd.DataFrame(rank_rows)
    avg = rank_df.groupby(['prog_group', 'planner'])['rank'].mean().reset_index()

    groups = sorted(avg['prog_group'].unique())
    group_labels = [prog_group_label(g) for g in groups]
    planners_avail = sorted(avg['planner'].unique())

    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for planner in planners_avail:
        sub = avg[avg['planner'] == planner]
        style = PLANNER_STYLE.get(planner, {})
        color = style.get('color', 'gray')
        raw_ls = style.get('ls', '-')
        ls = re.sub(r'[xod\^v<>sp\*hH+D|_]', '', raw_ls) or '-'
        label = style.get('label', planner)
        vals = [sub.loc[sub['prog_group'] == g, 'rank'].values[0] if g in sub['prog_group'].values else np.nan
                for g in groups]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', color=color, ls=ls, lw=1.5, label=label, markersize=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_ylabel('Avg rank')
    # ax.set_title('Average rank per scenario (radar)', pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    out_path = plot_dir / 'radar_rank.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Radar chart saved to {out_path}')


def _block_pivot(df, value_col, aggfunc, mask=None):
    """Pivot planner x prog_group for a given column (optionally masked)."""
    sub = df if mask is None else df[mask]
    return sub.pivot_table(index='planner', columns='prog_group',
                           values=value_col, aggfunc=aggfunc)


def print_block_heatmaps(df):
    """Per-block heatmaps: rank, feasibility, normalized length and time.

    Rows are planners (ordered by overall mean rank), columns are the scenario
    blocks (`prog00`..`prog09`).  Each cell is annotated with the underlying
    value; colour encodes the metric (green = better).  Length and time are
    normalized per block against the best planner of that block, so the colour
    shows the *relative* penalty irrespective of the block's absolute scale.
    """
    plot_dir = RESULTS_ROOT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(df['prog_group'].unique())
    col_labels = [prog_group_label(g) for g in groups]

    planners = _block_mean_rank(df).sort_values().index.tolist()
    labels = [PLANNER_STYLE.get(p, {}).get('label', p) for p in planners]

    # rank per block
    rank_rows = []
    for _, block in df.groupby(['prog_group', 'seed']):
        block = rank_within_seed(block)
        for _, r in block.iterrows():
            rank_rows.append(dict(prog_group=r['prog_group'],
                                  planner=r['planner'], rank=r['rank']))
    rank_p = (pd.DataFrame(rank_rows)
              .pivot_table(index='planner', columns='prog_group',
                           values='rank', aggfunc='mean'))

    feas_p = _block_pivot(df, 'feasible', 'mean')
    len_p = _block_pivot(df, 'length', 'mean',
                         mask=(df['feasible'] & df['success']))
    time_p = _block_pivot(df, 'time_s', 'mean')

    idx, cols = planners, groups
    rank_p = rank_p.reindex(index=idx, columns=cols)
    feas_p = feas_p.reindex(index=idx, columns=cols)
    len_p = len_p.reindex(index=idx, columns=cols)
    time_p = time_p.reindex(index=idx, columns=cols)

    # per-block normalization against the best planner
    len_norm = len_p.div(len_p.min(axis=0), axis=1)
    time_norm = time_p.div(time_p.min(axis=0), axis=1)
    time_log = np.log10(time_norm.replace(0, np.nan))

    panels = [
        (rank_p.values, 'Mean rank  (lower is better)', 'RdYlGn_r',
         rank_p.values, lambda v: f'{v:.1f}', None),
        (feas_p.values, 'Feasibility rate  (higher is better)', 'RdYlGn',
         feas_p.values, lambda v: f'{v*100:.0f}%', (0.0, 1.0)),
        (len_norm.values, 'Length vs. best of block (x)', 'RdYlGn_r',
         len_p.values, lambda v: f'{v:.2f}', None),
        (time_log.values, 'Computation cost  (time vs. best of block, log scale)',
         'RdYlGn_r', time_p.values, lambda v: f'{v:.2f}s', None),
    ]

    n = len(planners)
    fig, axes = plt.subplots(2, 2, figsize=(16, max(5.0, 0.5 * n + 3.0)))
    axes = axes.ravel()
    for ax, (data, title, cmap, ann, fmt, vlim) in zip(axes, panels):
        if vlim is not None:
            vmin, vmax = vlim
        else:
            finite = data[np.isfinite(data)]
            vmin = float(np.nanmin(finite)) if finite.size else 0.0
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for i in range(n):
            for j in range(len(cols)):
                av = ann[i, j]
                if av is None or (isinstance(av, float) and np.isnan(av)):
                    continue
                ax.text(j, i, fmt(av), ha='center', va='center', fontsize=7,
                        color='black')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if title.startswith('Computation cost'):
            cbar.set_label('log scale', fontsize=8)

    fig.tight_layout()
    out_path = plot_dir / 'block_heatmaps.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Per-block heatmaps saved to {out_path}')


def print_time_boxplot(df):
    plot_dir = RESULTS_ROOT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []
    medians = df.groupby('planner')['time_s'].median().sort_values()
    order = medians.index.tolist()

    for planner in order:
        vals = df.loc[df['planner'] == planner, 'time_s'].dropna()
        if len(vals) == 0:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clean = vals[(vals >= lower) & (vals <= upper)]
        data.append(clean)
        labels.append(PLANNER_STYLE.get(planner, {}).get('label', planner))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw=dict(height_ratios=[1, 2])
    )
    plt.subplots_adjust(hspace=0.08)

    colors = [PLANNER_STYLE.get(planner, {}).get('color', '#888888')
              for planner in order]

    for ax in (ax_bot, ax_top):
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        for median_line in bp['medians']:
            median_line.set_color('black')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('CPU time (s)')

    ax_bot.set_ylim(0, 16)
    ax_top.set_ylim(50, 270)
    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_bot.set_xticklabels(labels)
    ax_bot.tick_params(axis='x', rotation=25, labelsize=12)

    d = 0.5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle='none', color='k', clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

    # fig.suptitle('Execution time distribution by algorithm (outliers removed)',
                #  fontweight='bold')
    fig.tight_layout()
    out_path = plot_dir / 'boxplot_elapsed.pdf'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Boxplot saved to {out_path}')


def print_latex_table(df):
    rank_rows = []
    for _, seed_grp in df.groupby(['prog_group', 'seed']):
        seed_grp = rank_within_seed(seed_grp)
        for _, row in seed_grp.iterrows():
            rank_rows.append({
                'planner': row['planner'],
                'rank': row['rank'],
            })
    rank_df = pd.DataFrame(rank_rows)
    avg_rank = rank_df.groupby('planner')['rank'].mean()

    feas_rate = df.groupby('planner')['feasible'].mean()

    def iqr_mean_std(grp):
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = grp[(grp >= lo) & (grp <= hi)]
        return filtered.mean(), filtered.std()

    time_stats = df.groupby('planner')['time_s'].apply(lambda g: iqr_mean_std(g))
    avg_time = time_stats.apply(lambda x: x[0])
    std_time = time_stats.apply(lambda x: x[1])

    succ = df[df['success'] & df['feasible']]
    mean_len_per_scenario = succ.groupby(['prog_group', 'planner'])['length'].mean().unstack()
    # A planner only wins a block if it is feasible on *every* seed of that
    # block; otherwise its mean length would be computed over a biased subset
    # (only the easy seeds), letting a low-feasibility method appear "best".
    block_feas = df.groupby(['prog_group', 'planner'])['feasible'].mean().unstack()
    eligible = mean_len_per_scenario.where(block_feas.eq(1.0))
    n_scenarios = len(mean_len_per_scenario)
    block_best = eligible.min(axis=1)
    best_mask = eligible.eq(block_best, axis=0) & eligible.notna()
    best_count = best_mask.sum()
    norm_len_per_scenario = eligible.div(block_best, axis=0)
    avg_norm_len = norm_len_per_scenario.mean()

    planners_sorted = avg_rank.sort_values().index.tolist()
    best_col_label = f'Best {n_scenarios} scenarios'
    header = (f"{'Planner':40s} & {'Feasibility':12s} & {'Avg Time':18s}"
              f" & {best_col_label:22s} & {'Norm. Length':13s} & {'Avg Rank':8s} \\\\")
    print(header)
    print('\\hline')
    for p in planners_sorted:
        label = PLANNER_STYLE.get(p, {}).get('label', p)
        f = f'{feas_rate[p]:.2f}'
        t = f'{avg_time[p]:.2f}s $\\pm$ {std_time.get(p, 0):.2f}s'
        bc = f'{best_count.get(p, 0):.0f}/{n_scenarios}'
        nl = f'{avg_norm_len.get(p, 0):.3f}' if avg_norm_len.get(p, 0) > 0 else '---'
        r = f'{avg_rank[p]:.2f}'
        print(f'{label:40s} & {f:12s} & {t:10s} & {bc:22s} & {nl:13s} & {r:8s} \\\\')
    print()


def main():
    global RESULTS_ROOT, SELECT_BEST_GRID_CONFIG
    argv = sys.argv[1:]
    root_override = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--root' and i + 1 < len(argv):
            root_override = argv[i + 1]; i += 2
        elif a == '--no-best-config':
            SELECT_BEST_GRID_CONFIG = False; i += 1
        elif a == '--best-config':
            SELECT_BEST_GRID_CONFIG = True; i += 1
        elif not a.startswith('-'):
            root_override = a; i += 1
        else:
            i += 1
    if root_override:
        RESULTS_ROOT = Path(root_override)

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
        df = ensure_feasible(df)
        if df.empty:
            print(f'=== {exp_dir.name} ===  (vazio)')
            continue
        all_dfs.append(df)

    if not all_dfs:
        print('Nenhum dado para analisar.')
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['prog_group'] = combined['scenario'].apply(extract_prog_group)

    # ── Time accounting ──────────────────────────────────────────────
    # Prefer `cpu_time` (process CPU time, fair under joblib parallelism);
    # fall back to `elapsed` for summaries generated before that column
    # existed (where `elapsed` was wall-clock).
    if 'cpu_time' in combined.columns and combined['cpu_time'].notna().any():
        combined['time_s'] = combined['cpu_time'].fillna(combined['elapsed'])
        print('  [time] usando cpu_time (fallback: elapsed)')
    else:
        combined['time_s'] = combined['elapsed']
        print('  [time] cpu_time ausente; usando elapsed (wall-clock)')

    if SCENARIO_PREFIX:
        before = len(combined)
        combined = combined[combined['prog_group'].str.startswith(SCENARIO_PREFIX)]
        n_removed = before - len(combined)
        if n_removed:
            print(f'Filtrados {n_removed} registros fora do prefixo "{SCENARIO_PREFIX}"')

    # ── Reduce each swept variant to its best configuration ──────────
    # (uses the exhaustive grid from the paper experiment; see
    #  SELECT_BEST_GRID_CONFIG).  Must run after `time_s`/`prog_group` exist.
    if SELECT_BEST_GRID_CONFIG:
        print()
        combined, _picked = select_best_grid_configs(combined)

    # Keep only planners of interest (canonical names after the reduction).
    combined = filter_planners(combined)
    if combined.empty:
        print('Nenhum dado apos o filtro de planners.')
        return
    print()

    # Ranking por grupo (prog00, prog01, ...)
    print('=' * 70)
    print('=== RANKING POR GRUPO (media do rank por semente) ===')
    print_prog_ranking_table(combined)

    # Tabela de colocacoes por grupo
    print('=' * 70)
    print('=== COLOCACOES POR GRUPO ===')
    print_prog_placement_table(combined)

    # Ranking total
    print('=' * 70)
    print_overall_ranking(combined)

    # Colocacoes total
    print('=' * 70)
    print_placement_overall(combined)

    # Tempo medio
    print('=' * 70)
    print_time_stats(combined)

    # Boxplot
    print('=' * 70)
    print_time_boxplot(combined)

    # Radar chart
    print('=' * 70)
    print_radar_chart(combined)

    # Heatmaps por bloco (rank, factibilidade, comprimento, tempo)
    print('=' * 70)
    print('=== ANALISE POR BLOCO (heatmaps) ===')
    print_block_heatmaps(combined)

    # Tabela final LaTeX
    print('=' * 70)
    print('=== TABELA RESUMO (para LaTeX) ===')
    print_latex_table(combined)

    out = results_dir / 'combined_summary.csv'
    combined.to_csv(out, index=False)
    print(f'Combined CSV saved to {out}  ({len(combined)} records)')


if __name__ == '__main__':
    main()
