import pickle, re
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
    'bit_star_dubins':   dict(color='#FFEB3B', ls='-.', lw=2.0, label='BIT* Dubins'),
    'bit_star_theta':    dict(color='#E91E63', ls='--', lw=2.0, label='BIT* Theta'),
    'de2d_nurbs':        dict(color='#4CAF50', ls='-',  lw=2.5, label='DE2D_NURBS'),
    'pso2d_nurbs':       dict(color='#795548', ls=':',  lw=2.5, label='PSO2D_NURBS'),
    'rrt_star_asv':      dict(color='#9C27B0', ls='-d',  lw=2.0, label='RRT* ASV'),
}


def prog_group_label(g):
    m = re.match(r'prog(\d+)', g)
    if m:
        n = int(m.group(1))
        return f'scen-{n}obs'
    return g


RESULTS_ROOT = Path(__file__).parent / 'comparison_results'

SCENARIO_PREFIX = 'prog'

SELECTED_PLANNERS = [
    'rrt_star',
    'rrt_star_smooth',
    'rrt_star_dubins',
    'rrt_dubins_smooth',
    'modified_dubins_rrt_star',
    'modified_dubins_rrt_star_ccpoa',
    'bit_star_dubins',
    'de2d_nurbs',
    'pso2d_nurbs',
    'rrt_star_asv',
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
    'rrt_star_dubins', 'bit_star_dubins', 'rrt_dubins_smooth',
    'modified_dubins_rrt_star', 'modified_dubins_rrt_star_ccpoa',
    'rrt_star_asv',
}

NO_CURV_PLANNERS = {'rrt_star'}


def ensure_feasible(df):
    df = df.copy()
    if 'collision_free' not in df.columns:
        df['collision_free'] = True
    feas_list = []
    for _, row in df.iterrows():
        col_free = row.get('collision_free', True)
        if isinstance(col_free, float):
            col_free = bool(col_free) if not np.isnan(col_free) else True
        if not col_free:
            feas_list.append(False)
        elif row['planner'] in DUBINS_PLANNERS:
            feas_list.append(True)
        elif row['planner'] in NO_CURV_PLANNERS:
            feas_list.append(False)
        else:
            cv = row.get('cv')
            if cv is not None and not (isinstance(cv, float) and np.isnan(cv)):
                feas_list.append(float(cv) < CURV_FEAS_THRESHOLD)
            elif 'feasible' in df.columns and not pd.isna(row.get('feasible', np.nan)):
                feas_list.append(row['feasible'])
            else:
                feas_list.append(col_free)
    df['feasible'] = feas_list
    if 'cv' not in df.columns:
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
        ['feasible', 'length', 'elapsed'],
        ascending=[False, True, True],
        na_position='last',
    )
    grp = grp.copy()
    grp['rank'] = range(1, n + 1)
    grp.loc[~grp['feasible'], 'rank'] = n
    return grp


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
    print('  === TEMPO MEDIO POR ALGORITMO ===')
    rows = []
    for planner, g in df.groupby('planner'):
        elapsed = g['elapsed'].dropna()
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


def print_time_boxplot(df):
    plot_dir = RESULTS_ROOT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []
    medians = df.groupby('planner')['elapsed'].median().sort_values()
    order = medians.index.tolist()

    for planner in order:
        vals = df.loc[df['planner'] == planner, 'elapsed'].dropna()
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
        ax.set_ylabel('Time (s)')

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

    time_stats = df.groupby('planner')['elapsed'].apply(lambda g: iqr_mean_std(g))
    avg_time = time_stats.apply(lambda x: x[0])
    std_time = time_stats.apply(lambda x: x[1])

    succ = df[df['success'] & df['feasible']]
    mean_len_per_scenario = succ.groupby(['prog_group', 'planner'])['length'].mean().unstack()
    n_scenarios = len(mean_len_per_scenario)
    best_mask = mean_len_per_scenario.eq(mean_len_per_scenario.min(axis=1), axis=0)
    best_count = best_mask.sum()
    norm_len_per_scenario = mean_len_per_scenario.div(mean_len_per_scenario.min(axis=1), axis=0)
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
        df = filter_planners(df)
        df = ensure_feasible(df)
        if df.empty:
            print(f'=== {exp_dir.name} ===  (vazio apos filtro)')
            continue
        all_dfs.append(df)

    if not all_dfs:
        print('Nenhum dado para analisar.')
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['prog_group'] = combined['scenario'].apply(extract_prog_group)

    if SCENARIO_PREFIX:
        before = len(combined)
        combined = combined[combined['prog_group'].str.startswith(SCENARIO_PREFIX)]
        n_removed = before - len(combined)
        if n_removed:
            print(f'Filtrados {n_removed} registros fora do prefixo "{SCENARIO_PREFIX}"')
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

    # Tabela final LaTeX
    print('=' * 70)
    print('=== TABELA RESUMO (para LaTeX) ===')
    print_latex_table(combined)

    out = results_dir / 'combined_summary.csv'
    combined.to_csv(out, index=False)
    print(f'Combined CSV saved to {out}  ({len(combined)} records)')


if __name__ == '__main__':
    main()
