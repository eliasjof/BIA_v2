import sys, pickle, argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from scenario_config import ScenarioConfig


def _curvature_and_arc(pts):
    a = np.asarray(pts)
    # Menger curvature of circumcircle through (i-1, i, i+1)
    p0 = a[:-2]; p1 = a[1:-1]; p2 = a[2:]
    ax = p1[:, 0] - p0[:, 0]; ay = p1[:, 1] - p0[:, 1]
    bx = p2[:, 0] - p1[:, 0]; by = p2[:, 1] - p1[:, 1]
    cx = p2[:, 0] - p0[:, 0]; cy = p2[:, 1] - p0[:, 1]
    cross = np.abs(ax * by - bx * ay)
    a_len = np.hypot(ax, ay); b_len = np.hypot(bx, by); c_len = np.hypot(cx, cy)
    denom = a_len * b_len * c_len
    k = np.where(denom > 1e-12, 2.0 * cross / denom, 0.0)
    k = np.pad(k, (1, 1), constant_values=0.0)  # endpoints have no Menger curvature
    # Arc length
    diff = np.diff(a, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    s = np.zeros(len(a))
    s[1:] = np.cumsum(seg)
    return k, s


PLANNER_STYLE = {
    'rrt_star':          dict(color='#2196F3', ls='-x',  lw=1.5, label='RRT*'),
    'rrt_star_smooth':   dict(color='#00BCD4', ls='-', lw=2.0, label='RRT* + B-spline'),
    'rrt_star_dubins':   dict(color='#F44336', ls='-',  lw=1.5, label='RRT* Dubins'),
    'rrt_dubins_smooth': dict(color='#FF9800', ls='-.', lw=2.0, label='RRT* Dubins + B-spline'),
    'modified_dubins_rrt_star': dict(color="#00FF62", ls='--', lw=2.0, label='Modified Dubins-RRT*'),
    'modified_dubins_rrt_star_ccpoa': dict(color="#0A4922", ls='--', lw=2.0, label='Modified Dubins-RRT* + CCPOA'),
    'bit_star_dubins':   dict(color='#FFEB3B', ls='-.', lw=2.0, label='BIT* Dubins'),
    'bit_star_theta':    dict(color='#E91E63', ls='--', lw=2.0, label='BIT* Theta'),
    'de2d_nurbs':        dict(color='#4CAF50', ls='-',  lw=2.5, label='DE2D_NURBS'),
    'pso2d_nurbs':       dict(color='#795548', ls=':',  lw=2.5, label='PSO2D_NURBS'),
    'rrt_star_asv':      dict(color='#9C27B0', ls='-d',  lw=2.0, label='RRT* ASV'),
}


def _get_scenario_config(sc_cfg):
    """Return a namespace object from a config dict (from config.pkl)."""
    return SimpleNamespace(
        start=np.array([sc_cfg['start_x'], sc_cfg['start_y']]),
        goal=np.array([sc_cfg['goal_x'], sc_cfg['goal_y']]),
        th_start=sc_cfg.get('th_start', 0.0),
        th_goal=sc_cfg.get('th_goal', 0.0),
        kappa_max=sc_cfg['kappa_max'],
        radius=sc_cfg.get('radius', 0.073),
        xmin=sc_cfg['xmin'], xmax=sc_cfg['xmax'],
        ymin=sc_cfg['ymin'], ymax=sc_cfg['ymax'],
    )


def _get_scenario_config_from_df(df_sub):
    """Fallback: extract config from first row of scenario dataframe."""
    row = df_sub.iloc[0]
    return SimpleNamespace(
        start=np.array([row['start_x'], row['start_y']]),
        goal=np.array([row['goal_x'], row['goal_y']]),
        th_start=row.get('th_start', 0.0),
        th_goal=row.get('th_goal', 0.0),
        kappa_max=row['kappa_max'],
        radius=row.get('radius', 0.073),
        xmin=row['xmin'], xmax=row['xmax'],
        ymin=row['ymin'], ymax=row['ymax'],
    )


def get_workspace_config():
    """Fallback: create default ScenarioConfig when config.pkl / df cols absent."""
    c = ScenarioConfig()
    c.setup()
    return c


def read_df_safe(results_dir):
    """Try pickle, fall back to CSV (handles pandas version mismatches)."""
    pkl = results_dir / 'summary.pkl'
    csv = results_dir / 'summary.csv'
    if pkl.is_file():
        try:
            return pd.read_pickle(pkl)
        except Exception:
            print('  Warning: pickle version mismatch, falling back to CSV')
    if csv.is_file():
        df = pd.read_csv(csv)
        for col in ('success', 'collision_free', 'feasible'):
            if col in df.columns:
                df[col] = df[col].astype(bool)
        return df
    raise FileNotFoundError(f'No summary.pkl or summary.csv in {results_dir}')


def plot_scenario(scenario_label, df_sub, paths_store, obstacles,
                  start=None, goal=None, radius=0.073,
                  save_dir='plots', planners=None, sc_config=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if sc_config:
        ws = _get_scenario_config(sc_config)
    elif {'start_x', 'start_y', 'goal_x', 'goal_y',
          'kappa_max', 'xmin', 'xmax', 'ymin', 'ymax'}.issubset(df_sub.columns):
        ws = _get_scenario_config_from_df(df_sub)
    else:
        ws = get_workspace_config()
    if start is None:
        start = ws.start
    if goal is None:
        goal = ws.goal

    has_feas = 'feasible' in df_sub.columns

    if planners is not None:
        df_sub = df_sub[df_sub['planner'].isin(planners)]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.set_xlim(ws.xmin, ws.xmax)
    ax.set_ylim(ws.ymin, ws.ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    if obstacles is not None and len(obstacles):
        for ob in obstacles:
            ox, oy, size = ob[0], ob[1], ob[2]
            ax.add_patch(plt.Circle((ox, oy), size,
                                    color='#8BC34A', alpha=0.35,
                                    ec='#558B2F', lw=1))
            ax.add_patch(plt.Circle((ox, oy), size + radius,
                                    color='#FFCC80', alpha=0.15, ec='none'))

    ax.plot(start[0], start[1], 's', color='#2E7D32',
            markersize=10, zorder=5, label='Start')
    ax.plot(goal[0], goal[1], '*', color='#C62828',
            markersize=14, zorder=5, label='Goal')


    # setas de heading
    dx_s = 0.15 * np.cos(ws.th_start)
    dy_s = 0.15 * np.sin(ws.th_start)
    ax.arrow(ws.start[0], ws.start[1], dx_s, dy_s,
                head_width=0.05, head_length=0.05, color='#2E7D32')
    dx_g = 0.15 * np.cos(ws.th_goal)
    dy_g = 0.15 * np.sin(ws.th_goal)
    ax.arrow(ws.goal[0], ws.goal[1], dx_g, dy_g,
                head_width=0.05, head_length=0.05, color='#C62828')
    legend_lines = []
    legend_labels = []

    for _, row in df_sub.iterrows():
        pname = row['planner']
        pkey = f"{scenario_label}_{pname}"
        path = paths_store.get(pkey)

        style = PLANNER_STYLE.get(pname, dict(color='gray', ls='-', lw=1))

        if path is not None and len(path) >= 2:
            lw = style['lw']
            if has_feas and not row['feasible']:
                lw = style['lw']
                ax.plot(path[:, 0], path[:, 1], color=style['color'],
                        ls=style['ls'], lw=lw, alpha=0.5, zorder=4)
            else:
                ax.plot(path[:, 0], path[:, 1], color=style['color'],
                        ls=style['ls'], lw=lw, zorder=4)

        ok = row['success']
        length = row['length']
        col_free = row['collision_free']

        if ok and not np.isnan(length):
            parts = [f"L={length:.2f}m"]
            if has_feas:
                sym = '\u2713' if row['feasible'] else '\u2717'
                parts.append(f'CV={row["cv"]:.1e}' if not pd.isna(row['cv']) else 'CV=nan')
                parts.append(f'{sym}')
            if not col_free:
                parts.append("COL!")
            metric_str = f"  [{', '.join(parts)}]"
        elif not ok:
            metric_str = "  [FAILED]"
        else:
            metric_str = ""

        lbl = f"{style['label']}{metric_str}"
        leg_line = plt.Line2D([0], [0], color=style['color'], ls=style['ls'], lw=style['lw'])
        legend_lines.append(leg_line)
        legend_labels.append(lbl)

    ax.legend(legend_lines, legend_labels, fontsize=7, loc='upper left',
              framealpha=0.9, edgecolor='gray')
    


    # ── Curvature plot ──
    curv_fig, curv_ax = plt.subplots(figsize=(9, 5))
    curv_ax.axhline(ws.kappa_max, color='gray', ls='--', lw=1, label=f'$\\kappa_{{max}}={ws.kappa_max:.2f}$')
    for _, row in df_sub.iterrows():
        pname = row['planner']
        pkey = f"{scenario_label}_{pname}"
        path = paths_store.get(pkey)
        style = PLANNER_STYLE.get(pname, dict(color='gray', ls='-', lw=1.5))
        if path is not None and len(path) >= 3:
            k, s = _curvature_and_arc(path)
            curv_ax.plot(s, k, color=style['color'], ls=style['ls'], lw=style['lw'],
                         label=style['label'])
    curv_ax.set_xlabel('Arc length (m)', fontsize=12)
    curv_ax.set_ylabel('Curvature $\\kappa$ (m$^{-1}$)', fontsize=12)
    curv_ax.tick_params(labelsize=10)
    curv_ax.grid(True, alpha=0.3)
    curv_ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray')
    curv_fig.tight_layout()
    curv_path = save_dir / f'curvature_{scenario_label}.png'
    curv_fig.savefig(curv_path, dpi=150, bbox_inches='tight')
    plt.close(curv_fig)
    print(f'  Saved {curv_path}')

    ax.axis([-2,2, -2,2])
    fig.tight_layout()
    out_path = save_dir / f'{scenario_label}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')
    return out_path


def _load_pickle_safe(path):
    if not path.is_file():
        return {}
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'  Warning: could not load {path.name}: {e}')
        return {}


def plot_all(results_dir='comparison_results', scenario_filter=None, show=False,
             start=None, goal=None, radius=0.073, planners=None):
    results_dir = Path(results_dir)
    df = read_df_safe(results_dir)
    paths_store = _load_pickle_safe(results_dir / 'paths.pkl')
    obstacles_store = _load_pickle_safe(results_dir / 'obstacles.pkl')
    configs = _load_pickle_safe(results_dir / 'config.pkl')

    scenarios = df['scenario'].unique()
    if scenario_filter:
        scenarios = [s for s in scenarios if scenario_filter in s]

    plot_dir = results_dir / 'plots'

    print(f'Plotting {len(scenarios)} scenario(s) from {results_dir} ...')
    for sc in scenarios:
        df_sub = df[df['scenario'] == sc]
        obstacles = obstacles_store.get(sc, np.array([]))
        sc_cfg = configs.get(sc) if configs else None
        plot_scenario(sc, df_sub, paths_store, obstacles,
                      start=start, goal=goal, radius=radius,
                      save_dir=plot_dir, planners=planners,
                      sc_config=sc_cfg)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='no_obs',
                        help='Experiment subdirectory under comparison_results/')
    parser.add_argument('--filter', default=None,
                        help='Plot only scenarios whose name contains FILTER')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (blocks)')
    parser.add_argument('--planners', nargs='*', default=None,
                        help='(opcional) sobrescreve a lista fixa abaixo')
    args = parser.parse_args()

    # Lista de planners para plotar (comente/descomente os que desejar)
    planners = args.planners if args.planners is not None else [
        'rrt_star',
        'rrt_star_smooth',
        'rrt_star_dubins',
        'rrt_dubins_smooth',
        'modified_dubins_rrt_star',
        'modified_dubins_rrt_star_ccpoa',
        'bit_star_dubins',
        'bit_star_theta',
        'de2d_nurbs',
        'pso2d_nurbs',
        'rrt_star_asv',
    ]

    plot_all(results_dir=f'comparison_results/{args.exp}',
             scenario_filter=args.filter, show=args.show,
             planners=planners)
