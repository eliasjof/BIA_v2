import sys, pickle, argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
from scenario_config import ScenarioConfig


PLANNER_STYLE = {
    'rrt_star':          dict(color='#2196F3', ls='-',  lw=1.5, label='RRT*'),
    'rrt_star_smooth':   dict(color='#00BCD4', ls='--', lw=2.0, label='RRT* + B-spline'),
    'rrt_star_dubins':   dict(color='#F44336', ls='-',  lw=1.5, label='RRT* Dubins'),
    'rrt_dubins_smooth': dict(color='#FF9800', ls='--', lw=2.0, label='RRT* Dubins + B-spline'),
    'bit_star_dubins':   dict(color='#9C27B0', ls='-.', lw=2.0, label='BIT* Dubins'),
    'de2d_nurbs':        dict(color='#4CAF50', ls='-',  lw=2.5, label='DE2D_NURBS'),
    'pso2d_nurbs':       dict(color='#795548', ls=':',  lw=2.5, label='PSO2D_NURBS'),
}


def get_workspace_config():
    c = ScenarioConfig()
    c.setup()
    return c


def plot_scenario(scenario_label, df_sub, paths_store, obstacles,
                  start=None, goal=None, radius=0.073,
                  save_dir='plots'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ws = get_workspace_config()
    if start is None:
        start = ws.start
    if goal is None:
        goal = ws.goal

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

    legend_lines = []
    legend_labels = []

    for _, row in df_sub.iterrows():
        pname = row['planner']
        pkey = f"{scenario_label}_{pname}"
        path = paths_store.get(pkey)

        style = PLANNER_STYLE.get(pname, dict(color='gray', ls='-', lw=1))
        color = style['color']
        ls = style['ls']
        lw = style['lw']

        if path is not None and len(path) >= 2:
            ax.plot(path[:, 0], path[:, 1], color=color, ls=ls, lw=lw, zorder=4)

        ok = row['success']
        length = row['length']
        kappa = row['max_kappa']
        col_free = row['collision_free']

        if ok and not np.isnan(length):
            parts = [f"L={length:.2f}m"]
            if not np.isnan(kappa):
                parts.append(f"\u03ba={kappa:.2f}")
            if not col_free:
                parts.append("COL!")
            metric_str = f"  [{', '.join(parts)}]"
        elif not ok:
            metric_str = "  [FAILED]"
        else:
            metric_str = ""

        lbl = f"{style['label']}{metric_str}"
        leg_line = plt.Line2D([0], [0], color=color, ls=ls, lw=lw)
        legend_lines.append(leg_line)
        legend_labels.append(lbl)

    ax.legend(legend_lines, legend_labels, fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor='gray')
    ax.set_title(f'Scenario: {scenario_label}', fontsize=13, fontweight='bold')

    fig.tight_layout()
    out_path = save_dir / f'{scenario_label}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')
    return out_path


def plot_all(results_dir='comparison_results', scenario_filter=None, show=False,
             start=None, goal=None, radius=0.073):
    results_dir = Path(results_dir)
    df = pd.read_pickle(results_dir / 'summary.pkl')
    with open(results_dir / 'paths.pkl', 'rb') as f:
        paths_store = pickle.load(f)
    obstacles_store = {}
    obs_path = results_dir / 'obstacles.pkl'
    if obs_path.exists():
        with open(obs_path, 'rb') as f:
            obstacles_store = pickle.load(f)

    scenarios = df['scenario'].unique()
    if scenario_filter:
        scenarios = [s for s in scenarios if scenario_filter in s]

    plot_dir = results_dir / 'plots'

    print(f'Plotting {len(scenarios)} scenario(s) from {results_dir} ...')
    for sc in scenarios:
        df_sub = df[df['scenario'] == sc]
        obstacles = obstacles_store.get(sc, np.array([]))
        plot_scenario(sc, df_sub, paths_store, obstacles,
                      start=start, goal=goal, radius=radius,
                      save_dir=plot_dir)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='no_obs',
                        help='Experiment subdirectory under comparison_results/')
    parser.add_argument('--filter', default=None,
                        help='Plot only scenarios whose name contains FILTER')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (blocks)')
    args = parser.parse_args()
    plot_all(results_dir=f'comparison_results/{args.exp}',
             scenario_filter=args.filter, show=args.show)
