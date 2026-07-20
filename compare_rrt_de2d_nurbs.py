import pickle, time, copy, gc
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_config import ScenarioConfig
from rrt_based.rrt_planner import RRTPlanner, angle_mod
from de2d_nurbs import DE2D_NURBS


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────

def path_length(path):
    if path is None or len(path) < 2:
        return np.nan
    a = np.asarray(path)
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def max_curvature_numerical(path):
    if path is None or len(path) < 4:
        return np.nan
    a = np.asarray(path)
    dx = np.gradient(a[:, 0])
    dy = np.gradient(a[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-10)
    return float(k.max())


def is_collision_free(path, obstacle_list, robot_radius):
    if path is None or len(path) < 2:
        return False
    a = np.asarray(path)
    for ox, oy, size in obstacle_list:
        d = np.hypot(a[:, 0] - ox, a[:, 1] - oy)
        if (d <= size + robot_radius + 1e-9).any():
            return False
    return True


def check_collision_path(path, obstacle_list, robot_radius):
    return is_collision_free(path, obstacle_list, robot_radius)


# ──────────────────────────────────────────────
# Planner runners
# ──────────────────────────────────────────────

def run_rrt(config, planner_type='rrt_star', extra_kw=None, seed=None):
    kw = dict(
        max_iter=2000, connect_circle_dist=2.0,
        goal_sample_rate=20, search_until_max_iter=True,
    )
    if extra_kw:
        kw.update(extra_kw)
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    p = RRTPlanner(config, planner_type, **kw)
    p.run(animation=False)
    raw = p.get_best_path()
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw_arr = np.asarray(raw)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)

    return dict(path=raw_arr, raw_path=raw_arr, elapsed=elapsed, success=True,
                length=length, max_kappa=np.nan, collision_free=col_free)


def run_rrt_star(config, seed=None):
    return run_rrt(config, planner_type='rrt_star', seed=seed)


def run_rrt_star_dubins(config, seed=None):
    kw = dict(max_iter=300, connect_circle_dist=4.5,
              goal_sample_rate=20, path_resolution=0.05,
              random_yaw_strategy='toward_goal',
              search_until_max_iter=True)
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    p = RRTPlanner(config, 'rrt_star_dubins', **kw)
    p.run(animation=False)
    raw = p.get_best_path()
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw_arr = np.asarray(raw)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)

    try:
        k_anal, _ = p.planner.get_curvature_analytical()
        max_k = float(k_anal.max())
    except Exception:
        max_k = max_curvature_numerical(raw_arr)

    return dict(path=raw_arr, raw_path=raw_arr, elapsed=elapsed, success=True,
                length=length, max_kappa=max_k, collision_free=col_free)


def smooth_path_rrt(planner, raw_path, n_waypoints=20, s=0.015):
    if raw_path is None or len(raw_path) < 4:
        return None
    # Temporarily set _path so smooth_path() works
    planner._path = raw_path.tolist() if hasattr(raw_path, 'tolist') else list(raw_path)
    try:
        smooth = planner.smooth_path(n_waypoints=n_waypoints, s=s)
        return np.asarray(smooth) if smooth is not None else None
    except Exception:
        return None


def run_rrt_star_smooth(config, seed=None, n_waypoints=20, s=0.015):
    kw = dict(max_iter=2000, connect_circle_dist=2.0,
              goal_sample_rate=20, search_until_max_iter=True)
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    p = RRTPlanner(config, 'rrt_star', **kw)
    p.run(animation=False)
    raw = p.get_best_path()
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw_arr = np.asarray(raw)
    smooth_arr = smooth_path_rrt(p, raw_arr, n_waypoints, s)
    result = dict(raw_path=raw_arr, elapsed=elapsed)

    if smooth_arr is not None and len(smooth_arr) >= 2:
        length = path_length(smooth_arr)
        k_max = max_curvature_numerical(smooth_arr)
        col_free = is_collision_free(smooth_arr, config.obs, config.radius)
        result.update(path=smooth_arr, success=True,
                      length=length, max_kappa=k_max, collision_free=col_free)
    else:
        result.update(path=raw_arr, success=True,
                      length=path_length(raw_arr),
                      max_kappa=np.nan,
                      collision_free=is_collision_free(raw_arr, config.obs, config.radius))
    return result


def run_rrt_dubins_smooth(config, seed=None, n_waypoints=20, s=0.015):
    kw = dict(max_iter=300, connect_circle_dist=4.5,
              goal_sample_rate=20, path_resolution=0.05,
              random_yaw_strategy='toward_goal',
              search_until_max_iter=True)
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    p = RRTPlanner(config, 'rrt_star_dubins', **kw)
    p.run(animation=False)
    raw = p.get_best_path()
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw_arr = np.asarray(raw)
    # Analytical curvature from Dubins tree
    try:
        k_anal, _ = p.planner.get_curvature_analytical()
        max_k_raw = float(k_anal.max())
    except Exception:
        max_k_raw = max_curvature_numerical(raw_arr)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)
    result = dict(raw_path=raw_arr, elapsed=elapsed, success=True,
                  length=length, max_kappa=max_k_raw, collision_free=col_free)

    # Smoothed version
    smooth_arr = smooth_path_rrt(p, raw_arr, n_waypoints, s)
    if smooth_arr is not None and len(smooth_arr) >= 2:
        length_s = path_length(smooth_arr)
        k_s = max_curvature_numerical(smooth_arr)
        col_s = is_collision_free(smooth_arr, config.obs, config.radius)
        result_s = dict(path=smooth_arr, success=True,
                        length=length_s, max_kappa=k_s, collision_free=col_s)
        result_s.update(elapsed=elapsed, raw_path=raw_arr)
        return result_s
    return result


def run_de2d_nurbs(config, seed=None):
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    de = DE2D_NURBS(config)
    try:
        de.run()
    except Exception:
        return dict(path=None, raw_path=None, elapsed=time.perf_counter() - t0,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False)
    elapsed = time.perf_counter() - t0

    result = de.get_best_path()
    if result is None:
        return dict(path=None, raw_path=None, elapsed=elapsed,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False)

    pts = np.asarray(result['points'])
    length = path_length(pts)
    k_max = max_curvature_numerical(pts)
    col_free = is_collision_free(pts, config.obs, config.radius)

    return dict(path=pts, raw_path=pts, elapsed=elapsed, success=True,
                length=length, max_kappa=k_max, collision_free=col_free)


# ──────────────────────────────────────────────
# Scenario generation helpers
# ──────────────────────────────────────────────

PLANNERS = [
    ('rrt_star',         run_rrt_star),
    ('rrt_star_smooth',  run_rrt_star_smooth),
    ('rrt_star_dubins',  run_rrt_star_dubins),
    ('rrt_dubins_smooth', run_rrt_dubins_smooth),
    ('de2d_nurbs',       run_de2d_nurbs),
]


def make_scenario(seed, obs_fornecida=None, start=None, goal=None,
                  th_start=0.0, th_goal=0.0, radius=0.073, kappa_max=1/0.73):
    config = ScenarioConfig()
    config.seed = seed
    config.radius = radius
    config.kappa_max = kappa_max
    config.th_start = th_start
    config.th_goal = th_goal
    if start is not None:
        config.start = np.asarray(start, dtype=float)
    if goal is not None:
        config.goal = np.asarray(goal, dtype=float)
    if obs_fornecida is not None:
        config.obs_fornecida = list(obs_fornecida)
    return config


# ──────────────────────────────────────────────
# Main comparison loop
# ──────────────────────────────────────────────

def run_comparison(scenario_configs, planners=PLANNERS,
                   output_dir='comparison_results', verbose=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    paths_store = {}

    for sc_idx, sc in enumerate(scenario_configs):
        seed = sc['seed']
        obs = sc.get('obs_fornecida')
        label = sc.get('label', f'seed{seed}')

        if verbose:
            print(f'\n{"="*60}')
            print(f'Scenario {sc_idx+1}/{len(scenario_configs)}: {label}')
            print(f'  seed={seed}  obstacles={"yes" if obs else "no"}')

        for pname, runner_fn in planners:
            if verbose:
                print(f'  Running {pname:20s} ...', end=' ', flush=True)

            # Fresh config for each run to avoid _setup_done issues
            c = make_scenario(
                seed=seed, obs_fornecida=obs,
                start=sc.get('start'), goal=sc.get('goal'),
                th_start=sc.get('th_start', 0.0),
                th_goal=sc.get('th_goal', 0.0),
                radius=sc.get('radius', 0.073),
                kappa_max=sc.get('kappa_max', 1/0.73),
            )

            t_start = time.perf_counter()
            try:
                result = runner_fn(c, seed=seed)
            except Exception as e:
                if verbose:
                    print(f'ERROR: {e}')
                result = dict(path=None, raw_path=None, success=False,
                              length=np.nan, max_kappa=np.nan,
                              collision_free=False)
            total_time = time.perf_counter() - t_start
            result.setdefault('elapsed', total_time)
            result.setdefault('success', False)
            result.setdefault('length', np.nan)
            result.setdefault('max_kappa', np.nan)
            result.setdefault('collision_free', False)

            record = dict(
                scenario=label, seed=seed, planner=pname,
                success=result['success'],
                length=result.get('length', np.nan),
                max_kappa=result.get('max_kappa', np.nan),
                elapsed=result.get('elapsed', total_time),
                collision_free=result.get('collision_free', False),
                n_obs=len(c.obs) if hasattr(c, 'obs') else 0,
            )
            all_records.append(record)

            # Store path for later plotting
            path_key = f'{label}_{pname}'
            raw = result.get('raw_path')
            path_data = result.get('path')
            # Store the actual path (smoothed if applicable)
            if path_data is not None and len(path_data) >= 2:
                paths_store[path_key] = np.asarray(path_data)
            elif raw is not None and len(raw) >= 2:
                paths_store[path_key] = np.asarray(raw)
            else:
                paths_store[path_key] = None

            if verbose:
                status = 'OK' if record['success'] else 'FAIL'
                print(f'{status}  len={record["length"]:.2f}m  '
                      f'k={record["max_kappa"]:.2f}  '
                      f't={record["elapsed"]:.1f}s'
                      f'  col={"" if record["collision_free"] else "COL!"}')

    df = pd.DataFrame(all_records)
    return df, paths_store


def save_results(df, paths_store, output_dir='comparison_results'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_dir / 'summary.pkl')
    df.to_csv(output_dir / 'summary.csv', index=False)

    with open(output_dir / 'paths.pkl', 'wb') as f:
        pickle.dump(paths_store, f)

    print(f'\nResults saved to {output_dir}/')
    print(f'  summary.pkl / summary.csv  ({len(df)} records)')
    print(f'  paths.pkl                  ({len(paths_store)} entries)')


def load_results(output_dir='comparison_results'):
    output_dir = Path(output_dir)
    df = pd.read_pickle(output_dir / 'summary.pkl')
    with open(output_dir / 'paths.pkl', 'rb') as f:
        paths_store = pickle.load(f)
    return df, paths_store


# ──────────────────────────────────────────────
# Quick summary & plotting helpers
# ──────────────────────────────────────────────

def print_summary(df):
    print('\n=== SUMMARY ===')
    grouped = df.groupby('planner').agg(
        success_rate=('success', 'mean'),
        avg_length=('length', 'mean'),
        std_length=('length', 'std'),
        avg_kappa=('max_kappa', 'mean'),
        max_kappa=('max_kappa', 'max'),
        avg_time=('elapsed', 'mean'),
        collision_free_rate=('collision_free', 'mean'),
        count=('success', 'count'),
    ).round(3)
    print(grouped.to_string())


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def generate_chicane_obstacles():
    return [
        (0.6883004767862384, 0.3022825881539657, 0.14464214762976454),
        (-1.0917736086156131, -0.1126768217023627, 0.2784359135409691),
        (-0.9002903939515345, 0.26820105644434156, 0.10530719393677274),
        (-1.3207217472358725, 0.4514588866302939, 0.2618860913355653),
        (-0.4346374873469, -0.5239275669138156, 0.23962787899764537),
        (1.0290397406091127, 0.18269405408555595, 0.11934327536669281),
        (-0.3171168234468903, 0.07341651282804262, 0.29462315279587414),
    ]


def main():
    """Entry point — define scenarios and run comparison."""
    # Smaller seed set for quick validation
    seeds = [2, 4, 5]

    # 1) No-obstacle scenario
    scenarios = [
        dict(seed=s, obs_fornecida=[], label=f'no_obs_seed{s}')
        for s in seeds
    ]

    # 2) With obstacles (chicane from the problem)
    chicane_obs = generate_chicane_obstacles()
    scenarios += [
        dict(seed=s, obs_fornecida=chicane_obs, label=f'obs_chicane_seed{s}')
        for s in seeds
    ]

    df, paths_store = run_comparison(scenarios, output_dir='comparison_results')
    save_results(df, paths_store)
    print_summary(df)
    return df, paths_store


if __name__ == '__main__':
    main()
