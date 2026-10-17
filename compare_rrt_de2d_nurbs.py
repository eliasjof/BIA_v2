import os
import json
import pickle, time, copy, gc, sys
from pathlib import Path

SAFE_N_JOBS = 4

# Number of completed tasks between periodic checkpoint writes.  Lower = safer
# against power loss but more I/O.  Overridable via --save-every.
CHECKPOINT_SAVE_EVERY = 20
CHECKPOINT_NAME = 'checkpoint.pkl'

# Prevent BLAS/OpenMP oversubscription when running many joblib workers
# (e.g. n_jobs=21 on 32 cores).  Without this each worker may spawn its own
# BLAS threads -> N_workers * N_threads threads, which destroys both the
# wall-clock and the fairness of the timing.  Must be set before numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scenario_config import ScenarioConfig
from rrt_based.rrt_planner_modified import (
    RRTPlanner, angle_mod, ModifiedDubinsRRTStar, RRTStarASV,
    extend_path_start_goal, _extended_start_goal_dist)
from rrt_based.ccpoa import CCPOA
from de2d_nurbs import DE2D_NURBS
try:
    from pso2d_nurbs import PSO2D_NURBS
    _HAVE_PSO = True
except ImportError:
    _HAVE_PSO = False


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
    # Menger curvature: curvature of circumcircle through (i-1, i, i+1)
    # vectorized: area of triangle (i-1, i, i+1) gives curvature via 4*area/(a*b*c)
    p0 = a[:-2]  # i-1
    p1 = a[1:-1]  # i
    p2 = a[2:]   # i+1
    ax = p1[:, 0] - p0[:, 0]
    ay = p1[:, 1] - p0[:, 1]
    bx = p2[:, 0] - p1[:, 0]
    by = p2[:, 1] - p1[:, 1]
    cx = p2[:, 0] - p0[:, 0]
    cy = p2[:, 1] - p0[:, 1]
    cross = np.abs(ax * by - bx * ay)
    a_len = np.hypot(ax, ay)
    b_len = np.hypot(bx, by)
    c_len = np.hypot(cx, cy)
    denom = a_len * b_len * c_len
    k = np.zeros_like(denom)
    np.divide(2.0 * cross, denom, out=k, where=denom > 1e-12)
    return float(k.max())


def is_collision_free(path, obstacle_list, robot_radius):
    if path is None or len(path) < 2:
        return False
    a = np.asarray(path)
    for ox, oy, size in obstacle_list:
        d = np.hypot(a[:, 0] - ox, a[:, 1] - oy)
        if (d + 1e-9 < size + robot_radius).any():
            return False
    return True


def check_collision_path(path, obstacle_list, robot_radius):
    return is_collision_free(path, obstacle_list, robot_radius)


def apply_extended_start_goal(path, config, step_size=0.05):
    if path is None or not getattr(config, 'extended_start_goal', False):
        return path
    return extend_path_start_goal(
        path,
        (float(config.start[0]), float(config.start[1])),
        (float(config.goal[0]), float(config.goal[1])),
        float(config.th_start), float(config.th_goal),
        _extended_start_goal_dist(config),
        step_size=step_size)


def _numerical_curvature_violation(pts, kappa_max):
    a = np.asarray(pts)
    dx = np.gradient(a[:, 0])
    dy = np.gradient(a[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-8)
    return float(np.sum(np.maximum(0, k - kappa_max) ** 2))


# ──────────────────────────────────────────────
# Planner runners
# ──────────────────────────────────────────────

# Extension step (the "steer") for the non-Dubins RRT* variants.  This is
# `expand_dis` in RRTStar; `step_size` is only meaningful for Dubins/ASV.
RRT_EXPAND_DIS = 0.3


def run_rrt(config, planner_type='rrt_star', extra_kw=None, seed=None):
    kw = dict(
        max_iter=2000, connect_circle_dist=2.0,
        goal_sample_rate=20, search_until_max_iter=True,
    )
    if extra_kw:
        kw.update(extra_kw)
    if seed is not None:
        config.seed = seed
    # Non-Dubins RRT* variants: always use extended start/goal so the
    # trajectory reaches start / leaves goal with the desired headings.
    config.extended_start_goal = True

    t0 = time.perf_counter()
    p = RRTPlanner(config, planner_type, **kw)
    p.run(animation=False)
    raw = p.get_best_path()
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False,
                    cv_opt=np.nan)

    raw_arr = np.asarray(raw)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)

    return dict(path=raw_arr, raw_path=raw_arr, elapsed=elapsed, success=True,
                length=length, max_kappa=np.nan, collision_free=col_free,
                cv_opt=np.nan)


def run_rrt_star(config, seed=None):
    return run_rrt(config, planner_type='rrt_star', seed=seed)


def _dubins_kw(**overrides):
    kw = dict(max_iter=2000, connect_circle_dist=4.5, goal_sample_rate=20,
              path_resolution=0.05, random_yaw_strategy='toward_goal',
              search_until_max_iter=True)
    kw.update(overrides)
    return kw


def _run_dubins_raw_kw(config, kw, seed=None):
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


def run_rrt_star_dubins(config, seed=None):
    return _run_dubins_raw_kw(config, _dubins_kw(), seed)


def run_modified_dubins_rrt_star(config, seed=None, safety_radius_factor=0.0,
                                 eta1=0.3):
    if seed is not None:
        config.seed = seed
    config.setup()

    start = [float(config.start[0]), float(config.start[1]), float(config.th_start)]
    goal = [float(config.goal[0]), float(config.goal[1]), float(config.th_goal)]
    obstacle_list = list(config.obs)

    t0 = time.perf_counter()
    planner = ModifiedDubinsRRTStar(
        start=start, goal=goal,
        obstacle_list=obstacle_list,
        rand_area=[config.xmin, config.xmax],
        rand_area_x=[config.xmin, config.xmax],
        rand_area_y=[config.ymin, config.ymax],
        goal_sample_rate=20, max_iter=2000,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        eta1=eta1,
        safety_radius_factor=safety_radius_factor)
    raw = planner.planning(animation=False, search_until_max_iter=True)
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw = apply_extended_start_goal(raw, config)
    raw_arr = np.asarray(raw)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)

    try:
        k_anal, _ = planner.get_curvature_analytical()
        max_k = float(k_anal.max())
    except Exception:
        max_k = max_curvature_numerical(raw_arr)

    return dict(path=raw_arr, raw_path=raw_arr, elapsed=elapsed, success=True,
                length=length, max_kappa=max_k, collision_free=col_free)


def run_modified_dubins_rrt_star_ccpoa(config, seed=None, safety_radius_factor=0.0,
                                       eta1=0.3):
    if seed is not None:
        config.seed = seed
    config.setup()

    start = [float(config.start[0]), float(config.start[1]), float(config.th_start)]
    goal = [float(config.goal[0]), float(config.goal[1]), float(config.th_goal)]
    obstacle_list = list(config.obs)

    t0 = time.perf_counter()
    planner = ModifiedDubinsRRTStar(
        start=start, goal=goal,
        obstacle_list=obstacle_list,
        rand_area=[config.xmin, config.xmax],
        rand_area_x=[config.xmin, config.xmax],
        rand_area_y=[config.ymin, config.ymax],
        goal_sample_rate=20, max_iter=2000,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        eta1=eta1,
        safety_radius_factor=safety_radius_factor)
    raw = planner.planning(animation=False, search_until_max_iter=True)
    mdr_elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=mdr_elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    # Extract waypoints from the node path
    nodes_path = []
    best_idx = planner.search_best_goal_node()
    if best_idx is None:
        return dict(path=None, raw_path=None, elapsed=mdr_elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)
    node = planner.node_list[best_idx]
    while node is not None:
        nodes_path.append(node)
        node = node.parent
    nodes_path.reverse()

    if len(nodes_path) < 2:
        raw_ext = apply_extended_start_goal(raw, config)
        return dict(path=np.asarray(raw_ext), raw_path=raw, elapsed=mdr_elapsed,
                    success=True, length=path_length(raw_ext),
                    max_kappa=np.nan, collision_free=False)

    waypoints = np.array([[n.x, n.y] for n in nodes_path])

    ccpoa = CCPOA(curvature=config.kappa_max, max_iter=5, tol=1e-6,
                  step_size=0.05)
    opt_path, theta_opt = ccpoa.optimize(
        waypoints, config.th_start, config.th_goal)
    total_elapsed = time.perf_counter() - t0

    opt_path = apply_extended_start_goal(opt_path, config)
    opt_arr = np.asarray(opt_path)
    length = path_length(opt_arr)
    col_free = is_collision_free(opt_arr, config.obs, config.radius)

    try:
        k_anal, _ = planner.get_curvature_analytical()
        max_k = float(k_anal.max())
    except Exception:
        max_k = max_curvature_numerical(opt_arr)

    return dict(path=opt_arr, raw_path=raw, elapsed=total_elapsed, success=True,
                length=length, max_kappa=max_k, collision_free=col_free)


def run_rrt_star_asv(config, seed=None, cost_a=1.20, cost_b=1.5):
    if seed is not None:
        config.seed = seed
    config.setup()

    start = [float(config.start[0]), float(config.start[1]), float(config.th_start)]
    goal = [float(config.goal[0]), float(config.goal[1]), float(config.th_goal)]
    obstacle_list = list(config.obs)

    t0 = time.perf_counter()
    planner = RRTStarASV(
        start=start, goal=goal,
        obstacle_list=obstacle_list,
        rand_area=[config.xmin, config.xmax],
        rand_area_x=[config.xmin, config.xmax],
        rand_area_y=[config.ymin, config.ymax],
        goal_sample_rate=30, max_iter=2000,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        cost_a=cost_a, cost_b=cost_b, max_nodes=None)
    raw = planner.planning(animation=False, search_until_max_iter=True)
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

    raw = apply_extended_start_goal(raw, config)
    raw_arr = np.asarray(raw)
    length = path_length(raw_arr)
    col_free = is_collision_free(raw_arr, config.obs, config.radius)
    max_k = max_curvature_numerical(raw_arr)

    return dict(path=raw_arr, raw_path=raw_arr, elapsed=elapsed, success=True,
                length=length, max_kappa=max_k, collision_free=col_free)


def smooth_path_rrt(planner, raw_path, n_waypoints=15, s=0.08, n_points=250):
    # n_points: resolution of the evaluated B-spline curve.  These are the
    # points on which collision checking and curvature are computed, so a
    # smaller value speeds up the post-processing (e.g. 500 -> 250).
    if raw_path is None or len(raw_path) < 4:
        return None
    # Temporarily set _path so smooth_path() works
    planner._path = raw_path.tolist() if hasattr(raw_path, 'tolist') else list(raw_path)
    try:
        smooth = planner.smooth_path(n_waypoints=n_waypoints, s=s,
                                     n_points=n_points)
        return np.asarray(smooth) if smooth is not None else None
    except Exception:
        return None


def _run_rrt_star_smooth_kw(config, kw, seed=None, n_waypoints=15, s=0.08):
    """Generic smooth RRT* runner: plan with `kw`, then B-spline smooth.

    The start/goal extension is applied to the raw path *before* smoothing
    (via `config.extended_start_goal`): attaching the straight tangent stubs
    after smoothing would create a kink at the start/goal (since the B-spline
    tangent there differs from the stub direction) and inflate the curvature.
    """
    if seed is not None:
        config.seed = seed
    config.extended_start_goal = True

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
        result.update(path=smooth_arr, success=True,
                      length=path_length(smooth_arr),
                      max_kappa=max_curvature_numerical(smooth_arr),
                      collision_free=is_collision_free(smooth_arr, config.obs,
                                                       config.radius))
    else:
        result.update(path=raw_arr, success=True,
                      length=path_length(raw_arr),
                      max_kappa=max_curvature_numerical(raw_arr),
                      collision_free=is_collision_free(raw_arr, config.obs,
                                                       config.radius))
    return result


def _smooth_kw(**overrides):
    kw = dict(max_iter=2000, connect_circle_dist=2.0,
              goal_sample_rate=20, search_until_max_iter=True,
              expand_dis=RRT_EXPAND_DIS)
    kw.update(overrides)
    return kw


def run_rrt_star_smooth(config, seed=None, n_waypoints=15, s=0.08):
    return _run_rrt_star_smooth_kw(config, _smooth_kw(), seed, n_waypoints, s)


def run_rrt_dubins_smooth(config, seed=None, n_waypoints=15, s=0.08):
    return _run_dubins_smooth_kw(config, _dubins_kw(), seed, n_waypoints, s)


def _run_dubins_smooth_kw(config, kw, seed=None, n_waypoints=15, s=0.08):
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


def run_bit_star_dubins(config, seed=None):
    kw = dict(max_iter=2000, connect_circle_dist=4.5,
              goal_sample_rate=20, path_resolution=0.05,
              random_yaw_strategy='toward_goal',
              search_until_max_iter=True,
              batch_size=200)
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    p = RRTPlanner(config, 'bit_star_dubins', **kw)
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


# ---- RRT* variants: rewire radius (ccdist) and steer step (expand_dis) ----
# NOTE: for non-Dubins RRT* the extension step is `expand_dis` (used by
# `steer`), NOT `step_size` (that name only exists for the Dubins/ASV
# planners).  `RRT_EXPAND_DIS` keeps the steer step equal to the base
# rrt_star so that the ccdist variants isolate the rewire-radius effect.
def _rrt_star_kw(**overrides):
    kw = dict(expand_dis=RRT_EXPAND_DIS, connect_circle_dist=2.0,
              max_iter=3000, goal_sample_rate=20,
              search_until_max_iter=True)
    kw.update(overrides)
    return kw


def run_rrt_star_ccdist_1_5(config, seed=None):
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(connect_circle_dist=1.5),
                   seed=seed)


def run_rrt_star_ccdist_2_5(config, seed=None):
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(connect_circle_dist=2.5),
                   seed=seed)


def run_rrt_star_ccdist_3_0(config, seed=None):
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(connect_circle_dist=3.0),
                   seed=seed)


def run_rrt_star_expand_0_2(config, seed=None):
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(expand_dis=0.2),
                   seed=seed)


def run_rrt_star_expand_0_5(config, seed=None):
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(expand_dis=0.5),
                   seed=seed)


def run_rrt_star_smooth_ccdist_1_5(config, seed=None):
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(connect_circle_dist=1.5), seed)


def run_rrt_star_smooth_ccdist_2_5(config, seed=None):
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(connect_circle_dist=2.5), seed)


def run_rrt_star_smooth_ccdist_3_0(config, seed=None):
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(connect_circle_dist=3.0), seed)


def run_rrt_star_smooth_expand_0_2(config, seed=None):
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(expand_dis=0.2), seed)


def run_rrt_star_smooth_expand_0_5(config, seed=None):
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(expand_dis=0.5), seed)


# ---- Tuned variants (grid search via tune_rrt_star.py) ----
# The winning ("recommended") configuration of each variant is read from
# rrt_tuning/best_params.json, produced by tune_rrt_star.py (Friedman +
# Nemenyi / paired-Wilcoxon over the ccdist x expand x max_iter grid).  If
# that file is missing, the hard-coded fallbacks below are used.
_TUNING_JSON = Path(__file__).parent / 'rrt_tuning' / 'best_params.json'


def _load_recommended(variant, fallback):
    """Return (ccdist, expand_dis, max_iter) for a variant.

    Reads the 'recommended' entry of rrt_tuning/best_params.json; falls back
    to `fallback` if the file/entry is missing.
    """
    try:
        data = json.loads(_TUNING_JSON.read_text())
        rec = data[variant]['recommended']
        return (float(rec['connect_circle_dist']),
                float(rec['expand_dis']),
                int(rec['max_iter']))
    except Exception as exc:  # missing/partial file -> keep working
        print(f'[tuned] using fallback for {variant} ({exc})')
        return fallback


RRT_STAR_TUNED = _load_recommended('rrt_star', (3.0, 0.1, 2000))
RRT_STAR_SMOOTH_TUNED = _load_recommended('rrt_star_smooth', (1.0, 0.1, 500))


def run_rrt_star_tuned(config, seed=None):
    cc, ed, mi = RRT_STAR_TUNED
    return run_rrt(config, planner_type='rrt_star',
                   extra_kw=_rrt_star_kw(connect_circle_dist=cc,
                                         expand_dis=ed, max_iter=mi),
                   seed=seed)


def run_rrt_star_smooth_tuned(config, seed=None):
    cc, ed, mi = RRT_STAR_SMOOTH_TUNED
    return _run_rrt_star_smooth_kw(
        config, _smooth_kw(connect_circle_dist=cc, expand_dis=ed,
                           max_iter=mi), seed)


# ---- Full parameter-grid sweep ----
# Instead of benchmarking only the single tuned ("recommended") config per
# variant, this enumerates the *whole* grid used by tune_rrt_star.py
# (ccdist x expand_dis x max_iter) for both variants, so every combination is
# run side by side.  Adjust the SWEEP_* lists below to restrict the grid.
SWEEP_CCDIST = [1.0, 2.0, 3.0]
SWEEP_EXPAND = [0.1, 0.2, 0.3, 0.4]
SWEEP_MAXITER = [500, 1000, 2000, 4000]


def _make_rrt_star_grid_runner(ccdist, expand_dis, max_iter):
    def _runner(config, seed=None):
        return run_rrt(config, planner_type='rrt_star',
                       extra_kw=_rrt_star_kw(connect_circle_dist=ccdist,
                                             expand_dis=expand_dis,
                                             max_iter=max_iter),
                       seed=seed)
    return _runner


def _make_rrt_star_smooth_grid_runner(ccdist, expand_dis, max_iter):
    def _runner(config, seed=None):
        return _run_rrt_star_smooth_kw(
            config, _smooth_kw(connect_circle_dist=ccdist,
                               expand_dis=expand_dis,
                               max_iter=max_iter), seed)
    return _runner


def _grid_name(variant, ccdist, expand_dis, max_iter):
    return f'{variant}_cc{ccdist}_ed{expand_dis}_mi{max_iter}'


def build_grid_planners():
    """All (name, runner) pairs for the full ccdist x expand x max_iter grid."""
    planners = []
    for cc in SWEEP_CCDIST:
        for ed in SWEEP_EXPAND:
            for mi in SWEEP_MAXITER:
                planners.append((
                    _grid_name('rrt_star', cc, ed, mi),
                    _make_rrt_star_grid_runner(cc, ed, mi)))
                planners.append((
                    _grid_name('rrt_star_smooth', cc, ed, mi),
                    _make_rrt_star_smooth_grid_runner(cc, ed, mi)))
    return planners


GRID_PLANNERS = build_grid_planners()


# ---- Dubins parameter-grid sweep ----
# Excludes bit_star_dubins.  Sweeps the parameters that actually matter for
# each Dubins family:
#   * RRT*Dubins (raw / smooth): rewire radius (ccdist) x random_yaw_strategy
#   * ModifiedDubinsRRTStar (: + CCPOA): eta1 (ModifiedSample) x safety factor
#   * RRT*ASV: turning-cost weights (cost_a, cost_b)
SWEEP_DUBINS_CCDIST = [3.5, 4.5, 6.0]
SWEEP_DUBINS_YAW = ['uniform', 'toward_goal']
SWEEP_MDR_ETA1 = [0.1, 0.3, 0.5]
SWEEP_MDR_SAFETY = [0.0, 1.0, 2.0]
SWEEP_ASV_COST_A = [1.0, 1.2]
SWEEP_ASV_COST_B = [1.0, 1.5, 2.0]


def _make_dubins_grid_runner(ccdist, yaw, smooth):
    def _runner(config, seed=None):
        kw = _dubins_kw(connect_circle_dist=ccdist,
                        random_yaw_strategy=yaw)
        if smooth:
            return _run_dubins_smooth_kw(config, kw, seed)
        return _run_dubins_raw_kw(config, kw, seed)
    return _runner


def _make_mdr_grid_runner(eta1, safety, ccpoa):
    fn = (run_modified_dubins_rrt_star_ccpoa if ccpoa
          else run_modified_dubins_rrt_star)

    def _runner(config, seed=None):
        return fn(config, seed=seed, safety_radius_factor=safety, eta1=eta1)
    return _runner


def _make_asv_grid_runner(cost_a, cost_b):
    def _runner(config, seed=None):
        return run_rrt_star_asv(config, seed=seed, cost_a=cost_a, cost_b=cost_b)
    return _runner


def build_dubins_grid_planners():
    """All (name, runner) pairs for the Dubins parameter grid (no bit_star)."""
    planners = []
    for cc in SWEEP_DUBINS_CCDIST:
        for yaw in SWEEP_DUBINS_YAW:
            planners.append((
                f'rrt_star_dubins_cc{cc}_yaw{yaw}',
                _make_dubins_grid_runner(cc, yaw, smooth=False)))
            planners.append((
                f'rrt_dubins_smooth_cc{cc}_yaw{yaw}',
                _make_dubins_grid_runner(cc, yaw, smooth=True)))
    for e in SWEEP_MDR_ETA1:
        for sf in SWEEP_MDR_SAFETY:
            planners.append((
                f'modified_dubins_rrt_star_eta{e}_safe{sf}',
                _make_mdr_grid_runner(e, sf, ccpoa=False)))
            planners.append((
                f'modified_dubins_rrt_star_ccpoa_eta{e}_safe{sf}',
                _make_mdr_grid_runner(e, sf, ccpoa=True)))
    for ca in SWEEP_ASV_COST_A:
        for cb in SWEEP_ASV_COST_B:
            planners.append((
                f'rrt_star_asv_ca{ca}_cb{cb}',
                _make_asv_grid_runner(ca, cb)))
    return planners


DUBINS_GRID_PLANNERS = build_dubins_grid_planners()


# ---- Dubins RRT* variants with different rewire radius ----
def run_rrt_star_dubins_ccdist_3_5(config, seed=None):
    kw = dict(max_iter=2000, connect_circle_dist=3.5,
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


def run_rrt_star_dubins_ccdist_6_0(config, seed=None):
    kw = dict(max_iter=2000, connect_circle_dist=6.0,
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


def _optimizer_cv(agent):
    try:
        return float(agent.log_best_CV[-1]) if len(agent.log_best_CV) else np.nan
    except Exception:
        return np.nan


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
                    collision_free=False, cv_opt=np.nan)
    elapsed = time.perf_counter() - t0

    result = de.get_best_path()
    if result is None:
        return dict(path=None, raw_path=None, elapsed=elapsed,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False, cv_opt=np.nan)

    pts = np.asarray(result['points'])
    length = path_length(pts)
    k_max = max_curvature_numerical(pts)
    col_free = is_collision_free(pts, config.obs, config.radius)
    cv_opt = _optimizer_cv(de.agent)

    return dict(path=pts, raw_path=pts, elapsed=elapsed, success=True,
                length=length, max_kappa=k_max, collision_free=col_free,
                cv_opt=cv_opt)


def run_pso2d_nurbs(config, seed=None):
    if seed is not None:
        config.seed = seed

    t0 = time.perf_counter()
    config.n_generations = 400
    config.pop_size = 50
    config.alpha_obs = 1000
    config.alpha_kappa = 10.00
    config.alpha_workspace = 1.00

    pso = PSO2D_NURBS(config)
    try:
        pso.run()
    except Exception:
        return dict(path=None, raw_path=None, elapsed=time.perf_counter() - t0,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False, cv_opt=np.nan)
    elapsed = time.perf_counter() - t0
    # pso.plot_convergence()
    result = pso.get_best_path()
    if result is None:
        return dict(path=None, raw_path=None, elapsed=elapsed,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False, cv_opt=np.nan)

    pts = np.asarray(result['points'])
    length = path_length(pts)
    k_max = max_curvature_numerical(pts)
    col_free = is_collision_free(pts, config.obs, config.radius)

    return dict(path=pts, raw_path=pts, elapsed=elapsed, success=True,
                length=length, max_kappa=k_max, collision_free=col_free,
                cv_opt=_optimizer_cv(pso.agent))


# ──────────────────────────────────────────────
# Scenario generation helpers
# ──────────────────────────────────────────────


PLANNERS = [
    # Non-Dubins RRT* variants (all run with extended_start_goal = True)
    ('rrt_star',          run_rrt_star),
    ('rrt_star_smooth',   run_rrt_star_smooth),
    ('rrt_star_ccdist_1_5',      run_rrt_star_ccdist_1_5),
    ('rrt_star_ccdist_2_5',      run_rrt_star_ccdist_2_5),
    ('rrt_star_ccdist_3_0',      run_rrt_star_ccdist_3_0),
    ('rrt_star_smooth_ccdist_1_5', run_rrt_star_smooth_ccdist_1_5),
    ('rrt_star_smooth_ccdist_2_5', run_rrt_star_smooth_ccdist_2_5),
    ('rrt_star_smooth_ccdist_3_0', run_rrt_star_smooth_ccdist_3_0),
    # steer-step (expand_dis) sweep; base rrt_star uses 0.3
    ('rrt_star_expand_0_2',        run_rrt_star_expand_0_2),
    ('rrt_star_expand_0_5',        run_rrt_star_expand_0_5),
    ('rrt_star_smooth_expand_0_2', run_rrt_star_smooth_expand_0_2),
    ('rrt_star_smooth_expand_0_5', run_rrt_star_smooth_expand_0_5),
    # full parameter-grid sweep: every ccdist x expand_dis x max_iter combo
    # for rrt_star and rrt_star_smooth (see build_grid_planners below)
    *GRID_PLANNERS,
    # Dubins parameter-grid sweep (bit_star_dubins excluded):
    #   RRT*Dubins: ccdist x yaw | MDR/MDR+CCPOA: eta1 x safety | ASV: cost_a x cost_b
    *DUBINS_GRID_PLANNERS,
    ('de2d_nurbs',        run_de2d_nurbs),
    # ('pso2d_nurbs',       run_pso2d_nurbs),
]
# PLANNERS = [
#     ('rrt_star',          run_rrt_star),
#     ('rrt_star_smooth',   run_rrt_star_smooth),
#     ('rrt_star_dubins',   run_rrt_star_dubins),
#     ('rrt_dubins_smooth', run_rrt_dubins_smooth),
#     ('modified_dubins_rrt_star', run_modified_dubins_rrt_star),
#     ('modified_dubins_rrt_star_ccpoa', run_modified_dubins_rrt_star_ccpoa),
#     ('rrt_star_asv',      run_rrt_star_asv),
#     ('bit_star_dubins',   run_bit_star_dubins),
#     # RRT* rewire-radius / step-size variants
#     ('rrt_star_ccdist_1_5',      run_rrt_star_ccdist_1_5),
#     ('rrt_star_ccdist_2_5',      run_rrt_star_ccdist_2_5),
#     ('rrt_star_ccdist_3_0',      run_rrt_star_ccdist_3_0),
#     ('rrt_star_smooth_ccdist_1_5', run_rrt_star_smooth_ccdist_1_5),
#     ('rrt_star_smooth_ccdist_2_5', run_rrt_star_smooth_ccdist_2_5),
#     ('rrt_star_smooth_ccdist_3_0', run_rrt_star_smooth_ccdist_3_0),
#     # Dubins RRT* rewire-radius variants
#     ('rrt_star_dubins_ccdist_3_5', run_rrt_star_dubins_ccdist_3_5),
#     ('rrt_star_dubins_ccdist_6_0', run_rrt_star_dubins_ccdist_6_0),
#     ('de2d_nurbs',        run_de2d_nurbs),
#     ('pso2d_nurbs',       run_pso2d_nurbs),
# ]


def _run_single_task(sc, pname, runner_fn):
    seed = sc['seed']
    obs = sc.get('obs_list')
    label = sc.get('label', f'seed{seed}')

    c = make_scenario(
        seed=seed, obs_list=obs,
        start=sc.get('start'), goal=sc.get('goal'),
        th_start=sc.get('th_start', 0.0),
        th_goal=sc.get('th_goal', 0.0),
        radius=sc.get('radius', 0.073),
        kappa_max=sc.get('kappa_max', 1/0.73),
        n_generations=sc.get('n_generations', 400),
        pop_size=sc.get('pop_size', 100),
        nsampling=sc.get('nsampling', 120),
    )
    c.setup()

    # Measure both wall-clock and CPU time of the runner.  Under joblib
    # parallelism the wall-clock is inflated by CPU contention (unfair
    # between planners); CPU time (process time) reflects the actual work
    # and is the fair metric for comparing planners.
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        result = runner_fn(c, seed=seed)
    except Exception:
        result = dict(path=None, raw_path=None, success=False,
                      length=np.nan, max_kappa=np.nan,
                      collision_free=False)
    wall_time = time.perf_counter() - wall0
    cpu_time = time.process_time() - cpu0
    total_time = wall_time
    result.setdefault('elapsed', total_time)
    # Fair, contention-insensitive time used for cross-planner comparison.
    result.setdefault('cpu_time', cpu_time)
    result.setdefault('wall_time', wall_time)
    result.setdefault('success', False)
    result.setdefault('length', np.nan)
    result.setdefault('max_kappa', np.nan)
    result.setdefault('collision_free', False)

    pts = result.get('path')
    max_kappa = result.get('max_kappa', np.nan)
    collision_free = result.get('collision_free', False)
    cv_opt = result.get('cv_opt', np.nan)

    # --- Post-hoc feasibility for ALL planners ---
    # Use gradient-based curvature violation (same method as the
    # optimizer's individual_cost_function) for consistency.
    # Avoids discrepancies from different curvature estimators
    # (e.g. Menger vs gradient).
    if pts is not None and len(np.asarray(pts)) >= 2:
        kappa_viol = _numerical_curvature_violation(pts, c.kappa_max)
        kappa_respected = kappa_viol <= 1e-5
    else:
        kappa_viol = np.nan
        kappa_respected = False

    ws_viol = 0.0
    if pts is not None and len(np.asarray(pts)) >= 2:
        a = np.asarray(pts)
        ws_viol = float(np.sum(
            (a[:, 0] < c.xmin) | (a[:, 0] > c.xmax) |
            (a[:, 1] < c.ymin) | (a[:, 1] > c.ymax)
        ))

    obs_viol = 0.0
    if pts is not None and len(np.asarray(pts)) >= 2:
        a = np.asarray(pts)
        if hasattr(c, 'obs') and c.obs is not None:
            for ob in c.obs:
                ox, oy, size = ob[0], ob[1], ob[2]
                d = np.hypot(a[:, 0] - ox, a[:, 1] - oy)
                pen = np.maximum(0, size + c.radius - d)
                if np.any(pen > 0):
                    obs_viol += float(np.sum(pen ** 2))

    collision_free = result.get('collision_free', False)
    cv_val = kappa_viol + ws_viol + obs_viol
    if np.isnan(cv_val):
        cv_val = np.inf
    feasible = collision_free and kappa_respected and ws_viol < 0.5 and (obs_viol < 1e-5) and (cv_val <= 1e-5)
    if cv_val <= 1e-5:
        cv_val = 0.0
        feasible = collision_free and kappa_respected and ws_viol < 0.5 and (obs_viol < 1e-5)

    record = dict(
        scenario=label, seed=seed, planner=pname,
        success=result['success'],
        length=result.get('length', np.nan),
        max_kappa=result.get('max_kappa', np.nan),
        # `elapsed` = CPU time (fair under joblib parallelism).
        # `wall_time` keeps the (contention-inflated) wall-clock for reference.
        elapsed=result.get('cpu_time', result.get('elapsed', total_time)),
        cpu_time=result.get('cpu_time', np.nan),
        wall_time=result.get('wall_time', total_time),
        collision_free=result.get('collision_free', False),
        cv=cv_val,
        cv_opt=cv_opt,
        feasible=feasible,
        n_obs=len(c.obs) if hasattr(c, 'obs') else 0,
        kappa_max=c.kappa_max,
        radius=c.radius,
        nsampling=getattr(c, 'nsampling', np.nan),
        th_start=getattr(c, 'th_start', 0.0),
        th_goal=getattr(c, 'th_goal', 0.0),
        start_x=c.start[0], start_y=c.start[1],
        goal_x=c.goal[0], goal_y=c.goal[1],
        xmin=c.xmin, xmax=c.xmax,
        ymin=c.ymin, ymax=c.ymax,
    )

    path_key = f'{label}_{pname}'
    raw = result.get('raw_path')
    path_data = result.get('path')
    if path_data is not None and len(path_data) >= 2:
        path_arr = np.asarray(path_data)
    elif raw is not None and len(raw) >= 2:
        path_arr = np.asarray(raw)
    else:
        path_arr = None

    obs_arr = np.asarray(c.obs) if hasattr(c, 'obs') and len(c.obs) else np.array([])
    return record, path_key, path_arr, label, obs_arr


def make_scenario(seed, obs_list=None, start=None, goal=None,
                  th_start=0.0, th_goal=0.0, radius=0.073, kappa_max=1/0.73,
                  n_generations=400, pop_size=100, nsampling=120):
    config = ScenarioConfig()
    config.seed = seed
    config.radius = radius
    config.kappa_max = kappa_max
    config.th_start = th_start
    config.th_goal = th_goal
    config.n_generations = n_generations
    config.pop_size = pop_size
    config.nsampling = nsampling
    # 
    config.width = config.scale_x*2
    config.height = config.scale_y*2
    config.max_fes = None
    config.obstacle_type = 'circles'   # 'polygons' ou 'circles'
    
    
    
    
    config.lambda_i = config.radius
    config.lambda_f = config.radius
    config.alpha_workspace = 5
    config.alpha_obs = 200
    config.alpha_kappa = 10

    # print(f'  Scenario seed={seed}  radius={radius:.3f}  kappa_max={kappa_max:.3f},  n_generations={n_generations}  pop_size={pop_size}  nsampling={nsampling} lambda_i={config.lambda_i:.3f}  lambda_f={config.lambda_f:.3f}  alpha_workspace={config.alpha_workspace:.3f}  alpha_obs={config.alpha_obs:.3f}  alpha_kappa={config.alpha_kappa:.3f}')
    
    config.scale_x = 2.1
    config.scale_y = 2.1
    config.xmin = -2.1
    config.xmax = 2.1
    config.ymin = -2.1
    config.ymax = 2.1
    config.start = np.array([-1.4, -0.8])
    config.goal = np.array([1.4, 0.8])
    config.th_start = 0.0
    config.th_goal = 0.0
    config.safe_radius = 0.4
    if start is not None:
        config.start = np.asarray(start, dtype=float)
    if goal is not None:
        config.goal = np.asarray(goal, dtype=float)
    if obs_list is not None:
        config.obs_list = list(obs_list)
    return config


# ── Obstacle generators ───────────────────────

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


def generate_obstacle_pool(n_total=10, seed=42, center_size=0.35,
                           x_range=(-1.2, 1.2), y_range=(-0.8, 0.8),
                           r_range=(0.08, 0.25)):
    """Generate *n_total* obstacles; first is always a large central one."""
    rng = np.random.RandomState(seed)
    pool = [(0.0, 0.0, center_size)]
    for _ in range(n_total - 1):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        r = rng.uniform(*r_range)
        pool.append((x, y, r))
    return pool


def generate_random_obstacles(n, seed, x_range=(-1.5, 1.5), y_range=(-0.9, 0.9),
                              r_range=(0.08, 0.25)):
    """Generate *n* independent random obstacles (no central fixed one)."""
    rng = np.random.RandomState(seed)
    return [(rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(*r_range))
            for _ in range(n)]


# ── Scenario list builders ────────────────────

def scenarios_no_obstacles(seeds):
    return [dict(seed=s, obs_list=[], label=f'no_obs_seed{s}')
            for s in seeds]


def scenarios_fixed_obstacles(seeds, obstacles, label_prefix='obs'):
    return [dict(seed=s, obs_list=list(obstacles), label=f'{label_prefix}_seed{s}')
            for s in seeds]


def scenarios_progressive(seeds, max_obs=10, pool_seed=42, center_size=0.35,
                          label_prefix='prog'):
    """From 1 (central) up to *max_obs* obstacles, sliced from a fixed pool."""
    pool = generate_obstacle_pool(max_obs, pool_seed, center_size)
    sc = []
    for s in seeds:
        for k in range(0, max_obs + 1):
            sc.append(dict(seed=s, obs_list=pool[:k],
                           label=f'{label_prefix}{k:02d}_seed{s}'))
    return sc


def scenarios_random_count(seeds, counts, label_prefix='rand'):
    """Independent random obstacles for each count (different layout per count)."""
    sc = []
    for s in seeds:
        for n in counts:
            obs = generate_random_obstacles(n, seed=s * 100 + n)
            sc.append(dict(seed=s, obs_list=obs,
                           label=f'{label_prefix}{n:02d}_seed{s}'))
    return sc


# ──────────────────────────────────────────────
# Main comparison loop
# ──────────────────────────────────────────────

def _result_key(res):
    """Unique (scenario, planner) key of a result tuple for resume/de-dup."""
    rec = res[0]
    return (rec.get('scenario'), rec.get('planner'))


def _checkpoint_file(output_dir):
    return Path(output_dir) / CHECKPOINT_NAME


def _save_checkpoint(output_dir, results):
    """Atomically dump partial results.

    Written to a temp file and then `os.replace`d, so a crash/power loss
    during the write cannot leave a truncated/corrupt checkpoint.
    """
    path = _checkpoint_file(output_dir)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'wb') as f:
            pickle.dump(results, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as exc:
        print(f'  [checkpoint] WARNING: could not save {path}: {exc}')


def _load_checkpoint(output_dir):
    path = _checkpoint_file(output_dir)
    if not path.exists():
        return []
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as exc:
        print(f'  [checkpoint] WARNING: could not load {path}: {exc}')
        return []


def run_comparison(scenario_configs, planners=PLANNERS,
                   output_dir='comparison_results', verbose=True, n_jobs=1,
                   checkpoint=True, save_every=CHECKPOINT_SAVE_EVERY,
                   resume=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = [(sc, pname, runner_fn)
                 for sc in scenario_configs
                 for pname, runner_fn in planners]
    total = len(task_list)

    # Resume: keep every result already stored in the checkpoint and only run
    # the tasks that are still missing.
    results = []
    done = set()
    if checkpoint and resume:
        results = _load_checkpoint(output_dir)
        done = {_result_key(r) for r in results}
        if done:
            print(f'  [checkpoint] resuming: {len(done)} task(s) already done, '
                  f'{total - len(done)} remaining (from {_checkpoint_file(output_dir)}).')

    remaining = [
        args for args in task_list
        if (args[0].get('label', f'seed{args[0]["seed"]}'), args[1]) not in done
    ]
    n_saved = len(results)

    def _flush(force=False):
        """Write a checkpoint if `save_every` new results accumulated."""
        nonlocal n_saved
        if not checkpoint:
            return
        if force or (len(results) - n_saved) >= save_every:
            _save_checkpoint(output_dir, results)
            n_saved = len(results)
            if verbose:
                print(f'  [checkpoint] {len(results)}/{total} tasks saved to '
                      f'{_checkpoint_file(output_dir)}')

    if n_jobs == 1:
        n_rem = len(remaining)
        for idx, args in enumerate(remaining):
            if verbose:
                sc = args[0]
                label = sc.get('label', f'seed{sc["seed"]}')
                print(f'[{len(results)+1}/{total}] {args[1]:20s}  {label}',
                      end=' ', flush=True)
            res = _run_single_task(*args)
            results.append(res)
            if verbose:
                rec = res[0]
                ok = 'OK' if rec['success'] else 'FAIL'
                print(f'{ok}  len={rec["length"]:.2f}m  '
                      f'cpu={rec["cpu_time"]:.2f}s  wall={rec["wall_time"]:.2f}s'
                      f'  col={"" if rec["collision_free"] else "COL!"}')
            _flush()
    else:
        n_workers = min(n_jobs, len(remaining)) if remaining else 0
        if n_workers > SAFE_N_JOBS:
            print(f'  [WARNING] n_jobs={n_workers} > {SAFE_N_JOBS}: memory/cache '
                  f'contention inflates the per-task time (up to ~2x at '
                  f'n_jobs=21). Use -j {SAFE_N_JOBS} (or -j 1) for fair timing.')
        if verbose:
            print(f'Running {len(remaining)} tasks on {n_workers} workers '
                  f'({total - len(remaining)} already done) ...')
        if remaining:
            # loky (processes): the per-task CPU time is measured with
            # time.process_time() inside _run_single_task.  NOTE: this removes
            # the multithreading effect but NOT memory-bandwidth contention, so
            # keep n_jobs moderate (see SAFE_N_JOBS) for comparable timings.
            # Prefer `return_as='generator_unordered'` (joblib >= 1.4) so
            # results are yielded as they finish and checkpoints can be written
            # incrementally without waiting for a slow task.  Fall back to the
            # ordered generator (joblib >= 1.3) and finally to a single batch.
            result_iter = None
            for mode in ('generator_unordered', 'generator'):
                try:
                    result_iter = Parallel(n_jobs=n_workers, backend='loky',
                                           verbose=10, return_as=mode)(
                        delayed(_run_single_task)(*args) for args in remaining)
                    break
                except (TypeError, ValueError):
                    result_iter = None
            if result_iter is not None:
                try:
                    for res in result_iter:
                        results.append(res)
                        _flush()
                except Exception as exc:
                    _flush(force=True)
                    print(f'  [checkpoint] parallel run interrupted: {exc}. '
                          f'{len(results)}/{total} results checkpointed; '
                          f're-run to resume.')
                    raise
            else:
                # joblib too old for generators: single batch (no incremental
                # checkpoints during the parallel phase).
                batch = Parallel(n_jobs=n_workers, backend='loky', verbose=10)(
                    delayed(_run_single_task)(*args) for args in remaining)
                results.extend(batch)
                _flush(force=True)
        if verbose and results:
            df_tmp = pd.DataFrame([r[0] for r in results])
            print('\nCPU time (s) mean per planner:')
            print(df_tmp.groupby('planner')['cpu_time'].mean()
                  .sort_values().to_string())

    _flush(force=True)

    all_records = []
    paths_store = {}
    obstacles_store = {}
    for record, path_key, path_arr, label, obs_arr in results:
        all_records.append(record)
        if path_key not in paths_store:
            paths_store[path_key] = path_arr
        obstacles_store[label] = obs_arr

    df = pd.DataFrame(all_records)
    return df, paths_store, obstacles_store


def save_results(df, paths_store, obstacles_store=None, output_dir='comparison_results'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_dir / 'summary.pkl')
    df.to_csv(output_dir / 'summary.csv', index=False)

    with open(output_dir / 'paths.pkl', 'wb') as f:
        pickle.dump(paths_store, f)

    # Save config per scenario (first row per scenario)
    cfg_cols = ['kappa_max', 'radius', 'nsampling', 'th_start', 'th_goal',
                'start_x', 'start_y', 'goal_x', 'goal_y',
                'xmin', 'xmax', 'ymin', 'ymax']
    configs = {}
    for sc_label, grp in df.groupby('scenario'):
        row = grp.iloc[0]
        configs[sc_label] = {col: row[col] for col in cfg_cols if col in row}
    with open(output_dir / 'config.pkl', 'wb') as f:
        pickle.dump(configs, f)

    if obstacles_store:
        with open(output_dir / 'obstacles.pkl', 'wb') as f:
            pickle.dump(obstacles_store, f)

    print(f'\nResults saved to {output_dir}/')
    print(f'  summary.pkl / summary.csv  ({len(df)} records)')
    print(f'  config.pkl                 ({len(configs)} scenarios)')
    print(f'  paths.pkl                  ({len(paths_store)} entries)')
    if obstacles_store:
        print(f'  obstacles.pkl              ({len(obstacles_store)} entries)')


def load_results(output_dir='comparison_results'):
    output_dir = Path(output_dir)
    df = pd.read_pickle(output_dir / 'summary.pkl')
    with open(output_dir / 'paths.pkl', 'rb') as f:
        paths_store = pickle.load(f)
    obstacles_store = {}
    obs_path = output_dir / 'obstacles.pkl'
    if obs_path.exists():
        with open(obs_path, 'rb') as f:
            obstacles_store = pickle.load(f)
    return df, paths_store, obstacles_store


# ──────────────────────────────────────────────
# Quick summary & plotting helpers
# ──────────────────────────────────────────────

def print_summary(df):
    print('\n=== RANKING (lower avg_rank = better) ===')
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

    print('\n=== AGGREGATED METRICS ===')
    fea = df[df['feasible']]
    grp = df.groupby('planner')
    rows = []
    for name, g in grp:
        feas_mask = g['feasible']
        feasible_len = g.loc[feas_mask, 'length']
        row = dict(
            planner=name,
            success_rate=g['success'].mean(),
            feasible_rate=feas_mask.mean(),
            avg_length=g['length'].mean(),
            avg_length_fea=feasible_len.mean() if len(feasible_len) else np.nan,
            avg_cv=g['cv'].mean(),
            avg_cpu_time=g['elapsed'].mean(),   # elapsed == CPU time
        )
        if 'wall_time' in g.columns:
            row['avg_wall_time'] = g['wall_time'].mean()
        rows.append(row)
    agg = pd.DataFrame(rows).set_index('planner').round(3)
    agg = agg.sort_values(['feasible_rate', 'avg_length'], ascending=[False, True])
    print(agg.to_string())


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main():
    """Entry point — define scenarios and run comparison.

    To customise, edit the ``experiments`` list below.  Each entry is a
    ``(name, scenario_list)`` pair; the comparison runs for each entry and
    saves results under ``comparison_results/<name>/``.

    Usage
    -----
      python compare_rrt_de2d_nurbs.py             # sequential
      python compare_rrt_de2d_nurbs.py -j 4        # 4 workers
      python compare_rrt_de2d_nurbs.py -j -1       # all cores
      python compare_rrt_de2d_nurbs.py --save-every 5   # checkpoint every 5 tasks
      python compare_rrt_de2d_nurbs.py --fresh     # ignore/restart checkpoint
      python compare_rrt_de2d_nurbs.py --no-checkpoint

    Progress is checkpointed to ``<output_dir>/checkpoint.pkl`` every
    ``--save-every`` tasks (default 20) and resumed automatically on the next
    run, so an interruption (e.g. power loss) does not restart from scratch.
    Use ``--fresh`` to start over, or delete that file.
    """
    n_jobs = 4
    checkpoint = True
    resume = True
    save_every = CHECKPOINT_SAVE_EVERY
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '-j' and i + 1 < len(argv):
            n_jobs = int(argv[i + 1]); i += 2
        elif a.startswith('-j') and len(a) > 2:
            n_jobs = int(a[2:]); i += 1
        elif a == '--fresh':
            resume = False; i += 1
        elif a == '--no-checkpoint':
            checkpoint = False; i += 1
        elif a == '--save-every' and i + 1 < len(argv):
            save_every = int(argv[i + 1]); i += 2
        else:
            i += 1

    seeds = list(range(30)) #[2, 4, 5, 10, 21, 40, 50, 60, 70, 80]  # random seeds for scenarios

    experiments = [
        # # # 1) No obstacles
        # ('no_obs', scenarios_no_obstacles(seeds)),

        # # 2) Fixed chicane (7 obstacles)
        # ('chicane', scenarios_fixed_obstacles(
        #     seeds, generate_chicane_obstacles(), 'chicane')),

        # 3) 1 → 5 obstacles from a progressive pool
        ('progressive', scenarios_progressive(
            seeds, max_obs=9, pool_seed=32, center_size=0.3)),
    ]

    # ── Uncomment any line below to add more experiments ──

    # 1..10 obstacles with independent random layouts
    # experiments.append(
    #     ('random', scenarios_random_count(seeds, range(1, 11)))
    # )

    # Custom chicane with different robot radius
    # from copy import deepcopy
    # for s in seeds:
    #     sc = dict(seed=s, obs_list=generate_chicane_obstacles(),
    #               radius=0.05, label=f'chicane_tight_seed{s}')
    #     experiments.append(('chicane_tight', [sc]))

    for exp_name, scenario_list in experiments:
        print(f'\n{"#"*70}')
        print(f'# Experiment: {exp_name}  ({len(scenario_list)} scenarios)')
        print(f'{exp_name}: {len(scenario_list) * len(PLANNERS)} tasks')
        print(f'n_jobs={n_jobs}')
        print(f'{"#"*70}')
        out_dir = f'comparison_results_paper/{exp_name}'
        df, ps, obs_store = run_comparison(
            scenario_list, output_dir=out_dir, n_jobs=n_jobs,
            checkpoint=checkpoint, save_every=save_every, resume=resume,
        )
        save_results(df, ps, obs_store, output_dir=out_dir)
        print_summary(df)


if __name__ == '__main__':
    main()
