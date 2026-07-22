import pickle, time, copy, gc, sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scenario_config import ScenarioConfig
from rrt_based.rrt_planner_modified import RRTPlanner, angle_mod, ModifiedDubinsRRTStar, RRTStarASV
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
    dx = np.gradient(a[:, 0])
    dy = np.gradient(a[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-8)
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


def run_rrt_star_dubins(config, seed=None):
    kw = dict(max_iter=2000, connect_circle_dist=4.5,
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


def run_modified_dubins_rrt_star(config, seed=None, safety_radius_factor=0.0):
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
        goal_sample_rate=20, max_iter=1500,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        eta1=0.3,
        safety_radius_factor=safety_radius_factor)
    raw = planner.planning(animation=False, search_until_max_iter=True)
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

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


def run_modified_dubins_rrt_star_ccpoa(config, seed=None, safety_radius_factor=0.0):
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
        goal_sample_rate=20, max_iter=1500,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        eta1=0.3,
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
        return dict(path=np.asarray(raw), raw_path=raw, elapsed=mdr_elapsed,
                    success=True, length=path_length(raw),
                    max_kappa=np.nan, collision_free=False)

    waypoints = np.array([[n.x, n.y] for n in nodes_path])

    ccpoa = CCPOA(curvature=config.kappa_max, max_iter=5, tol=1e-6,
                  step_size=0.05)
    opt_path, theta_opt = ccpoa.optimize(
        waypoints, config.th_start, config.th_goal)
    total_elapsed = time.perf_counter() - t0

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


def run_rrt_star_asv(config, seed=None, turning_cost_weight=0.5):
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
        goal_sample_rate=20, max_iter=500,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        turning_cost_weight=turning_cost_weight)
    raw = planner.planning(animation=False, search_until_max_iter=True)
    elapsed = time.perf_counter() - t0

    if raw is None:
        return dict(path=None, raw_path=None, elapsed=elapsed, success=False,
                    length=np.nan, max_kappa=np.nan, collision_free=False)

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
    kw = dict(max_iter=2000, connect_circle_dist=4.5,
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


def run_bit_star_dubins(config, seed=None):
    kw = dict(max_iter=5000, connect_circle_dist=4.5,
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
    pso = PSO2D_NURBS(config)
    try:
        pso.run()
    except Exception:
        return dict(path=None, raw_path=None, elapsed=time.perf_counter() - t0,
                    success=False, length=np.nan, max_kappa=np.nan,
                    collision_free=False, cv_opt=np.nan)
    elapsed = time.perf_counter() - t0

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
    # ('rrt_star',          run_rrt_star),
    # ('rrt_star_smooth',   run_rrt_star_smooth),
    # ('rrt_star_dubins',   run_rrt_star_dubins),
    # ('rrt_dubins_smooth', run_rrt_dubins_smooth),
    ('modified_dubins_rrt_star', run_modified_dubins_rrt_star),
    ('modified_dubins_rrt_star_ccpoa', run_modified_dubins_rrt_star_ccpoa),
    ('rrt_star_asv',      run_rrt_star_asv),
    ('bit_star_dubins',   run_bit_star_dubins),
    ('de2d_nurbs',        run_de2d_nurbs),
    # ('pso2d_nurbs',       run_pso2d_nurbs),
]


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
        n_generations=sc.get('n_generations', 450),
        pop_size=sc.get('pop_size', 100),
        nsampling=sc.get('nsampling', 200),
    )
    c.setup()

    t_start = time.perf_counter()
    try:
        result = runner_fn(c, seed=seed)
    except Exception:
        result = dict(path=None, raw_path=None, success=False,
                      length=np.nan, max_kappa=np.nan,
                      collision_free=False)
    total_time = time.perf_counter() - t_start
    result.setdefault('elapsed', total_time)
    result.setdefault('success', False)
    result.setdefault('length', np.nan)
    result.setdefault('max_kappa', np.nan)
    result.setdefault('collision_free', False)

    pts = result.get('path')
    max_kappa = result.get('max_kappa', np.nan)
    collision_free = result.get('collision_free', False)
    cv_opt = result.get('cv_opt', np.nan)

    # --- Post-hoc feasibility for ALL planners ---
    # (not the optimizer's internal CV, which can be > 1e-6 for
    #  visually-feasible paths due to numerical curvature artifacts)
    if not np.isnan(max_kappa):
        kappa_respected = max_kappa <= c.kappa_max * 1.05 + 1e-6
        kappa_viol = 0.0 if kappa_respected else (max_kappa - c.kappa_max) ** 2
    elif pts is not None and len(np.asarray(pts)) >= 2:
        kappa_viol = _numerical_curvature_violation(pts, c.kappa_max)
        kappa_respected = kappa_viol < 1e-6
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

    feasible = (obs_viol < 1e-6) and kappa_respected and ws_viol < 0.5
    cv_val = 0.0 if feasible else kappa_viol + ws_viol + obs_viol

    record = dict(
        scenario=label, seed=seed, planner=pname,
        success=result['success'],
        length=result.get('length', np.nan),
        max_kappa=result.get('max_kappa', np.nan),
        elapsed=result.get('elapsed', total_time),
        collision_free=result.get('collision_free', False),
        cv=cv_val,
        cv_opt=cv_opt,
        feasible=feasible,
        n_obs=len(c.obs) if hasattr(c, 'obs') else 0,
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
                  n_generations=450, pop_size=100, nsampling=120):
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

def run_comparison(scenario_configs, planners=PLANNERS,
                   output_dir='comparison_results', verbose=True, n_jobs=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = [(sc, pname, runner_fn)
                 for sc in scenario_configs
                 for pname, runner_fn in planners]

    if n_jobs == 1:
        results = []
        n_tasks = len(task_list)
        for idx, args in enumerate(task_list):
            if verbose:
                sc = args[0]
                label = sc.get('label', f'seed{sc["seed"]}')
                print(f'[{idx+1}/{n_tasks}] {args[1]:20s}  {label}', end=' ', flush=True)
            rec, pkey, parr, lbl, oarr = _run_single_task(*args)
            results.append((rec, pkey, parr, lbl, oarr))
            if verbose:
                ok = 'OK' if rec['success'] else 'FAIL'
                print(f'{ok}  len={rec["length"]:.2f}m  t={rec["elapsed"]:.1f}s'
                      f'  col={"" if rec["collision_free"] else "COL!"}')
    else:
        n_jobs = min(n_jobs, len(task_list))
        if verbose:
            print(f'Running {len(task_list)} tasks on {n_jobs} workers ...')
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_single_task)(*args) for args in task_list
        )

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

    if obstacles_store:
        with open(output_dir / 'obstacles.pkl', 'wb') as f:
            pickle.dump(obstacles_store, f)

    print(f'\nResults saved to {output_dir}/')
    print(f'  summary.pkl / summary.csv  ({len(df)} records)')
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
        rows.append(dict(
            planner=name,
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
    """
    n_jobs = 21
    argv = sys.argv[1:]
    if argv and argv[0] == '-j' and len(argv) > 1:
        n_jobs = int(argv[1])
    elif argv and argv[0].startswith('-j') and len(argv[0]) > 2:
        n_jobs = int(argv[0][2:])

    seeds = list(range(1)) #[2, 4, 5, 10, 21, 40, 50, 60, 70, 80]  # random seeds for scenarios

    experiments = [
        # # # 1) No obstacles
        ('no_obs', scenarios_no_obstacles(seeds)),

        # # 2) Fixed chicane (7 obstacles)
        # ('chicane', scenarios_fixed_obstacles(
        #     seeds, generate_chicane_obstacles(), 'chicane')),

        # 3) 1 → 5 obstacles from a progressive pool
        # ('progressive', scenarios_progressive(
            # seeds, max_obs=9, pool_seed=22, center_size=0.3)),
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
        out_dir = f'comparison_results/{exp_name}'
        df, ps, obs_store = run_comparison(
            scenario_list, output_dir=out_dir, n_jobs=n_jobs,
        )
        save_results(df, ps, obs_store, output_dir=out_dir)
        print_summary(df)


if __name__ == '__main__':
    main()
