import sys, pathlib, math, time, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scenario_config import ScenarioConfig
from rrt_based.rrt_planner_modified import RRTStarASV, angle_mod


def run(com_obstaculo=False, max_iter=500, seed=None):
    # ── cenário ─────────────────────────────────────────────
    config = ScenarioConfig()
    config.radius = 0.073
    config.kappa_max = 1.0 / 0.73
    config.start = [-1.4, -0.8]
    config.goal = [1.4, 0.8]
    config.th_start = 0.0
    config.th_goal = 0.0

    if com_obstaculo:
        config.obs_list = [(0.0, 0.0, 0.3)]
    else:
        config.obs_list = []
    if seed is not None:
        config.seed = seed
    config.setup()

    # ── planner ─────────────────────────────────────────────
    planner = RRTStarASV(
        start=[float(c) for c in config.start] + [config.th_start],
        goal=[float(c) for c in config.goal] + [config.th_goal],
        obstacle_list=list(config.obs),
        rand_area=[config.xmin, config.xmax],
        rand_area_x=[config.xmin, config.xmax],
        rand_area_y=[config.ymin, config.ymax],
        goal_sample_rate=20,
        max_iter=max_iter,
        connect_circle_dist=4.5,
        robot_radius=config.radius,
        step_size=0.05,
        curvature=config.kappa_max,
        turning_cost_weight=0.0,
    )

    # ── plan ────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw = planner.planning(animation=False, search_until_max_iter=True)
    elapsed = time.perf_counter() - t0

    if raw is None:
        print("Falha — nenhum caminho encontrado")
        return

    path = np.asarray(raw)
    comprimento = sum(
        math.hypot(path[i, 0] - path[i - 1, 0], path[i, 1] - path[i - 1, 1])
        for i in range(1, len(path))
    )

    # ── métricas ────────────────────────────────────────────
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curv = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-10)

    cab_start = math.atan2(dy[0], dx[0])
    cab_end = math.atan2(dy[-1], dx[-1])

    col_free = True
    for ox, oy, size in config.obs:
        d = np.hypot(path[:, 0] - ox, path[:, 1] - oy)
        if (d + 1e-9 < size + config.radius).any():
            col_free = False
            break

    # ── resultado ───────────────────────────────────────────
    obst_str = "COM obstáculo" if com_obstaculo else "sem obstáculo"
    print(f"\n{'='*50}")
    print(f"RRT*ASV — {obst_str}  (seed={seed})")
    print(f"{'='*50}")
    print(f"  Tempo         : {elapsed:.1f}s")
    print(f"  Comprimento   : {comprimento:.3f}m")
    print(f"  Sucesso       : sim" if raw is not None else "não")
    print(f"  Colisão-free  : {col_free}")
    print(f"  Curvatura máx : {curv.max():.3f}  (limite={config.kappa_max:.3f})")
    print(f"  Ptos acima lim: {(curv > config.kappa_max).sum()}")
    print(f"  Heading start : {math.degrees(cab_start):.1f}°  (desejado={math.degrees(config.th_start):.1f}°)")
    print(f"  Heading end   : {math.degrees(cab_end):.1f}°  (desejado={math.degrees(config.th_goal):.1f}°)")
    print(f"  Nós na árvore : {len(planner.node_list)}")
    print(f"  Pontos no caminho: {len(path)}")

    # ── comparação com Dubins ótima ────────────────────────
    from rrt_based.dubins_path_planner import plan_dubins_path as dubins
    px, py, _, _, lengths = dubins(
        config.start[0], config.start[1], config.th_start,
        config.goal[0], config.goal[1], config.th_goal,
        config.kappa_max, step_size=0.05,
    )
    dub_len = sum(abs(c) for c in lengths)
    razao = comprimento / dub_len if dub_len > 0 else float("inf")
    print(f"  Dubins ótima  : {dub_len:.3f}m")
    print(f"  Razão         : {razao:.2f}x")

    # ── primeiros / últimos pontos ──────────────────────────
    print(f"\n  Primeiros 3: {path[:3].tolist()}")
    print(f"  Últimos  3: {path[-3:].tolist()}")
    print()


if __name__ == "__main__":
    print(">>> Teste 1: ambiente livre, 500 iterações")
    run(com_obstaculo=False, max_iter=500, seed=42)

    print(">>> Teste 2: com obstáculo, 500 iterações")
    run(com_obstaculo=True, max_iter=500, seed=42)

    print(">>> Teste 3: ambiente livre, 1000 iterações")
    run(com_obstaculo=False, max_iter=1000, seed=7)
