import sys, pathlib, math, time, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from scenario_config import ScenarioConfig
from rrt_based.rrt_planner_modified import RRTStarASV, angle_mod


def run(com_obstaculo=False, max_iter=2500, seed=None, plot=False):
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
        cost_a=1.2, cost_b=1.5, max_nodes=None,
    )

    # ── plan ────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw = planner.planning(animation=True, search_until_max_iter=True)
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

    # ── plot ────────────────────────────────────────────────
    if plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        # árvore
        for n in planner.node_list:
            if n.parent is not None and hasattr(n, 'path_x') and n.path_x is not None and len(n.path_x) > 1:
                ax.plot(n.path_x, n.path_y, color="gray", linewidth=0.3, alpha=0.4)
        # caminho
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="RRT*ASV")
        # Dubins ótima
        ax.plot(px, py, "r--", linewidth=1.5, alpha=0.7, label="Dubins ótima")
        # obstáculos
        for ox, oy, size in config.obs:
            circle = plt.Circle((ox, oy), size, color="orange", alpha=0.4, label="_")
            ax.add_patch(circle)
            circle2 = plt.Circle((ox, oy), size + config.radius,
                                 color="orange", fill=False, linestyle=":", linewidth=1)
            ax.add_patch(circle2)
        # start / goal
        ax.plot(config.start[0], config.start[1], "go", markersize=8, label="Start")
        ax.plot(config.goal[0], config.goal[1], "ro", markersize=8, label="Goal")
        # setas de heading
        dx_s = 0.15 * math.cos(config.th_start)
        dy_s = 0.15 * math.sin(config.th_start)
        ax.arrow(config.start[0], config.start[1], dx_s, dy_s,
                 head_width=0.05, head_length=0.05, color="green")
        dx_g = 0.15 * math.cos(config.th_goal)
        dy_g = 0.15 * math.sin(config.th_goal)
        ax.arrow(config.goal[0], config.goal[1], dx_g, dy_g,
                 head_width=0.05, head_length=0.05, color="red")

        obst_str = "com obstáculo" if com_obstaculo else "sem obstáculo"
        ax.set_title(f"RRT*ASV — {obst_str}  ({len(planner.node_list)} nós, {comprimento:.2f}m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)


if __name__ == "__main__":
    # print(">>> Teste 1: ambiente livre, 500 iterações")
    # run(com_obstaculo=False, max_iter=500, seed=42, plot=True)

    print(">>> Teste 2: com obstáculo, 5000 iterações")
    run(com_obstaculo=True, max_iter=5000, seed=42, plot=True)

    # print(">>> Teste 3: ambiente livre, 1000 iterações")
    # run(com_obstaculo=False, max_iter=1000, seed=7, plot=True)
