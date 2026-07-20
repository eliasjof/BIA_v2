import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scenario_config import ScenarioConfig
from rrt_based.rrt_planner import RRTPlanner
from pso2d_nurbs import PSO2D_NURBS


def exemplo_rrt_star(config, ax=None, curv_ax=None):
    print("RRT* CLASSICO")
    p = RRTPlanner(config, 'rrt_star',
                   max_iter=2000, expand_dis=0.2,
                   path_resolution=0.02, goal_sample_rate=30)
    p.run(animation=False)
    path = p.get_best_path()
    print(f"  {'OK' if path else 'FAIL'}")
    smooth = p.smooth_path()
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax, curv_ax = axes[0], axes[1]
        p.draw_scenario(ax=ax, show=False)
    if path:
        raw = np.array(path)
        ax.plot(raw[:, 0], raw[:, 1], 'b-', lw=1.5, alpha=0.4, label='RRT* (raw)')
        ax.plot(smooth[:, 0], smooth[:, 1], 'r-', lw=2, label='B-Spline')
    ax.legend(fontsize=10)
    ax.set_title("RRT* Classico")
    if own_fig:
        p.plot_curvature(smoothed=False, ax=curv_ax)
        p.plot_curvature(smoothed=True, ax=axes[2])
        axes[1].set_title("Curvatura (raw)")
        axes[2].set_title("Curvatura (raw + smoothed)")
        fig.tight_layout()
        return fig
    return None


def exemplo_rrt_star_dubins(config, ax=None, curv_ax=None):
    print("RRT* DUBINS")
    p = RRTPlanner(
        config, 'rrt_star_dubins',
        random_yaw_strategy='toward_goal',
        max_iter=900, connect_circle_dist=8.5,
        step_size=0.02, goal_sample_rate=40,
        goal_xy_th=0.10, goal_yaw_th=np.deg2rad(5),
    )
    p.run(animation=False)
    path = p.get_best_path()
    print(f"  {'OK' if path else 'FAIL'}")
    own_fig = ax is None
    if not path:
        print("  FAIL")
        if own_fig:
            return fig
        return None
    a = np.array(path)
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax, curv_ax = axes
        p.draw_scenario(ax=ax, show=False)
    comp = np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))
    style = dict(color='#F44336', ls='-', lw=1.5, label=f'RRT* Dubins ({comp:.2f}m)')
    ax.plot(a[:, 0], a[:, 1], **style)
    ax.legend(fontsize=10)
    ax.set_title("RRT* Dubins")
    if own_fig:
        ax.axis([config.xmin, config.xmax, config.ymin, config.ymax])
        fig.tight_layout()
        return fig
    return p, a


def exemplo_bit_star_dubins(config, ax=None, curv_ax=None):
    print("BIT* DUBINS")
    p = RRTPlanner(
        config, 'bit_star_dubins',
        random_yaw_strategy='toward_goal',
        max_iter=1500, connect_circle_dist=4.5,
        step_size=0.05, goal_sample_rate=20,
        batch_size=200, goal_xy_th=0.10,
        goal_yaw_th=np.deg2rad(5),
    )
    p.run(animation=False)
    path = p.get_best_path()
    print(f"  {'OK' if path else 'FAIL'}")
    own_fig = ax is None
    if not path:
        print("  FAIL")
        if own_fig:
            return fig
        return None
    a = np.array(path)
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax, curv_ax = axes
        p.draw_scenario(ax=ax, show=False)
    comp = np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))
    style = dict(color='#9C27B0', ls='-.', lw=2.0, label=f'BIT* Dubins ({comp:.2f}m)')
    ax.plot(a[:, 0], a[:, 1], **style)
    ax.legend(fontsize=10)
    ax.set_title("BIT* Dubins")
    if own_fig:
        ax.axis([config.xmin, config.xmax, config.ymin, config.ymax])
        fig.tight_layout()
        return fig
    return p, a


def exemplo_de2d_nurbs(config, ax=None, curv_ax=None):
    print("DE2D_NURBS")
    from de2d_nurbs import DE2D_NURBS
    de = DE2D_NURBS(config)
    de.run()
    result = de.get_best_path()
    if result is None:
        print("  FAIL")
        return None
    pts = result["points"]
    comp = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    print(f"  OK  comp={comp:.2f}m")
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax, curv_ax = axes
        de.draw_scenario(ax=ax, show=False)
    style = dict(color='#4CAF50', ls='-', lw=2.5, label=f'DE2D_NURBS ({comp:.2f}m)')
    ax.plot(pts[:, 0], pts[:, 1], **style)
    ax.legend(fontsize=10)
    ax.set_title("DE2D_NURBS")
    if own_fig:
        ax.axis([config.xmin, config.xmax, config.ymin, config.ymax])
        fig.tight_layout()
        return fig
    return de, pts


def exemplo_pso2d_nurbs(config, ax=None, curv_ax=None):
    print("PSO2D_NURBS")
    pso = PSO2D_NURBS(config)
    pso.run()
    result = pso.get_best_path()
    if result is None:
        print("  FAIL")
        return None
    pts = result["points"]
    comp = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    print(f"  OK  comp={comp:.2f}m")
    own_fig = ax is None
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax, curv_ax = axes
        pso.draw_scenario(ax=ax, show=False)
    style = dict(color='#795548', ls=':', lw=2.5, label=f'PSO2D_NURBS ({comp:.2f}m)')
    ax.plot(pts[:, 0], pts[:, 1], **style)
    ax.legend(fontsize=10)
    ax.set_title("PSO2D_NURBS")
    if own_fig:
        ax.axis([config.xmin, config.xmax, config.ymin, config.ymax])
        fig.tight_layout()
        return fig
    return pso, pts


def exemplo_bit_star_theta(config, ax=None, curv_ax=None):
    print("BIT* THETA")
    p = RRTPlanner(
        config, 'bit_star_theta',
        random_yaw_strategy='toward_goal',
        max_iter=1500, connect_circle_dist=4.5,
        step_size=0.05, goal_sample_rate=20,
        batch_size=200, goal_xy_th=0.10,
        goal_yaw_th=np.deg2rad(5),
    )
    p.run(animation=False)
    path = p.get_best_path()
    print(f"  {'OK' if path else 'FAIL'}")
    own_fig = ax is None
    if not path:
        print("  FAIL")
        if own_fig:
            return fig
        return None
    a = np.array(path)
    if own_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax, curv_ax = axes
        p.draw_scenario(ax=ax, show=False)
    comp = np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))
    style = dict(color='#E91E63', ls='--', lw=2.0, label=f'BIT* Theta ({comp:.2f}m)')
    ax.plot(a[:, 0], a[:, 1], **style)
    ax.legend(fontsize=10)
    ax.set_title("BIT* Theta")
    if own_fig:
        ax.axis([config.xmin, config.xmax, config.ymin, config.ymax])
        fig.tight_layout()
        return fig
    return p, a


def _compute_curvature_and_arc(pts):
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-10)
    diff = np.diff(pts, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    s = np.zeros(len(pts))
    s[1:] = np.cumsum(seg)
    return k, s


if __name__ == '__main__':
    config = ScenarioConfig()
    config.seed = 2
    config.obstacle_type = "circles"
    config.occupancy_rate = 0.2
    config.start = np.array([-1.4, -0.8])
    config.goal = np.array([1.4, 0.8])
    config.th_start = np.deg2rad(0)
    config.th_goal = np.deg2rad(0)
    config.radius = 0.073
    config.kappa_max = 1 / 0.73
    config.max_fes = None
    config.n_generations = 200
    config.pop_size = 100
    config.nsampling = 120
    config.scale_x = 2.0
    config.scale_y = 2.0
    config.xmin = -2.0
    config.xmax = 2.0
    config.ymin = -2.0
    config.ymax = 2.0
    config.lambda_f = config.radius
    config.lambda_i = config.radius

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    # Desenha o cenário uma vez
    config.draw_scenario(ax=ax, show=True)

    # ── Cada figura separada (True) ou um único figure comparativo (False) ──
    MODO_FIGURAS_SEPARADAS = False

    planners = [
        # exemplo_bit_star_dubins,
        exemplo_de2d_nurbs,
        # exemplo_pso2d_nurbs,
        exemplo_rrt_star,
        # exemplo_rrt_star_dubins,
        # exemplo_bit_star_theta,
    ]

    # Cores que combinam com as cores dos paths
    planner_colors = {
        exemplo_bit_star_dubins: '#9C27B0',
        exemplo_de2d_nurbs:      '#4CAF50',
        exemplo_pso2d_nurbs:     '#795548',
        exemplo_rrt_star:        '#2196F3',
        exemplo_rrt_star_dubins: '#F44336',
        exemplo_bit_star_theta:  '#E91E63',
    }

    if MODO_FIGURAS_SEPARADAS:
        figs = [p(config) for p in planners]
        figs = [f for f in figs if f is not None]
    else:
        fig, (ax, curv_ax) = plt.subplots(1, 2, figsize=(16, 6))
        # Desenha o cenário uma vez
        config.draw_scenario(ax=ax, show=False)
        ax.legend_.remove()

        # Executa os planejadores e coleta resultados
        results = []
        for fn in planners:
            if fn == exemplo_pso2d_nurbs:
                config.n_generations = 200
                config.pop_size = 30
                config.alpha_kappa = 0.1
                config.alpha_obs = 500.0
                config.alpha_workspace = 1.0
            elif fn == exemplo_de2d_nurbs:
                config.n_generations = 200
                config.pop_size = 100
                config.alpha_kappa = 0.5
                config.alpha_obs = 10.0
                config.alpha_workspace = 10.0
            
            r = fn(config, ax=ax)
            if r is not None:
                results.append((fn, r[0], r[1]))

        ax.set_xlim(config.xmin, config.xmax)
        ax.set_ylim(config.ymin, config.ymax)
        ax.set_aspect('equal')
        # ax.set_title("Comparação de Planejadores")

        # Curvatura: analítica para Dubins, numérica para os demais
        for i, (fn, planner, pts) in enumerate(results):
            color = planner_colors.get(fn, f'C{i}')
            inner = getattr(planner, 'planner', None)
            if inner is not None and hasattr(inner, 'get_curvature_analytical'):
                k, s = inner.get_curvature_analytical()
            else:
                k, s = _compute_curvature_and_arc(pts)
            label = fn.__name__.replace('exemplo_', '').replace('_', ' ')
            curv_ax.plot(s, k, color=color, linewidth=1.5, label=label)
        curv_ax.axhline(y=config.kappa_max, color='r', linestyle='--',
                        linewidth=1.5, label=rf'$\kappa_{{\mathrm{{max}}}}$ = {config.kappa_max}')
        curv_ax.set_xlabel('Arc length', fontsize=12)
        curv_ax.set_ylabel('Curvature', fontsize=12)
        # curv_ax.set_title("Curvatura", fontsize=14)
        curv_ax.legend(fontsize=10)
        curv_ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs = [fig]

    if figs:
        plt.show()
