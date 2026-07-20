import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scenario_config import ScenarioConfig
from rrt_based.rrt_planner import RRTPlanner


def exemplo_rrt_star():
    """RRT* classico — funciona em cenarios apertados (sem orientacao)."""
    print("=" * 60)
    print("RRT* CLASSICO (geometrico, sem orientacao)")
    print("=" * 60)

    config = ScenarioConfig()
    config.obs_fornecida = [(0.6883004767862384, 0.3022825881539657, 0.14464214762976454),(-1.0917736086156131, -0.1126768217023627, 0.2784359135409691),(-0.9002903939515345, 0.26820105644434156, 0.10530719393677274),(-1.3207217472358725, 0.4514588866302939, 0.2618860913355653),(-0.4346374873469, -0.5239275669138156, 0.23962787899764537),(1.0290397406091127, 0.18269405408555595, 0.11934327536669281),(-0.3171168234468903, 0.07341651282804262, 0.29462315279587414)]
    config.start = np.array([-1.5, -1.0])
    config.goal = np.array([1.5, 1.0])
    config.th_start = 0.0
    config.th_goal = 0.0
    config.radius = 0.073
    config.kappa_max = 1/0.73

    p = RRTPlanner(config, 'rrt_star',
                   max_iter=2000, expand_dis=0.2,
                   path_resolution=0.05, goal_sample_rate=30)
    p.run(animation=False)
    path = p.get_best_path()

    print(f"Caminho: {'OK' if path else 'FAIL'}")

    smooth = p.smooth_path()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p.draw_scenario(ax=axes[0], show=False)
    if path:
        raw = np.array(path)
        axes[0].plot(raw[:, 0], raw[:, 1], 'b-', lw=1.5, alpha=0.4, label='RRT* (raw)')
        axes[0].plot(smooth[:, 0], smooth[:, 1], 'r-', lw=2, label='B-Spline')
    axes[0].set_title("RRT* Classico - Trajetoria")
    axes[0].legend()
    p.plot_curvature(smoothed=False, ax=axes[1])
    axes[1].set_title("Curvatura (raw)")
    p.plot_curvature(smoothed=True, ax=axes[2])
    axes[2].set_title("Curvatura (raw + smoothed)")
    plt.tight_layout()
    plt.show()


def exemplo_rrt_star_dubins():
    

    config = ScenarioConfig()
    # config.obs_fornecida = [
    #     [-0.8, -0.2, 0.4],
    #     [0.0, 0.6, 0.4],
    #     [0.0, -0.9, 0.4],
    #     [0.8, 0.2, 0.4]
    # ]
    config.obs_fornecida = [(0.6883004767862384, 0.3022825881539657, 0.14464214762976454),(-1.0917736086156131, -0.1126768217023627, 0.2784359135409691),(-0.9002903939515345, 0.26820105644434156, 0.10530719393677274),(-1.3207217472358725, 0.4514588866302939, 0.2618860913355653),(-0.4346374873469, -0.5239275669138156, 0.23962787899764537),(1.0290397406091127, 0.18269405408555595, 0.11934327536669281),(-0.3171168234468903, 0.07341651282804262, 0.29462315279587414)]
    # config.obs_fornecida = []
    config.scale_x = [-2.1, 1.6]
    config.scale_y = [-2.1, 1.6]
    config.start = np.array([-1.5, -1.0])
    config.goal = np.array([1.5, 1.0])
    config.th_start = np.deg2rad(0)
    config.th_goal = np.deg2rad(45)
    config.radius = 0.073
    config.kappa_max = 1/0.73

    p = RRTPlanner(
    config,
    'rrt_star_dubins',
    random_yaw_strategy='toward_goal',
    max_iter=900,
    connect_circle_dist=5.5,
    step_size=0.02,
    goal_sample_rate=40,
    goal_xy_th=0.10,
    goal_yaw_th=np.deg2rad(5),
    rand_area_x=[-3.0, 3],
    rand_area_y=[-3, 3]
)
    p.run(animation=False)
    path = p.get_best_path()

    print(f"Caminho: {'OK' if path else 'FAIL'}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p.draw_scenario(ax=axes[0], show=False)
    if path:
        a = np.array(path)
        comp = np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1))
        axes[0].plot(a[:, 0], a[:, 1], 'r-', lw=2, label=f'RRT* Dubins ({comp:.2f}m)')
        axes[0].legend()
    axes[0].set_title("RRT* Dubins - Trajetoria")
    axes[0].axis([-3,3,-3,3])
    p.plot_curvature(ax=axes[1])   
    axes[1].set_title("Curvatura")
    plt.tight_layout()    
    plt.show()


if __name__ == '__main__':
    # exemplo_rrt_star()
    exemplo_rrt_star_dubins()
