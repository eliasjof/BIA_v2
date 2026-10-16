import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
from scipy.interpolate import splprep, splev

# Assumindo que a estrutura de pastas permite importar os módulos necessários
sys.path.append(str(pathlib.Path(__file__).parent.parent)) 

from rrt_star_dubins import RRTStarDubins
from rrt_star import RRTStar

def run_rrt_dubins(start, goal, obstacle_list, rand_area=[-2.0, 1.8]):
    """
    Interface para rodar o RRT* com Dubins Path
    """
    print("Iniciando RRT* com Dubins para Robô Ackerman")
    
    # Configuração do RRT*
    rrt_planner = RRTStarDubins(
        start=start, 
        goal=goal, 
        obstacle_list=obstacle_list, 
        rand_area=rand_area,
        max_iter=900, 
        connect_circle_dist=4.5,
        robot_radius=0.073, 
        step_size=0.05,
    )
    
    # Curvatura = 1.0 / raio_minimo (0.73)
    rrt_planner.curvature = 1 / 0.73
    rrt_planner.goal_sample_rate = 15
    
    path = rrt_planner.planning(animation=False)
    return path, rrt_planner

def main():
    # Definições do usuário
    xlim = [-2.1, 1.6]
    ylim = [-2.1, 1.6]
    figsize = (8, 8)
    obstacle_list = []
    # Cenário de "Chicane" (Curva em S apertada) para forçar o corner-cutting
    # obstacle_list = [
    #     [-0.8, -0.2, 0.4],
    #     [0.0, 0.6, 0.4],
    #     [0.0, -0.9, 0.4],
    #     [0.8, 0.2, 0.4]
    # ]
    # obstacle_list = [(0.6883004767862384, 0.3022825881539657, 0.14464214762976454),(-1.0917736086156131, -0.1126768217023627, 0.2784359135409691),(-0.9002903939515345, 0.26820105644434156, 0.10530719393677274),(-1.3207217472358725, 0.4514588866302939, 0.2618860913355653),(-0.4346374873469, -0.5239275669138156, 0.23962787899764537),(1.0290397406091127, 0.18269405408555595, 0.11934327536669281),(-0.3171168234468903, 0.07341651282804262, 0.29462315279587414)]
    start = [-1.80, -0.50, np.deg2rad(0.0)]
    goal = [1.50, 0.80, np.deg2rad(45.0)]

    path, planner = run_rrt_dubins(start, goal, obstacle_list, rand_area=[xlim[0], xlim[1]])

    # Executar RRT* clássico para comparação (Parâmetros ajustados para o cenário)
    # Executar RRT* clássico para comparação (Parâmetros ajustados para o cenário)
    rrt_classic = RRTStar(start=[start[0], start[1]], goal=[goal[0], goal[1]], 
                          obstacle_list=obstacle_list, rand_area=[xlim[0], xlim[1]], 
                          expand_dis=0.2,
                          path_resolution=0.05,
                          connect_circle_dist=0.8,
                          goal_sample_rate=30,
                          max_iter=2000,
                          search_until_max_iter=True,
                          robot_radius=0.073)
    path_rrt = rrt_classic.planning(animation=True)


    if not path:
        print("Falha ao encontrar caminho.")
        return

    print("Caminho encontrado com sucesso!")
    
    # Calcula o comprimento total do caminho
    total_dist = sum(np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1))
    print(f"Comprimento total do caminho: {total_dist:.4f} metros")

    # ==========================================================
    # RECUPERAÇÃO CORRETA DO NÓ OBJETIVO (GOAL)
    # ==========================================================
    best_goal_node = None
    if hasattr(planner, 'search_best_goal_node'):
        best_goal_index = planner.search_best_goal_node()
        if best_goal_index is not None:
            best_goal_node = planner.node_list[best_goal_index]
    
    # Fallback genérico: procura o nó mais próximo do goal
    if best_goal_node is None:
        min_dist = float('inf')
        for n in planner.node_list:
            dist = np.hypot(n.x - goal[0], n.y - goal[1])
            if dist < min_dist:
                min_dist = dist
                best_goal_node = n

    # Reconstrói a lista de nós do início ao fim
    nodes_path = []
    node = best_goal_node
    while node is not None:
        nodes_path.append(node)
        node = node.parent
    nodes_path.reverse()

    # ==========================================================
    # GRÁFICO 1: CAMINHO ORIGINAL DUBINS
    # ==========================================================
    plt.figure(figsize=figsize)
    color_map = {'L': 'blue', 'R': 'orange', 'S': 'red'}
    
    for n in nodes_path:
        if n.parent is None: continue
        total_len = sum(abs(c) for c in n.course_lengths)
        n_points = len(n.path_x)
        curr_idx = 0
        
        for i, mode in enumerate(n.mode):
            segment_len = abs(n.course_lengths[i])
            num_points = int(n_points * (segment_len / total_len)) if total_len > 0 else 0
            if i == len(n.mode) - 1:
                num_points = n_points - curr_idx
            
            end_idx = min(curr_idx + num_points + 1, n_points)
            plt.plot(n.path_x[curr_idx:end_idx], n.path_y[curr_idx:end_idx], 
                     color=color_map.get(mode, 'black'), linewidth=3)
            curr_idx = end_idx - 1

    for (ox, oy, size) in obstacle_list:
        plt.gca().add_patch(plt.Circle((ox, oy), radius=size, color="green", alpha=0.3))
        plt.gca().add_patch(plt.Circle((ox, oy), radius=size + 0.073, color="tan", alpha=0.3))
        
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("1º Estágio: Caminho Dubins (Azul=L, Laranja=R, Vermelho=S)")
    plt.show()

    # ==========================================================
    # GRÁFICO 2: NORMA DA CURVATURA DO DUBINS
    # ==========================================================
    plt.figure(figsize=(8, 4))
    all_curvatures = []
    
    for n in nodes_path:
        if hasattr(n, 'mode'):
            for i, mode in enumerate(n.mode):
                c = planner.curvature if mode != 'S' else 0
                if mode == 'R': c = -planner.curvature
                all_curvatures.extend([c] * max(1, int(len(n.path_x) / len(n.mode))))

    plt.plot(np.abs(all_curvatures), label="Norma da Curvatura", color='blue')
    plt.axhline(y=planner.curvature, color='r', linestyle='--', label="Limite Mecânico do Robô")
    plt.xlabel("Passos da trajetória")
    plt.ylabel("Norma da Curvatura $\kappa$ [1/m]")
    plt.title("Análise da Norma da Curvatura (1º Estágio - Dubins)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ==========================================================
    # GRÁFICO 3: DEMONSTRAÇÃO DO "CORNER-CUTTING"
    # ==========================================================
    plt.figure(figsize=figsize)
    
    # ... (Dubins path setup) ...
    # (Existing logic for Dubins path smooth_x, smooth_y)

    # 3. Adicionar suavização B-Spline para o RRT* Clássico
    full_x = [p[0] for p in path]
    full_y = [p[1] for p in path]
    
    step = max(1, len(full_x) // 20)
    wx = full_x[::step]
    wy = full_y[::step]
    if wx[-1] != full_x[-1]: 
        wx.append(full_x[-1])
        wy.append(full_y[-1])

    weights = np.ones(len(wx))
    weights[0] = 1000.0   
    weights[-1] = 1000.0  
    
    tck, u = splprep([wx, wy], w=weights, s=0.015, k=3) 
    u_new = np.linspace(0, 1, 500)
    smooth_x, smooth_y = splev(u_new, tck)

    # 3. Adicionar suavização B-Spline para o RRT* Clássico
    # Adicionar obstáculos ao gráfico
    for (ox, oy, size) in obstacle_list:
        plt.gca().add_patch(plt.Circle((ox, oy), radius=size, color="green", alpha=0.3))
        plt.gca().add_patch(plt.Circle((ox, oy), radius=size + 0.073, color="tan", alpha=0.3))

    if path_rrt:
        rrt_x = [p[0] for p in path_rrt]
        rrt_y = [p[1] for p in path_rrt]
        
        step_rrt = max(1, len(rrt_x) // 20)
        wx_rrt = rrt_x[::step_rrt]
        wy_rrt = rrt_y[::step_rrt]
        
        # Suavização B-Spline para o RRT*
        if len(wx_rrt) > 3:
            try:
                tck_rrt, _ = splprep([wx_rrt, wy_rrt], s=0.015, k=3)
                u_rrt = np.linspace(0, 1, 500)
                smooth_x_rrt, smooth_y_rrt = splev(u_rrt, tck_rrt)
                plt.plot(smooth_x_rrt, smooth_y_rrt, color='green', linestyle='--', linewidth=2, label='B-Spline (RRT* Suavizado)')
                
                # Curvatura para o RRT* suavizado
                dx_rrt, dy_rrt = splev(u_rrt, tck_rrt, der=1)
                ddx_rrt, ddy_rrt = splev(u_rrt, tck_rrt, der=2)
                curv_rrt = np.abs(dx_rrt * ddy_rrt - dy_rrt * ddx_rrt) / ((dx_rrt**2 + dy_rrt**2)**1.5)
            except Exception as e:
                print(f"Aviso: Não foi possível calcular a B-Spline do RRT*: {e}")
                curv_rrt = None
        else:
            curv_rrt = None
            
        plt.plot(rrt_x, rrt_y, color='green', linestyle=':', linewidth=1, alpha=0.5, label='RRT* Clássico')


    plt.plot(full_x, full_y, color='blue', linestyle='--', linewidth=2, zorder=3, label='1º Estágio: Dubins')
    plt.plot(smooth_x, smooth_y, color='red', linewidth=3, zorder=4, label='2º Estágio: B-Spline (Corner-cutting)')
    plt.scatter(wx, wy, color='black', s=30, zorder=5, label='Waypoints extraídos')

    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title("O Problema do 'Corner-Cutting' em Planejadores Desacoplados")
    plt.legend(loc='upper right', fontsize=9)
    plt.show()

    # ==========================================================
    # GRÁFICO 4: NORMA DA CURVATURA (B-SPLINE E RRT*)
    # ==========================================================
    plt.figure(figsize=(8, 4))
    
    # Curvatura B-Spline original
    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)
    curvature_smooth = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
    plt.plot(u_new, curvature_smooth, color='red', linewidth=2, label="Curvatura (Dubins + B-Spline)")
    
    # Curvatura RRT* suavizado (se existir)
    if 'curv_rrt' in locals():
        # Normaliza u_rrt para que o plot fique visualmente comparável
        plt.plot(u_rrt, curv_rrt, color='green', linewidth=2, label="Curvatura (RRT* + B-Spline)")
    
    plt.axhline(y=planner.curvature, color='black', linestyle='--', linewidth=2, label="Limite Mecânico do Robô")
    
    plt.xlabel("Parâmetro da curva (u)")
    plt.ylabel("Norma da Curvatura $\kappa$ [1/m]")
    plt.title("Comparação da Norma da Curvatura")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()