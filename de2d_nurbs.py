import numpy as np
# from lshade_cop import LSHADE_COP
from lshade_cop_vectorized import LSHADE_COP
import matplotlib.pyplot as plt
from geomdl import NURBS, BSpline
from shapely.geometry import Polygon, Point, LineString
from __utils import *


class ScenarioConfig:
    """Configuração do cenário de planejamento de caminho.

    Parâmetros:
        width / height         : dimensões do workspace (centrado em 0,0).
        radius                 : raio do robô.
        kappa_max              : curvatura máxima permitida.
        lambd                  : comprimento de referência para os vetores
                                 tangentes inicial/final (default = radius*2).
        degree                 : grau da curva NURBS.
        num_free_ctrlpts       : número de pontos de controle livres.
        num_static_ctrlpts     : número de pontos de controle fixos
                                 (calculado automaticamente: 2*degree + 4).
        space_dim              : dimensão do espaço (2 ou 3).
        nsampling              : número de pontos amostrados na curva para
                                 avaliação de colisão (menor = mais rápido).
        line_width             : espessura das linhas nos plots.
        pop_size               : tamanho da população do LSHADE.
        max_fes                : máximo de avaliações da função objetivo.
        start                  : ponto inicial [x, y].
        goal                   : ponto final [x, y].
        th_start / th_goal     : ângulo de orientação inicial/final (rad).
        safe_radius            : raio de segurança ao redor de start/goal
                                 (nenhum obstáculo é gerado dentro dele).
        obstacle_type          : 'polygons' (polígonos convexos aleatórios) ou
                                 'circles' (círculos). Ignorado se obs_fornecida.
        n_obstacles            : número de obstáculos (apenas polygons).
        poly_size              : raio médio dos polígonos.
        r_min / r_max          : raio mínimo/máximo dos círculos.
        occupancy_rate         : fração do workspace a ser ocupada por
                                 círculos (0..1, apenas circles).
        seed                   : semente aleatória para reprodutibilidade.
        scale_x / scale_y      : escala dos limites da população (busca).
        debug                  : modo debug do LSHADE (0 = desligado).
        obs_fornecida          : lista de obstáculos fornecida pelo usuário.
                                 Se None, obstáculos são gerados aleatoriamente.
                                 Para círculos: lista de (x, y, r).
                                 Para polígonos: lista de shapely.Polygon.
        expanded_obs_fornecida : lista de obstáculos expandidos (com raio do
                                 robô). Se None, é calculado automaticamente.
    """
    def __init__(self):
        self.width = 3.2
        self.height = 2.0
        self.radius = 0.15 / 2
        self.kappa_max = 8.0
        self.lambd = self.radius * 2
        self.degree = 5
        self.num_free_ctrlpts = 4
        self.num_static_ctrlpts = 2 * self.degree + 4
        self.space_dim = 2
        self.nsampling = 50
        self.line_width = 2
        self.pop_size = 100
        self.max_fes = None
        self.n_generations = 200
        self.start = np.array([-1.4, -0.8])
        self.goal = np.array([1.4, 0.8])
        self.th_start = 0.0
        self.th_goal = 0.0
        self.safe_radius = 0.4
        self.obstacle_type = 'polygons'
        self.n_obstacles = 8
        self.poly_size = 0.3
        self.r_min = 0.1
        self.r_max = 0.3
        self.occupancy_rate = 0.15
        self.seed = 2
        self.scale_x = 1.5
        self.scale_y = 1.0
        self.debug = 0
        self.obs_fornecida = None
        self.expanded_obs_fornecida = None
        self._setup_done = False

    def setup(self):
        if self._setup_done:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.xmin = -self.width / 2
        self.xmax = self.width / 2
        self.ymin = -self.height / 2
        self.ymax = self.height / 2
        self.workspace = Polygon([
            (self.xmin, self.ymin),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
            (self.xmin, self.ymax)
        ])
        self._a_min = 2 * np.pi * self.radius**2
        self._a_max = 0.2
        safe_zones = [
            Point(self.start).buffer(self.safe_radius),
            Point(self.goal).buffer(self.safe_radius)
        ]
        if self.obs_fornecida is not None:
            self.obs = self.obs_fornecida
        elif self.obstacle_type == 'circles':
            self.obs = generate_circles_fast(
                self.occupancy_rate, self.xmin, self.xmax,
                self.ymin, self.ymax, self.r_min, self.r_max,
                start_point=self.start, end_point=self.goal,
                safe_radius=self.safe_radius)
        else:
            self.obs = generate_polygons(
                self.n_obstacles, self.xmin, self.xmax,
                self.ymin, self.ymax, self.poly_size,
                safe_zones, self._a_min, self._a_max)
        if self.expanded_obs_fornecida is not None:
            self.expanded_obs = self.expanded_obs_fornecida
        elif self.obstacle_type == 'circles' or (
                self.obs_fornecida is not None and
                len(self.obs) > 0 and
                isinstance(self.obs[0], (tuple, list)) and
                len(self.obs[0]) == 3):
            self.expanded_obs = [
                (*c[:2], c[2] + self.radius) for c in self.obs
            ]
        else:
            self.expanded_obs = [
                poly.buffer(self.radius) for poly in self.obs
            ]
        self._setup_done = True


class DE2D_NURBS:
    def __init__(self, config=None):
        self.config = config if config else ScenarioConfig()
        self.agent = None
        self._set_seed()

    def _set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    @staticmethod
    def generate_straightline(pti, ptf, gammai=0, gammaf=0, thi=0, thf=0,
                               num_points=5, vi=0.01, vf=0.01,
                               dimension=2, degree=5):
        dxi = np.cos(gammai) * vi
        dyi = np.sin(gammai) * vi
        pti2x = pti[0] + dxi * 2 / 3
        pti2y = pti[1] + dyi * 2 / 3
        if dimension == 3:
            pti2z = pti[2] + np.tan(thi) * vi * 2 / 3
        pti3x = pti2x + dxi * 1 / 3
        pti3y = pti2y + dyi * 1 / 3
        if dimension == 3:
            pti3z = pti2z + np.tan(thi) * vi * 1 / 3
        dxf = np.cos(gammaf) * vf
        dyf = np.sin(gammaf) * vf
        ptf2x = ptf[0] - dxf * 2 / 3
        ptf2y = ptf[1] - dyf * 2 / 3
        if dimension == 3:
            ptf2z = ptf[2] - np.tan(thf) * vf * 2 / 3
        ptf3x = ptf2x - dxf * 1 / 3
        ptf3y = ptf2y - dyf * 1 / 3
        if dimension == 3:
            ptf3z = ptf2z - np.tan(thf) * vf * 1 / 3
        line_points = []
        for _ in range(degree):
            line_points.extend(pti)
        if dimension == 3:
            line_points.extend([pti2x, pti2y, pti2z])
            point1 = [pti3x, pti3y, pti3z]
            point2 = [ptf2x, ptf2y, ptf2z]
            point3 = [ptf3x, ptf3y, ptf3z]
        elif dimension == 2:
            line_points.extend([pti2x, pti2y])
            point1 = [pti3x, pti3y]
            point2 = [ptf2x, ptf2y]
            point3 = [ptf3x, ptf3y]
        x = np.linspace(point1[0], point3[0], num_points + 2)
        y = np.linspace(point1[1], point3[1], num_points + 2)
        if dimension == 3:
            z = np.linspace(point1[2], point3[2], num_points + 2)
        if dimension == 3:
            for xi, yi, zi in zip(x, y, z):
                line_points.extend([xi, yi, zi])
        elif dimension == 2:
            for xi, yi in zip(x, y):
                line_points.extend([xi, yi])
        line_points.extend(point2)
        for _ in range(degree):
            line_points.extend(ptf)
        if dimension == 3:
            line_points = np.array(line_points).reshape(-1, 3)
            return line_points, [x, y, z]
        elif dimension == 2:
            line_points = np.array(line_points).reshape(-1, 2)
            return line_points, [x, y]

    def get_static_params(self, th_start=None, th_goal=None,
                          p_start=None, p_goal=None, agent_id=1):
        c = self.config
        c.setup()
        th_start = th_start if th_start is not None else c.th_start
        th_goal = th_goal if th_goal is not None else c.th_goal
        p_start = p_start if p_start is not None else c.start
        p_goal = p_goal if p_goal is not None else c.goal
        initial_ctrlpts, _ = self.generate_straightline(
            p_start, p_goal, gammai=th_start, gammaf=th_goal,
            thi=0, thf=0, num_points=c.num_free_ctrlpts,
            vi=c.lambd, vf=c.lambd, dimension=c.space_dim, degree=c.degree)
        n_total = c.num_free_ctrlpts + c.num_static_ctrlpts
        return {
            "th_start": th_start,
            "th_goal": th_goal,
            "p_start": p_start,
            "p_goal": p_goal,
            "initial_ctrlpts": initial_ctrlpts,
            "initial_weights": np.ones(n_total),
            "knots": list(np.linspace(0, 1, n_total + c.degree + 1)),
            "degree": c.degree,
            "num_free_ctrlpts": c.num_free_ctrlpts,
            "num_static_ctrlpts": c.num_static_ctrlpts,
            "kappa_max": c.kappa_max,
            "radius": c.radius,
            "space_dim": c.space_dim,
            "obstacles": c.obs,
            "expanded_obstacles": c.expanded_obs,
            "obstacle_type": c.obstacle_type,
            "xmin_env": c.xmin,
            "xmax_env": c.xmax,
            "ymin_env": c.ymin,
            "ymax_env": c.ymax,
            "workspace": c.workspace,
            "id": agent_id,
        }

    @staticmethod
    def get_nurbs(params, deltaP, deltaw, Nsampling=100):
        curve = NURBS.Curve()
        curve.degree = params['degree']
        ctrlpts = np.array(params['initial_ctrlpts'])
        s = params['num_static_ctrlpts'] // 2
        n = params['num_free_ctrlpts']
        ctrlpts[s:s + n] += deltaP
        curve.ctrlpts = ctrlpts.tolist()
        weights = np.array(params['initial_weights'])
        weights[s:s + n] += deltaw
        weights = weights / np.max(weights)
        curve.weights = weights
        curve.knotvector = params['knots']
        curve.delta = 1 / Nsampling
        return curve

    @staticmethod
    def get_bspline(params, deltaP, Nsampling=100):
        curve = BSpline.Curve()
        curve.degree = params['degree']
        ctrlpts = np.array(params['initial_ctrlpts'])
        s = params['num_static_ctrlpts'] // 2
        n = params['num_free_ctrlpts']
        ctrlpts[s:s + n] += deltaP
        curve.ctrlpts = ctrlpts.tolist()
        curve.knotvector = params['knots']
        curve.delta = 1 / Nsampling
        curve.evaluate()
        return curve

    @staticmethod
    def get_curvature_diff(points_curve):
        dx = np.gradient(points_curve[:, 0])
        dy = np.gradient(points_curve[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        norm_d = np.sqrt(dx**2 + dy**2)
        cross = dx * d2y - dy * d2x
        curvature = np.abs(cross) / (norm_d**3 + 1e-8)
        return curvature

    @staticmethod
    def compute_nurbs_basis(knot_vector, degree, nsampling):
        n_ctrlpts = len(knot_vector) - degree - 1
        eval_pts = np.linspace(knot_vector[degree], knot_vector[-degree-1], nsampling)
        N = np.zeros((nsampling, n_ctrlpts))
        for i_u, u in enumerate(eval_pts):
            if u >= knot_vector[-1]:
                span = n_ctrlpts - 1
            elif u <= knot_vector[0]:
                span = degree
            else:
                span = np.searchsorted(knot_vector, u, side='right') - 1
                span = max(degree, min(span, n_ctrlpts - 1))
            left = np.zeros(degree + 1)
            right = np.zeros(degree + 1)
            N_local = np.zeros(degree + 1)
            N_local[0] = 1.0
            for j in range(1, degree + 1):
                left[j] = u - knot_vector[span + 1 - j]
                right[j] = knot_vector[span + j] - u
                saved = 0.0
                for r in range(j):
                    temp = N_local[r] / (right[r+1] + left[j-r])
                    N_local[r] = saved + right[r+1] * temp
                    saved = left[j-r] * temp
                N_local[j] = saved
            N[i_u, span - degree:span + 1] = N_local
        return N, eval_pts

    @staticmethod
    def evaluate_nurbs_batch(ctrlpts_batch, weights_batch, basis_matrix):
        N = basis_matrix[np.newaxis, :, :]
        W = weights_batch[:, np.newaxis, :]
        weighted = N * W
        denom = weighted.sum(axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        R = weighted / denom
        points = np.matmul(R, ctrlpts_batch)
        return points

    @staticmethod
    def individual_cost_function(pop, shared_pop=[], static_params=None):
        if static_params is None:
            raise ValueError("static_params must be provided")

        if "basis_matrix" not in static_params:
            N, _ = DE2D_NURBS.compute_nurbs_basis(
                static_params["knots"], static_params["degree"],
                static_params.get("nsampling", 100))
            static_params["basis_matrix"] = N

        kappa_max = static_params["kappa_max"]
        r = static_params["radius"]
        n_free = static_params["num_free_ctrlpts"]
        space_dim = static_params["space_dim"]
        n_static = static_params["num_static_ctrlpts"]
        s = n_static // 2

        # ========== Extract all deltaP / deltaw ==========
        dim = n_free * space_dim
        deltaP_all = pop[:, :dim].reshape(pop.shape[0], n_free, space_dim)
        deltaw_all = pop[:, dim:]

        # ========== Build control points and weights for all ==========
        base_c = np.asarray(static_params["initial_ctrlpts"])
        base_w = np.asarray(static_params["initial_weights"])
        ctrlpts_all = np.broadcast_to(base_c, (pop.shape[0], *base_c.shape)).copy()
        ctrlpts_all[:, s:s+n_free] += deltaP_all
        weights_all = np.broadcast_to(base_w, (pop.shape[0], *base_w.shape)).copy()
        weights_all[:, s:s+n_free] += deltaw_all
        wmax = weights_all.max(axis=1, keepdims=True)
        wmax[wmax == 0] = 1.0
        weights_all = weights_all / wmax

        # ========== Evaluate all curves at once ==========
        points_all = DE2D_NURBS.evaluate_nurbs_batch(
            ctrlpts_all, weights_all, static_params["basis_matrix"])

        # ========== Length (vectorized) ==========
        diff_all = np.diff(points_all, axis=1)
        length = np.linalg.norm(diff_all, axis=-1).sum(axis=-1)

        # ========== Curvature (vectorized) ==========
        dx = np.gradient(points_all[:, :, 0], axis=1)
        dy = np.gradient(points_all[:, :, 1], axis=1)
        d2x = np.gradient(dx, axis=1)
        d2y = np.gradient(dy, axis=1)
        norm_d = np.sqrt(dx**2 + dy**2)
        cross = dx * d2y - dy * d2x
        kappa_all = np.abs(cross) / (norm_d**3 + 1e-8)
        kappa_all = np.where(kappa_all <= kappa_max, kappa_max, kappa_all)
        h_kappa = 0.05 * np.sum((kappa_all - kappa_max)**2, axis=1)

        # ========== Workspace (vectorized — rectangle) ==========
        xmin_env = static_params.get("xmin_env", -2)
        xmax_env = static_params.get("xmax_env", 2)
        ymin_env = static_params.get("ymin_env", -2)
        ymax_env = static_params.get("ymax_env", 2)
        pts_x = points_all[:, :, 0]
        pts_y = points_all[:, :, 1]
        inside_pt = (pts_x >= xmin_env) & (pts_x <= xmax_env) & \
                    (pts_y >= ymin_env) & (pts_y <= ymax_env)
        inside_seg = inside_pt[:, :-1] & inside_pt[:, 1:]
        h_workspace = np.sum(~inside_seg, axis=1) * 1000 * 0.01

        # ========== Collision (per individual, Shapely-dependent) ==========
        h_obs = np.zeros(pop.shape[0])
        expanded_obs = static_params["expanded_obstacles"]
        is_circle = (
            len(expanded_obs) > 0 and
            isinstance(expanded_obs[0], (tuple, list)) and
            len(expanded_obs[0]) == 3
        )
        for i in range(pop.shape[0]):
            pts = points_all[i]
            if is_circle:
                dists = check_collisions_cylinderBT(pts, expanded_obs, r)
                n_segs = len(dists)
                inside_len = np.sum(dists) if n_segs > 0 else 0.0
            else:
                n_segs, inside_len, _, _ = detailed_collision_with_polygons(
                    pts, expanded_obs)
            h_obs[i] = 10 * inside_len if n_segs > 0 else 0.0

        f = length
        g = np.zeros((pop.shape[0], 1))
        h = np.zeros((pop.shape[0], 3))
        h[:, 0] = h_kappa * f
        h[:, 1] = h_obs * f
        h[:, 2] = h_workspace * f
        g[g < 1e-18] = 0
        return f, g, h

    @staticmethod
    def get_points_from_solution(agent, solution, nsampling=100):
        static_params = agent.static_params
        dim = static_params["num_free_ctrlpts"] * static_params["space_dim"]
        deltaP = solution[:dim].reshape(
            (static_params["num_free_ctrlpts"], static_params["space_dim"]))
        deltaw = solution[dim:]
        curve = DE2D_NURBS.get_nurbs(static_params, deltaP, deltaw,
                                      Nsampling=nsampling)
        curve.evaluate()
        points_curve_i = np.array(curve.evalpts)
        return (points_curve_i, deltaP, deltaw,
                curve.ctrlpts, curve.weights, curve.knotvector)

    @staticmethod
    def draw_solutions(agents, iteration, ax=None,
                       show_initial_solution=False, show_info=False,
                       line_width=2):
        
         
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        for agent in agents:
            if iteration <0:
                iteration = len(agent.log_best_f) + iteration
            static_params = agent.static_params
            cv = np.array(agent.log_best_CV)
            cv[cv <= 1e-5] = 0
            feasible = np.where(cv == 0)[0]
            feasible_iter = feasible[0] if len(feasible) else -1
            for ii, ind in enumerate(agent.log_population[iteration]):
                pts, _, _, _, _, _ = DE2D_NURBS.get_points_from_solution(
                    agent, ind, nsampling=static_params.get("nsampling", 100))
                if ii == 0:
                    ax.plot(pts[:, 0], pts[:, 1], "-x",
                            color=agent.color, alpha=0.8,
                            label=f'Agent {static_params["id"]} Solutions')
                    points_curve_best = pts.copy()
                    if show_initial_solution:
                        dim = (static_params["num_free_ctrlpts"] *
                               static_params["space_dim"])
                        ind_0 = np.zeros(dim + static_params["num_free_ctrlpts"])
                        pts0, _, _, _, _, _ = DE2D_NURBS.get_points_from_solution(
                            agent, ind_0)
                        ax.plot(pts0[:, 0], pts0[:, 1], '--k',
                                alpha=0.8, label='Initial solution')
                else:
                    ax.plot(pts[:, 0], pts[:, 1],
                            color=agent.color, alpha=0.1)
            if show_info:
                text = (f'NURBS\nBest Fitness: {agent.log_best_f[iteration]:.4f}\n'
                        f'Best CV: {agent.log_best_CV[iteration]:.4f}')
                if feasible_iter >= 0:
                    text += f'\nFeasible after {feasible_iter} generations'
                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.7))
            ax.plot(static_params["p_start"][0], static_params["p_start"][1],
                    'go', markersize=4)
            ax.plot(static_params["p_goal"][0], static_params["p_goal"][1],
                    'ro', markersize=4)
            obs_type = static_params.get("obstacle_type", "polygons")
            for ob in static_params["obstacles"]:
                if obs_type == "circles":
                    draw_disc(p=np.array([ob[0], ob[1]]), r=ob[2],
                              ax=ax, color='gray', alpha=0.5)
                else:
                    x, y = ob.exterior.xy
                    ax.fill(x, y, alpha=0.5, edgecolor='k',
                            linewidth=line_width)
            if obs_type == "circles":
                for ob in static_params["expanded_obstacles"]:
                    draw_disc(p=np.array([ob[0], ob[1]]), r=ob[2],
                              ax=ax, color='gray', alpha=0.3)
            else:
                exp_obs = [poly.buffer(static_params["radius"])
                           for poly in static_params["obstacles"]]
                _, _, inside_seg, outside_seg = \
                    detailed_collision_with_polygons(
                        points_curve_best, exp_obs)
                for poly in exp_obs:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.4, edgecolor='k',
                            linewidth=line_width)
                for seg in outside_seg:
                    x, y = seg.xy
                    ax.plot(x, y, 'b-', linewidth=2)
                for seg in inside_seg:
                    x, y = seg.xy
                    ax.plot(x, y, 'r-', linewidth=2)
            x, y = static_params["workspace"].exterior.xy
            ax.plot(x, y, 'k-', linewidth=line_width, label="Workspace")
        sp = agents[0].static_params if agents else {}
        ax.set_title(f'Iteration {iteration}')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_xlim([sp.get("xmin_env", -2), sp.get("xmax_env", 2)])
        ax.set_ylim([sp.get("ymin_env", -1), sp.get("ymax_env", 1)])
        ax.set_aspect('equal')
        ax.grid()

    def setup_agent(self, agent_id=1):
        c = self.config
        static_params = self.get_static_params(agent_id=agent_id)
        dim = c.num_free_ctrlpts * c.space_dim + c.num_free_ctrlpts
        xmin_list = [[-c.scale_x, -c.scale_y]] * c.num_free_ctrlpts
        xmin_list.append([1e-5] * c.num_free_ctrlpts)
        xmin_pop = np.array(flatten_list(xmin_list))
        xmax_list = [[c.scale_x, c.scale_y]] * c.num_free_ctrlpts
        xmax_list.append([2.0] * c.num_free_ctrlpts)
        xmax_pop = np.array(flatten_list(xmax_list))
        initial_pop = xmin_pop + np.random.rand(c.pop_size, dim) * (
            xmax_pop - xmin_pop)
        for pop_i in initial_pop:
            w = pop_i[c.num_free_ctrlpts * c.space_dim:]
            pop_i[c.num_free_ctrlpts * c.space_dim:] = w / np.max(w)
        ctrlpts_i = np.zeros(c.num_free_ctrlpts * c.space_dim)
        weights_i = np.zeros(c.num_free_ctrlpts)
        initial_pop[0] = np.concatenate([ctrlpts_i, weights_i])
        static_params["nsampling"] = c.nsampling
        agent = LSHADE_COP(
            pop_size=c.pop_size,
            dim=dim,
            xmin=xmin_pop,
            xmax=xmax_pop,
            max_fes=c.max_fes,
            n_generations=c.n_generations,
            func=self.individual_cost_function,
            static_params=static_params,
            initial_pop=initial_pop.copy(),
            type_mean=2,
            type_mutation=1,
            type_sharing='best',
            DEBUG=c.debug,
            color=randomColor(n=1),
        )
        self.agent = agent
        return agent

    def run(self):
        if self.agent is None:
            self.setup_agent()
        self.agent.init_population(shared_pop=[])
        self.agent.run(shared_pop=[])
        return self.agent

    def draw_scenario(self, ax=None, show=True):
        """Desenha o cenário (obstáculos, workspace, start/goal) sem a curva."""
        c = self.config
        c.setup()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        x, y = c.workspace.exterior.xy
        ax.plot(x, y, 'k-', linewidth=c.line_width, label='Workspace')
        if c.obstacle_type == 'circles':
            for ob in c.obs:
                draw_disc(p=np.array([ob[0], ob[1]]), r=ob[2],
                          ax=ax, color='gray', alpha=0.5)
            for ob in c.expanded_obs:
                draw_disc(p=np.array([ob[0], ob[1]]), r=ob[2],
                          ax=ax, color='gray', alpha=0.3)
        else:
            for poly in c.obs:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, edgecolor='k',
                        linewidth=c.line_width)
            for poly in c.expanded_obs:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.3, edgecolor='k',
                        linewidth=c.line_width)
        ax.plot(c.start[0], c.start[1], 'go', markersize=8, label='Start')
        ax.plot(c.goal[0], c.goal[1], 'ro', markersize=8, label='Goal')        
        ax.set_xlim(c.xmin, c.xmax)
        ax.set_ylim(c.ymin, c.ymax)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid()
        ax.legend(fontsize=10)
        ax.set_title('Scenario')
        if show:
            plt.show()
        return ax

    def get_best_path(self, nsampling=None):
        if self.agent is None:
            return None
        c = self.config
        ns = nsampling or c.nsampling
        pts, _, _, ctrlpts, weights, knots = self.get_points_from_solution(
            self.agent, self.agent.log_population[-1][0], nsampling=ns)
        return {
            "points": pts,
            "ctrlpts": ctrlpts,
            "weights": weights,
            "knots": knots,
        }

    def plot_convergence(self, figsize=(10, 6)):
        if self.agent is None:
            return
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(self.agent.log_best_f, color='blue', label='Best Fitness')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Best Fitness', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid()
        ax2 = ax1.twinx()
        cv = np.array(self.agent.log_best_CV)
        cv[cv <= 1e-5] = 0
        ax2.plot(cv, color='red', label='Best CV', linestyle='--')
        ax2.set_ylabel('Best CV', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.set_title('Convergence')
        fig.tight_layout()
        plt.show()
        return fig, ax1, ax2

    def plot_result(self, iteration=-1, show_initial=True, show_info=True):
        fig, ax = plt.subplots(figsize=(10.8, 6))
        self.draw_solutions([self.agent], iteration=iteration, ax=ax,
                            show_initial_solution=show_initial,
                            show_info=show_info,
                            line_width=self.config.line_width)
        plt.show()
        return ax

    def plot_curvature(self, iteration=-1, ax=None, figsize=(8, 4),
                       nsampling=None):
        if self.agent is None:
            return None

        if iteration < 0:
            iteration = len(self.agent.log_best_f) + iteration

        kappa_max = self.agent.static_params["kappa_max"]
        solution = self.agent.log_population[iteration][0]
        ns = nsampling or self.config.nsampling #max(200, self.config.nsampling * 5)
        pts, _, _, _, _, _ = self.get_points_from_solution(
            self.agent, solution, nsampling=ns)

        curvature = self.get_curvature_diff(pts)

        diff = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diff, axis=1)
        arc_length = np.zeros(len(pts))
        arc_length[1:] = np.cumsum(seg_lengths)

        show = ax is None
        if show:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(arc_length, curvature, 'b-', linewidth=1.5,
                label='Curvature')
        ax.axhline(y=kappa_max, color='r', linestyle='--', linewidth=1.5,
                   label=rf'$\kappa_{{\mathrm{{max}}}}$ = {kappa_max}')

        ax.set_xlabel('Arc length', fontsize=12)
        ax.set_ylabel('Curvature', fontsize=12)
        ax.set_title(f'Curvature (generation {iteration})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig = ax.figure
        if show:
            fig.tight_layout()
            plt.show()
        return fig, ax

    def save_result(self, filepath=None, num_points=4000):
        import os
        best = self.get_best_path(nsampling=num_points)
        if best is None:
            return
        self.agent.static_params["best_curve_pts"] = best["points"]
        self.agent.static_params["best_ctrlpts"] = best["ctrlpts"]
        self.agent.static_params["best_weights"] = best["weights"]
        self.agent.static_params["best_knotvector"] = best["knots"]
        data = np.array(self.agent.static_params, dtype=object)
        fpath = filepath or f'../results/test_{self.config.seed}/agent_{self.config.seed}.npy'
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        np.save(fpath, data, allow_pickle=True)
        return fpath
