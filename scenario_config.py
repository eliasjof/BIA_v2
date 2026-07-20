import random
import numpy as np
from shapely.geometry import Polygon, Point
from __utils import generate_circles_fast, generate_polygons


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
        if not hasattr(self, 'xmin'):
            self.xmin = -self.width / 2
        if not hasattr(self, 'xmax'):
            self.xmax = self.width / 2
        if not hasattr(self, 'ymin'):
            self.ymin = -self.height / 2
        if not hasattr(self, 'ymax'):
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
