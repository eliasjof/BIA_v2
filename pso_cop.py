import numpy as np
import random as _random
import time


class PSO:
    def __init__(self, pop_size, dim, max_fes=None, xmin=None, xmax=None,
                 func=None, static_params=None, initial_pop=None,
                 n_generations=None, c1=2.05, c2=2.05, w=0.7, w_min=0.4,
                 color='blue', id=0, DEBUG=0, **kwargs):
        self.pop_size = pop_size
        self.dim = dim
        self.xmin = np.asarray(xmin) if xmin is not None else np.zeros(dim)
        self.xmax = np.asarray(xmax) if xmax is not None else np.ones(dim)
        self.func = func
        self.static_params = static_params.copy() if static_params else {}
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.w_min = w_min
        self.color = color
        self.id = id
        self.DEBUG = DEBUG

        if n_generations is not None:
            self.n_generations = n_generations
            self.max_fes = n_generations * pop_size
        else:
            self.n_generations = max_fes // pop_size if max_fes else 1000
            self.max_fes = max_fes if max_fes else self.n_generations * pop_size

        self.fes = 0
        self.gen = 0

        if initial_pop is not None:
            self.P_c = initial_pop.copy()
        else:
            self.P_c = xmin + np.random.rand(pop_size, dim) * (xmax - xmin)

        self.V = np.zeros_like(self.P_c)
        v_max = 0.2 * (self.xmax - self.xmin)
        self.v_max = np.broadcast_to(v_max, (pop_size, dim))

        self.pbest_pos = self.P_c.copy()
        self.pbest_f = np.full(pop_size, np.inf)
        self.pbest_CV = np.full(pop_size, np.inf)
        self.gbest_pos = self.P_c[0].copy()
        self.gbest_f = np.inf
        self.gbest_CV = np.inf

        self.log_best_f = []
        self.log_best_CV = []
        self.log_population = []
        self.log_best_solutions = []

    def get_CV(self, g, h, tolerance=1e-5):
        g_adj = np.maximum(0, g)
        g_adj[g_adj < 1e-18] = 0
        h_adj = np.maximum(0, np.abs(h) - tolerance)
        CV = np.sum(g_adj, axis=1) + np.sum(h_adj, axis=1)
        return CV

    def evaluate(self, pop):
        f, g, h = self.func(pop, shared_pop=[], static_params=self.static_params)
        CV = self.get_CV(g, h)
        self.fes += pop.shape[0]
        return f, CV

    def init_population(self, shared_pop=[]):
        f, CV = self.evaluate(self.P_c)
        for i in range(self.pop_size):
            self.pbest_f[i] = f[i]
            self.pbest_CV[i] = CV[i]
            self.pbest_pos[i] = self.P_c[i].copy()

        best_idx = self._select_best(f, CV)
        self.gbest_pos = self.P_c[best_idx].copy()
        self.gbest_f = f[best_idx]
        self.gbest_CV = CV[best_idx]

    def _select_best(self, f, CV):
        feasible = CV <= 1e-5
        if np.any(feasible):
            candidates = np.where(feasible)[0]
            return candidates[np.argmin(f[feasible])]
        else:
            return np.argmin(CV)

    def update_gbest(self):
        for i in range(self.pop_size):
            if self.pbest_CV[i] < 1e-5 and self.gbest_CV >= 1e-5:
                self.gbest_pos = self.pbest_pos[i].copy()
                self.gbest_f = self.pbest_f[i]
                self.gbest_CV = self.pbest_CV[i]
            elif self.pbest_CV[i] < 1e-5 and self.gbest_CV < 1e-5:
                if self.pbest_f[i] < self.gbest_f:
                    self.gbest_pos = self.pbest_pos[i].copy()
                    self.gbest_f = self.pbest_f[i]
                    self.gbest_CV = self.pbest_CV[i]
            elif self.pbest_CV[i] >= 1e-5 and self.gbest_CV >= 1e-5:
                if self.pbest_CV[i] < self.gbest_CV:
                    self.gbest_pos = self.pbest_pos[i].copy()
                    self.gbest_f = self.pbest_f[i]
                    self.gbest_CV = self.pbest_CV[i]

    def run(self, shared_pop=[]):
        if self.fes == 0:
            self.init_population(shared_pop=shared_pop)

        for gen in range(self.n_generations):
            self.gen = gen

            # Adaptive inertia weight (linear decay)
            w_curr = self.w - (self.w - self.w_min) * gen / self.n_generations

            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)

            self.V = (w_curr * self.V
                      + self.c1 * r1 * (self.pbest_pos - self.P_c)
                      + self.c2 * r2 * (self.gbest_pos - self.P_c))

            v_scale = np.broadcast_to(self.v_max, self.V.shape)
            self.V = np.clip(self.V, -v_scale, v_scale)

            self.P_c = self.P_c + self.V
            self.P_c = np.clip(self.P_c, self.xmin, self.xmax)

            f_new, CV_new = self.evaluate(self.P_c)

            for i in range(self.pop_size):
                better_by_feasibility = (
                    (CV_new[i] < 1e-5 and self.pbest_CV[i] >= 1e-5) or
                    (CV_new[i] < 1e-5 and self.pbest_CV[i] < 1e-5 and f_new[i] < self.pbest_f[i]) or
                    (CV_new[i] >= 1e-5 and self.pbest_CV[i] >= 1e-5 and CV_new[i] < self.pbest_CV[i])
                )
                if better_by_feasibility:
                    self.pbest_f[i] = f_new[i]
                    self.pbest_CV[i] = CV_new[i]
                    self.pbest_pos[i] = self.P_c[i].copy()

            old_gbest_f = self.gbest_f
            self.update_gbest()

            self.log_best_f.append(self.gbest_f)
            self.log_best_CV.append(self.gbest_CV)
            self.log_population.append(self.P_c.copy())

            if self.DEBUG and (gen % 10 == 0 or gen == self.n_generations - 1):
                feasible = CV_new <= 1e-5
                n_feas = np.sum(feasible)
                print(f"[PSO] Gen {gen:4d}/{self.n_generations} | f_best={self.gbest_f:.4e} "
                      f"CV_best={self.gbest_CV:.4e} n_feas={n_feas}/{self.pop_size} "
                      f"fes={self.fes}/{self.max_fes}")

        return self.P_c

    def printD(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)
