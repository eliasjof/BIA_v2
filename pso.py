import numpy as np
from mealpy import PSO as MealpyPSO, FloatVar
from mealpy.swarm_based.PSO import HPSO_TVAC
# from mealpy.swarm_based.PSO import PPSO

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
        self.initial_pop = initial_pop
        self.P_c = initial_pop.copy() if initial_pop is not None else None

        self.log_best_f = []
        self.log_best_CV = []
        self.log_population = []
        self.log_best_solutions = []
        self.gbest_pos = None
        self.gbest_f = np.inf
        self.gbest_CV = np.inf

    def get_CV(self, g, h, tolerance=1e-5):
        g_adj = np.maximum(0, g)
        g_adj[g_adj < 1e-18] = 0
        h_adj = np.maximum(0, np.abs(h) - tolerance)
        CV = np.sum(g_adj, axis=1) + np.sum(h_adj, axis=1)
        return CV

    def init_population(self, shared_pop=[]):
        pass

    def run(self, shared_pop=[]):
        if self.func is None:
            return

        def obj_func(solution):
            pop = solution.reshape(1, -1)
            f, g, h = self.func(pop, shared_pop=[],
                                static_params=self.static_params)
            CV = self.get_CV(g, h)
            cv_val = CV[0]
            # print(cv_val)
            return f[0] + cv_val if cv_val > 1e-5 else f[0]

        problem = {
            "bounds": FloatVar(lb=self.xmin, ub=self.xmax),
            "obj_func": obj_func,
            "minmax": "min",
            "save_population": True,
            "log_to": None,
        }
        model = HPSO_TVAC(
            epoch=self.n_generations,
            pop_size=self.pop_size,
            c1=self.c1, c2=self.c2, w=self.w,

        )
        # model = PPSO(epoch=self.n_generations, pop_size=self.pop_size, c1=self.c1, c2=self.c2, w=self.w)
        # model = MealpyPSO.OriginalPSO(
        #     epoch=self.n_generations,
        #     pop_size=self.pop_size,
        #     c1=self.c1, c2=self.c2, w=self.w,
        # )
        model.solve(
            problem,
            starting_solutions=self.initial_pop,
        )

        self.gbest_pos = model.g_best.solution
        self.gbest_f = model.g_best.target.fitness

        f, g, h = self.func(np.array([self.gbest_pos]), shared_pop=[],
                            static_params=self.static_params)
        cv_arr = self.get_CV(g, h)
        self.gbest_CV = float(cv_arr[0])

        self.log_best_f = []
        self.log_best_CV = []
        for agent in model.history.list_global_best:
            self.log_best_f.append(agent.target.fitness)
            f_i, g_i, h_i = self.func(np.array([agent.solution]), shared_pop=[],
                                      static_params=self.static_params)
            cv_i = self.get_CV(g_i, h_i)[0]
            self.log_best_CV.append(float(cv_i))

        self.P_c = np.array([agent.solution for agent in model.pop])
        self.log_population = [
            np.array([agent.solution for agent in pop])
            for pop in model.history.list_population
        ]

    def printD(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)
