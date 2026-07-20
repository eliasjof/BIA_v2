import numpy as np
from de2d_nurbs import DE2D_NURBS
from pso import PSO
from __utils import *


class PSO2D_NURBS(DE2D_NURBS):
    def setup_agent(self, agent_id=1):
        c = self.config
        static_params = self.get_static_params(agent_id=agent_id)
        dim = c.num_free_ctrlpts * c.space_dim + c.num_free_ctrlpts
        xmin_list = [[-c.scale_x, -c.scale_y]] * c.num_free_ctrlpts
        xmin_list.append([1e-6] * c.num_free_ctrlpts)
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
        agent = PSO(
            pop_size=c.pop_size,
            dim=dim,
            xmin=xmin_pop,
            xmax=xmax_pop,
            max_fes=c.max_fes,
            n_generations=c.n_generations,
            func=self.individual_cost_function,
            static_params=static_params,
            initial_pop=initial_pop.copy(),
            c1=2.05,
            c2=2.05,
            w=0.7,
            w_min=0.4,
            DEBUG=c.debug,
            color=randomColor(n=1),
        )
        self.agent = agent
        return agent

    def get_best_path(self, nsampling=None):
        if self.agent is None or self.agent.gbest_pos is None:
            return None
        c = self.config
        ns = nsampling or c.nsampling
        pts, _, _, ctrlpts, weights, knots = self.get_points_from_solution(
            self.agent, self.agent.gbest_pos, nsampling=ns)
        return {
            "points": pts,
            "ctrlpts": ctrlpts,
            "weights": weights,
            "knots": knots,
        }
