import numpy as np
import sys
import pathlib
from scipy.spatial import KDTree
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))  # root dir
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import dubins_path_planner
from rrt_star import RRTStar
from plot import plot_arrow
from matplotlib import pyplot as plt
import random
import copy
show_animation = True


class RRTStarDubins(RRTStar):
    """
    Class for RRT star planning with Dubins path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=10,
                 max_iter=200,
                 connect_circle_dist=50.0,
                 robot_radius=0.0,
                 step_size=0.1,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        robot_radius: robot body modeled as circle with given radius
        step_size: step size for dubins path

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist
        self.step_size = step_size

        self.robot_radius = robot_radius
        self.curvature = 1.0 / 0.73  # for dubins path
        self.goal_yaw_th = np.deg2rad(0.5)
        self.goal_xy_th = self.robot_radius  # Tolerância igual ao raio do robô

    def planning(self, animation=True, search_until_max_iter=True):
        """
        RRT Star planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=100 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.axis("equal")
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = \
            dubins_path_planner.plan_dubins_path(
                from_node.x, from_node.y, from_node.yaw,
                to_node.x, to_node.y, to_node.yaw, self.curvature,
                step_size=self.step_size)

        # Convert path_yaw from course_lengths which are actually used in Dubins planning
        # ensure we re-calculate or get the segments correctly
        
        # O problema provável é que px, py, pyaw retornados pelo Dubins já contêm os pontos
        # baseados no step_size interno do dubins_path_planner.
        # Vamos verificar se o retorno do dubins_path_planner está consistente com o step_size.
        
        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.mode = mode
        new_node.course_lengths = course_lengths
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node
        
        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = dubins_path_planner.plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        cost = sum([abs(c) for c in course_lengths])

        return from_node.cost + cost

    def get_random_node(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand),
                            random.uniform(-np.pi, np.pi)
                            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)

        return rnd

    def search_best_goal_node(self):
        goal_indexes = []
        # Usa tolerâncias originais (mais restritas)
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                if abs(node.yaw - self.end.yaw) <= self.goal_yaw_th:
                    # Fazemos a verificação Dubins apenas se o nó estiver perto
                    px, _, _, _, _ = dubins_path_planner.plan_dubins_path(
                        node.x, node.y, node.yaw, 
                        self.end.x, self.end.y, self.end.yaw, 
                        self.curvature)
                    if len(px) > 1:
                        goal_indexes.append(i)
        
        if not goal_indexes:
            return None

        # Escolhe o nó com menor custo acumulado até o goal
        min_cost = min([self.node_list[i].cost for i in goal_indexes])
        for i in goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    def generate_final_course(self, goal_index):
        print("final")
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path