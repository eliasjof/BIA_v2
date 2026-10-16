"""
RRT* (RRT Star) path planning

author: Atsushi Sakai (@Atsushi_twi), extended version
"""

import math
import random
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))
from rrt import RRT


class RRTStar(RRT):
    """
    Class for RRT* planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=0.5,
                 path_resolution=0.1,
                 goal_sample_rate=20,
                 max_iter=500,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=0.0,
                 ):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        self.play_area = None

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not self.search_until_max_iter) and new_node:
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

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        r = min(r, self.connect_circle_dist)
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_indexes = [i for i, d in enumerate(dist_list) if d <= r ** 2]
        return near_indexes

    def choose_parent(self, new_node, near_indexes):
        if not near_indexes:
            return None

        best_cost = float('inf')
        best_parent_index = -1

        for i in near_indexes:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                cost = self.calc_new_cost(near_node, new_node)
                if cost < best_cost:
                    best_cost = cost
                    best_parent_index = i

        if best_parent_index == -1:
            return None

        new_node = self.steer(
            self.node_list[best_parent_index], new_node)
        new_node.cost = best_cost
        return new_node

    def rewire(self, new_node, near_indexes):
        for i in near_indexes:
            near_node = self.node_list[i]
            t_node = self.steer(new_node, near_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                cost = self.calc_new_cost(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost

    def search_best_goal_node(self):
        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.expand_dis:
                goal_indexes.append(i)

        if not goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in goal_indexes])
        for i in goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_index]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand,
                  self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    obstacle_list = [
        (5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
        (9, 5, 2), (8, 10, 1)
    ]

    rrt_star = RRTStar(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacle_list,
        expand_dis=0.5,
        path_resolution=0.1,
        goal_sample_rate=20,
        max_iter=500,
        connect_circle_dist=50.0,
        search_until_max_iter=False,
        robot_radius=0.0)
    path = rrt_star.planning(animation=True)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        rrt_star.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()
