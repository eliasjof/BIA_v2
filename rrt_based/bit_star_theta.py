"""
BIT* (Batch Informed Trees) with straight-line steering in (x, y, theta).

Classic BIT* formulation with configuration space (x, y, theta).
Steering connects two configurations with a straight line;
theta along the path is the direction of motion.

Reference:
    Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2015).
    Batch Informed Trees (BIT*): Sampling-based Optimal Planning via the
    Heuristically Guided Search of Implicit Random Geometric Graphs.
"""

import math
import random
import copy
import heapq
import sys
import pathlib
from math import sin, cos, atan2, sqrt, acos, pi, hypot

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).parent))
from plot import plot_arrow


def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


class BITStarTheta:
    class Node:
        def __init__(self, x, y, yaw=0.0):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.cost = float('inf')
            self.parent = None
            self.path_x = []
            self.path_y = []
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=20, max_iter=500,
                 connect_circle_dist=2.0, robot_radius=0.0,
                 step_size=0.1, batch_size=200,
                 search_until_max_iter=True, random_yaw_strategy='toward_goal',
                 rand_area_x=None, rand_area_y=None):
        self.start = self.Node(
            start[0], start[1], start[2] if len(start) > 2 else 0.0)
        self.goal = self.Node(
            goal[0], goal[1], goal[2] if len(goal) > 2 else 0.0)

        if rand_area_x is not None:
            self.min_rand_x, self.max_rand_x = rand_area_x
            if rand_area_y is not None:
                self.min_rand_y, self.max_rand_y = rand_area_y
            else:
                self.min_rand_y, self.max_rand_y = rand_area_x
            self.min_rand = self.min_rand_x
            self.max_rand = self.max_rand_x
        else:
            self.min_rand = rand_area[0]
            self.max_rand = rand_area[1]
            self.min_rand_x = self.min_rand_y = self.min_rand
            self.max_rand_x = self.max_rand_y = self.max_rand

        self.obstacle_list = obstacle_list
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.batch_size = batch_size
        self.search_until_max_iter = search_until_max_iter
        self.random_yaw_strategy = random_yaw_strategy

        self.goal_xy_th = self.robot_radius
        self.goal_yaw_th = np.deg2rad(30)

        self.vertices = []
        self.samples = {}
        self.queue = []
        self.sample_to_vertex = {}
        self.solution_cost = float('inf')
        self.solution_vertex_idx = None
        self.solution_goal_path = None

    # ── Heuristics ────────────────────────────────────

    @staticmethod
    def euclidean_dist(a, b):
        return hypot(a.x - b.x, a.y - b.y)

    def heuristic(self, node):
        return self.euclidean_dist(node, self.goal)

    def edge_heuristic(self, a, b):
        return self.euclidean_dist(a, b)

    def f_hat(self, v, s):
        return v.cost + self.edge_heuristic(v, s) + self.heuristic(s)

    # ── Sampling ───────────────────────────────────────

    def _sample_state(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.uniform(self.min_rand_x, self.max_rand_x)
            y = random.uniform(self.min_rand_y, self.max_rand_y)
            if self.random_yaw_strategy == 'toward_goal':
                yaw = math.atan2(self.goal.y - y, self.goal.x - x)
            else:
                yaw = random.uniform(-np.pi, np.pi)
        else:
            x, y = self.goal.x, self.goal.y
            yaw = self.goal.yaw
        return self.Node(x, y, yaw)

    def _sample_batch(self):
        self.samples = {}
        self.sample_to_vertex = {}
        for _ in range(self.batch_size):
            s = self._sample_state()
            self.samples[id(s)] = s

    # ── Queue ──────────────────────────────────────────

    def _build_queue(self):
        self.queue = []
        for v_idx, v in enumerate(self.vertices):
            for s_id, s in self.samples.items():
                f = self.f_hat(v, s)
                if f < self.solution_cost:
                    heapq.heappush(self.queue, (f, v_idx, s_id))

    # ── Bezier steering (position + orientation) ──────

    @staticmethod
    def _cubic_bezier(p0, p1, p2, p3, n):
        t = np.linspace(0, 1, n)
        b = (1 - t)**3 * np.array(p0).reshape(-1, 1) \
          + 3 * (1 - t)**2 * t * np.array(p1).reshape(-1, 1) \
          + 3 * (1 - t) * t**2 * np.array(p2).reshape(-1, 1) \
          + t**3 * np.array(p3).reshape(-1, 1)
        return b.T

    def _steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = hypot(dx, dy)
        if d < 1e-6:
            return None

        alpha = d / 3.0
        p0 = (from_node.x, from_node.y)
        p3 = (to_node.x, to_node.y)
        p1 = (from_node.x + alpha * cos(from_node.yaw),
              from_node.y + alpha * sin(from_node.yaw))
        p2 = (to_node.x - alpha * cos(to_node.yaw),
              to_node.y - alpha * sin(to_node.yaw))

        n = max(8, int(d / self.step_size))
        pts = self._cubic_bezier(p0, p1, p2, p3, n)
        pts[-1] = np.array([to_node.x, to_node.y])
        first_dir = np.array([cos(from_node.yaw), sin(from_node.yaw)])
        last_dir = np.array([cos(to_node.yaw), sin(to_node.yaw)])
        pts[1] = pts[0] + self.step_size * first_dir
        pts[-2] = pts[-1] - self.step_size * last_dir

        t = np.linspace(0, 1, n)
        v0 = np.array(p1) - np.array(p0)
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        db = (3 * (1 - t)**2 * v0.reshape(-1, 1)
            + 6 * (1 - t) * t * v1.reshape(-1, 1)
            + 3 * t**2 * v2.reshape(-1, 1)).T
        yaws = np.arctan2(db[:, 1], db[:, 0])
        yaws[-1] = to_node.yaw

        seg = np.diff(pts, axis=0)
        total_len = np.linalg.norm(seg, axis=-1).sum()

        return {
            'x': pts[:, 0].tolist(),
            'y': pts[:, 1].tolist(),
            'yaw': yaws.tolist(),
            'cost': total_len,
        }

    def _check_collision(self, path_data):
        if path_data is None:
            return False
        for ox, oy, size in self.obstacle_list:
            for px, py in zip(path_data['x'], path_data['y']):
                if (px - ox) ** 2 + (py - oy) ** 2 <= (size + self.robot_radius) ** 2:
                    return False
        return True

    # ── Near-neighbor search ──────────────────────────

    def _find_near_indices(self, node, radius=None):
        if radius is None:
            n = len(self.vertices)
            r = self.connect_circle_dist * math.sqrt(math.log(n + 1) / (n + 1))
            r = min(r, self.connect_circle_dist)
        else:
            r = radius
        near_indices = []
        for i, v in enumerate(self.vertices):
            if self.euclidean_dist(v, node) <= r:
                near_indices.append(i)
        return near_indices

    # ── Rewiring ───────────────────────────────────────

    def _is_ancestor(self, ancestor, node):
        while node is not None:
            if node is ancestor:
                return True
            node = node.parent
        return False

    def _rewire(self, new_idx):
        new_node = self.vertices[new_idx]
        near_indices = self._find_near_indices(new_node)

        for i in near_indices:
            if i == new_idx:
                continue
            near_node = self.vertices[i]

            if self._is_ancestor(near_node, new_node):
                continue

            path = self._steer(new_node, near_node)
            if path is None:
                continue
            new_cost = new_node.cost + path['cost']
            if new_cost < near_node.cost:
                if self._check_collision(path):
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    near_node.path_x = path['x']
                    near_node.path_y = path['y']
                    near_node.path_yaw = path['yaw']

        for i in near_indices:
            if i == new_idx:
                continue
            near_node = self.vertices[i]

            if self._is_ancestor(new_node, near_node):
                continue

            path = self._steer(near_node, new_node)
            if path is None:
                continue
            new_cost = near_node.cost + path['cost']
            if new_cost < new_node.cost:
                if self._check_collision(path):
                    new_node.parent = near_node
                    new_node.cost = new_cost
                    new_node.path_x = path['x']
                    new_node.path_y = path['y']
                    new_node.path_yaw = path['yaw']

    # ── Goal checking ─────────────────────────────────

    def _check_goal(self, node_idx):
        node = self.vertices[node_idx]

        d = self.euclidean_dist(node, self.goal)
        yaw_diff = abs(angle_mod(node.yaw - self.goal.yaw))

        if d <= self.goal_xy_th and yaw_diff <= self.goal_yaw_th:
            return True, None, node.cost

        path = self._steer(node, self.goal)
        if path is None:
            return False, None, float('inf')
        if not self._check_collision(path):
            return False, None, float('inf')

        total = node.cost + path['cost']
        return True, path, total

    # ── Final path reconstruction ──────────────────────

    def generate_final_course(self, goal_index):
        if goal_index is None:
            return None

        node_path = []
        n = self.vertices[goal_index]
        while n is not None:
            node_path.append(n)
            n = n.parent
        node_path.reverse()

        path = [[node_path[0].x, node_path[0].y]]
        for n in node_path[1:]:
            if len(n.path_x) > 0:
                for x, y in zip(n.path_x[1:], n.path_y[1:]):
                    path.append([x, y])
            else:
                path.append([n.x, n.y])

        if self.solution_goal_path is not None:
            gp = self.solution_goal_path
            for x, y in zip(gp['x'][1:], gp['y'][1:]):
                path.append([x, y])
        else:
            last = self.vertices[goal_index]
            d = self.euclidean_dist(last, self.goal)
            if d > self.goal_xy_th:
                path.append([self.goal.x, self.goal.y])

        return path

    # ── Main planning loop ─────────────────────────────

    def planning(self, animation=True, search_until_max_iter=None):
        if search_until_max_iter is None:
            search_until_max_iter = self.search_until_max_iter

        self.start.cost = 0.0
        self.vertices = [self.start]
        self.solution_cost = float('inf')
        self.solution_vertex_idx = None
        self.solution_goal_path = None

        direct = self._steer(self.start, self.goal)
        if direct is not None and self._check_collision(direct):
            self.solution_cost = direct['cost']
            self.solution_vertex_idx = 0
            self.solution_goal_path = direct
            self.start.parent = None
            path = [[self.start.x, self.start.y]]
            for x, y in zip(direct['x'][1:], direct['y'][1:]):
                path.append([x, y])
            return path

        self._sample_batch()
        self._build_queue()

        solution_stable = False

        for iteration in range(self.max_iter):
            if not self.queue:
                if solution_stable:
                    break
                if not self.samples:
                    self._sample_batch()
                    solution_stable = False
                elif len(self.samples) < self.batch_size // 2:
                    self._sample_batch()
                    solution_stable = False
                self._build_queue()
                if not self.queue:
                    if self.solution_vertex_idx is not None:
                        solution_stable = True
                    continue

            while self.queue:
                f_hat, v_idx, s_id = heapq.heappop(self.queue)

                if f_hat >= self.solution_cost:
                    self.queue = []
                    solution_stable = True
                    break

                actual_idx = self.sample_to_vertex.get(s_id, None)

                if actual_idx is not None:
                    if s_id in self.samples:
                        continue
                    v = self.vertices[v_idx]
                    s = self.vertices[actual_idx]

                    if v.cost + self.edge_heuristic(v, s) >= s.cost:
                        continue

                    path = self._steer(v, s)
                    if path is not None and self._check_collision(path):
                        new_cost = v.cost + path['cost']
                        if new_cost < s.cost:
                            s.parent = v
                            s.cost = new_cost
                            s.path_x = path['x']
                            s.path_y = path['y']
                            s.path_yaw = path['yaw']
                    continue

                if s_id not in self.samples:
                    continue

                s = self.samples[s_id]
                v = self.vertices[v_idx]

                path = self._steer(v, s)
                if path is not None and self._check_collision(path):
                    new_node = copy.deepcopy(s)
                    new_node.cost = v.cost + path['cost']
                    new_node.parent = v
                    new_node.path_x = path['x']
                    new_node.path_y = path['y']
                    new_node.path_yaw = path['yaw']

                    new_idx = len(self.vertices)
                    self.vertices.append(new_node)
                    self.sample_to_vertex[s_id] = new_idx
                    del self.samples[s_id]

                    self._rewire(new_idx)

                    reachable, goal_path, total = self._check_goal(new_idx)
                    if reachable and total < self.solution_cost:
                        self.solution_cost = total
                        self.solution_vertex_idx = new_idx
                        self.solution_goal_path = goal_path

                if not self.queue:
                    break

            if not search_until_max_iter and self.solution_vertex_idx is not None:
                break

        if self.solution_vertex_idx is not None:
            return self.generate_final_course(self.solution_vertex_idx)
        return None

    @property
    def node_list(self):
        return self.vertices

    def search_best_goal_node(self):
        return self.solution_vertex_idx

    # ── Drawing ─────────────────────────────────────────

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in self.vertices:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g", lw=0.5, alpha=0.6)

        for s in self.samples.values():
            plt.plot(s.x, s.y, ".", color="gray", alpha=0.3, markersize=2)

        for ox, oy, size in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=100 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([self.min_rand, self.max_rand,
                  self.min_rand, self.max_rand])
        plt.grid(True)
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.goal.x, self.goal.y, self.goal.yaw)
        plt.axis("equal")
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.goal.x, self.goal.y, self.goal.yaw)

    # ── Curvature (zero for straight-line segments) ────

    def get_curvature_analytical(self):
        if self.solution_vertex_idx is None:
            return np.array([0.0]), np.array([0.0])

        node_path = []
        n = self.vertices[self.solution_vertex_idx]
        while n is not None:
            node_path.append(n)
            n = n.parent
        node_path.reverse()

        arc_positions = [0.0]
        total_len = 0.0

        for n in node_path[1:]:
            seg_len = hypot(n.x - n.parent.x, n.y - n.parent.y) if n.parent else 0.0
            if seg_len < 1e-8:
                continue
            npts = max(2, int(seg_len / self.step_size))
            seg_step = seg_len / npts
            for _ in range(npts):
                total_len += seg_step
                arc_positions.append(total_len)

        node = node_path[-1]
        d_g = self.euclidean_dist(node, self.goal)
        if d_g > self.goal_xy_th and self.solution_goal_path is not None:
            gp = self.solution_goal_path
            seg_len = gp['cost']
            npts = max(2, int(seg_len / self.step_size))
            seg_step = seg_len / npts
            for _ in range(npts):
                total_len += seg_step
                arc_positions.append(total_len)

        s = np.array(arc_positions)
        k = np.zeros_like(s)
        return k, s
