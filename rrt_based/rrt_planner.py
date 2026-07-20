import math
import random
import copy
from math import sin, cos, atan2, sqrt, acos, pi, hypot

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from scenario_config import ScenarioConfig


# ═══════════════════════════════════════════════════════════════
# 1. ANGLE UTILITIES
# ═══════════════════════════════════════════════════════════════

def rot_mat_2d(angle):
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


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


# ═══════════════════════════════════════════════════════════════
# 2. DUBINS PATH PLANNER
# ═══════════════════════════════════════════════════════════════

def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                     step_size=0.1, selected_types=None):
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw
    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size,
        planning_funcs)
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)
    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "R"]
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "L"]
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL}


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                      step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature
    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)
    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None
    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue
        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                       curvature, step_size)
    lengths = [length / curvature for length in lengths]
    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)
        if mode == "L":
            path_yaw.append(origin_yaw + length)
        elif mode == "R":
            path_yaw.append(origin_yaw - length)
    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]
    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]
        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size
        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)
    return p_x, p_y, p_yaw


# ═══════════════════════════════════════════════════════════════
# 3. PLOT UTILITIES
# ═══════════════════════════════════════════════════════════════

def plot_arrow(x, y, yaw, arrow_length=1.0,
               origin_point_plot_style="xr",
               head_width=0.1, fc="r", ec="k", **kwargs):
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, head_width=head_width,
                       fc=fc, ec=ec, **kwargs)
    else:
        plt.arrow(x, y,
                  arrow_length * math.cos(yaw),
                  arrow_length * math.sin(yaw),
                  head_width=head_width, fc=fc, ec=ec, **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


# ═══════════════════════════════════════════════════════════════
# 4. RRT
# ═══════════════════════════════════════════════════════════════

class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=3.0, path_resolution=0.5,
                 goal_sample_rate=5, max_iter=500,
                 play_area=None, robot_radius=0.0):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)
            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)
            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)
            if animation and i % 5:
                self.draw_graph(rnd_node)
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

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        # plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None:
            return True
        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False
        return True

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):
        if node is None:
            return False
        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= (size + robot_radius) ** 2:
                return False
        return True

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


# ═══════════════════════════════════════════════════════════════
# 5. RRT*
# ═══════════════════════════════════════════════════════════════

class RRTStar(RRT):
    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=0.5, path_resolution=0.1,
                 goal_sample_rate=20, max_iter=500,
                 connect_circle_dist=50.0, search_until_max_iter=False,
                 robot_radius=0.0):
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
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
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
        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
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
            if t_node and self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                cost = self.calc_new_cost(near_node, new_node)
                if cost < best_cost:
                    best_cost = cost
                    best_parent_index = i
        if best_parent_index == -1:
            return None
        new_node = self.steer(self.node_list[best_parent_index], new_node)
        new_node.cost = best_cost
        return new_node

    def rewire(self, new_node, near_indexes):
        for i in near_indexes:
            near_node = self.node_list[i]
            t_node = self.steer(new_node, near_node)
            if t_node and self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                cost = self.calc_new_cost(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.path_x = t_node.path_x
                    near_node.path_y = t_node.path_y
                    near_node.cost = cost

    def search_best_goal_node(self):

        candidates = []

        for i, node in enumerate(self.node_list):

            d = self.calc_dist_to_goal(node.x, node.y)

            if d > self.goal_xy_th:
                continue

            yaw_diff = abs(angle_mod(node.yaw - self.end.yaw))

            if yaw_diff > self.goal_yaw_th:
                continue

            px, py, _, _, course_lengths = plan_dubins_path(
                node.x,
                node.y,
                node.yaw,
                self.end.x,
                self.end.y,
                self.end.yaw,
                self.curvature,
                step_size=self.step_size
            )

            if len(px) <= 1:
                continue

            temp = RRT.Node(0, 0)
            temp.path_x = px
            temp.path_y = py

            if not self.check_collision(
                temp,
                self.obstacle_list,
                self.robot_radius
            ):
                continue

            total_cost = node.cost + sum(abs(c) for c in course_lengths)

            candidates.append((i, total_cost))

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1])[0]

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


# ═══════════════════════════════════════════════════════════════
# 6. RRT* DUBINS
# ═══════════════════════════════════════════════════════════════

class RRTStarDubins(RRTStar):
    class Node(RRTStar.Node):
        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 goal_sample_rate=10, max_iter=200,
                 connect_circle_dist=50.0, robot_radius=0.0,
                 step_size=0.1, random_yaw_strategy='uniform',
                 rand_area_x=None, rand_area_y=None):
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        if rand_area_x is not None:
            self.min_rand_x, self.max_rand_x = rand_area_x
            self.min_rand_y, self.max_rand_y = rand_area_y if rand_area_y is not None else rand_area_x
            self.min_rand = self.min_rand_x
            self.max_rand = self.max_rand_x
        else:
            self.min_rand = rand_area[0]
            self.max_rand = rand_area[1]
            self.min_rand_x = self.min_rand_y = self.min_rand
            self.max_rand_x = self.max_rand_y = self.max_rand
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist
        self.step_size = step_size
        self.robot_radius = robot_radius
        self.curvature = 1.0 / 0.73
        self.goal_yaw_th = np.deg2rad(30)
        self.goal_xy_th = self.robot_radius
        self.play_area = None
        self.random_yaw_strategy = random_yaw_strategy

    def planning(self, animation=True, search_until_max_iter=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            # FIX #1: honor `random_yaw_strategy`. Previously `rnd.yaw` (set
            # in get_random_node according to the chosen strategy) was
            # discarded and always overwritten with the nearest->rnd
            # direction, making the strategy parameter a no-op.
            if self.calc_dist_to_goal(rnd.x, rnd.y) < 1e-10:
                target_yaw = self.end.yaw
            elif self.random_yaw_strategy == 'toward_goal':
                target_yaw = rnd.yaw
            else:
                target_yaw = math.atan2(rnd.y - nearest_node.y, rnd.x - nearest_node.x)
            new_node = self.steer(nearest_node, rnd, target_yaw=target_yaw)
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)
            if animation and i % 5 == 0:
                self.draw_graph(rnd)
            if (not search_until_max_iter) and new_node:
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)
        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        return None

    def draw_graph(self, rnd=None):
        plt.clf()
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
        # FIX: removed the redundant call to plot_start_goal_arrow() that
        # used to happen right before draw_graph() in planning(). draw_graph
        # itself calls plt.clf() and then plot_start_goal_arrow() below, so
        # the outer call was always wiped out and had zero effect.
        self.plot_start_goal_arrow()
        plt.axis("equal")
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        plot_arrow(self.start.x, self.start.y, self.start.yaw)
        plot_arrow(self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node, target_yaw=None):
        if target_yaw is None:
            target_yaw = to_node.yaw
        px, py, pyaw, mode, course_lengths = plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, target_yaw, self.curvature,
            step_size=self.step_size)
        # FIX #2: guard against a degenerate/invalid Dubins path (e.g. when
        # from_node and to_node coincide). Without this check a path of
        # length 1 was silently accepted, letting duplicate/zero-cost nodes
        # into the tree. This mirrors the reference implementation's
        # `if len(px) <= 1: return None`.
        if len(px) <= 1:
            return None
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
        # FIX: pass step_size consistently instead of silently falling back
        # to plan_dubins_path's default (0.1). Doesn't change the resulting
        # cost value (which only depends on d1/d2/d3, not sampling density)
        # but keeps behavior consistent with the rest of the class.
        _, _, _, _, course_lengths = plan_dubins_path(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature,
            step_size=self.step_size)
        cost = sum([abs(c) for c in course_lengths])
        return from_node.cost + cost

    def rewire(self, new_node, near_indexes):
        # FIX: override RRTStar.rewire so that, when a neighbor is rewired
        # through new_node, its path_yaw/mode/course_lengths are updated
        # too (the inherited version only updated path_x/path_y, leaving
        # those attributes stale/inconsistent with the new parent).
        for i in near_indexes:
            near_node = self.node_list[i]
            t_node = self.steer(new_node, near_node)
            if t_node and self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                cost = self.calc_new_cost(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.path_x = t_node.path_x
                    near_node.path_y = t_node.path_y
                    near_node.path_yaw = t_node.path_yaw
                    near_node.mode = t_node.mode
                    near_node.course_lengths = t_node.course_lengths
                    near_node.cost = cost

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.uniform(self.min_rand_x, self.max_rand_x)
            y = random.uniform(self.min_rand_y, self.max_rand_y)
            if self.random_yaw_strategy == 'toward_goal':
                yaw = math.atan2(self.end.y - y, self.end.x - x)
            else:
                yaw = random.uniform(-np.pi, np.pi)
            rnd = self.Node(x, y, yaw)
        else:
            rnd = self.Node(self.end.x, self.end.y, self.end.yaw)
        return rnd

    def search_best_goal_node(self):
        candidates = []
        for (i, node) in enumerate(self.node_list):
            d = self.calc_dist_to_goal(node.x, node.y)
            if d > self.goal_xy_th:
                continue
            # FIX #3: actually use self.goal_yaw_th (previously configurable
            # via RRTPlanner.run(goal_yaw_th=...) but never read anywhere).
            # If the node's yaw is already within the goal yaw tolerance we
            # accept a zero-extra-cost connection; otherwise we fall back to
            # computing the exact Dubins connection cost to the goal pose.
            yaw_diff = abs(angle_mod(node.yaw - self.end.yaw))
            if yaw_diff <= self.goal_yaw_th:
                candidates.append((i, node.cost, 0.0))
                continue
            px, py, _, _, course_lengths = plan_dubins_path(
                node.x, node.y, node.yaw,
                self.end.x, self.end.y, self.end.yaw,
                self.curvature, step_size=self.step_size)
            if len(px) <= 1:
                continue
            temp = RRT.Node(0, 0)
            temp.path_x = px
            temp.path_y = py
            if not self.check_collision(temp, self.obstacle_list, self.robot_radius):
                continue
            dubins_cost = sum(abs(c) for c in course_lengths)
            candidates.append((i, node.cost, dubins_cost))
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda x: x[1] + x[2])[0]
        return best_idx

    def generate_final_course(self, goal_index):
        node = self.node_list[goal_index]

        node_path = []
        n = node
        while n.parent is not None:
            node_path.append(n)
            n = n.parent
        node_path.append(n)
        node_path.reverse()

        path = [[node_path[0].x, node_path[0].y]]
        for n in node_path[1:]:
            if len(n.path_x) > 0:
                path.extend([x, y] for x, y in zip(n.path_x[1:], n.path_y[1:]))
            else:
                path.append([n.x, n.y])

        d_g = self.calc_dist_to_goal(node.x, node.y)
        yaw_err = abs(angle_mod(node.yaw - self.end.yaw))
        if d_g >= self.goal_xy_th or yaw_err >= self.goal_yaw_th:
            px, py, _, _, _ = plan_dubins_path(
                node.x, node.y, node.yaw,
                self.end.x, self.end.y, self.end.yaw,
                self.curvature, step_size=self.step_size)
            if len(px) > 1:
                path.extend([x, y] for x, y in zip(px[1:], py[1:]))

        return path

    def get_curvature_analytical(self):
        best_idx = self.search_best_goal_node()
        if best_idx is None:
            return np.array([0.0]), np.array([0.0])
        node_path = []
        n = self.node_list[best_idx]
        while n.parent is not None:
            node_path.append(n)
            n = n.parent
        node_path.append(n)
        node_path.reverse()

        curvatures = []
        arc_positions = [0.0]

        for n in node_path[1:]:
            mode = getattr(n, 'mode', [])
            lengths = getattr(n, 'course_lengths', [])
            if not mode or not lengths:
                continue
            for m, l in zip(mode, lengths):
                k = self.curvature if m in ('L', 'R') else 0.0
                npts = max(1, int(abs(l) / self.step_size))
                seg_step = abs(l) / npts
                curvatures.extend([k] * npts)
                for _ in range(npts):
                    arc_positions.append(arc_positions[-1] + seg_step)

        node = node_path[-1]
        d_g = self.calc_dist_to_goal(node.x, node.y)
        if d_g >= self.goal_xy_th:
            _, _, _, mode_f, lengths_f = plan_dubins_path(
                node.x, node.y, node.yaw,
                self.end.x, self.end.y, self.end.yaw,
                self.curvature, step_size=self.step_size)
            for m, l in zip(mode_f, lengths_f):
                k = self.curvature if m in ('L', 'R') else 0.0
                npts = max(1, int(abs(l) / self.step_size))
                seg_step = abs(l) / npts
                curvatures.extend([k] * npts)
                for _ in range(npts):
                    arc_positions.append(arc_positions[-1] + seg_step)

        s = np.array(arc_positions)
        k = np.array(curvatures)
        if len(k) < len(s):
            k = np.append(k, k[-1] if len(k) > 0 else 0.0)
        elif len(s) < len(k):
            s = np.append(s, s[-1])
        return k[:len(s)], s[:len(k)]


# ═══════════════════════════════════════════════════════════════
# 7. RRT PLANNER WRAPPER
# ═══════════════════════════════════════════════════════════════

class RRTPlanner:
    def __init__(self, config=None, planner_type='rrt_star', **kwargs):
        self.config = config if config else ScenarioConfig()
        self.planner_type = planner_type
        self.planner = None
        self._path = None
        self._nodes_path = None
        self._best_goal_node = None
        self._planner_kwargs = kwargs

    def setup(self):
        self.config.setup()

    def run(self, animation=False, **kwargs):
        self.config.setup()
        c = self.config

        base = dict(
            goal_sample_rate=20,
            max_iter=2000,
            connect_circle_dist=2.0,
            search_until_max_iter=True,
            robot_radius=c.radius,
        )

        if self.planner_type == 'rrt_star_dubins':
            dubins_curvature = c.kappa_max
            base.update(dict(
                max_iter=1000,
                connect_circle_dist=4.5,
                dubins_curvature=dubins_curvature,
            ))

        base.update(self._planner_kwargs)
        base.update(kwargs)
        p = base

        obstacle_list = list(c.obs)

        if self.planner_type == 'rrt_star_dubins':
            start = [float(c.start[0]), float(c.start[1]), float(c.th_start)]
            goal = [float(c.goal[0]), float(c.goal[1]), float(c.th_goal)]
            path_resolution = p.get('path_resolution', 0.05)
            self.planner = RRTStarDubins(
                start=start, goal=goal,
                obstacle_list=obstacle_list,
                rand_area=[c.xmin, c.xmax],
                rand_area_x=[c.xmin, c.xmax],
                rand_area_y=[c.ymin, c.ymax],
                goal_sample_rate=p['goal_sample_rate'],
                max_iter=p['max_iter'],
                connect_circle_dist=p['connect_circle_dist'],
                robot_radius=p['robot_radius'],
                step_size=path_resolution,
                random_yaw_strategy=p.get('random_yaw_strategy', 'uniform'),
            )
            self.planner.curvature = p['dubins_curvature']
            self.planner.goal_sample_rate = p['goal_sample_rate']
            self.planner.goal_xy_th = p.get('goal_xy_th', 0.2)
            self.planner.goal_yaw_th = p.get('goal_yaw_th', np.deg2rad(60))
        else:
            start = [float(c.start[0]), float(c.start[1])]
            goal = [float(c.goal[0]), float(c.goal[1])]
            expand_dis = p.get('expand_dis', 0.3)
            path_resolution = p.get('path_resolution', 0.05)
            self.planner = RRTStar(
                start=start, goal=goal,
                obstacle_list=obstacle_list,
                rand_area=[c.xmin, c.xmax],
                expand_dis=expand_dis,
                path_resolution=path_resolution,
                goal_sample_rate=p['goal_sample_rate'],
                max_iter=p['max_iter'],
                connect_circle_dist=p['connect_circle_dist'],
                search_until_max_iter=p['search_until_max_iter'],
                robot_radius=p['robot_radius'],
            )

        path = self.planner.planning(animation=animation)
        self._path = path
        self._extract_best_nodes()
        return self.planner

    def _extract_best_nodes(self):
        if self.planner is None or self._path is None:
            return
        best_goal_node = None
        if hasattr(self.planner, 'search_best_goal_node'):
            best_index = self.planner.search_best_goal_node()
            if best_index is not None:
                best_goal_node = self.planner.node_list[best_index]
        if best_goal_node is None:
            goal = self.config.goal
            min_dist = float('inf')
            for n in self.planner.node_list:
                dist = np.hypot(n.x - goal[0], n.y - goal[1])
                if dist < min_dist:
                    min_dist = dist
                    best_goal_node = n
        self._best_goal_node = best_goal_node
        if best_goal_node is not None:
            nodes_path = []
            node = best_goal_node
            while node is not None:
                nodes_path.append(node)
                node = node.parent
            nodes_path.reverse()
            self._nodes_path = nodes_path
        else:
            self._nodes_path = []

    def get_best_path(self):
        return self._path

    def draw_scenario(self, ax=None, show=True):
        c = self.config
        c.setup()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        for ob in c.obs:
            ax.add_patch(plt.Circle(
                (ob[0], ob[1]), radius=ob[2], color="green", alpha=0.3))
        for ob in c.expanded_obs:
            ax.add_patch(plt.Circle(
                (ob[0], ob[1]), radius=ob[2], color="tan", alpha=0.3))
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

    @staticmethod
    def _curvature_and_arc(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        k = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-10)
        diff = np.diff(path, axis=0)
        seg = np.linalg.norm(diff, axis=1)
        s = np.zeros(len(path))
        s[1:] = np.cumsum(seg)
        return k, s

    def smooth_path(self, n_points=500, s=0.015, k=3, n_waypoints=20):
        if self._path is None:
            return None
        from scipy.interpolate import splprep, splev
        full = np.array(self._path)
        step = max(1, len(full) // n_waypoints)
        wx = list(full[::step, 0])
        wy = list(full[::step, 1])
        if wx[-1] != full[-1, 0]:
            wx.append(full[-1, 0])
            wy.append(full[-1, 1])
        w = np.ones(len(wx))
        w[0] = 1000.0
        w[-1] = 1000.0
        tck, _ = splprep([wx, wy], w=w, s=s, k=min(k, len(wx) - 1))
        u = np.linspace(0, 1, n_points)
        smooth = np.column_stack(splev(u, tck))
        return smooth

    def plot_result(self, show=True, smoothed=False, **plot_kwargs):
        # NOTE: the original file defined `plot_result` twice; the second
        # definition (kept here, with the `smoothed` option) silently
        # shadowed the first one. Removed the duplicate to avoid confusion.
        c = self.config
        c.setup()
        fig, ax = plt.subplots(figsize=(10.8, 6))
        self.draw_scenario(ax=ax, show=False)
        if self._path is not None:
            raw = np.array(self._path)
            ax.plot(raw[:, 0], raw[:, 1], 'b-', linewidth=1.5,
                    label=f'{self.planner_type} (raw)', alpha=0.5, **plot_kwargs)
        if smoothed:
            smooth = self.smooth_path()
            if smooth is not None:
                ax.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2,
                        label=f'{self.planner_type} (smoothed)', **plot_kwargs)
        ax.set_title(f'{self.planner_type} - Result')
        if show:
            plt.show()
        return ax

    def plot_curvature(self, nsampling=None, ax=None, figsize=(8, 4),
                       smoothed=False):
        if self._path is None:
            return None

        show = ax is None
        if show:
            fig, ax = plt.subplots(figsize=figsize)

        kappa_max = getattr(self.config, 'kappa_max', 8.0)

        # Use analytical curvature when available (Dubins segments),
        # otherwise fall back to numerical differentiation (RRT/RRTStar)
        if hasattr(self.planner, 'get_curvature_analytical'):
            k_raw, s_raw = self.planner.get_curvature_analytical()
        else:
            raw = np.array(self._path)
            k_raw, s_raw = self._curvature_and_arc(raw)
        ax.plot(s_raw, k_raw, 'b-', linewidth=1.5, alpha=0.5,
                label=f'{self.planner_type} (raw)')

        # smoothed path curvature
        if smoothed:
            smooth = self.smooth_path()
            if smooth is not None:
                k_sm, s_sm = self._curvature_and_arc(smooth)
                ax.plot(s_sm, k_sm, 'r-', linewidth=2,
                        label=f'{self.planner_type} (smoothed)')

        ax.axhline(y=kappa_max, color='r', linestyle='--', linewidth=1.5,
                   label=rf'$\kappa_{{\mathrm{{max}}}}$ = {kappa_max}')
        ax.set_xlabel('Arc length', fontsize=12)
        ax.set_ylabel('Curvature', fontsize=12)
        ax.set_title(f'{self.planner_type} - Curvature', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        # FIX: `ax.set_axis(...)` doesn't exist in matplotlib (AttributeError)
        # and the hardcoded [-3,3,-3,3] range didn't match arc-length/curvature
        # data anyway. Removed; the plot now auto-scales to the data.
        fig = ax.figure
        if show:
            fig.tight_layout()
            plt.show()
        return fig, ax, k_raw

    def get_dubins_node_path(self):
        return self._nodes_path