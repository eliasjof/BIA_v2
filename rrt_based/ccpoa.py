import numpy as np
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
from rrt_planner_modified import plan_dubins_path


class CCPOA:
    """Curvature-Constrained Path Optimization Algorithm (CCPOA).

    Keeps waypoint positions fixed and optimises heading angles at
    intermediate waypoints by iteratively solving the Three-Point
    Dubins Problem (3PDP) for each waypoint.

    Based on Section 3.2 of Wang et al. (2026).
    """

    def __init__(self, curvature, max_iter=100, tol=1e-6, step_size=0.05):
        self.curvature = curvature
        self.r = 1.0 / curvature if curvature > 0 else float('inf')
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, waypoints, theta_start, theta_goal):
        """Optimise heading angles for a sequence of waypoints.

        Parameters
        ----------
        waypoints : ndarray, shape (N, 2)
            Ordered waypoint positions from the MDR planner.
        theta_start : float
            Prescribed initial heading (rad).
        theta_goal : float
            Prescribed final heading (rad).

        Returns
        -------
        optimized_path : ndarray, shape (M, 2)
            Concatenated Dubins path after heading optimisation.
        theta : ndarray, shape (N,)
            Optimised heading at each waypoint.
        """
        theta = self.initial_guess(waypoints, theta_start, theta_goal)
        old_length = float('inf')

        for _ in range(self.max_iter):
            theta_old = theta.copy()

            for i in range(1, len(waypoints) - 1):
                theta[i] = self.solve_3pdp(
                    waypoints[i - 1], waypoints[i], waypoints[i + 1],
                    theta[i - 1], theta[i + 1],
                )

            path = self.build_dubins_path(waypoints, theta)
            new_length = self.path_length(path)

            if abs(old_length - new_length) < self.tol:
                break

            if new_length > old_length:
                theta = theta_old
                break

            old_length = new_length

        optimized_path = self.build_dubins_path(waypoints, theta)
        return optimized_path, theta

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------

    def initial_guess(self, waypoints, theta_start, theta_goal):
        """Initialise intermediate headings using the average direction
        from the previous to the next waypoint."""
        n = len(waypoints)
        theta = np.zeros(n)
        theta[0] = theta_start
        theta[-1] = theta_goal
        for i in range(1, n - 1):
            dx = waypoints[i + 1, 0] - waypoints[i - 1, 0]
            dy = waypoints[i + 1, 1] - waypoints[i - 1, 1]
            theta[i] = np.arctan2(dy, dx)
        return theta

    # ------------------------------------------------------------------
    # Three-Point Dubins Problem  (3PDP)
    # ------------------------------------------------------------------

    def _dubins_length(self, x1, y1, t1, x2, y2, t2):
        """Return the total arc length of a Dubins path between two
        configurations."""
        _, _, _, _, course_lengths = plan_dubins_path(
            x1, y1, t1, x2, y2, t2,
            self.curvature, step_size=self.step_size)
        return sum(abs(c) for c in course_lengths)

    @staticmethod
    def _golden_section_search(f, a, b, tol=1e-7):
        phi = (1 + 5 ** 0.5) / 2
        inv_phi = 1 / phi
        c = b - (b - a) * inv_phi
        d = a + (b - a) * inv_phi
        fc, fd = f(c), f(d)
        while abs(b - a) > tol:
            if fc < fd:
                b, d = d, c
                fd = fc
                c = b - (b - a) * inv_phi
                fc = f(c)
            else:
                a, c = c, d
                fc = fd
                d = a + (b - a) * inv_phi
                fd = f(d)
        return (a + b) / 2

    def solve_3pdp(self, z_prev, z_mid, z_next, theta_prev, theta_next):
        """Solve the three-point Dubins problem for one intermediate waypoint.

        Returns the heading at *z_mid* that minimises the sum of the two
        Dubins segments.
        """
        def total_length(tm):
            l1 = self._dubins_length(
                z_prev[0], z_prev[1], theta_prev,
                z_mid[0], z_mid[1], tm)
            l2 = self._dubins_length(
                z_mid[0], z_mid[1], tm,
                z_next[0], z_next[1], theta_next)
            return l1 + l2

        best_tm = self._golden_section_search(total_length, 0, 2 * np.pi)
        return float(best_tm % (2 * np.pi))

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def build_dubins_path(self, waypoints, theta):
        """Concatenate Dubins curves between consecutive configurations."""
        path = [[waypoints[0, 0], waypoints[0, 1]]]
        for i in range(len(waypoints) - 1):
            px, py, _, _, _ = plan_dubins_path(
                waypoints[i, 0], waypoints[i, 1], theta[i],
                waypoints[i + 1, 0], waypoints[i + 1, 1], theta[i + 1],
                self.curvature, step_size=self.step_size)
            if len(px) > 1:
                path.extend([x, y] for x, y in zip(px[1:], py[1:]))
            else:
                path.append([waypoints[i + 1, 0], waypoints[i + 1, 1]])
        return np.array(path)

    @staticmethod
    def path_length(path):
        if path is None or len(path) < 2:
            return 0.0
        return float(np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))))
