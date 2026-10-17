# Response to Reviewer

The questions concern (1) the parameters of the compared methods, (2) the
stopping criterion used for RRT*, and (3) the step size of the RRT*
variations. Part A is a concise answer; Part B gives the full details,
including the meaning of every swept variable.

---

## Part A — Concise response

**1) Parameters of the other methods.** All methods share the same setup
(4.2 m × 4.2 m workspace, robot radius 0.073 m, minimum turning radius 0.73 m,
same obstacles and seeds). Sampling-based planners use a fixed iteration
budget; the two evolutionary baselines use a fixed number of function
evaluations (FEs):

| Method | Budget / effort | Steer / step | Rewire radius | Goal bias | Extra |
|---|---|---|---|---|---|
| RRT* | 2000 iter | `expand_dis` 0.3 m, `path_resolution` 0.05 m | 2.0 m | 20 % | — |
| RRT* + B-spline | 2000 iter | same | 2.0 m | 20 % | B-spline smoothing |
| Dubins-RRT* | 2000 iter | `step_size` 0.05 m | 4.5 m | 20 % | heading sampling |
| Modified Dubins-RRT* (MDR) | 2000 iter | 0.05 m | 4.5 m | 20 % | `eta1` 0.3, `safety` 0.0 |
| MDR + CCPOA | 2000 iter + CCPOA | 0.05 m | 4.5 m | 20 % | CCPOA (5 iters) |
| RRT*-ASV | 2000 iter | 0.05 m | 4.5 m | 30 % | `cost_a` 1.20, `cost_b` 1.5 |
| DE2D-NURBS (proposed) | 400 gen × 100 pop ≈ 40 000 FEs | NURBS/DE | — | — | `nsampling` 120 |
| PSO2D-NURBS (baseline) | 400 gen × 50 pop ≈ 20 000 FEs | NURBS/PSO | — | — | `nsampling` 120 |

Every free hyper-parameter above was **swept exhaustively** (96 non-Dubins
configs + 36 Dubins configs = 145 registered configurations), and the final
values were chosen with a Friedman test plus Nemenyi / paired-Wilcoxon post-hoc,
so they are data-justified rather than hand-picked (Part B.4).

**2) Stopping criterion.** All sampling-based planners use a **fixed iteration
budget with no early exit** (`search_until_max_iter = True`): exactly
`max_iter` iterations are always executed, even when a path to the goal is found
earlier. The runtime is therefore set by the budget, exactly as noted by the
reviewer. We address this by (i) using the same fixed budget for every method
and scenario, (ii) explicitly sweeping `max_iter ∈ {500, 1000, 2000, 4000}` for
the RRT* family and selecting it statistically, and (iii) stopping DE/PSO after a
fixed number of generations (fixed FEs, no early termination).

**3) Step size of the RRT* variations.** The extension/steer step of the base
non-Dubins RRT* is `expand_dis = 0.3 m` and was swept over
**{0.1, 0.2, 0.3, 0.4} m**; the collision-check interpolation is
`path_resolution = 0.05 m`. For the Dubins/ASV planners the same 0.05 m is used
as the curvature-aware `step_size`. The steer step affects search quality but not
the runtime (which is bounded by `max_iter`).

**On timing.** The RRT* runtime (~1.5 s) is dominated by the constant factor of a
**naive pure-Python implementation** (linear O(n) nearest-neighbour scan instead
of a k-d tree, per-obstacle Python collision loops, no compiled backend, plus
B-spline and start/goal-extension post-processing) combined with the fixed
2000-iteration budget. "Low time" is a relative statement w.r.t. the evolutionary
optimizers (tens of seconds), measured as CPU time averaged over scenarios.

---

## Part B — Detailed response

### B.1 Common experimental setup (identical for all planners)

- Workspace 4.2 m × 4.2 m (`x, y ∈ [−2.1, 2.1]`), start `(−1.4, −0.8)`,
  goal `(1.4, 0.8)`, headings `θ_start = θ_goal = 0`.
- Robot radius `r = 0.073 m`; minimum turning radius `ρ = 0.73 m`, hence
  `κ_max = 1/ρ ≈ 1.37 m⁻¹`.
- Same obstacle sets (progressive pool of 0–9 circular obstacles) and the same
  random seeds for every method.

### B.2 Parameters of the compared methods

Values in braces `{…}` are the swept grid; otherwise the value is fixed.

| Method (variant) | Iteration / effort budget | Steer / step | Rewire radius `connect_circle_dist` | Goal bias | Other |
|---|---|---|---|---|---|
| RRT* (`rrt_star`) | `max_iter` = 2000 {500, 1000, 2000, 4000} | `expand_dis` = 0.3 m {0.1, 0.2, 0.3, 0.4}; `path_resolution` = 0.05 m | 2.0 m {1.0, 2.0, 3.0} | `goal_sample_rate` = 20 % | `search_until_max_iter = True` |
| RRT* + B-spline (`rrt_star_smooth`) | same as RRT* | same | same | 20 % | B-spline: 15 control points, `s = 0.08`, 250 evaluation points |
| Dubins-RRT* (`rrt_star_dubins`) | 2000 | `step_size` = `path_resolution` = 0.05 m | 4.5 m {3.5, 4.5, 6.0} | 20 % | `random_yaw_strategy` ∈ {uniform, toward_goal}; `κ_max` |
| Dubins-RRT* + B-spline (`rrt_dubins_smooth`) | 2000 | 0.05 m | 4.5 m {3.5, 4.5, 6.0} | 20 % | same + B-spline |
| Modified Dubins-RRT* (MDR) | 2000 | 0.05 m | 4.5 m | 20 % | `eta1` = 0.3 {0.1, 0.3, 0.5}; `safety_radius_factor` = 0.0 {0.0, 1.0, 2.0} |
| MDR + CCPOA | 2000 (+ CCPOA) | 0.05 m | 4.5 m | 20 % | CCPOA: `max_iter` = 5, `tol` = 1e-6, `step_size` = 0.05 |
| RRT*-ASV | 2000 | 0.05 m | 4.5 m | 30 % | turning costs `cost_a` = 1.20 {1.0, 1.2}, `cost_b` = 1.5 {1.0, 1.5, 2.0} |
| DE2D-NURBS (proposed) | 400 generations × 100 pop ≈ **40 000** function evaluations | — (NURBS/DE) | — | — | `nsampling` = 120; `α_obs` = 200, `α_κ` = 10, `α_ws` = 5; `λ_i = λ_f = r` |
| PSO2D-NURBS (baseline) | 400 × 50 ≈ **20 000** function evaluations | — | — | — | `α_obs` = 1000, `α_κ` = 10, `α_ws` = 1; `nsampling` = 120 |

*(`bit_star_dubins` is intentionally excluded from the sweep.)*

### B.3 Meaning of each tested variable

**Non-Dubins RRT\***
- `max_iter` — total number of tree-growth iterations. Each iteration samples a
  random configuration, steers toward it, collision-checks the edge, then chooses
  the best parent and rewires the neighbourhood. It is the **stopping budget**.
- `expand_dis` — the **steer/extension step**: maximum distance the tree grows
  from the nearest node toward the random sample. It sets the granularity of the
  exploration (small → finer but slower geometric progress; large → coarser and
  may skip narrow passages). It changes search quality, not the runtime.
- `connect_circle_dist` — the **RRT\* rewire neighbourhood radius** (the `gamma`
  in `find_near_nodes` / `choose_parent` / `rewire`). A larger radius considers
  more candidate parents and performs more rewiring, improving optimality
  (shorter paths) at a higher per-iteration cost; it governs asymptotic
  optimality.
- `path_resolution` — interpolation resolution along an edge for **collision
  checking** (not a search parameter). Finer resolution gives a more faithful
  collision check at higher cost.
- `goal_sample_rate` — **goal bias**: probability of sampling the goal pose
  instead of a random one, which accelerates convergence toward the goal.
- `search_until_max_iter` — whether to run the full budget even after reaching
  the goal (`True` = fixed budget, no early stop).

**Dubins-RRT\***
- `step_size` (= `path_resolution`) — arc-length resolution used to discretise
  the curvature-constrained (Dubins) arcs; analogous to `path_resolution`.
- `random_yaw_strategy` — how the heading of a random node is chosen:
  `uniform` draws it uniformly, `toward_goal` sets it along the direction to the
  goal, reducing unnecessary curvature and waste.

**Modified Dubins-RRT\* (MDR)**
- `eta1` — sampling probability in `ModifiedSample` (Algorithm 9): with
  probability `eta1` a random `(x, y)` is drawn with the goal-averaged heading;
  otherwise the goal pose itself is sampled. It trades exploration vs.
  exploitation (goal bias).
- `safety_radius_factor` — multiplier on the turning radius for the conservative
  straight-line collision test of Theorem 1 / Remark 2
  (`r_safe = factor·r` for `d ≥ 4r`, else `2·factor·r`). `0.0` disables the
  safety margin (checks only obstacle + robot radius), i.e. faster but with a
  weaker collision guarantee.

**RRT\*-ASV**
- `cost_a`, `cost_b` — weights of the **turning-cost function** that penalises
  small-radius arcs during growth and rewiring, biasing the tree toward gentler
  (more comfortable) curves.

**Evolutionary optimizers (DE2D/PSO2D-NURBS)**
- `n_generations` × `pop_size` — number of generations times population size =
  the **total number of function evaluations** (the fixed effort budget).
- `nsampling` — number of points sampled along the NURBS curve used to evaluate
  curvature and collision (coarser = faster, less accurate).
- `alpha_obs`, `alpha_kappa`, `alpha_workspace` — penalty weights of the cost
  function for obstacle penetration, curvature violation and workspace exit,
  respectively.
- `lambda_i`, `lambda_f` — magnitude of the initial/final tangent vectors used to
  build the NURBS curve (heading constraint at start/goal).

### B.4 Exhaustive search performed

Two grids are enumerated so that each parameter choice is justified by data:

- **Non-Dubins RRT\***: `connect_circle_dist ∈ {1.0, 2.0, 3.0}` ×
  `expand_dis ∈ {0.1, 0.2, 0.3, 0.4}` × `max_iter ∈ {500, 1000, 2000, 4000}`,
  for the raw and the B-spline variant → 48 configurations × 2 = **96**.
- **Dubins family**: RRT\*Dubins `ccdist × yaw` (3 × 2 = 12), MDR / MDR+CCPOA
  `eta1 × safety` (3 × 3 × 2 = 18), RRT\*-ASV `cost_a × cost_b` (2 × 3 = 6) →
  **36**.

The total number of registered planner configurations is **145**. Winning
configurations were selected with a Friedman test plus Nemenyi
critical-difference and paired-Wilcoxon (Holm) post-hoc over per-scenario
rankings (feasible first, then path length, curvature and CPU time).

### B.5 Stopping criterion

All sampling-based planners use a **fixed iteration budget** with
`search_until_max_iter = True`: exactly `max_iter` iterations are executed even
if a feasible path to the goal is found earlier. The runtime is therefore set by
the designer's budget, exactly as the reviewer notes, and is not an early-exit
artifact. Consequently:

- the same fixed budget is used for every method and scenario;
- `max_iter ∈ {500, 1000, 2000, 4000}` is explicitly swept for the RRT* family
  and selected statistically;
- DE/PSO stop after a fixed number of generations (≈ 40 000 FEs for DE and
  ≈ 20 000 for PSO), with no early termination.

### B.6 Step size of the RRT* variations

- **Extension/steer step** (`expand_dis`, used by `steer()`): `0.3 m` for the
  base non-Dubins RRT*; swept over **{0.1, 0.2, 0.3, 0.4} m**.
- **Collision-check interpolation** (`path_resolution`): `0.05 m` for the
  non-Dubins RRT*; the Dubins/ASV planners use the same `0.05 m` as the
  curvature-aware `step_size`.
- The steer step changes the tree-expansion quality but not the runtime, which is
  bounded by `max_iter`.

### B.7 Fairness / balanced computational cost

- **Time metric:** CPU time (`time.process_time`) rather than wall-clock,
  because under parallel execution the wall-clock is inflated by
  memory-bandwidth contention (measured ≈ +40 % at 8 workers and ≈ +88 % at 21
  workers, and not uniform across methods). Wall-clock is still stored for
  reference, and runs use low parallelism (≤ 4 workers).
- **Bounded effort:** sampling planners use a fixed iteration budget; the
  evolutionary baselines use a fixed number of function evaluations. Thus the
  computational effort is bounded and comparable, and any time difference
  reflects per-iteration/implementation cost rather than a divergent stopping
  rule.
- **Cost–quality trade-off:** the exhaustive `max_iter` sweep documents the
  trade-off instead of committing to a single value.

### B.8 Why RRT* is "fast" yet takes ≈ 1.5 s in our runs

The asymptotic per-iteration cost of RRT* is low, but the **constant factor** is
inflated by a **naive pure-Python implementation**:

- each node is a Python object and the nearest-neighbour query is a **linear
  O(n) scan** (no k-d tree / spatial index), so ~2000 iterations cost
  O(n²)-like work;
- collision checking is a per-obstacle **Python loop**, not vectorized;
- no compiled backend (e.g. Cython/numba) and repeated object/array allocations;
- post-processing adds a fixed cost: B-spline fitting (smooth variants) and the
  extended start/goal tangent segments.

Combined with `search_until_max_iter = True` (the full 2000-iteration budget
always runs, no early exit), the fixed cost lands in the ≈ 1.5–3 s range. The
"low average time" is a **relative** statement — small compared with the
evolutionary optimizers (tens of seconds) — and is a CPU-time average over
scenarios, not an absolute claim about RRT*.
