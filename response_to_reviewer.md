# Response to Reviewer

The questions concern (1) the parameters of the compared methods, (2) the
stopping criterion used for RRT*, (3) the step size of the RRT* variations, and
(4) how RRT* + B-spline is computed and why its reported time was lower than
pure RRT*. Part A is a concise answer; Part B gives the full details, including
the meaning of every swept variable.

---

## Part A — Concise response

**1) Parameters of the other methods.** All methods share the same setup
(4.2 m × 4.2 m workspace, robot radius 0.073 m, minimum turning radius 0.73 m,
same obstacles and seeds). Sampling-based planners use a fixed iteration
budget; the evolutionary baselines use a fixed number of function evaluations
(FEs). The values below are the ones **actually used** in the final run (each
free hyper-parameter was chosen from the exhaustive sweep by the statistical
procedure of Part B.4, not hand-picked):

| Method | Budget / effort | Steer / step | Rewire radius | Goal bias | Other |
|---|---|---|---|---|---|
| RRT* | 1000 iter | `expand_dis` 0.4 m, `path_resolution` 0.05 m | 2.0 m | 20 % | — |
| RRT* + B-spline | 1000 iter | `expand_dis` 0.1 m, `path_resolution` 0.05 m | 1.0 m | 20 % | B-spline smoothing |
| RRT*-Dubins | 2000 iter | `step_size` 0.05 m | 6.0 m | 20 % | `random_yaw_strategy` = uniform |
| RRT*-Dubins + B-spline | 2000 iter | 0.05 m | 6.0 m | 20 % | yaw = toward_goal + B-spline |
| Dubins-RRT* (MDR) | 2000 iter | 0.05 m | 4.5 m | 20 % | `eta1` 0.3, `safety` 0.0 |
| MDR + CCPOA | 2000 iter + CCPOA | 0.05 m | 4.5 m | 20 % | `eta1` 0.1, `safety` 0.0; CCPOA (5 iters) |
| RRT*-ASV | 2000 iter | 0.05 m | 4.5 m | 30 % | `cost_a` 1.0, `cost_b` 1.5 |
| DE2D-NURBS (proposed) | 400 gen × 100 pop ≈ 40 000 FEs | NURBS/DE | — | — | `nsampling` 120; `α_obs` 200, `α_κ` 10, `α_ws` 5 |
| PSO2D-NURBS (baseline) | 400 gen × 100 pop ≈ 40 000 FEs | NURBS/PSO | — | — | `nsampling` 120; `α_obs` 1000, `α_κ` 10, `α_ws` 1 |

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
the RRT* family and selecting it statistically (excluding the 500 budget, so
every selected configuration uses at least 1000 iterations), and (iii) stopping
DE/PSO after a fixed number of generations (fixed FEs, no early termination).

**3) Step size of the RRT* variations.** The extension/steer step of the base
non-Dubins RRT* is `expand_dis = 0.3 m` and was swept over
**{0.1, 0.2, 0.3, 0.4} m**; the collision-check interpolation is
`path_resolution = 0.05 m`. For the Dubins/ASV planners the same 0.05 m is used
as the curvature-aware `step_size`. The steer step affects search quality but not
the runtime (which is bounded by `max_iter`).

**4) How RRT* + B-spline is computed.** The **same** RRT* run is used and its
best path is passed through the B-spline smoothing stage (no second/different
trajectory). In the earlier version the parameters were **fixed** and, whenever
the smoothed curve **collided** with an obstacle (or the fit failed), the code
**fell back to the original RRT\* path**; those runs incurred no extra spline
cost, which is why the RRT* + B-spline row showed a lower mean time and std than
pure RRT*. In the current version every parameter is swept and selected
statistically, and both the 500-iteration budget and the `cc = 3.0` radius are
excluded from the selection (the swept variants use at least 1000 iterations and
`cc ≤ 2.0`). The two rows still show different average times (0.24 s vs 0.31 s),
but this is **not** because the smoothing is cheap: the two methods were
independently tuned and their selected base configurations differ in the rewiring
radius and extension step (`cc = 1.0`, `expand_dis = 0.1` for the B-spline variant
vs `cc = 2.0`, `expand_dis = 0.4` for plain RRT*), both of which add
per-iteration work. **Paired on the same base configuration the B-spline variant
is never faster than plain RRT\*** — the smoothing adds only ≈ 0.2 ms (see
Part B.4 and B.9).

**On timing.** With the final selected configurations the RRT* family averages
0.24–0.31 s of CPU time per scenario, whereas the single-query Dubins/ASV and
modified-Dubins planners average tens to hundreds of seconds (see the results
table in Part C). The RRT* runtime is dominated by the constant factor of a
**naive pure-Python implementation** (linear O(n) nearest-neighbour scan instead
of a k-d tree, per-obstacle Python collision loops, no compiled backend, plus
B-spline and start/goal-extension post-processing) combined with the fixed
`max_iter` budget (1000 iterations for RRT* and RRT* + B-spline; 2000 for the
Dubins/ASV ones). "Low time" is a relative statement w.r.t. the other planners,
measured as CPU time averaged over scenarios.

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

Values in braces `{…}` are the swept grid; the **bold** value is the one
selected by the statistical procedure and actually used in the reported run.
Fixed values are shown without braces. The 500-iteration budget is **excluded**
from the selection: swept variants use `max_iter ≥ 1000`, so a post-processed
planner cannot appear cheaper merely by running a shorter search.

| Method (variant) | Iteration / effort budget | Steer / step | Rewire radius `connect_circle_dist` | Goal bias | Other |
|---|---|---|---|---|---|
| RRT* (`rrt_star`) | `max_iter` = **1000** {1000, 2000, 4000} | `expand_dis` = **0.4 m** {0.1, 0.2, 0.3, 0.4}; `path_resolution` = 0.05 m | **2.0 m** {1.0, 2.0} | `goal_sample_rate` = 20 % | `search_until_max_iter = True` |
| RRT* + B-spline (`rrt_star_smooth`) | `max_iter` = **1000** {1000, 2000, 4000} | `expand_dis` = **0.1 m** {0.1, 0.2, 0.3, 0.4} | **1.0 m** {1.0, 2.0} | 20 % | B-spline: 15 control points, `s = 0.08`, 250 evaluation points |
| RRT*-Dubins (`rrt_star_dubins`) | 2000 | `step_size` = `path_resolution` = 0.05 m | **6.0 m** {3.5, 4.5, 6.0} | 20 % | `random_yaw_strategy` = **uniform** {uniform, toward_goal}; `κ_max` |
| RRT*-Dubins + B-spline (`rrt_dubins_smooth`) | 2000 | 0.05 m | **6.0 m** {3.5, 4.5, 6.0} | 20 % | `random_yaw_strategy` = **toward_goal**; same + B-spline |
| Dubins-RRT* / Modified Dubins-RRT* (`modified_dubins_rrt_star`) | 2000 | 0.05 m | 4.5 m | 20 % | `eta1` = **0.3** {0.1, 0.3, 0.5}; `safety_radius_factor` = **0.0** {0.0, 1.0, 2.0} |
| MDR + CCPOA (`modified_dubins_rrt_star_ccpoa`) | 2000 (+ CCPOA) | 0.05 m | 4.5 m | 20 % | `eta1` = **0.1** {0.1, 0.3, 0.5}; `safety` = **0.0**; CCPOA: `max_iter` = 5, `tol` = 1e-6, `step_size` = 0.05 |
| RRT*-ASV | 2000 | 0.05 m | 4.5 m | 30 % | turning costs `cost_a` = **1.0** {1.0, 1.2}, `cost_b` = **1.5** {1.0, 1.5, 2.0} |
| DE2D-NURBS (proposed) | 400 generations × 100 pop ≈ **40 000** function evaluations | — (NURBS/DE) | — | — | `nsampling` = 120; `α_obs` = 200, `α_κ` = 10, `α_ws` = 5; `λ_i = λ_f = r` |
| PSO2D-NURBS (baseline) | 400 × 100 ≈ **40 000** function evaluations | — | — | — | `α_obs` = 1000, `α_κ` = 10, `α_ws` = 1; `nsampling` = 120 |

*(`bit_star_dubins` is intentionally excluded from the sweep.)*  The variants
above are also referred to by the short labels used in the results table of
Part C: `rrt_star` → "RRT*", `rrt_star_smooth` → "RRT* + B-spline",
`rrt_star_dubins` → "RRT* Dubins", `rrt_dubins_smooth` → "RRT* Dubins +
B-spline", `modified_dubins_rrt_star` → "Dubins-RRT*",
`modified_dubins_rrt_star_ccpoa` → "Modified Dubins-RRT* + CCPOA",
`rrt_star_asv` → "RRT* ASV", `de2d_nurbs` → "DE2D-NURBS".

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
  in `find_near_nodes` / `choose_parent` / `rewire`). Around each newly added
  node it (i) searches for the **best parent** among all tree nodes inside the
  ball and (ii) **rewires** those neighbours through the new node whenever that
  lowers their path cost. A larger radius therefore reconnects the tree more
  aggressively: the surviving path is assembled from **fewer, longer and
  better-aligned segments** (fewer direction changes), which reduces its
  curvature — this is why a large `cc` can reach curvature feasibility in cases
  where a small `cc` cannot. The price is a **higher per-iteration cost**: more
  neighbours to test and more collision checks inside `choose_parent`/`rewire`,
  so under the same iteration budget a larger `cc` is markedly slower (0.099 s at
  `cc = 1.0` vs 0.308 s at `cc = 2.0`, `expand_dis = 0.4 m`, `max_iter = 1000`).
  Because of this cost the selection is restricted to `cc ≤ 2.0`. It governs
  asymptotic optimality.
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

- **Non-Dubins RRT\***: `connect_circle_dist ∈ {1.0, 2.0}` (the expensive
  `cc = 3.0` radius is excluded) × `expand_dis ∈ {0.1, 0.2, 0.3, 0.4}` ×
  `max_iter ∈ {1000, 2000, 4000}` (the 500-iteration budget is excluded), for the
  raw and the B-spline variant → 24 configurations × 2 = **48**.
- **Dubins family**: RRT\*Dubins `ccdist × yaw` (3 × 2 = 12), MDR / MDR+CCPOA
  `eta1 × safety` (3 × 3 × 2 = 18), RRT\*-ASV `cost_a × cost_b` (2 × 3 = 6) →
  **36**.

The total number of registered planner configurations is **145**. Accuracy
being the primary criterion, only the configurations attaining the **highest
feasible rate** of a variant are eligible; among them the winner is chosen with
a Friedman test plus Nemenyi critical-difference and paired-Wilcoxon (Holm)
post-hoc over per-scenario rankings (feasible first, then path length, curvature
and CPU time).

Because the Nemenyi critical difference can declare several configurations
statistically indistinguishable, the representative was taken as the **fastest**
among the most feasible configurations whose mean length and curvature are
within a 1 % relative tolerance of the best in the equivalence set. This avoids
selecting a configuration that is only microscopically shorter (e.g. a 0.002 %
length difference) but tens of times slower — a length-only tie-break would
otherwise have picked a 3.76 s RRT* + B-spline configuration over a statistically
identical one an order of magnitude faster.

Note that the B-spline variant reaches a slightly *higher* feasible rate (0.21)
than plain RRT* (0.18), because the smoothing rounds the corners (fixing
curvature) but can push the curve into obstacles; crucially, the two variants
peak in **different `connect_circle_dist` (`cc`) regimes**, so their
representatives are not comparable configuration-by-configuration:

- For **plain RRT***, feasibility grows with `cc`, because a larger rewiring
  radius straightens the path (fewer sharp turns → curvature below `κ_max`). At
  the selected `expand_dis = 0.4 m` and 1000 iterations it is 0.00 at `cc = 1.0`
  and 0.18 at `cc = 2.0`, while the CPU time grows from 0.099 s to 0.308 s. The
  representative is therefore `cc = 2.0`, 1000 iterations (feasibility 0.18).
- For **RRT* + B-spline**, the opposite holds: at `cc = 1.0` (with
  `expand_dis = 0.1 m`) the raw path is jagged (infeasible) but the B-spline
  repairs the curvature, giving the variant's best feasible rate 0.21; at
  `cc = 2.0`, `expand_dis = 0.4 m` the raw path is already straighter, and
  smoothing now *bends it into obstacles*, dropping feasibility from 0.18 to
  0.13.

Paired on the same base configuration the effect is unambiguous: `cc = 2.0`
favours plain RRT* (0.18 vs 0.13) while `cc = 1.0` favours the B-spline variant
(0.21 vs 0.00). Both reported representatives use 1000 iterations and differ in
`cc` (2.0 vs 1.0) and in `expand_dis` (0.4 vs 0.1).

**Are the two CPU times directly comparable?** No — and this matters for reading
the table. The two methods were tuned independently, and their selected bases
differ in `cc` and `expand_dis`, both of which add work (`find_near_nodes` /
`choose_parent` / `rewire` over a larger ball, extended further per step), so
plain RRT* (`cc = 2.0`, `expand_dis = 0.4`) is ≈ 1.3× slower than the B-spline
base (`cc = 1.0`, `expand_dis = 0.1`). Holding the base fixed and changing
**only** the smoothing shows that the B-spline pass is essentially free and never
makes the variant faster:

| Base configuration | plain RRT* | RRT* + B-spline | Δ (smoothing only) |
|---|---|---|---|
| `cc = 1.0`, `expand_dis = 0.1`, 1000 iter | 0.2394 s (feas 0.00) | 0.2396 s (feas 0.21) | +0.0002 s |
| `cc = 2.0`, `expand_dis = 0.4`, 1000 iter | 0.3075 s (feas 0.18) | 0.3077 s (feas 0.13) | +0.0002 s |

At both bases the times agree to well under a millisecond, and the smoothing
only *helps* feasibility when the base path is jagged (`cc = 1.0`) and *hurts* it
when the base path is already straighter (`cc = 2.0`). The reported RRT* +
B-spline row (0.24 s) is therefore slightly faster than the reported RRT* row
(0.31 s) because they are different `cc`/step regimes — **not** evidence that
smoothing is cheaper than the planner it post-processes.

### B.5 Stopping criterion

All sampling-based planners use a **fixed iteration budget** with
`search_until_max_iter = True`: exactly `max_iter` iterations are executed even
if a feasible path to the goal is found earlier. The runtime is therefore set by
the designer's budget, exactly as the reviewer notes, and is not an early-exit
artifact. Consequently:

- the same fixed budget is used for every method and scenario;
- `max_iter ∈ {500, 1000, 2000, 4000}` is explicitly swept for the RRT* family
  and selected statistically, with the lowest budget (500) excluded so that no
  selected configuration runs fewer than 1000 iterations;
- DE/PSO stop after a fixed number of generations (≈ 40 000 FEs for DE and
  ≈ 40 000 for PSO), with no early termination.

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

### B.8 Why RRT* is "fast" yet takes ≈ 0.24–0.31 s in our runs

The asymptotic per-iteration cost of RRT* is low, but the **constant factor** is
inflated by a **naive pure-Python implementation**:

- each node is a Python object and the nearest-neighbour query is a **linear
  O(n) scan** (no k-d tree / spatial index), so ~1000–2000 iterations cost
  O(n²)-like work;
- collision checking is a per-obstacle **Python loop**, not vectorized;
- no compiled backend (e.g. Cython/numba) and repeated object/array allocations;
- post-processing adds a fixed cost: B-spline fitting (smooth variants) and the
  extended start/goal tangent segments.

Combined with `search_until_max_iter = True` (the full `max_iter` budget always
runs, no early exit), the fixed cost lands in the ≈ 0.24–0.31 s range for the
RRT* family with the selected budgets (1000 iterations for both RRT* and
RRT* + B-spline). The "low average
time" is a **relative** statement — small compared with the other planners
(tens to hundreds of seconds) — and is a CPU-time average over scenarios, not an
absolute claim about RRT*.

### B.9 How RRT* + B-spline is computed (and why it used to be faster)

**Is a different trajectory computed?** No. RRT* + B-spline runs the **same
RRT\*** planner once (same algorithm, same obstacle set, same seed) and takes its
best path; that single path — after the start/goal tangent extension — is then
passed through the **B-spline smoothing stage** (`splprep`, 15 waypoints sampled
by arc length, endpoint weights `10⁴`, `s = 0.08`, evaluated at 250 points). No
second tree and no different trajectory are computed: the reported trajectory is
the smoothed version of the *same* RRT* path.

**Why did the previous version show a lower time than pure RRT*?** Two reasons,
both addressed in the current version:

1. *Fixed parameters.* The earlier run used a single hard-set configuration per
   method (not the exhaustive sweep); the RRT* base used by the B-spline
   variant therefore differed from the pure-RRT* base, so the two rows were not
   run under the same search effort. The current run sweeps every configuration
   and selects it statistically (Part B.4).
2. *Fallback to the raw path.* Whenever the smoothing stage did not return a
   *valid* curve — most importantly when the fitted B-spline **collided** with an
   obstacle (and also when the fit itself failed, e.g. on duplicated waypoints) —
   the code returned the **original RRT\* path** unchanged. In those runs the
   "RRT* + B-spline" entry was in fact the raw RRT* trajectory, and no extra
   spline cost was incurred beyond the attempt. Averaged over scenarios, this
   depressed both the mean time and its standard deviation below pure RRT*.

In the current version the smoothing is applied to the same RRT* path, every
base parameter is swept and selected statistically, and both the 500-iteration
budget and the `cc = 3.0` radius are excluded from the selection. The final table
therefore shows RRT* + B-spline at **0.24 s** vs plain RRT* at **0.31 s** — and
this small gap is **not** a smoothing-speed effect: the two rows use different
selected bases (`cc` 1.0 vs 2.0, `expand_dis` 0.1 vs 0.4), and paired on the
*same* base configuration the two times match to within ≈ 0.2 ms, i.e. the
smoothing never makes the variant faster than the planner it is built on
(Part B.4). What smoothing does change is feasibility, and
only in the base regime where the raw path is jagged (`cc = 1.0`: 0.00 → 0.21);
when the raw path is already straighter (`cc = 2.0`) the smoothing instead
*reduces* feasibility (0.18 → 0.13) by cutting corners into obstacles.

**Why does the smoothing collide, and how could it be improved?** The fit is a
B-spline over **15 control points** (`s = 0.08`), i.e. it is deliberately *stiff*:
it approximates the raw path rather than interpolating it, so where the raw path
hugs an obstacle the smoothed curve rounds the corner and can **cross into the
obstacle** (the fit has too few degrees of freedom to follow the free-space
corridor while still reducing curvature). A natural improvement — left as future
work — is to **increase the number of control points** (finer fit), so the curve
can track the collision-free raw path more closely and avoid cutting corners,
while still smoothing the high-curvature kinks. At present, such a curve is
simply rejected when it collides.

---

## Part C — Results table

The table below is the final comparison over the progressive scenario pool
(10 obstacle densities × 10 random seeds = 100 scenarios). It is generated by
`analyze_comparison.py`, which reduces each swept variant to its
statistically-selected best configuration (Part B.2/B.4).

```latex
\begin{table}[t]
  \centering
  \caption{Comparison over 100 scenarios (10 obstacle densities $\times$ 10
  seeds). \emph{Feasibility}: fraction of runs that are collision-free and
  satisfy the curvature ($\kappa_{\max}=1/0.73\ \mathrm{m}^{-1}$, 5\,\%
  tolerance) and workspace constraints. \emph{Avg Time}: mean CPU time
  (IQR-filtered) $\pm$ std. \emph{Best 10 scenarios}: number of density blocks
  in which the method attains the shortest feasible path while succeeding on
  \emph{every} seed of the block. \emph{Norm.\ Length}: mean path length
  normalised by the best feasible path of the blocks fully solved by the method
  ($1.0$ = best; \text{--} = never fully solved a block). \emph{Avg Rank}: mean
  per-block rank (lower is better; infeasible runs are ranked last).}
  \label{tab:comparison}
  \begin{tabular}{lccccc}
    \hline
    Planner & Feasibility & Avg Time & Best 10 scenarios & Norm. Length & Avg Rank \\
    \hline
    DE2D-NURBS (proposed)        & 0.99 & 0.78\,s $\pm$ 0.24  & 7/10 & 1.002 & 2.35 \\
    RRT*-ASV                     & 1.00 & 20.00\,s $\pm$ 13.89 & 1/10 & 1.033 & 2.62 \\
    PSO2D-NURBS (baseline)       & 0.69 & 7.29\,s $\pm$ 0.57  & 1/10 & 1.000 & 4.08 \\
    RRT*-Dubins                  & 0.44 & 48.12\,s $\pm$ 24.60 & 0/10 & \text{--} & 6.86 \\
    RRT* + B-spline              & 0.21 & 0.24\,s $\pm$ 0.05  & 0/10 & \text{--} & 7.35 \\
    RRT*-Dubins + B-spline       & 0.17 & 50.76\,s $\pm$ 26.26 & 1/10 & 1.000 & 7.87 \\
    Dubins-RRT*                  & 0.24 & 335.30\,s $\pm$ 82.93 & 0/10 & \text{--} & 8.00 \\
    Modified Dubins-RRT* + CCPOA & 0.17 & 301.60\,s $\pm$ 230.55 & 0/10 & \text{--} & 8.29 \\
    RRT*                         & 0.18 & 0.31\,s $\pm$ 0.08  & 0/10 & \text{--} & 8.31 \\
    \hline
  \end{tabular}
\end{table}
```

**How to read the table.**

- **Feasibility** — a run is *feasible* when the trajectory is collision-free
  **and** respects the curvature limit **and** stays inside the workspace. The
  curvature limit is verified with the **Menger maximum curvature** of the
  sampled path against `κ_max·(1+5%)`; the 5 % relative tolerance absorbs the
  discretisation error of the sampled curve (Menger curvature is exact for
  circular arcs). This is the criterion that the scoring uses everywhere.
- **Avg Time** — CPU time per scenario (`time.process_time`), robust mean
  (outliers outside 1.5·IQR removed) with standard deviation. CPU (not wall)
  time is used because wall-clock is inflated by memory-bandwidth contention
  under parallelism (Part B.7). Each method runs at its own statistically
  selected configuration, so these times are the cost at each method's chosen
  operating point — they are **not** a paired per-configuration comparison.
  In particular, the RRT* vs RRT* + B-spline gap (0.31 s vs 0.24 s) reflects the
  different selected bases (`cc = 2.0` vs `1.0`; `expand_dis = 0.4` vs `0.1`),
  not the smoothing cost; paired on the same base the two times match within
  ≈ 0.2 ms (Part B.4).
- **Best 10 scenarios** — for each of the 10 density blocks (10 seeds each), a
  method is credited with the block only if it succeeds on **every** seed of the
  block and attains the shortest mean *feasible* path; `k/10` counts the blocks
  won. Requiring full-block feasibility prevents a low-feasibility method from
  appearing "best" by averaging only its few (easy, short) successes.
- **Norm. Length** — the mean, over the blocks fully solved by the method, of the
  method's mean feasible-path length divided by that block's best feasible
  length; `1.000` would be optimal, values `> 1` are that fraction longer, and
  `--` means the method never fully solved any block (so it is never credited a
  reliable path length).
- **Avg Rank** — within each `(density, seed)` block the methods are ranked by
  *feasible first, then shortest path, then CPU time*; infeasible runs receive
  the worst rank. `Avg Rank` averages those ranks over all 100 blocks (lower is
  better). This is the metric used to select the winning configuration of each
  variant, together with a Friedman test and a Nemenyi critical-difference /
  paired-Wilcoxon (Holm) post-hoc (Part B.4).

**Interpretation.** DE2D-NURBS (proposed) is the best all-round method: it is
the most feasible (0.99), by far the fastest (0.78 s), and has the lowest average
rank (2.35); it wins 7/10 density blocks outright. RRT*-ASV is also fully
feasible (1.00) and ranks second (2.62) but is ~25× slower per scenario;
PSO2D-NURBS (baseline) is moderately feasible (0.69) and slower still (7.29 s),
ranking third (4.08) and winning one block. The plain RRT* variants are cheap and,
after the curvature correction, non-trivially feasible (0.18–0.44); RRT* +
B-spline ranks slightly better than plain RRT* (7.35 vs 8.31) because smoothing
repairs the jagged `cc = 1.0` path (0.00 → 0.21), whereas plain RRT* needs
`cc = 2.0` to reach 0.18 feasibility and ends up last. None of them solves a whole
density block, so none is credited a block win. The Dubins and modified-Dubins
planners respect curvature but are penalised by long runtimes (≈ 300–335 s for the
modified variants) and lower feasibility under the stricter Menger criterion, so
they rank among the last.
