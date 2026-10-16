# Conversation Summary – Budget, Step‑Size & Rewire Radius Choices

## 1. Parameter table (current script)

| Planner (key) | Function | `max_iter` / generations | `goal_sample_rate` | `connect_circle_dist` | `step_size` / `path_resolution` | Curvature `kappa_max` | Post‑process | Stop criterion |
|---|---|---|---|---|---|---|---|---|
| `rrt_star` | `run_rrt_star → run_rrt` | 2000 | 20 | **2.0** | *default of `RRTPlanner` (~0.5 m)* | `config.kappa_max` | – | max‑iter |
| `rrt_star_smooth` | `run_rrt_star_smooth` | 2000 | 20 | **2.0** | *default* | `config.kappa_max` | **B‑spline** (`n_waypoints=20, s=0.015`) | max‑iter + one spline |
| `rrt_star_dubins` | `run_rrt_star_dubins` | 2000 | 20 | **4.5** | `path_resolution=0.05` | `config.kappa_max` | – | max‑iter |
| `modified_dubins_rrt_star_ccpoa` | `run_modified_dubins_rrt_star_ccpoa` | 1500 (RRT*) + 5 (CCPOA) | 20 | **4.5** | `step_size=0.05` (both) | `config.kappa_max` | CCPOA (5 iters) | max‑iter RRT* + CCPOA |
| `rrt_star_asv` | `run_rrt_star_asv` | 5000 | 30 | **4.5** | `step_size=0.05` | `config.kappa_max` | – | max‑iter |
| `de2d_nurbs` | `run_de2d_nurbs` | 450 gen × 100 pop = 45 k evals | – | – | – (DE) | `config.kappa_max` | – | fixed generations |
| `pso2d_nurbs` | `run_pso2d_nurbs` | 500 gen × 50 pop = 25 k evals | – | – | – (PSO) | `config.kappa_max` | – | fixed generations |

## 2. Step‑size adequacy
- **Dubins‑based planners** (`path_resolution` / `step_size = 0.05 m`) → appropriate (≈ 0.7 × robot radius, matches minimum turning radius 0.73 m).
- **Plain RRT\*** uses the *default* step‑size of `RRTPlanner` (~0.5 m). For a 4.2 × 4.2 m workspace with obstacles as small as 0.08 m, this is **too large**; it can skip narrow passages. Recommended: **0.10–0.15 m** (≈ 1.5–2 × robot radius).

## 3. Rewire radius (`connect_circle_dist`)
- Theoretical asymptotically optimal radius for 2‑D RRT\* with 2000 nodes ≈ 3–4 m. Current **2.0 m** is conservative → slower convergence.
- For Dubins (3‑D state) 4.5 m is near the theoretical value.
- **Worth testing variations** to show the choice is not arbitrary.

### Suggested grid
| Planner | `connect_circle_dist` values |
|---|---|
| `rrt_star` / `rrt_star_smooth` | 1.5 m, **2.0 m**, 3.0 m |
| Dubins variants | 3.5 m, **4.5 m**, 6.0 m |

## 4. Budget justification (already documented)
- Sampling‑based planners: fixed `max_iter` (2000 / 5000) → comparable wall‑clock (~2 s vs 30‑100 s).
- Evolutionary planners: fixed evaluations (≈ 30‑45 k) → literature‑based convergence region.

## 5. Next steps (to be implemented)
1. Reduce RRT* step‑size to 0.10 m (expose via `extra_kw`).
2. Add new planner entries in `PLANNERS` for each `connect_circle_dist` variant (and optionally step‑size variants).
3. Keep all other parameters identical so the only difference is the hyper‑parameter under test.
4. Re‑run the *progressive* experiment (30 seeds) – extra ≈ 5 min total.

--- 

*Generated automatically from the chat with the AI assistant on 2026‑10‑15.*