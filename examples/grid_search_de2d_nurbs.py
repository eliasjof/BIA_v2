import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from scenario_config import ScenarioConfig
from de2d_nurbs import DE2D_NURBS
import time


def _compute_curvature_and_arc(pts):
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    k = np.abs(dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5 + 1e-10)
    return k


def evaluate_result(de, config):
    result = de.get_best_path()
    if result is None:
        return None
    pts = result["points"]
    comp = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    k = _compute_curvature_and_arc(pts)
    kappa_viol = float(np.sum(np.maximum(0, k - config.kappa_max)**2))
    ws_viol = float(np.sum(
        (pts[:, 0] < config.xmin) | (pts[:, 0] > config.xmax) |
        (pts[:, 1] < config.ymin) | (pts[:, 1] > config.ymax)
    ))
    from __utils import check_collisions_cylinderBT, detailed_collision_with_polygons
    config.setup()
    obs = config.expanded_obs
    is_circle = (len(obs) > 0 and isinstance(obs[0], (tuple, list)) and len(obs[0]) == 3)
    if is_circle:
        dists = check_collisions_cylinderBT(pts, obs, config.radius)
        obs_viol = float(np.sum(dists)) if len(dists) > 0 else 0.0
    else:
        _, inside_len, _, _ = detailed_collision_with_polygons(pts, obs)
        obs_viol = float(inside_len)
    return {
        "success": True,
        "comprimento": comp,
        "kappa_viol": kappa_viol,
        "obs_viol": obs_viol,
        "ws_viol": ws_viol,
        "total_viol": kappa_viol + obs_viol + ws_viol,
    }


def plot_results(agg):
    aw_vals = sorted(agg["alpha_workspace"].unique())
    ak_vals = sorted(agg["alpha_kappa"].unique())
    ao_vals = sorted(agg["alpha_obs"].unique())
    n_aw = len(aw_vals)

    fig, axes = plt.subplots(1, n_aw, figsize=(6 * n_aw, 4.5), squeeze=False,
                              constrained_layout=True)
    vmin = agg["comp_medio"].min()
    vmax = agg["comp_medio"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, aw in enumerate(aw_vals):
        ax = axes[0, idx]
        sub = agg[agg["alpha_workspace"] == aw].pivot_table(
            index="alpha_obs", columns="alpha_kappa", values="comp_medio",
            aggfunc="mean")
        sub = sub.reindex(index=ao_vals, columns=ak_vals)
        im = ax.imshow(sub.values, aspect="auto", origin="lower", norm=norm,
                       cmap="viridis_r")
        ax.set_xticks(range(len(ak_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in ak_vals])
        ax.set_yticks(range(len(ao_vals)))
        ax.set_yticklabels([f"{v:.0f}" for v in ao_vals])
        ax.set_xlabel(r"$\alpha_{\kappa}$")
        ax.set_ylabel(r"$\alpha_{obs}$")
        ax.set_title(rf"$\alpha_{{workspace}}$ = {aw}")
        for i in range(len(ao_vals)):
            for j in range(len(ak_vals)):
                val = sub.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="w" if norm(val) > 0.5 else "k")

    fig.suptitle("Comprimento médio do caminho", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    # ── Top combos (score normalizado) ──
    validos = agg[agg["success_rate"] == 1.0].copy()
    if len(validos) > 0:
        c_min, c_max = validos["comp_medio"].min(), validos["comp_medio"].max()
        v_min, v_max = validos["total_viol_medio"].min(), validos["total_viol_medio"].max()
        c_rng = max(c_max - c_min, 1e-12)
        v_rng = max(v_max - v_min, 1e-12)
        validos["score"] = (
            (validos["comp_medio"] - c_min) / c_rng +
            (validos["total_viol_medio"] - v_min) / v_rng
        )
        top = validos.sort_values("score").head(10).reset_index(drop=True)
        top["label"] = top.apply(
            lambda r: rf"$\kappa$={r['alpha_kappa']:.1f} obs={r['alpha_obs']:.0f} "
                      rf"ws={r['alpha_workspace']:.1f}", axis=1)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(top))
        w = 0.3
        ax2.bar(x - w, top["comp_medio"], w, label="Comprimento", color="steelblue")
        ax2.bar(x, top["kappa_viol_medio"], w, label=r"Viol. $\kappa$", color="coral")
        ax2.bar(x + w, top["obs_viol_medio"], w, label="Viol. obs", color="goldenrod")
        ax2.set_xticks(x)
        ax2.set_xticklabels(top["label"], fontsize=8, rotation=25, ha="right")
        ax2.set_ylabel("Valor")
        ax2.set_title("Top 10 combinações (score = comp_norm + viol_norm)")
        ax2.legend(fontsize=9)
        fig2.tight_layout()

    plt.show()


def _run_single(params, run, n_generations, run_configs, idx):
    scen = run_configs[run]
    config = ScenarioConfig()
    config.seed = scen["seed"]
    config.occupancy_rate = scen["occupancy"]
    config.start = np.array([-1.4, -0.8])
    config.goal = np.array([1.4, 0.8])
    config.th_start = np.deg2rad(0)
    config.th_goal = np.deg2rad(0)
    config.radius = 0.073
    config.kappa_max = 1 / 0.73
    config.n_generations = n_generations
    config.pop_size = params["pop_size"]
    config.scale_x = 2.0
    config.scale_y = 2.0
    config.xmin = -2.0
    config.xmax = 2.0
    config.ymin = -2.0
    config.ymax = 2.0
    config.lambda_i = config.radius
    config.lambda_f = config.radius
    config.alpha_kappa = params["alpha_kappa"]
    config.alpha_obs = params["alpha_obs"]
    config.alpha_workspace = params["alpha_workspace"]

    de = DE2D_NURBS(config)
    de.run()

    eval = evaluate_result(de, config)
    row = {
        "pop_size": params["pop_size"],
        "alpha_kappa": params["alpha_kappa"],
        "alpha_obs": params["alpha_obs"],
        "alpha_workspace": params["alpha_workspace"],
        "run": run,
        "occupancy": scen["occupancy"],
    }
    if eval is None:
        row["success"] = False
    else:
        row.update(eval)

    print(f"[{idx + 1}] pop={params['pop_size']}, "
          f"ak={params['alpha_kappa']:.1f}, "
          f"ao={params['alpha_obs']:.0f}, "
          f"aw={params['alpha_workspace']:.1f} "
          f"-> {'OK' if eval else 'FAIL'} ")
    return row


def main():
    param_grid = {
        "pop_size": [50, 80, 100, 120, 150],
        "alpha_kappa": [0.1, 0.5, 1.0, 10.0],
        "alpha_obs": [1.0, 10.0, 100.0, 500.0],
        "alpha_workspace": [0.1, 1.0, 10.0],
    }
    N_RUNS = 10
    N_GENERATIONS = 200
    N_JOBS = 20
    BASE_SEED = 10

    # Cada run usa seed e ocupação diferentes — TODAS as combinações
    # enfrentam o MESMO conjunto de cenários, permitindo comparação justa.
    run_configs = [
        {"seed": BASE_SEED + r, "occupancy": round(0.05 * (r + 1), 2)}
        for r in range(N_RUNS)
    ]

    grid = list(ParameterGrid(param_grid))
    total = len(grid) * N_RUNS
    print(f"Param grid: {len(grid)} combos x {N_RUNS} runs = {total}")
    print(f"Scenario configs por run:")
    for r, sc in enumerate(run_configs):
        print(f"  run {r}: seed={sc['seed']:2d}, occupancy={sc['occupancy']:.2f}")
    print(f"n_jobs={N_JOBS}")
    print()

    t0 = time.time()

    tasks = []
    idx = 0
    for params in grid:
        for run in range(N_RUNS):
            tasks.append((params, run, N_GENERATIONS, run_configs, idx))
            idx += 1

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_run_single)(*args) for args in tasks
    )

    df = pd.DataFrame(results)
    csv_path = pathlib.Path(__file__).parent / "grid_search_de_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    agg = df.groupby(["pop_size", "alpha_kappa", "alpha_obs", "alpha_workspace"]).agg(
        success_rate=("success", "mean"),
        comp_medio=("comprimento", "mean"),
        comp_std=("comprimento", "std"),
        kappa_viol_medio=("kappa_viol", "mean"),
        obs_viol_medio=("obs_viol", "mean"),
        ws_viol_medio=("ws_viol", "mean"),
        total_viol_medio=("total_viol", "mean"),
    ).reset_index()

    agg = agg.sort_values("comp_medio")
    print("\n=== Resultados agregados (ordenados por comprimento) ===")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print(agg.to_string(index=False))

    validos = agg[agg["success_rate"] == 1.0].copy()
    if len(validos) > 0:
        validos["score"] = validos["comp_medio"] * (1 + validos["total_viol_medio"])
        best = validos.sort_values("score").head(5)
        print("\n=== Top 5 (sem falhas, menor score = comp_norm + viol_norm) ===")
        print(best.to_string(index=False))
    else:
        print("\nNenhuma combinação teve 100% de sucesso.")

    plot_results(agg)

    total_elapsed = time.time() - t0
    print(f"\nTempo total: {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
