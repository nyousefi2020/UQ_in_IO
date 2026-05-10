import os
import pickle
import time
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import gamma
from utils import run_mcmc_var, normalize_c

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


_env_lp = os.environ.get("ALG1_LP_BACKEND", "").strip().lower()
lp_backend = _env_lp if _env_lp in ("highs", "gurobi") else "gurobi"

# === PARAMETERS (SET FOR EACH RUN) ===
dim = 20
sigma_actual = 0.1  # variance (y ~ N(x, sigma_actual*I))
n_constraints = 75  
n_points = 1000
n_simulations = 100
n_chains = 2
n_iterations = 15000
n_adaptive = n_iterations // 4
_n_cpu = multiprocessing.cpu_count()
sim_workers = max(1, _n_cpu)  # same for all n_points (full logical CPUs)
c_actual = normalize_c(np.ones(dim))

# Choose c-proposal for Algorithm1/LP:
# - "gauss_fullcov" (default): Gaussian RW with adapted full covariance in (c, eta), then normalize c
# - "sym_gauss": isotropic Gaussian RW on sphere + RW on eta=log(sigma_sq)
# - "vmf": vMF RW on sphere centered at current c with inferred kappa + RW on eta
c_proposal = "gauss_fullcov"

prior_density_log = np.log(gamma(dim / 2) / (2 * np.pi ** (dim / 2)))
main_columns = ["Simulation", "Chain", "Optimal", "Cost", "Sigma", "Posterior", "Prior", "Accept"]
if c_proposal == "vmf":
    main_columns = ["Simulation", "Chain", "Optimal", "Cost", "Sigma", "Kappa", "Posterior", "Prior", "Accept"]

# === LOAD DATA ===
data_dir = os.path.join(_SCRIPT_DIR, "data", f"dim{dim}")
filename_data = os.path.join(data_dir, f"s{sigma_actual}_n{n_points}_m{n_constraints}_data.pkl")
with open(filename_data, "rb") as f:
    data_dict = pickle.load(f)

_check = data_dict[0]
if len(_check) != n_points:
    raise ValueError(
        f"Pickle {filename_data!r} has len(sim 0)={len(_check)} but run_sim.py n_points={n_points}. "
        f"Regenerate data with data_generator.ipynb using the same n_points (and sigma_actual, dim)."
    )

# === OUTPUT FILES ===
result_dir = os.path.join(_SCRIPT_DIR, "results", f"dim{dim}")
os.makedirs(result_dir, exist_ok=True)
filename_results = os.path.join(result_dir, f"s{sigma_actual}_n{n_points}_m{n_constraints}_results.csv")
filename_runs = os.path.join(result_dir, f"s{sigma_actual}_n{n_points}_m{n_constraints}_runs.csv")

base_cols = ["Simulation", "Coverage", "Mean-angle", "Semi-angle", "R-hat", "Number of iterations", "Time_sec"]
rhat_cost_cols = [f"Rhat_c{j+1}" for j in range(dim)]
save_runs = True

# Do NOT write CSV headers at module import: worker processes re-import this file and would truncate outputs.


def run_simulation(sim_id):
    from utils import run_mcmc_var  # avoid pickling issues on some platforms

    try:
        t0 = time.perf_counter()
        result = run_mcmc_var(
            sim_id,
            data_dict[sim_id],
            dim,
            c_actual,
            sigma_actual,
            n_chains,
            n_adaptive,
            n_iterations,
            prior_density_log,
            main_columns,
            correct=False,
            save_runs=save_runs,
            lp_backend=lp_backend,
            c_proposal=c_proposal,
        )
        time_sec = time.perf_counter() - t0
        rhat_cost = result["Rhat_cost"]
        summary = {
            "Simulation": sim_id + 1,
            "Coverage": result["covered"],
            "Mean-angle": result["meanangle"],
            "Semi-angle": result["semiangle"],
            "R-hat": result["Rhat"],
            "Number of iterations": result["iterations"],
            "Time_sec": time_sec,
        }
        for j in range(dim):
            summary[f"Rhat_c{j+1}"] = rhat_cost[j]
        return (summary, result["df"])
    except Exception as e:
        print(f"Simulation {sim_id + 1} failed: {e}")
        return None


if __name__ == "__main__":
    proposal_info = f"c_proposal={c_proposal}"
    if c_proposal == "vmf":
        proposal_info += ", infer_kappa=True"
    print(
        f"Running {n_simulations} simulations: {sim_workers} sim workers (parallel processes), "
        f"lp_backend={lp_backend}, {proposal_info}"
    )

    pd.DataFrame(columns=base_cols + rhat_cost_cols).to_csv(filename_results, index=False)
    if save_runs:
        pd.DataFrame(columns=main_columns).to_csv(filename_runs, index=False)

    with ProcessPoolExecutor(max_workers=sim_workers) as executor:
        futures = [executor.submit(run_simulation, sim) for sim in range(n_simulations)]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                summary, df_run = result
                pd.DataFrame([summary]).to_csv(filename_results, mode="a", header=False, index=False)
                if save_runs:
                    if df_run is None:
                        print(
                            f"Warning: simulation {summary.get('Simulation', '?')} finished but df is None "
                            "(save_runs issue in worker)"
                        )
                    else:
                        df_run.to_csv(filename_runs, mode="a", header=False, index=False)

            if i % 10 == 0:
                print(f"\u2705 Simulations {i - 9} to {i} done")
