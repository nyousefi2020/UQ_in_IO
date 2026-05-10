import os
import pickle
import time
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import gamma
from utils import run_mcmc_var, normalize_c, solve_io_all  # Make sure utils.py is in the same folder

# Resolve paths relative to this file (so the script runs from any working directory).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === PARAMETERS (SET FOR EACH RUN) ===
dim = 20
n_constraints = 75  # (> dim + 1 for feasibility)
kappa_actual = 10
n_points = 1000
n_simulations = 100
n_chains = 2
n_iterations = 100000
n_adaptive = n_iterations // 4
c_actual = normalize_c(np.ones(dim))

prior_density_log = np.log(gamma(dim / 2) / (2 * np.pi ** (dim / 2)))
main_columns = ['Simulation','Chain','x','Cost','Kappa','Posterior','Prior','Accept']

# Choose c-proposal for Algorithm2/LP:
# - "gauss_fullcov" (default): Gaussian RW with adapted full covariance, then normalize c
# - "sym_gauss": isotropic Gaussian RW + normalize (symmetric)
# - "vmf": vMF RW centered at current c with current kappa
c_proposal = "gauss_fullcov"

# === LOAD DATA ===
data_dir = os.path.join(_SCRIPT_DIR, "data", f"dim{dim}")
filename_data = os.path.join(
    data_dir, f"k{kappa_actual}_n{n_points}_m{n_constraints}_data.pkl"
)
with open(filename_data, 'rb') as f:
    data_dict = pickle.load(f)

# === OUTPUT FILES ===
result_dir = os.path.join(_SCRIPT_DIR, "results", f"dim{dim}")
os.makedirs(result_dir, exist_ok=True)
filename_results = os.path.join(
    result_dir, f'results_k{kappa_actual}_n{n_points}_m{n_constraints}.csv'
)
filename_runs = os.path.join(
    result_dir, f'runs_k{kappa_actual}_n{n_points}_m{n_constraints}.csv'
)

# Initialize output files (overwrite if they exist to avoid duplicates)
base_cols = ["Simulation", "Coverage", "Mean-angle", "Semi-angle", "R-hat", "Number of iterations", "Time_sec"]
rhat_cost_cols = [f"Rhat_c{j+1}" for j in range(dim)]
pd.DataFrame(columns=base_cols + rhat_cost_cols).to_csv(filename_results, index=False)
pd.DataFrame(columns=main_columns).to_csv(filename_runs, index=False)

# === FUNCTION TO RUN ONE SIMULATION ===
def run_simulation(sim_id):
    from utils import run_mcmc_var, solve_io_all  # avoid pickling issues on some platforms
    try:
        t0 = time.perf_counter()
        raw = solve_io_all(data_dict[sim_id])
        c_optimals = np.array([c for c in raw if c is not None], dtype=float)
        result = run_mcmc_var(sim_id, c_optimals, dim, c_actual, kappa_actual,
                              n_chains, n_adaptive, n_iterations, prior_density_log,
                              main_columns, correct=False, c_proposal=c_proposal)
        time_sec = time.perf_counter() - t0
        rhat_cost = result['Rhat_cost']
        summary = {
            "Simulation": sim_id + 1,
            "Coverage": result['covered'],
            "Mean-angle": result['meanangle'],
            "Semi-angle": result['semiangle'],
            "R-hat": result['Rhat'],
            "Number of iterations": result['iterations'],
            "Time_sec": time_sec,
        }
        for j in range(dim):
            summary[f"Rhat_c{j+1}"] = rhat_cost[j]
        return (summary, result['df'])
    except Exception as e:
        print(f"Simulation {sim_id + 1} failed: {e}")
        return None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    sim_workers = multiprocessing.cpu_count()
    print(f"Running {n_simulations} simulations using {sim_workers} worker processes (c_proposal={c_proposal})")

    with ProcessPoolExecutor(max_workers=sim_workers) as executor:
        futures = [executor.submit(run_simulation, sim) for sim in range(n_simulations)]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                summary, df_run = result
                pd.DataFrame([summary]).to_csv(filename_results, mode='a', header=False, index=False)
                df_run.to_csv(filename_runs, mode='a', header=False, index=False)

            if i % 10 == 0:
                print(f"✅ Simulations {i - 9} to {i} done")
