import os
import pickle
import time
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import gamma
from utils import run_mcmc_var, normalize_c  # Make sure utils.py is in the same folder

# Resolve paths relative to this file (so the script runs from any working directory).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === PARAMETERS (SET FOR EACH RUN) ===
# For speed: reduce n_points (main cost = n_points QP solves per iteration)
dim = 20
sigma_actual = 0.1  # variance (y ~ N(x, sigma_actual*I))
n_points = 1000  
n_simulations = 100
n_chains = 2
n_iterations = 20000
n_adaptive = n_iterations // 4 
_n_cpu = multiprocessing.cpu_count()
_use_qp_parallel = n_points >= 50  # parallelize QPs when n_points is large
qp_workers = max(2, _n_cpu // 2) if _use_qp_parallel else 1
sim_workers = max(1, _n_cpu // 2) if _use_qp_parallel else _n_cpu
k = 0   
r = np.ones(dim)
c_actual = normalize_c(np.ones(dim))

prior_density_log = np.log(gamma(dim/2)/(2 * np.pi**(dim/2)))
main_columns = ['Simulation','Chain','Optimal','Cost','Sigma','Posterior','Prior','Accept']

# Choose c-proposal for Algorithm1/QP:
# - "gauss_fullcov" (default): current method, adapt full covariance in (c, eta) then normalize c
# - "sym_gauss": isotropic Gaussian RW on sphere for c, independent RW for eta
# - "vmf": vMF RW on sphere for c, independent RW for eta (proposal-only; kappa_prop tuned internally)
c_proposal = "gauss_fullcov"

# === LOAD DATA ===
data_dir = os.path.join(_SCRIPT_DIR, "data", f"dim{dim}")
filename_data = os.path.join(data_dir, f"s{sigma_actual}_n{n_points}_data.pkl")
with open(filename_data, 'rb') as f:
    data_dict = pickle.load(f)

# === OUTPUT FILES ===
result_dir = os.path.join(_SCRIPT_DIR, "results", f"dim{dim}")
os.makedirs(result_dir, exist_ok=True)
filename_results = os.path.join(result_dir, f's{sigma_actual}_n{n_points}_results.csv')
filename_runs = os.path.join(result_dir, f's{sigma_actual}_n{n_points}_runs.csv')

base_cols = ["Simulation", "Coverage", "Mean-angle", "Semi-angle", "R-hat", "Number of iterations", "Time_sec"]
rhat_cost_cols = [f"Rhat_c{j+1}" for j in range(dim)]
pd.DataFrame(columns=base_cols + rhat_cost_cols).to_csv(filename_results, index=False)
save_runs = False  # set True to save runs CSV (for trace plots); False = faster, less I/O
if save_runs:
    pd.DataFrame(columns=main_columns).to_csv(filename_runs, index=False)

# === FUNCTION TO RUN ONE SIMULATION ===
def run_simulation(sim_id):
    from utils import run_mcmc_var  # avoid pickling issues on some platforms
    try:
        t0 = time.perf_counter()
        result = run_mcmc_var(sim_id, r, k, data_dict[sim_id], dim, c_actual, sigma_actual,
                              n_chains, n_adaptive, n_iterations, prior_density_log,
                              main_columns, correct=False, qp_workers=qp_workers, save_runs=save_runs,
                              c_proposal=c_proposal)
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
    print(
        f"Running {n_simulations} simulations: {sim_workers} sim workers, {qp_workers} QP threads each "
        f"(c_proposal={c_proposal})"
    )

    with ProcessPoolExecutor(max_workers=sim_workers) as executor:
        futures = [executor.submit(run_simulation, sim) for sim in range(n_simulations)]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                summary, df_run = result
                pd.DataFrame([summary]).to_csv(filename_results, mode='a', header=False, index=False)
                if save_runs and df_run is not None:
                    df_run.to_csv(filename_runs, mode='a', header=False, index=False)

            if i % 10 == 0:
                print(f"\u2705 Simulations {i - 9} to {i} done")
