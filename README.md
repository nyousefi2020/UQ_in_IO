## UQ in IO — Code for Numerical Experiments

This folder contains the code used to generate the numerical experiments for the paper.

### Folder structure
- `Algorithm1/LP/`: Algorithm 1 for LP experiments (`run_sim.py`, `utils.py`, `data_generator.ipynb`)
- `Algorithm1/QP/`: Algorithm 1 for QP experiments (`run_sim.py`, `utils.py`, `data_generator.ipynb`)
- `Algorithm2/LP/`: Algorithm 2 for LP experiments (`run_sim.py`, `utils.py`, `data_generator.ipynb`)
- `Algorithm2/QP/`: Algorithm 2 for QP experiments (`run_sim.py`, `utils.py`, `data_generator.ipynb`)

Each subfolder follows the same pattern:
- `data_generator.ipynb` generates synthetic data files under `data/`
- `run_sim.py` runs Monte Carlo simulations and writes outputs under `results/`
- `utils.py` contains the optimization and MCMC routines

### Requirements
- Python 3.x
- Packages listed in `requirements.txt`
- Gurobi (via `gurobipy`) is used for the forward optimization problems; a working Gurobi installation/license is required.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### How to run
1) Generate data (per module) by running the notebook:
- open `data_generator.ipynb` in the relevant subfolder and run all cells

2) Run the simulation script (per module):

```bash
python run_sim.py
```

By default, `run_sim.py` uses multiprocessing to run independent simulation replicates in parallel (one replicate per worker process). Output CSVs are written to `results/` in the same module folder.

### Outputs
Each `run_sim.py` writes:
- a `*_results.csv` summary (coverage, widths/angles, \(\hat R\), iterations, and `Time_sec`)
- optionally a `*_runs.csv` containing per-iteration chain draws (see `save_runs` flags where present)

