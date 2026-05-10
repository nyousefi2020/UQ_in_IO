import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import multivariate_normal, halfcauchy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Utility Functions
# -------------------------------

def normalize_c(c):
    return c / np.linalg.norm(c)

def get_cand_c_sym_gauss(c, var_c):
    """
    Symmetric isotropic Gaussian random-walk on the sphere:
      c' = (c + eps) / ||c + eps||,  eps ~ N(0, var_c I).
    """
    c = np.asarray(c, dtype=float).ravel()
    eps = np.random.normal(0.0, np.sqrt(float(var_c)), size=c.shape[0])
    return normalize_c(c + eps)

def get_cand_c_vmf(c, kappa_prop):
    """
    von Mises-Fisher random-walk on the sphere centered at current c.
    Symmetric proposal: q(c'|c) = q(c|c').
    """
    from scipy.stats import vonmises_fisher
    c = normalize_c(np.asarray(c, dtype=float))
    samples = vonmises_fisher.rvs(c, float(kappa_prop), size=1)
    return normalize_c(np.asarray(samples).ravel())

def get_cand(c, sigma, var_jump):
    """Propose in (c, sigma) space jointly. var_jump last dim is for sigma. Returns (c_new, sigma_new)."""
    dim = c.shape[0]
    mean = np.append(c, sigma)
    cand = np.random.multivariate_normal(mean, var_jump)
    c_new = cand[:-1]
    sigma_new = cand[-1]
    return normalize_c(c_new), sigma_new


def get_cand_log_sigma_sq(c, sigma_sq, var_jump):
    """Propose in (c, eta) space where eta=log(sigma_sq). var_jump last dim is for eta.
    Returns (c_new, sigma_sq_new). sigma_sq = variance."""
    dim = c.shape[0]
    eta = np.log(sigma_sq)
    mean = np.append(c, eta)
    cand = np.random.multivariate_normal(mean, var_jump)
    c_new = normalize_c(cand[:-1])
    eta_new = cand[-1]
    sigma_sq_new = float(np.exp(eta_new))
    return c_new, sigma_sq_new

def get_cand_eta(eta, var_eta):
    """Gaussian random-walk in eta=log(sigma_sq). Returns eta_new."""
    return float(np.random.normal(float(eta), np.sqrt(float(var_eta))))

def psrf(chains_list, n_chains, split_chains=True):
    n_samples = len(chains_list) // n_chains
    chains = chains_list.values.reshape(n_chains, n_samples)
    n_chains_eff = n_chains
    if split_chains:
        n_samples = n_samples // 2
        chains = chains[:, :n_samples * 2].reshape(n_chains, n_samples, 2)
        n_chains_eff = 2 * n_chains  # each half is a chain
    mean_chain = np.mean(chains, axis=1)
    mean_all = np.mean(mean_chain)
    B = n_samples / (n_chains_eff - 1) * np.sum((mean_chain - mean_all) ** 2)
    W = (1.0 / n_chains_eff) * np.sum(np.var(chains, axis=1))
    var_theta = (n_samples - 1) / n_samples * W + 1.0 / n_samples * B
    return np.sqrt(var_theta / W)


def psrf_cost(cost_vectors, n_chains, split_chains=True):
    """
    R-hat for each component of the cost vector.
    """
    V = np.asarray(cost_vectors, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    dim = V.shape[1]
    rhats = np.zeros(dim)
    for j in range(dim):
        rhats[j] = psrf(pd.Series(V[:, j]), n_chains, split_chains=split_chains)
    return rhats

# -------------------------------
# QP Solver with Warm Start
# -------------------------------

class QPSolver:
    def __init__(self, dim):
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.model.Params.FeasibilityTol = 1e-6
        self.model.Params.Threads = 1  # we parallelize across QPs; avoid oversubscription
        self.model.Params.BarIterLimit = 1500  # cap barrier iterations; most QPs converge in < 200
        self.x = self.model.addMVar(shape=dim, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="x")
        self.constr = self.model.addConstr(0 <= 0)
        self.model.setObjective(0, GRB.MINIMIZE)

    def solve(self, Q, r, k, c, x0=None):
        try:
            self.model.remove(self.constr)
            self.model.update()
            self.constr = self.model.addConstr(self.x @ Q @ self.x + r @ self.x + k <= 0)
            self.model.setObjective(c @ self.x, GRB.MINIMIZE)
            if x0 is not None:
                self.x.Start = x0
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                return self.x.X
            else:
                return None
        except Exception as e:
            print(f"Gurobi error: {e}")
            return None

# -------------------------------
# Closed-Form QP Solver (k=0, r=ones)
# -------------------------------

def closed_form_qp(Q, r, c):
    """
    Closed-form solution for min c'x s.t. x'Qx + r'x <= 0 (r = ones, k=0).
    x* = -0.5 Q^{-1} r - 0.5 * sqrt(r'Q^{-1}r / c'Q^{-1}c) * Q^{-1}c
    Uses Cholesky (Q is PSD) for ~1.5x speed vs general solve. Returns x* or None if invalid.
    """
    Q = np.asarray(Q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    try:
        c_factor = cho_factor(Q, lower=True)
        Qinv_r = cho_solve(c_factor, r)
        Qinv_c = cho_solve(c_factor, c)
        r_Qinv_r = float(r @ Qinv_r)
        c_Qinv_c = float(c @ Qinv_c)
        if c_Qinv_c <= 1e-15 or r_Qinv_r <= 1e-15:
            return None
        scalar = np.sqrt(r_Qinv_r / c_Qinv_c)
        return -0.5 * Qinv_r - 0.5 * scalar * Qinv_c
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return None


def _solve_one_qp(args):
    """Worker for parallel QP solve. args = (Q, r, c)."""
    Q, r, c = args
    return closed_form_qp(Q, r, c)


def solve_all_qp_closed_form(data, r, c, n_workers=1, Q_list=None):
    """Solve QP for each (Q,y) in data using closed-form. No Gurobi needed.
    n_workers>1 uses ThreadPoolExecutor for parallel solves (faster on multi-core).
    Q_list: optional pre-extracted [data[i][0] for i in range(len(data))] to avoid repeated indexing."""
    if Q_list is None:
        Q_list = [data[i][0] for i in range(len(data))]
    n_pts = len(Q_list)
    if n_workers <= 1 or n_pts < 8:
        return [closed_form_qp(Q_list[i], r, c) for i in range(n_pts)]
    args_list = [(Q_list[i], r, c) for i in range(n_pts)]
    n_workers = min(n_workers, n_pts)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_solve_one_qp, args_list))


# -------------------------------
# Parallel QP Solve with Warm Start (Gurobi)
# -------------------------------

def solve_all_qp_parallel_warm(data, r, k, c, dim, x_starts=None, solver_pool=None):
    """Solve QP for each (Q,y) in data. Reuse solver_pool if provided (avoids model creation overhead)."""
    n_pts = len(data)
    indexed_data = [(data[i][0], data[i][1], i) for i in range(n_pts)]

    def solve_one(args):
        Q, _, idx = args
        if solver_pool is not None:
            solver = solver_pool[idx]
        else:
            solver = QPSolver(dim)
        x0 = x_starts[idx] if x_starts is not None else None
        return solver.solve(Q, r, k, c, x0=x0)

    max_workers = min(n_pts, 64)  # cap threads to avoid oversubscription
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(solve_one, indexed_data))
    return results

# -------------------------------
# Data Generator
# -------------------------------

def generate_qp_data_var(dim, n_points, c_star, sigma_sq, r, k):
    """Generate (Q, y) data. sigma_sq = variance (y ~ N(x, sigma_sq*I)).
    Uses closed-form when k=0 and r=ones; else Gurobi fo_qp."""
    use_closed_form = (k == 0 and np.allclose(r, np.ones(dim)))
    data_points = []
    for _ in range(n_points):
        while True:
            Q = generate_psd_matrix(dim, min_eig=1, max_eig=5)
            if use_closed_form:
                x = closed_form_qp(Q, r, c_star)
            else:
                x = fo_qp(Q, r, k, c_star)
            if x is not None:
                break
        y = np.random.multivariate_normal(x, sigma_sq * np.identity(dim))
        data_points.append((Q, y))
    return data_points

def generate_psd_matrix(dim, min_eig=1, max_eig=5, scale=1):
    U, _ = np.linalg.qr(np.random.randn(dim, dim))
    eigenvalues = np.linspace(min_eig, max_eig, dim)
    Q = U @ np.diag(eigenvalues) @ U.T
    return scale * Q

def fo_qp(Q, r, k, c):
    dim = c.shape[0]
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.FeasibilityTol = 1e-6
    x = model.addMVar(shape=dim, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    model.setObjective(c @ x, GRB.MINIMIZE)
    model.addConstr(x @ Q @ x + r @ x + k <= 0)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return x.X
    return None

# -------------------------------
# Posterior Evaluation
# -------------------------------
# sigma_sq = variance (y ~ N(x, sigma_sq*I)). We sample variance.
PRIOR_SIGMA_SCALE = 5  # half-Cauchy(0, scale) on std_dev=sqrt(sigma_sq); larger = weaker prior

def posterior_log_var(dim, data, optimals, sigma_sq, prior, Y=None):
    """Log posterior for joint (c, sigma_sq) sampling. sigma_sq = variance. Prior: half-Cauchy on sqrt(sigma_sq)."""
    if Y is None:
        Y = np.array([data[i][1] for i in range(len(data))])
    X = np.array(optimals)
    diff = Y - X
    sum_sq = np.sum(diff ** 2, axis=1)
    log_const = dim * np.log(2 * np.pi * sigma_sq)
    posterior = float(-0.5 * np.sum(log_const + sum_sq / sigma_sq))
    std_dev = np.sqrt(sigma_sq)
    prior_var = halfcauchy.logpdf(std_dev, 0, PRIOR_SIGMA_SCALE) - 0.5 * np.log(sigma_sq) - np.log(2)
    return posterior + prior + prior_var


def posterior_log_var_log_sigma_sq(dim, data, optimals, sigma_sq, prior, Y=None):
    """Log posterior when proposing in eta=log(sigma_sq) space. sigma_sq = variance.
    Prior on eta: half-Cauchy on sqrt(exp(eta)), Jacobian for log(sigma_sq).
    Pass Y precomputed to avoid repeated extraction (faster in MCMC loop)."""
    if Y is None:
        Y = np.array([data[i][1] for i in range(len(data))])
    X = np.array(optimals)
    diff = Y - X
    sum_sq = np.sum(diff ** 2, axis=1)
    log_const = dim * np.log(2 * np.pi * sigma_sq)
    posterior = float(-0.5 * np.sum(log_const + sum_sq / sigma_sq))
    std_dev = np.sqrt(sigma_sq)
    # Prior on eta=log(sigma_sq): log p(eta) = log p(v) + log(v). p(v) from half-Cauchy on sqrt(v).
    prior_var = halfcauchy.logpdf(std_dev, 0, PRIOR_SIGMA_SCALE) - 0.5 * np.log(sigma_sq) - np.log(2) + np.log(sigma_sq)
    return posterior + prior + prior_var

# -------------------------------
# MCMC with Parallel Chains
# -------------------------------
# sigma_sq = variance. We propose in (c, eta) with eta=log(sigma_sq). Storage uses sigma_sq.
# Column 'Sigma' stores variance (sigma_sq) for backward compat.

def run_single_chain(chain_id, sim_number, r, k, data, dim, c_actual, sigma_sq_actual,
                     n_adaptive, n_iterations, prior_density_log, main_columns, correct,
                     qp_workers=1, c_proposal="gauss_fullcov"):
    # Precompute Y and Q_list once (data doesn't change)
    n_pts = len(data)
    Y_data = np.array([data[i][1] for i in range(n_pts)])
    Q_list = [data[i][0] for i in range(n_pts)]
    # Use closed-form QP solver (no Gurobi)
    c_init = normalize_c(np.random.uniform(-1,1,dim)) 
    x_init = solve_all_qp_closed_form(data, r, c_init, n_workers=qp_workers, Q_list=Q_list)
    sigma_sq_init = halfcauchy.rvs(0, PRIOR_SIGMA_SCALE) ** 2  # variance from std_dev^2
    # Proposal configuration:
    # - "gauss_fullcov" (default): current method, adapt full covariance in (c, eta) then normalize c
    # - "sym_gauss": isotropic Gaussian RW on sphere for c, independent RW for eta
    # - "vmf": vMF RW on sphere for c, independent RW for eta
    c_proposal = str(c_proposal)
    jump_scale = np.diag([10.0] * dim + [0.5])  # used by gauss_fullcov (c, eta) covariance
    var_c = 0.10  # used by sym_gauss
    var_eta = 0.50  # used by sym_gauss/vmf (eta=log(sigma_sq))
    kappa_prop = 50.0  # used by vmf (proposal concentration)
    post_init = posterior_log_var_log_sigma_sq(dim, data, x_init, sigma_sq_init, prior_density_log, Y=Y_data)
    prior_init = 0
    list_store = [[sim_number+1, chain_id+1, x_init, c_init, sigma_sq_init, post_init, prior_init, 1]]

    tau = 1
    accept_ratio = 0
    max_adaptive_rounds = 50  # safety cap to prevent infinite loops
    adaptive_round = 0
    while (accept_ratio < 0.25 or accept_ratio > 0.40) and adaptive_round < max_adaptive_rounds:
        adaptive_round += 1
        for i in range(1, n_adaptive):
            if c_proposal == "gauss_fullcov":
                cand_c, cand_sigma_sq = get_cand_log_sigma_sq(list_store[i-1][3], list_store[i-1][4], jump_scale)
            else:
                # factorized proposal: c-step on sphere + eta-step (eta=log(sigma_sq))
                curr_c = list_store[i-1][3]
                curr_sigma_sq = float(list_store[i-1][4])
                curr_eta = float(np.log(max(curr_sigma_sq, 1e-12)))
                if c_proposal == "sym_gauss":
                    cand_c = get_cand_c_sym_gauss(curr_c, var_c)
                elif c_proposal == "vmf":
                    cand_c = get_cand_c_vmf(curr_c, kappa_prop)
                else:
                    raise ValueError(f"Unknown c_proposal: {c_proposal!r}")
                cand_eta = get_cand_eta(curr_eta, var_eta)
                cand_sigma_sq = float(np.exp(cand_eta))
            cand_mean = solve_all_qp_closed_form(data, r, cand_c, n_workers=qp_workers, Q_list=Q_list)
            if any(x is None for x in cand_mean):
                list_store.append([*list_store[i-1][:7], 0])
                continue
            cand_post = posterior_log_var_log_sigma_sq(dim, data, cand_mean, cand_sigma_sq, prior_density_log, Y=Y_data)
            cand_prior = 0
            log_accept_ratio = cand_post - list_store[i-1][5] + (cand_prior - list_store[i-1][6] if correct else 0)
            if np.log(np.random.uniform()) <= log_accept_ratio:
                list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, cand_post, cand_prior, 1])
            else:
                list_store.append([*list_store[i-1][:7], 0])
        last_row = list_store[-1]
        accept_ratio = np.mean([list_store[i][7] for i in range(len(list_store))])
        if c_proposal == "gauss_fullcov":
            cost_arr = np.array([list_store[i][3] for i in range(len(list_store))])
            eta_arr = np.log(np.array([list_store[i][4] for i in range(len(list_store))]))  # eta = log(sigma_sq)
            combined = np.column_stack([cost_arr, eta_arr])  # (c, eta) for proposal cov
            cov_mat = np.cov(combined, rowvar=False)
            cov_mat = (cov_mat + cov_mat.T) / 2 + np.eye(dim+1) * 1e-8
            tau *= 0.5 if accept_ratio < 0.22 else (3 if accept_ratio > 0.42 else 1)
            jump_scale = tau * (cov_mat + np.eye(dim+1))
        elif c_proposal == "sym_gauss":
            scale = 0.5 if accept_ratio < 0.22 else (3.0 if accept_ratio > 0.42 else 1.0)
            var_c = float(np.clip(var_c * scale, 1e-8, 1e2))
            var_eta = float(np.clip(var_eta * scale, 1e-8, 1e2))
        elif c_proposal == "vmf":
            # low acceptance => make proposals narrower (increase kappa_prop)
            # high acceptance => widen proposals (decrease kappa_prop)
            if accept_ratio < 0.22:
                kappa_prop *= 3.0
            elif accept_ratio > 0.42:
                kappa_prop *= 0.5
            kappa_prop = float(np.clip(kappa_prop, 1e-2, 1e6))
            scale = 0.5 if accept_ratio < 0.22 else (3.0 if accept_ratio > 0.42 else 1.0)
            var_eta = float(np.clip(var_eta * scale, 1e-8, 1e2))
        else:
            raise ValueError(f"Unknown c_proposal: {c_proposal!r}")

    list_store = [last_row]
    for i in range(1, n_iterations):
        if c_proposal == "gauss_fullcov":
            cand_c, cand_sigma_sq = get_cand_log_sigma_sq(list_store[i-1][3], list_store[i-1][4], jump_scale)
        else:
            curr_c = list_store[i-1][3]
            curr_sigma_sq = float(list_store[i-1][4])
            curr_eta = float(np.log(max(curr_sigma_sq, 1e-12)))
            if c_proposal == "sym_gauss":
                cand_c = get_cand_c_sym_gauss(curr_c, var_c)
            elif c_proposal == "vmf":
                cand_c = get_cand_c_vmf(curr_c, kappa_prop)
            else:
                raise ValueError(f"Unknown c_proposal: {c_proposal!r}")
            cand_eta = get_cand_eta(curr_eta, var_eta)
            cand_sigma_sq = float(np.exp(cand_eta))
        cand_mean = solve_all_qp_closed_form(data, r, cand_c, n_workers=qp_workers, Q_list=Q_list)
        if any(x is None for x in cand_mean):
            list_store.append([*list_store[i-1][:7], 0])
            continue
        cand_post = posterior_log_var_log_sigma_sq(dim, data, cand_mean, cand_sigma_sq, prior_density_log, Y=Y_data)
        cand_prior = 0
        log_accept_ratio = cand_post - list_store[i-1][5] + (cand_prior - list_store[i-1][6] if correct else 0)
        if np.log(np.random.uniform()) <= log_accept_ratio:
            list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, cand_post, cand_prior, 1])
        else:
            list_store.append([*list_store[i-1][:7], 0])
    return list_store

def run_mcmc_var(sim_number, r, k, data, dim, c_actual, sigma_sq_actual, n_chains, n_adaptive, n_iterations,
                 prior_density_log, main_columns, correct=False, qp_workers=1, save_runs=True,
                 c_proposal="gauss_fullcov"):
    """sigma_sq_actual = variance (y ~ N(x, sigma_sq*I)). save_runs=False skips building df for efficiency."""
    all_chains = [
        run_single_chain(chain, sim_number, r, k, data, dim, c_actual, sigma_sq_actual,
                         n_adaptive, n_iterations, prior_density_log, main_columns, correct,
                         qp_workers, c_proposal=c_proposal)
        for chain in range(n_chains)
    ]

    list_store_all = [row for chain_data in all_chains for row in chain_data]
    # list_store row: [Sim, Chain, Optimal, Cost, Sigma, Posterior, Prior, Accept] (indices 0-7)
    cost_arr = np.array([np.asarray(row[3]).ravel() for row in list_store_all])
    sigma_arr = np.array([row[4] for row in list_store_all])

    r_hat = psrf(pd.Series(sigma_arr), n_chains)
    r_hat_cost = psrf_cost(cost_arr, n_chains)

    # Elliptical cone (align with Algorithm 2 QP)
    theta_samples = np.array([normalize_c(c) for c in cost_arr])
    cone = build_elliptical_cone(theta_samples, alpha=0.95, ridge=1e-8, shrink=0.0)
    width = cone_width(cone, mode='rms')
    c_actual_normalized = normalize_c(c_actual)
    covered = bool(in_elliptical_cone(c_actual_normalized, cone))
    mean_direction = cone["mu_hat"]
    angle_mean = np.arccos(np.clip(np.dot(mean_direction, c_actual_normalized), -1.0, 1.0))

    out = {
        "covered": covered,
        "meanangle": angle_mean,
        "semiangle": width,
        "Rhat": r_hat,
        "Rhat_cost": r_hat_cost,
        "iterations": n_iterations,
    }
    if save_runs:
        df = pd.DataFrame(list_store_all, columns=main_columns)
        out["df"] = df
    else:
        out["df"] = None
    return out



# utils.py
# Elliptical-cone credible regions on the sphere + run summarizer
# ---------------------------------------------------------------
# Dependencies: numpy, pandas, re
# You provide psrf_fn when calling summarize_df_runs (or add it here).

import re

EPS = 1e-12

# ----------------- small helpers -----------------
def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n if n > EPS else 1.0)

def _tangent_basis(mu_hat):
    """
    Orthonormal basis B (p x (p-1)) for the plane orthogonal to mu_hat.
    Simple Gram–Schmidt; robust for p up to a few hundred.
    """
    mu_hat = _normalize(mu_hat)
    p = mu_hat.size
    B_cols = []
    drop = int(np.argmax(np.abs(mu_hat)))  # improves conditioning
    for i in range(p):
        if i == drop:
            continue
        v = np.zeros(p); v[i] = 1.0
        # orthogonalize to mu_hat and previous columns
        v = v - (v @ mu_hat) * mu_hat
        for b in B_cols:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-10:
            B_cols.append(v / n)
    return np.column_stack(B_cols)  # (p, p-1)

def _log_map_coords(mu_hat, B, u):
    """Log map to tangent coords at mu_hat: u (unit) -> y in R^{p-1}."""
    u = _normalize(u)
    dot = float(np.clip(mu_hat @ u, -1.0, 1.0))
    phi = np.arccos(dot)  # geodesic angle
    if phi < 1e-12:
        return np.zeros(B.shape[1])
    tang = u - dot * mu_hat
    s = np.linalg.norm(tang)
    if s < 1e-12:
        return np.zeros(B.shape[1])
    w = tang / s
    return (B.T @ w) * phi

def _ensure_vec(x, idx=None, expected_len=None):
    """
    Return a 1D float numpy array from a cell that may be:
    - np.ndarray
    - list/tuple (numbers OR arrays) -> if list-of-arrays, pick idx-th (or first)
    - string "[1 2 3]" or "[1, 2, 3]"
    - string "array([..])"
    - string "[array([..]), array([..]), ...]" -> parse all arrays, pick idx-th (or best match)
    - numeric scalar (fallback)
    idx : if the cell encodes a sequence of vectors (list-of-arrays), pick idx-th (clamped)
    expected_len : if many candidates, prefer the one matching this length
    """
    # numpy array
    if isinstance(x, np.ndarray):
        return x.astype(float).ravel()

    # list/tuple
    if isinstance(x, (list, tuple)):
        # list of arrays?
        if len(x) > 0 and all(isinstance(el, (np.ndarray, list, tuple)) for el in x):
            k = 0 if idx is None else min(max(int(idx), 0), len(x) - 1)
            cand = np.asarray(x[k], dtype=float).ravel()
            if expected_len is not None and cand.size != expected_len:
                for el in x:
                    if np.size(el) == expected_len:
                        return np.asarray(el, dtype=float).ravel()
            return cand
        # plain list of numbers
        try:
            return np.asarray(x, dtype=float).ravel()
        except Exception:
            pass  # fall through

    # string variants
    if isinstance(x, str):
        s = x.strip()

        # CASE A: "[array(...), array(...), ...]"  (list of arrays as a string)
        if "array(" in s and s.count("array(") > 1:
            blocks = re.findall(r'array\(\s*(\[[^\]]*\])', s, flags=re.S)
            cands = []
            for b in blocks:
                bb = b.strip()[1:-1]  # drop [ ]
                bb = bb.replace(",", " ")
                arr = np.fromstring(bb, sep=" ")
                if arr.size > 0:
                    cands.append(arr)
            if len(cands) > 0:
                if idx is not None:
                    k = min(max(int(idx), 0), len(cands) - 1)
                    cand = cands[k]
                    if expected_len is not None and cand.size != expected_len:
                        for el in cands:
                            if el.size == expected_len:
                                return el.ravel()
                    return cand.ravel()
                if expected_len is not None:
                    for el in cands:
                        if el.size == expected_len:
                            return el.ravel()
                return cands[0].ravel()

        # CASE B: single "array([ ... ])"
        if s.startswith("array("):
            left = s.find("("); right = s.rfind(")")
            inner = s[left+1:right] if right > left else s
            m = re.search(r'\[([^\]]+)\]', inner, flags=re.S)
            numstr = m.group(1) if m else inner
            numstr = numstr.replace(",", " ")
            arr = np.fromstring(numstr, sep=" ")
            if arr.size > 0:
                return arr.ravel()

        # CASE C: bracketed numbers "[ ... ]" (commas optional)
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            numstr = s[1:-1].replace(",", " ")
            arr = np.fromstring(numstr, sep=" ")
            if arr.size > 0:
                return arr.ravel()

        # CASE D: generic numbers in any text
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        if len(nums) > 0:
            arr = np.array([float(t) for t in nums], dtype=float)
            if expected_len is not None and arr.size >= expected_len:
                return arr[:expected_len].ravel()
            return arr.ravel()

        # last resort: single float
        try:
            return np.array([float(s)], dtype=float)
        except Exception:
            raise ValueError(f"Could not parse vector from string: {x!r}")

    # numeric scalar fallback
    try:
        return np.array([float(x)], dtype=float)
    except Exception:
        raise ValueError(f"Unsupported cell type for vector: {type(x)} -> {x!r}")

# -------------- cone core (build, width, membership) --------------
def build_elliptical_cone(theta_samples, alpha=0.95, ridge=1e-8, shrink=0.0):
    """
    Build an elliptical credible region on the sphere from unit vectors.
    theta_samples: (T, p) array, rows are unit vectors (we re-normalize anyway)
    alpha: credibility (e.g., 0.95) used as empirical Mahalanobis quantile
    ridge: small diagonal added to covariance in the tangent plane
    shrink: optional scalar in [0,1]; 0=no shrink, 1=full identity shrinkage
    Returns {'mu_hat','B','y_mean','S_inv','kappa'} defining the cone.
    """
    theta = np.array(theta_samples, dtype=float)
    theta = theta / np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), EPS)

    # axis (mean direction)
    R = theta.sum(axis=0)
    mu_hat = _normalize(R if np.linalg.norm(R) > 1e-12 else theta[0])

    # tangent basis and tangent coords
    B = _tangent_basis(mu_hat)
    Y = np.array([_log_map_coords(mu_hat, B, u) for u in theta])
    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean

    # covariance in tangent plane (with ridge + optional shrink)
    Tm1 = max(len(Y) - 1, 1)
    S = (Yc.T @ Yc) / Tm1
    p1 = S.shape[0]
    if shrink > 0.0:
        tr = np.trace(S) / max(p1, 1)
        S = (1.0 - shrink) * S + shrink * tr * np.eye(p1)
    S = S + ridge * np.eye(p1)

    S_inv = np.linalg.pinv(S)
    # empirical Mahalanobis^2 distances => alpha-quantile
    d = np.einsum('bi,ij,bj->b', Yc, S_inv, Yc)
    kappa = float(np.quantile(d, alpha))

    return {"mu_hat": mu_hat, "B": B, "y_mean": y_mean, "S_inv": S_inv, "kappa": kappa}

def cone_width_rad(cone):
    """One-number cone width: worst principal half-angle (radians)."""
    S_inv, kappa = cone["S_inv"], float(cone["kappa"])
    evals_inv = np.linalg.eigvalsh(S_inv)
    evals_inv = np.maximum(evals_inv, 1e-15)
    phi_axes = np.sqrt(kappa / evals_inv)
    return float(phi_axes.max())

# --- add this in utils.py after cone_width_rad (or anywhere above summarize_df_runs) ---
def cone_width(cone, theta_samples=None, mode='rms', angle_q=0.95):
    """
    Compute a width scalar from a cone.
    mode in {'max','rms','eq','q'}:
      - 'max': worst principal half-angle.
      - 'rms': root-mean-square principal half-angle (smooth contraction).
      - 'eq' : determinant/volume-equivalent (geometric mean of principal angles).
      - 'q'  : empirical angle quantile of samples to mu_hat; needs theta_samples.
    Returns width in radians.
    """
    import numpy as np

    S_inv, kappa = cone["S_inv"], float(cone["kappa"])
    evals_inv = np.linalg.eigvalsh(S_inv)
    evals_inv = np.maximum(evals_inv, 1e-15)
    lam = 1.0 / evals_inv                      # eigenvalues of S (tangent covariance)
    phi_axes = np.sqrt(kappa * lam)            # principal half-angles (radians)

    mode = mode.lower()
    if mode == 'max':
        return float(phi_axes.max())
    if mode == 'rms':
        return float(np.sqrt(np.mean(phi_axes**2)))
    if mode == 'eq':
        return float(np.exp(np.mean(np.log(np.maximum(phi_axes, 1e-15)))))

    if mode == 'q':
        if theta_samples is None:
            raise ValueError("cone_width(mode='q') needs theta_samples.")
        V = np.array(theta_samples, dtype=float)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        mu = cone["mu_hat"]
        dots = np.clip(V @ mu, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.quantile(ang, angle_q))

    raise ValueError(f"Unknown width mode: {mode}")


def in_elliptical_cone(u, cone):
    """Membership test for unit vector u. Returns 0/1."""
    mu_hat, B = cone["mu_hat"], cone["B"]
    y_mean, S_inv, kappa = cone["y_mean"], cone["S_inv"], cone["kappa"]
    u = _normalize(_ensure_vec(u, expected_len=B.shape[0] + 1))
    y = _log_map_coords(mu_hat, B, u)
    yc = y - y_mean
    d = float(yc @ (S_inv @ yc))
    return int(d <= kappa + 1e-12)

def cone_membership_and_width(theta_samples, alpha, u, ridge=1e-8, shrink=0.0):
    """Build cone and return (width_rad, membership_0_1, cone)."""
    cone = build_elliptical_cone(theta_samples, alpha=alpha, ridge=ridge, shrink=shrink)
    return cone_width_rad(cone), in_elliptical_cone(u, cone), cone

# -------------- run-level summarizer (100 sims x 2 chains) --------------
def summarize_df_runs(
    df_runs,
    alpha=0.95,
    psrf_fn=None,
    out_csv_path=None,
    cost_col='Cost',
    sigma_col=None,
    test_vector_col='Optimal',  # ignored if test_vector_u is provided
    ridge=1e-8,
    shrink=0.0,
    test_vector_u=None,         # SAME u for all simulations (optional)
    test_vector_map=None,       # dict {sim_id: u} (optional; ignored if test_vector_u set)
    width_mode='rms',           # 'max' | 'rms' | 'eq' | 'q'
    angle_q=0.95,               # used only if width_mode='q'
):
    """
    Build one elliptical-cone credible region per simulation and summarize.

    df_runs columns:
      - 'Simulation' (simulation id)
      - 'Chain' (chain id)
      - optional 'Iter' (iteration index for stable ordering)
      - cost_col (vectors; may be strings like 'array([...])' or '[array(...), ...]')
      - sigma_col (scalars for PSRF; optional)
      - test_vector_col (vector to check coverage; ignored if test_vector_u provided)

    Returns a DataFrame with one row per Simulation:
      Simulation, alpha, width_mode, width_rad, width_deg, coverage, rhat
    Also writes CSV if out_csv_path is provided.
    """
    import numpy as np
    import pandas as pd

    # auto-detect sigma column if not provided
    if sigma_col is None:
        if 'Sigma' in df_runs.columns:
            sigma_col = 'Sigma'
        elif 'sigma' in df_runs.columns:
            sigma_col = 'sigma'
        else:
            sigma_col = None  # allowed; rhat will be NaN

    need_iter = 'Iter' if 'Iter' in df_runs.columns else None

    summaries = []
    for sim_id, g in df_runs.groupby('Simulation', sort=True):
        theta_list = []
        chains_sigma = []
        expected_len = None  # inferred from first parsed cost vector

        # per-chain: order rows, parse vectors, normalize
        for ch_id, gc in g.groupby('Chain', sort=True):
            gc = gc.sort_values(need_iter) if need_iter else gc.sort_index()

            # parse cost vectors; if a cell encodes a list-of-arrays string,
            # use idx=k so the k-th iteration picks the k-th array
            cost_cells = gc[cost_col].tolist()
            vecs = []
            for k, cell in enumerate(cost_cells):
                v = _ensure_vec(cell, idx=k, expected_len=expected_len)
                if expected_len is None:
                    expected_len = v.size
                vecs.append(v)

            V = np.vstack(vecs)  # (n_draws, p)
            # normalize rows defensively to unit length
            norms = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)
            V = V / norms
            theta_list.append(V)

            # collect sigma series for PSRF if available
            if sigma_col is not None and sigma_col in gc.columns:
                chains_sigma.append(np.asarray(gc[sigma_col].values, dtype=float))

        # stack all chains' theta samples
        theta_samples = np.vstack(theta_list)  # (n_chains * n_draws, p)

        # build cone
        cone = build_elliptical_cone(theta_samples, alpha=alpha, ridge=ridge, shrink=shrink)

        # width (choose your mode)
        if width_mode == 'q':
            width = cone_width(cone, theta_samples=theta_samples, mode='q', angle_q=angle_q)
        else:
            width = cone_width(cone, mode=width_mode)

        # ---------- COVERAGE / MEMBERSHIP ----------
        coverage = np.nan
        p = theta_samples.shape[1]

        if test_vector_u is not None:
            # SAME u for all simulations
            u = _ensure_vec(test_vector_u, expected_len=p)
            coverage = bool(in_elliptical_cone(u, cone))

        elif test_vector_map is not None and sim_id in test_vector_map:
            # u provided per-simulation
            u = _ensure_vec(test_vector_map[sim_id], expected_len=p)
            coverage = bool(in_elliptical_cone(u, cone))

        elif test_vector_col is not None and test_vector_col in g.columns:
            # take u from a column in the DataFrame (e.g., 'Optimal')
            ref_series = g[test_vector_col].dropna()
            if len(ref_series) > 0:
                u = _ensure_vec(ref_series.iloc[0], idx=0, expected_len=p)
                coverage = bool(in_elliptical_cone(u, cone))
        # -------------------------------------------

        # rhat via user-supplied psrf_fn on sigma (if available)
        rhat = np.nan
        if psrf_fn is not None and len(chains_sigma) >= 2:
            try:
                sigma_concat = np.concatenate(chains_sigma, axis=0)
                # psrf expects a "chains_list" (pandas Series) and n_chains
                rhat = float(psrf_fn(pd.Series(sigma_concat), n_chains=len(chains_sigma), split_chains=True))
            except Exception:
                rhat = np.nan  # keep going even if PSRF fails

        summaries.append({
            "Simulation": sim_id,
            "alpha": alpha,
            "width_mode": width_mode,
            "width_rad": width,
            "width_deg": width * 180/np.pi,
            "coverage": coverage,
            "rhat": rhat,
        })

    out = pd.DataFrame(summaries).sort_values("Simulation").reset_index(drop=True)
    if out_csv_path:
        out.to_csv(out_csv_path, index=False)
    return out
