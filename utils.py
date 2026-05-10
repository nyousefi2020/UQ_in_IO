import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import multivariate_normal, halfcauchy, expon, vonmises_fisher
from scipy.special import gamma
from scipy.optimize import linprog
from scipy.stats import vonmises_fisher

# -------------------------------
# Utility Functions
# -------------------------------

def normalize_c(c):
    return c / np.linalg.norm(c)


def get_cand(c, sigma, var_jump):
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


def get_cand_c_sym_gauss(c, var_c):
    """Symmetric isotropic Gaussian RW on sphere: c'=(c+eps)/||c+eps||, eps~N(0,var_c I)."""
    c = np.asarray(c, dtype=float).ravel()
    v = float(max(var_c, 1e-12))
    eps = np.random.normal(0.0, np.sqrt(v), size=c.size)
    return normalize_c(c + eps)


def get_cand_c_vmf(c, kappa):
    """vMF RW on sphere: c' ~ vMF(mu=c/||c||, kappa). Symmetric for fixed kappa."""
    mu = normalize_c(np.asarray(c, dtype=float).ravel())
    k = float(max(kappa, 1e-12))
    s = vonmises_fisher(mu, k).rvs(size=1)
    return normalize_c(np.asarray(s).ravel())


# -------------------------------
# kappa prior (used only when c_proposal="vmf")
# -------------------------------
KAPPA_PRIOR_SCALE = 10.0
KAPPA_MIN = 1e-4
KAPPA_MAX = 1.0e4


def log_prior_kappa(kappa):
    kappa = float(kappa)
    if kappa < KAPPA_MIN or kappa > KAPPA_MAX:
        return -np.inf
    return float(expon.logpdf(kappa, scale=KAPPA_PRIOR_SCALE))


def log_post_joint_vmf(dim, data, optimals, c, sigma_sq, prior_c_log, Y, kappa):
    """Unnormalized log posterior including prior on kappa (likelihood does not identify kappa)."""
    return float(posterior_log_var_log_sigma_sq(dim, data, optimals, sigma_sq, prior_c_log, Y=Y) + log_prior_kappa(kappa))


def vmf_propose_c(c_unit, kappa, rng=None):
    """c' ~ vMF(mu=c/||c||, kappa). Symmetric kernel for fixed kappa."""
    mu = normalize_c(np.asarray(c_unit, dtype=float).ravel())
    k = float(np.clip(kappa, KAPPA_MIN, KAPPA_MAX))
    dist = vonmises_fisher(mu, k)
    s = dist.rvs(size=1, random_state=rng)
    return normalize_c(np.asarray(s).ravel())


def _mcmc_three_step_update_vmf(
    sim_number,
    chain_id,
    prev_row,
    dim,
    data,
    Y_data,
    A_list,
    b_list,
    solver_cls,
    prior_density_log,
    var_log_kappa,
    var_eta,
):
    """One scan for vmf-mode: update kappa (prior-only), then c ~ vMF(c,kappa), then eta=log(sigma_sq)."""
    sid, cid = sim_number + 1, chain_id + 1
    x, c, sig, kappa = prev_row[2], prev_row[3], prev_row[4], prev_row[5]
    post = prev_row[6]
    prior = prev_row[7]

    acc_k = acc_c = acc_e = False

    # (1) kappa | prior only
    eta_k = float(np.log(kappa))
    eta_ks = eta_k + float(np.random.normal(0.0, np.sqrt(var_log_kappa)))
    kappa_s = float(np.exp(eta_ks))
    if KAPPA_MIN <= kappa_s <= KAPPA_MAX:
        log_a = log_prior_kappa(kappa_s) - log_prior_kappa(kappa)
        if np.log(np.random.uniform()) <= log_a:
            kappa = kappa_s
            post = log_post_joint_vmf(dim, data, x, c, sig, prior_density_log, Y_data, kappa)
            acc_k = True

    # (2) c | vMF(c, kappa)
    c_s = vmf_propose_c(c, kappa)
    x_s = solve_all_lp_parallel_warm(
        data, c_s, dim, x, A_list=A_list, b_list=b_list, solver_cls=solver_cls
    )
    if not any(v is None for v in x_s):
        post_s = log_post_joint_vmf(dim, data, x_s, c_s, sig, prior_density_log, Y_data, kappa)
        log_a = post_s - post
        if np.log(np.random.uniform()) <= log_a:
            c, x, post = c_s, x_s, post_s
            acc_c = True

    # (3) eta = log(sigma_sq)
    eta = float(np.log(sig))
    eta_s = eta + float(np.random.normal(0.0, np.sqrt(var_eta)))
    sig_s = float(np.exp(eta_s))
    if sig_s > 0.0:
        post_s = log_post_joint_vmf(dim, data, x, c, sig_s, prior_density_log, Y_data, kappa)
        log_a = post_s - post
        if np.log(np.random.uniform()) <= log_a:
            sig, post = sig_s, post_s
            acc_e = True

    acc_any = acc_k or acc_c or acc_e
    return [sid, cid, x, c, sig, kappa, post, prior, 1 if acc_any else 0], acc_k, acc_c, acc_e


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
    """R-hat for each component of the cost vector."""
    V = np.asarray(cost_vectors, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    dim = V.shape[1]
    rhats = np.zeros(dim)
    for j in range(dim):
        rhats[j] = psrf(pd.Series(V[:, j]), n_chains, split_chains=split_chains)
    return rhats

# -------------------------------
# LP Solver with Warm Start
# -------------------------------

# Single-phase tie-break (no second optimization): replace min c'x with min (c + ε w)'x for fixed w.
# Breaks alternate-optima ties deterministically. Smaller ε stays closer to the true argmin of c'x;
# if ε is too large, the minimizer can differ from the pure-LP one when c'x gaps are tiny.
_LP_TIE_BREAK_EPS = 1e-10


def _lp_effective_cost(c, dim):
    """Return c + ε w with deterministic w = (1,…,dim); same rule for Gurobi and HiGHS."""
    c = np.asarray(c, dtype=np.float64).ravel()
    w = np.arange(1, int(dim) + 1, dtype=np.float64)
    return c + _LP_TIE_BREAK_EPS * w


def _lp_params_standard(model):
    """
    Single Gurobi profile for all LPs in this module (dynamic and fixed A,b).
    Keeps Method/Presolve fixed: changing these (e.g. Method=-1, Presolve=1) can pick different
    optimal vertices on degenerate problems; we also perturb c with a tiny fixed w (see _lp_effective_cost).
    """
    model.Params.OutputFlag = 0
    model.Params.FeasibilityTol = 1e-6
    model.Params.OptimalityTol = 1e-6
    model.Params.Threads = 1  # parallelize across LPs in Python, not inside each model
    model.Params.Method = 1  # dual simplex — matches legacy behaviour
    model.Params.Presolve = 2


class LPSolver:
    def __init__(self, dim, env=None):
        self._dim = int(dim)
        self.model = gp.Model(env=env) if env is not None else gp.Model()
        _lp_params_standard(self.model)
        self.x = self.model.addMVar(shape=dim, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="x")
        self.constr = self.model.addConstr(0 <= 0)
        self.model.setObjective(0.0, GRB.MINIMIZE)

    def solve(self, A, b, c, x0=None):
        try:
            self.model.remove(self.constr)
            self.model.update()
            self.constr = self.model.addConstr(A @ self.x >= b)
            c_eff = _lp_effective_cost(c, self._dim)
            self.model.setObjective(c_eff @ self.x, GRB.MINIMIZE)
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



class LPSolverHiGHS:
    """
    Same problem as LPSolver: min c @ x  s.t.  A x >= b,  -1 <= x <= 1 (via perturbed c_eff; see _lp_effective_cost).
    Uses SciPy's HiGHS (no Gurobi). Warm start (x0) is ignored. Match backend with data generation.
    """
    def __init__(self, dim):
        self.dim = int(dim)

    def solve(self, A, b, c, x0=None):
        try:
            A = np.asarray(A, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64).ravel()
            c = np.asarray(c, dtype=np.float64).ravel()
            c_eff = _lp_effective_cost(c, self.dim)
            res = linprog(
                c_eff,
                A_ub=-A,
                b_ub=-b,
                bounds=[(-1.0, 1.0)] * self.dim,
                method="highs",
                options={"disp": False},
            )
            if res.success and res.x is not None:
                return np.asarray(res.x, dtype=np.float64)
            return None
        except Exception as e:
            print(f"HiGHS LP error: {e}")
            return None


def lp_solver_factory(backend):
    """backend: 'gurobi' | 'highs'"""
    b = (backend or "gurobi").lower()
    if b == "highs":
        return LPSolverHiGHS
    return LPSolver

# -------------------------------
# Batch LP solve with warm start (sequential; one reused solver)
# -------------------------------
# A_list, b_list optional pre-extraction to avoid repeated indexing in the MCMC loop.

def solve_all_lp_parallel_warm(data, c, dim, x_starts=None, A_list=None, b_list=None,
                                solver_pool=None, solver_cls=LPSolver):
    """Solve LP for each (A,b,y) in data sequentially, reusing one solver.
    solver_pool: if provided, uses solver_pool[0]; else builds one solver_cls(dim)."""
    n_pts = len(data)
    if A_list is None:
        A_list = [data[i][0] for i in range(n_pts)]
    if b_list is None:
        b_list = [data[i][1] for i in range(n_pts)]
    if solver_pool is not None and len(solver_pool) > 0:
        solver = solver_pool[0]
    else:
        solver = solver_cls(dim)
    return [
        solver.solve(A_list[i], b_list[i], c, x0=x_starts[i] if x_starts else None)
        for i in range(n_pts)
    ]



# -------------------------------
# Data Generator
# -------------------------------

def generate_lp_data_var(dim, n_constraints, n_points, c_star, sigma_sq, lp_backend="gurobi"):
    """Generate (A, b, y) data. sigma_sq = variance (y ~ N(x, sigma_sq*I)).
    lp_backend: 'gurobi' | 'highs' — use the same value as in MCMC so x* matches the forward map.

    Rows of A are random directions (Gaussian + row scaling), not all in the positive orthant, so
    halfspaces Ax >= b close the feasible set from many sides inside [-1,1]^d. A random interior
    reference z and margins give b = A z - margin so z is strictly feasible and rejection is rare."""
    solver_cls = lp_solver_factory(lp_backend)
    solver = solver_cls(dim)
    data_points = []
    for _ in range(n_points):
        while True:
            A = np.random.randn(n_constraints, dim)
            rn = np.linalg.norm(A, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-8)
            A = (A / rn) * np.random.uniform(0.7, 1.4, size=(n_constraints, 1))
            z = np.random.uniform(-0.72, 0.72, size=(dim, 1))
            margins = np.random.uniform(0.04, 0.28, size=(n_constraints, 1))
            b = A @ z - margins
            x = solver.solve(A, b, c_star)
            if x is not None:
                break
        y = np.random.multivariate_normal(x, sigma_sq * np.identity(dim))
        data_points.append((A, b, y))
    return data_points

def fo_lp(A, b, c):
    dim = c.shape[0]
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.FeasibilityTol = 1e-6
    x = model.addMVar(shape=dim, vtype=GRB.CONTINUOUS, lb=-1, ub=1)
    c_eff = _lp_effective_cost(c, dim)
    model.setObjective(c_eff @ x, GRB.MINIMIZE)
    model.addConstr(A @ x >= b)
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
        Y = np.array([data[i][2] for i in range(len(data))])
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
        Y = np.array([data[i][2] for i in range(len(data))])
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
# MCMC
# -------------------------------
# sigma_sq = variance. We propose in (c, eta) with eta=log(sigma_sq). Storage uses sigma_sq.
# Column 'Sigma' stores variance (sigma_sq) for backward compat.
# Adaptive Metropolis matches Algorithm1/QP/utils.py; only the forward map differs (batch LP vs batch QP).

def run_single_chain(chain_id, sim_number, data, dim, c_actual, sigma_sq_actual,
                    n_adaptive, n_iterations, prior_density_log, main_columns, correct,
                    lp_backend="gurobi", c_proposal="gauss_fullcov"):
    """
    Same MCMC structure as QP run_single_chain: get_cand_log_sigma_sq, cumulative adaptive tuning,
    then sampling. Forward map: solve_all_lp_parallel_warm on data entries (A,b,y), not closed-form QP.
    lp_backend: 'gurobi' | 'highs' — match data generation. LPSolverFixed is not used (vertex consistency).
    """
    n_pts = len(data)
    Y_data = np.array([data[i][2] for i in range(n_pts)])
    A_list = [data[i][0] for i in range(n_pts)]
    b_list = [data[i][1] for i in range(n_pts)]
    solver_cls = lp_solver_factory(lp_backend)

    # --- vMF mode: infer kappa (prior-only) + vMF(c,kappa) RW + RW on eta ---
    if c_proposal == "vmf":
        n_pts = len(data)
        Y_data = np.array([data[i][2] for i in range(n_pts)])
        A_list = [data[i][0] for i in range(n_pts)]
        b_list = [data[i][1] for i in range(n_pts)]

        c_init = normalize_c(np.random.uniform(-1, 1, dim))
        x_starts = [None] * n_pts
        x_init = solve_all_lp_parallel_warm(
            data, c_init, dim, x_starts, A_list=A_list, b_list=b_list, solver_cls=solver_cls
        )
        sigma_sq_init = float(halfcauchy.rvs(0, PRIOR_SIGMA_SCALE) ** 2)
        kappa_init = float(np.clip(expon.rvs(scale=KAPPA_PRIOR_SCALE), KAPPA_MIN, KAPPA_MAX))
        prior_init = 0
        post_init = log_post_joint_vmf(dim, data, x_init, c_init, sigma_sq_init, prior_density_log, Y_data, kappa_init)
        # Rows: Simulation, Chain, Optimal, Cost, Sigma, Kappa, Posterior, Prior, Accept
        list_store = [[sim_number + 1, chain_id + 1, x_init, c_init, sigma_sq_init, kappa_init, post_init, prior_init, 1]]

        var_eta = 0.5
        var_log_kappa = 0.25
        tau_eta = tau_kappa = 1.0
        accept_ratio = 0.0
        max_adaptive_rounds = 50
        adaptive_round = 0
        while (accept_ratio < 0.25 or accept_ratio > 0.40) and adaptive_round < max_adaptive_rounds:
            adaptive_round += 1
            acc_k_list, acc_c_list, acc_e_list = [], [], []
            for _i in range(1, n_adaptive):
                prev = list_store[-1]
                new_row, ak, ac, ae = _mcmc_three_step_update_vmf(
                    sim_number,
                    chain_id,
                    prev,
                    dim,
                    data,
                    Y_data,
                    A_list,
                    b_list,
                    solver_cls,
                    prior_density_log,
                    var_log_kappa,
                    var_eta,
                )
                list_store.append(new_row)
                acc_k_list.append(ak)
                acc_c_list.append(ac)
                acc_e_list.append(ae)
            last_row = list_store[-1]
            eta_arr = np.log(np.array([list_store[i][4] for i in range(len(list_store))], dtype=float))
            var_eta_emp = max(float(np.var(eta_arr)), 1e-8)
            acc_k = float(np.mean(acc_k_list))
            acc_c = float(np.mean(acc_c_list))
            acc_e = float(np.mean(acc_e_list))
            accept_ratio = float(np.mean([list_store[i][8] for i in range(len(list_store))]))
            tau_eta *= 0.5 if acc_e < 0.22 else (3.0 if acc_e > 0.42 else 1.0)
            var_eta = tau_eta * var_eta_emp
            lk_arr = np.log(np.array([list_store[i][5] for i in range(len(list_store))], dtype=float))
            var_lk_emp = max(float(np.var(lk_arr)), 1e-8)
            tau_kappa *= 0.5 if acc_k < 0.22 else (3.0 if acc_k > 0.42 else 1.0)
            var_log_kappa = tau_kappa * var_lk_emp

        list_store = [last_row]
        for _i in range(1, n_iterations):
            prev = list_store[-1]
            new_row, _, _, _ = _mcmc_three_step_update_vmf(
                sim_number,
                chain_id,
                prev,
                dim,
                data,
                Y_data,
                A_list,
                b_list,
                solver_cls,
                prior_density_log,
                var_log_kappa,
                var_eta,
            )
            list_store.append(new_row)
        return list_store

    c_init = normalize_c(np.random.uniform(-1, 1, dim))
    x_starts = [None] * n_pts
    x_init = solve_all_lp_parallel_warm(
        data, c_init, dim, x_starts, A_list=A_list, b_list=b_list, solver_cls=solver_cls
    )
    sigma_sq_init = halfcauchy.rvs(0, PRIOR_SIGMA_SCALE) ** 2
    jump_scale = np.diag([10.0] * dim + [0.5])  # used when c_proposal="gauss_fullcov"
    var_c = 0.10  # used when c_proposal="sym_gauss"
    var_eta = 0.50  # used when c_proposal in {"sym_gauss","vmf"} (RW on eta=log(sigma_sq))
    if c_proposal == "vmf":
        # Infer kappa (prior-only) to adapt vMF concentration
        kappa_init = float(np.clip(expon.rvs(scale=KAPPA_PRIOR_SCALE), KAPPA_MIN, KAPPA_MAX))
        post_init = log_post_joint_vmf(dim, data, x_init, c_init, sigma_sq_init, prior_density_log, Y_data, kappa_init)
    else:
        kappa_init = None
        post_init = posterior_log_var_log_sigma_sq(dim, data, x_init, sigma_sq_init, prior_density_log, Y=Y_data)
    prior_init = 0
    if c_proposal == "vmf":
        # Rows: Simulation, Chain, Optimal, Cost, Sigma, Kappa, Posterior, Prior, Accept
        list_store = [[sim_number+1, chain_id+1, x_init, c_init, sigma_sq_init, kappa_init, post_init, prior_init, 1]]
    else:
        list_store = [[sim_number+1, chain_id+1, x_init, c_init, sigma_sq_init, post_init, prior_init, 1]]

    tau = 1
    accept_ratio = 0
    var_log_kappa = 0.25  # only used when c_proposal="vmf"
    max_adaptive_rounds = 50  # safety cap to prevent infinite loops (same as Algorithm1/QP)
    adaptive_round = 0
    while (accept_ratio < 0.25 or accept_ratio > 0.40) and adaptive_round < max_adaptive_rounds:
        adaptive_round += 1
        for i in range(1, n_adaptive):
            prev_c, prev_sigma_sq = list_store[i-1][3], list_store[i-1][4]
            prev_post = list_store[i-1][5] if c_proposal != "vmf" else list_store[i-1][6]

            if c_proposal == "sym_gauss":
                cand_c = get_cand_c_sym_gauss(prev_c, var_c)
                eta = float(np.log(prev_sigma_sq))
                eta_s = eta + float(np.random.normal(0.0, np.sqrt(var_eta)))
                cand_sigma_sq = float(np.exp(eta_s))
            elif c_proposal == "vmf":
                # (1) kappa update (prior-only MH on log kappa)
                prev_kappa = list_store[i-1][5]
                lk = float(np.log(prev_kappa))
                lk_s = lk + float(np.random.normal(0.0, np.sqrt(var_log_kappa)))
                cand_kappa = float(np.exp(lk_s))
                kappa = prev_kappa
                post = prev_post
                x = list_store[i-1][2]
                c = prev_c
                sig = prev_sigma_sq
                if KAPPA_MIN <= cand_kappa <= KAPPA_MAX:
                    log_a_k = log_prior_kappa(cand_kappa) - log_prior_kappa(prev_kappa)
                    if np.log(np.random.uniform()) <= log_a_k:
                        kappa = cand_kappa
                        post = log_post_joint_vmf(dim, data, x, c, sig, prior_density_log, Y_data, kappa)

                # (2) c update: vMF(c, kappa)
                cand_c = get_cand_c_vmf(c, kappa)
                # (3) eta update
                eta = float(np.log(sig))
                eta_s = eta + float(np.random.normal(0.0, np.sqrt(var_eta)))
                cand_sigma_sq = float(np.exp(eta_s))
            else:
                # default: Gaussian full-cov RW in (c, eta)
                cand_c, cand_sigma_sq = get_cand_log_sigma_sq(prev_c, prev_sigma_sq, jump_scale)
            x_starts = list_store[i-1][2]
            cand_mean = solve_all_lp_parallel_warm(
                data, cand_c, dim, x_starts, A_list=A_list, b_list=b_list, solver_cls=solver_cls
            )
            if any(x is None for x in cand_mean):
                if c_proposal == "vmf":
                    list_store.append([*list_store[i-1][:8], 0])
                else:
                    list_store.append([*list_store[i-1][:7], 0])
                continue
            if c_proposal == "vmf":
                # Use kappa after possible MH update above
                cand_post = log_post_joint_vmf(dim, data, cand_mean, cand_c, cand_sigma_sq, prior_density_log, Y_data, kappa)
            else:
                cand_post = posterior_log_var_log_sigma_sq(dim, data, cand_mean, cand_sigma_sq, prior_density_log, Y=Y_data)
            cand_prior = 0
            log_accept_ratio = cand_post - prev_post + (cand_prior - list_store[i-1][6] if (correct and c_proposal != "vmf") else 0)
            if np.log(np.random.uniform()) <= log_accept_ratio:
                if c_proposal == "vmf":
                    list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, kappa, cand_post, cand_prior, 1])
                else:
                    list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, cand_post, cand_prior, 1])
            else:
                if c_proposal == "vmf":
                    list_store.append([*list_store[i-1][:8], 0])
                else:
                    list_store.append([*list_store[i-1][:7], 0])
        last_row = list_store[-1]
        accept_ratio = np.mean([list_store[i][-1] for i in range(len(list_store))])
        tau *= 0.5 if accept_ratio < 0.22 else (3 if accept_ratio > 0.42 else 1)
        if c_proposal == "gauss_fullcov":
            cost_arr = np.array([list_store[i][3] for i in range(len(list_store))])
            eta_arr = np.log(np.array([list_store[i][4] for i in range(len(list_store))]))
            combined = np.column_stack([cost_arr, eta_arr])
            cov_mat = np.cov(combined, rowvar=False)
            cov_mat = (cov_mat + cov_mat.T) / 2 + np.eye(dim+1) * 1e-8
            jump_scale = tau * (cov_mat + np.eye(dim+1))
        elif c_proposal == "sym_gauss":
            var_c = float(np.clip(tau * var_c, 1e-8, 10.0))
            var_eta = float(np.clip(tau * var_eta, 1e-8, 10.0))
        else:  # vmf
            var_eta = float(np.clip(tau * var_eta, 1e-8, 10.0))
            var_log_kappa = float(np.clip(tau * var_log_kappa, 1e-8, 10.0))

    list_store = [last_row]
    for i in range(1, n_iterations):
        prev_c, prev_sigma_sq = list_store[i-1][3], list_store[i-1][4]
        prev_post = list_store[i-1][5] if c_proposal != "vmf" else list_store[i-1][6]
        if c_proposal == "sym_gauss":
            cand_c = get_cand_c_sym_gauss(prev_c, var_c)
            eta = float(np.log(prev_sigma_sq))
            eta_s = eta + float(np.random.normal(0.0, np.sqrt(var_eta)))
            cand_sigma_sq = float(np.exp(eta_s))
        elif c_proposal == "vmf":
            prev_kappa = list_store[i-1][5]
            lk = float(np.log(prev_kappa))
            lk_s = lk + float(np.random.normal(0.0, np.sqrt(var_log_kappa)))
            cand_kappa = float(np.exp(lk_s))
            kappa = prev_kappa
            post = prev_post
            x = list_store[i-1][2]
            c = prev_c
            sig = prev_sigma_sq
            if KAPPA_MIN <= cand_kappa <= KAPPA_MAX:
                log_a_k = log_prior_kappa(cand_kappa) - log_prior_kappa(prev_kappa)
                if np.log(np.random.uniform()) <= log_a_k:
                    kappa = cand_kappa
                    post = log_post_joint_vmf(dim, data, x, c, sig, prior_density_log, Y_data, kappa)

            cand_c = get_cand_c_vmf(c, kappa)
            eta = float(np.log(sig))
            eta_s = eta + float(np.random.normal(0.0, np.sqrt(var_eta)))
            cand_sigma_sq = float(np.exp(eta_s))
        else:
            cand_c, cand_sigma_sq = get_cand_log_sigma_sq(prev_c, prev_sigma_sq, jump_scale)
        x_starts = list_store[i-1][2]
        cand_mean = solve_all_lp_parallel_warm(
            data, cand_c, dim, x_starts, A_list=A_list, b_list=b_list, solver_cls=solver_cls
        )
        if any(x is None for x in cand_mean):
            if c_proposal == "vmf":
                list_store.append([*list_store[i-1][:8], 0])
            else:
                list_store.append([*list_store[i-1][:7], 0])
            continue
        if c_proposal == "vmf":
            cand_post = log_post_joint_vmf(dim, data, cand_mean, cand_c, cand_sigma_sq, prior_density_log, Y_data, kappa)
        else:
            cand_post = posterior_log_var_log_sigma_sq(dim, data, cand_mean, cand_sigma_sq, prior_density_log, Y=Y_data)
        cand_prior = 0
        log_accept_ratio = cand_post - prev_post + (cand_prior - list_store[i-1][6] if (correct and c_proposal != "vmf") else 0)
        if np.log(np.random.uniform()) <= log_accept_ratio:
            if c_proposal == "vmf":
                list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, kappa, cand_post, cand_prior, 1])
            else:
                list_store.append([sim_number+1, chain_id+1, cand_mean, cand_c, cand_sigma_sq, cand_post, cand_prior, 1])
        else:
            if c_proposal == "vmf":
                list_store.append([*list_store[i-1][:8], 0])
            else:
                list_store.append([*list_store[i-1][:7], 0])
    return list_store


def run_mcmc_var(sim_number, data, dim, c_actual, sigma_sq_actual, n_chains, n_adaptive, n_iterations,
                 prior_density_log, main_columns, correct=False, save_runs=True,
                 lp_backend="gurobi", c_proposal="gauss_fullcov"):
    """sigma_sq_actual = variance (y ~ N(x, sigma_sq*I)). save_runs=False skips building df for efficiency.
    lp_backend: 'gurobi' | 'highs' — SciPy HiGHS is often much faster (no license); match data generation."""
    all_chains = [
        run_single_chain(
            chain,
            sim_number,
            data,
            dim,
            c_actual,
            sigma_sq_actual,
            n_adaptive,
            n_iterations,
            prior_density_log,
            main_columns,
            correct,
            lp_backend,
            c_proposal,
        )
        for chain in range(n_chains)
    ]

    list_store_all = [row for chain_data in all_chains for row in chain_data]
    cost_arr = np.array([np.asarray(row[3]).ravel() for row in list_store_all])
    sigma_arr = np.array([row[4] for row in list_store_all])

    r_hat = psrf(pd.Series(sigma_arr), n_chains)
    r_hat_cost = psrf_cost(cost_arr, n_chains)

    # Elliptical cone (align with QP)
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
