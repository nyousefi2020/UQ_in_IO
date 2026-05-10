import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import multivariate_normal,halfcauchy,vonmises_fisher,gamma


def fo_lp(A,b,c):
    """
    Solves forward problem  <<min cx s.t Ax>=b, -1<=x<=1>>
    
    Parameters:
    - A (matrix)
    - b (vector)
    - c (vector)    
    
    Returns: 
    - x: optimal solution
    """
    try:
        m,n = A.shape
        model = gp.Model()
        model.Params.LogToConsole = 0
        x = model.addMVar(shape=n, lb=-1, ub=1, vtype=GRB.CONTINUOUS)
        model.setObjective(c@x, GRB.MINIMIZE)
        model.addConstr(A@x>=b)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            return x.X 
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def io_lp(A,b,x,tol=1e-2):
    m, n = A.shape
    G = []
    # Ax >= b
    for i in range(m):
        if np.isclose(A[i] @ x, b[i], atol=tol):
            G.append(A[i])

    # Box constraints
    for j in range(n):
        if np.isclose(x[j], 1, atol=tol):
            g = np.zeros(n)
            g[j] = 1
            G.append(g)
        elif np.isclose(x[j], -1, atol=tol):
            g = np.zeros(n)
            g[j] = -1
            G.append(g)
    if len(G)==0:
        return None
    c = np.mean(G, axis=0)
    return normalize_c(c)


def solve_io_all(data):
    """
    Finds inversely optimized c vectors by solving IO(u,x) for all u values when data is of form (u,y).

    Parameters:
    - data (list): list of (A,b,y,c)

    Returns:
    - cost_vectors: list of inversely optimized cost vectors
    """
    cost_vectors = [io_lp(data_point[0],data_point[1],data_point[2]) for data_point in data]
    return cost_vectors


def normalize_c(c): 
    """
    Normalizes a given vector.

    Parameters:
    - c (vector)

    Returns:
    - a normalized c
    """
    return c/np.linalg.norm(c)


def generate_lp_data_var(dim, n_constraints, n_points, c_star, kappa):
    """
    Generates data points of format (u,y), u={A,b}

    Parameters:
    - dim (int)
    - n_constraints (int)
    - n_points (int)
    - c_star (vector)
    - kappa (float>0): concentration parameter 

    Returns:
    - data_points: list of data points of format (A,b,y,c)
    """
    data_points = []
    for i in range(n_points):
        while True: # so that it generates the number of points we want
            # Generate a feasible polytope inside [-1,1]^dim.
            # Match Algorithm1/LP: random halfspaces with a strictly-feasible interior point z.
            A = np.random.randn(n_constraints, dim)
            rn = np.linalg.norm(A, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-8)
            A = (A / rn) * np.random.uniform(0.7, 1.4, size=(n_constraints, 1))

            # Pick an interior point well inside the box and set b so z is feasible with margin.
            z = np.random.uniform(-0.72, 0.72, size=(dim,))
            margins = np.random.uniform(0.04, 0.28, size=(n_constraints,))
            b = A @ z - margins
            c = vonmises_fisher.rvs(c_star, kappa)
            x = fo_lp(A,b,c[0])            
            if x is not None:
                break
        y = x
        data_points.append((A,b,y,c[0]))
    return data_points


def posterior_log_var(dim, c_optimals, c_cand, kappa, prior):
    """
    Finds log of unnormalized posterior density for general case.

    Parameters: 
    - dim (int)
    - c_optimals (list of vectors): list of inversely optimized cost vectors (c's)
    - c_cand (vector): mean direction candidate vector
    - kappa (float)
    - prior (float): prior probability 

    Returns:
    - log of posterior density 
    """
    c_optimals = [c for c in c_optimals if c is not None]
    if len(c_optimals) == 0:
        raise ValueError("All c_optimals vectors are NaN or invalid. ")

    posterior = sum(vonmises_fisher.logpdf(c_optimals, c_cand, kappa))
    prior_kappa = gamma.logpdf(kappa, a=2, scale=1.5)
    return posterior + prior + prior_kappa


def get_cand_c(c,var_jump):
    """
    Generates a random cost vector from jumping dist.

    Parameters:
    - c (vector): current cost vector
    - var_jump (matrix): variance matrix of the jumping distribution

    Returns:
    - candidate cost vector (normalized)
    """
    c_new = np.random.multivariate_normal(c, var_jump)
    return normalize_c(c_new)


def get_cand_c_sym_gauss(c, var_c):
    """
    Symmetric isotropic Gaussian random-walk on the sphere:
      c' = (c + eps) / ||c + eps||,  eps ~ N(0, var_c I).
    """
    c = np.asarray(c, dtype=float).ravel()
    v = float(max(var_c, 1e-12))
    eps = np.random.normal(0.0, np.sqrt(v), size=c.size)
    return normalize_c(c + eps)


def get_cand_c_vmf(c, kappa):
    """
    vMF random-walk on the sphere centered at current c:
      c' ~ vMF(mu=c/||c||, kappa).
    Symmetric kernel in (c, c') for fixed kappa.
    """
    mu = normalize_c(np.asarray(c, dtype=float).ravel())
    k = float(max(kappa, 1e-12))
    s = vonmises_fisher.rvs(mu, k, size=1)
    return normalize_c(np.asarray(s).ravel())


def get_kappa(kappa, var):
    """
    Generates a positive kappa using a log-normal random walk.

    Parameters:
    - kappa (positive): current kappa value
    - var (float): variance of the log-space jumping distribution

    Returns:
    - candidate kappa (positive)
    """
    kappa_safe = max(float(kappa), 1e-12)
    log_kappa = np.log(kappa_safe)
    cand_log = np.random.normal(log_kappa, np.sqrt(var))
    return float(np.exp(cand_log))


def psrf(chains_list, n_chains, split_chains=True):
    """
    Calculates the potential scale reduction factor (PSRF) for a set of MCMC chains

    Parameters: 
    - chains_list: series containing all chains
    - n_chains (int): number of chains
    - split_chains (0/1): a boolean indicating whether to split each chain in half before calculating the PSRF (default=True)
    
    Returns
    - psrf (float): PSRF, if <1.2 ok, close to 1 best, >1.2 bad
    """
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
    var_theta = (n_samples - 1) / n_samples * W + 1 / n_samples * B
    psrf = np.sqrt(var_theta / W)
    return psrf


def psrf_cost(cost_vectors, n_chains, split_chains=True):
    """
    R-hat for each component of the cost vector.

    Parameters:
    - cost_vectors: array (n_samples, dim) or list of vectors; rows ordered by chain (chain0, chain1, ...)
    - n_chains (int)
    - split_chains (bool): same as psrf

    Returns:
    - rhat_per_component: array of shape (dim,) — R-hat for c_1, c_2, ..., c_d
    """
    V = np.asarray(cost_vectors, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    elif V.ndim == 2 and V.shape[1] == 1:
        pass
    dim = V.shape[1]
    rhats = np.zeros(dim)
    for j in range(dim):
        rhats[j] = psrf(pd.Series(V[:, j]), n_chains, split_chains=split_chains)
    return rhats


# ============= Elliptical Cone Functions (from Algorithm2/QP) =============
EPS = 1e-12

def _normalize(v):
    """Normalize a vector to unit length."""
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
    - numeric scalar (fallback)
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
            pass
    
    # string variants
    if isinstance(x, str):
        s = x.strip()
        # bracketed numbers "[ ... ]" (commas optional)
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            numstr = s[1:-1].replace(",", " ")
            arr = np.fromstring(numstr, sep=" ")
            if arr.size > 0:
                return arr.ravel()
        # generic numbers in any text
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
# ============================================================================


# To refer: main_columns = ['Simulation','Chain','x','Cost','Kappa','Posterior','Prior','Accept'] 
def run_mcmc_var(
    sim_number,
    c_optimals,
    dim,
    c_actual,
    kappa_actual,
    n_chains,
    n_adaptive,
    n_iterations,
    prior_density_log,
    main_columns,
    correct=False,
    c_proposal="gauss_fullcov",
):
    """
    Runs mcmc Algorithm 2, including adaptive and sampling phases, for general case. 

    Parameters:
    - sim_number (int)
    - c_optimals (list): list of inversely optimized cost vectors (c's)
    - dim (int)
    - c_actual (vector): actual mean direction
    - kappa_actual (float)
    - n_chains (int)
    - n_adaptive (int)
    - n_iterations (int)
    - prior_density_log (float)
    - main_columns (list): list of column names for saving simulations
    - correct (T/F): True if we correct for prior density 

    Returns:
    - desired outcomes (coverage, etc.) and a df of the runs
    """
    list_store_all = []  # collects all chains in a simulation
    for chain in range(n_chains):
        c_init = normalize_c(np.random.uniform(-1,1,dim)) 
        kappa_init = gamma.rvs(a=2, scale=1.5) # random from a gamma dist
        # c proposal parameters
        jump_scale = 10*np.identity(dim) # cov of Gaussian full-cov jumping dist
        var_c = 0.10  # scalar variance for symmetric Gaussian RW-on-sphere
        var_kappa = 1.0  # variance for log(kappa) random-walk
        post_init = posterior_log_var(dim, c_optimals, c_init, kappa_init, prior_density_log)
        prior_init = 0
        list_store = []
        list_store.append([sim_number+1, chain+1, 'x', c_init, kappa_init, post_init, prior_init, 1])
        
        # ADAPTIVE
        tau_c = 1.0
        tau_kappa = 1.0
        accept_ratio = 0
        max_adaptive_rounds = 50  # Maximum number of adaptive rounds to prevent infinite loops
        adaptive_round = 0
        while (accept_ratio<0.25 or accept_ratio>0.40) and adaptive_round < max_adaptive_rounds:
            adaptive_round += 1
            acc_c_list, acc_k_list = [], []
            for i in range(1,n_adaptive):
                prev_c = list_store[i-1][3]
                prev_kappa = list_store[i-1][4]
                prev_post = list_store[i-1][5]

                # (1) kappa RW in log-space (MH includes Jacobian term)
                cand_kappa = get_kappa(prev_kappa, var_kappa)
                cand_post_k = posterior_log_var(dim, c_optimals, prev_c, cand_kappa, prior_density_log)
                log_jacobian = np.log(cand_kappa) - np.log(prev_kappa)
                log_a_k = cand_post_k - prev_post + log_jacobian
                if np.log(np.random.uniform()) <= log_a_k:
                    kappa = cand_kappa
                    post_k = cand_post_k
                    acc_k = 1
                else:
                    kappa = prev_kappa
                    post_k = prev_post
                    acc_k = 0

                # (2) c proposal given current kappa (symmetric => no q-ratio)
                if c_proposal == "sym_gauss":
                    cand_c = get_cand_c_sym_gauss(prev_c, var_c)
                elif c_proposal == "vmf":
                    cand_c = get_cand_c_vmf(prev_c, kappa)
                else:
                    cand_c = get_cand_c(prev_c, jump_scale)

                cand_post_c = posterior_log_var(dim, c_optimals, cand_c, kappa, prior_density_log)
                log_a_c = cand_post_c - post_k
                if np.log(np.random.uniform()) <= log_a_c:
                    c = cand_c
                    post = cand_post_c
                    acc_c = 1
                else:
                    c = prev_c
                    post = post_k
                    acc_c = 0

                cand_prior = 0
                acc_any = 1 if (acc_k or acc_c) else 0
                list_store.append([sim_number+1, chain+1, 'x', c, kappa, post, cand_prior, acc_any])
                acc_c_list.append(acc_c)
                acc_k_list.append(acc_k)
        
            df_adaptive = pd.DataFrame(list_store, columns = main_columns)
            last_row = df_adaptive.iloc[len(df_adaptive)-1].tolist()
            accept_ratio = df_adaptive.Accept.mean()
            acc_c = float(np.mean(acc_c_list)) if len(acc_c_list) else 0.0
            acc_k = float(np.mean(acc_k_list)) if len(acc_k_list) else 0.0
            tau_c *= 0.5 if acc_c < 0.22 else (3.0 if acc_c > 0.42 else 1.0)
            tau_kappa *= 0.5 if acc_k < 0.22 else (3.0 if acc_k > 0.42 else 1.0)
            if c_proposal == "gauss_fullcov":
                df_adaptive[[f"c_{i}" for i in range(1,dim+1)]] = pd.DataFrame(df_adaptive['Cost'].tolist(), index= df_adaptive.index)
                cov_mat = np.cov(df_adaptive.iloc[:, -dim:].values, rowvar=False)
                cov_mat = (cov_mat + cov_mat.T) / 2
                cov_mat = cov_mat + np.eye(dim) * 1e-8
                jump_scale = tau_c*(cov_mat + np.eye(dim))
            elif c_proposal == "sym_gauss":
                var_c = float(np.clip(tau_c * var_c, 1e-8, 10.0))
            # adapt log-kappa RW variance using kappa-acceptance
            var_kappa = float(np.clip(tau_kappa * var_kappa, 1e-6, 1e2))
        
        # SAMPLING
        list_store = []
        list_store.append(last_row)
        
        for i in range(1,n_iterations):
            prev_c = list_store[i-1][3]
            prev_kappa = list_store[i-1][4]
            prev_post = list_store[i-1][5]

            # (1) kappa update
            cand_kappa = get_kappa(prev_kappa, var_kappa)
            cand_post_k = posterior_log_var(dim, c_optimals, prev_c, cand_kappa, prior_density_log)
            log_jacobian = np.log(cand_kappa) - np.log(prev_kappa)
            log_a_k = cand_post_k - prev_post + log_jacobian
            if np.log(np.random.uniform()) <= log_a_k:
                kappa = cand_kappa
                post_k = cand_post_k
                acc_k = 1
            else:
                kappa = prev_kappa
                post_k = prev_post
                acc_k = 0

            # (2) c update
            if c_proposal == "sym_gauss":
                cand_c = get_cand_c_sym_gauss(prev_c, var_c)
            elif c_proposal == "vmf":
                cand_c = get_cand_c_vmf(prev_c, kappa)
            else:
                cand_c = get_cand_c(prev_c, jump_scale)
            cand_post_c = posterior_log_var(dim, c_optimals, cand_c, kappa, prior_density_log)
            log_a_c = cand_post_c - post_k
            if np.log(np.random.uniform()) <= log_a_c:
                c = cand_c
                post = cand_post_c
                acc_c = 1
            else:
                c = prev_c
                post = post_k
                acc_c = 0

            cand_prior = 0
            acc_any = 1 if (acc_k or acc_c) else 0
            list_store.append([sim_number+1, chain+1, 'x', c, kappa, post, cand_prior, acc_any])
        # list_store = list_store[len(list_store) // 2:] # uncomment if want to discard half of data
        list_store_all.extend(list_store)
    df = pd.DataFrame(list_store_all, columns = main_columns) 

    # Use elliptical cone functions (like Algorithm2/QP)
    r_hat = psrf(df.Kappa, n_chains)
    cost_arr = np.array([np.asarray(c).ravel() for c in df.Cost.values])
    r_hat_cost = psrf_cost(cost_arr, n_chains)

    # Extract cost vectors (theta samples) - normalize them
    theta_samples = np.array([normalize_c(c) for c in df.Cost.values])
    
    # Build elliptical cone
    cone = build_elliptical_cone(theta_samples, alpha=0.95, ridge=1e-8, shrink=0.0)
    
    # Compute width using 'rms' mode (root-mean-square principal half-angle, for paper)
    width = cone_width(cone, mode='rms')
    
    # Check coverage using elliptical cone membership test
    c_actual_normalized = normalize_c(c_actual)
    covered = bool(in_elliptical_cone(c_actual_normalized, cone))
    
    # Compute mean angle from mean direction to actual
    mean_direction = cone["mu_hat"]
    angle_mean = np.arccos(np.clip(np.dot(mean_direction, c_actual_normalized), -1.0, 1.0))
    return {
        "covered": covered,
        "meanangle": angle_mean, 
        "semiangle": width,
        "Rhat": r_hat,
        "Rhat_cost": r_hat_cost,
        "iterations": n_iterations, 
        "df": df
        }


# ============= Summarize DF Runs (from Algorithm2/QP) =============
def summarize_df_runs(
    df_runs,
    alpha=0.95,
    psrf_fn=None,
    out_csv_path=None,
    cost_col='Cost',
    kappa_col=None,  # Algorithm2 uses Kappa
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
    
    For Algorithm2: uses Kappa column instead of Sigma.

    df_runs columns:
      - 'Simulation' (simulation id)
      - 'Chain' (chain id)
      - optional 'Iter' (iteration index for stable ordering)
      - cost_col (vectors; may be strings like 'array([...])' or '[array(...), ...]')
      - kappa_col (scalars for PSRF; optional)
      - test_vector_col (vector to check coverage; ignored if test_vector_u provided)

    Returns a DataFrame with one row per Simulation:
      Simulation, alpha, width_mode, width_rad, width_deg, coverage, rhat
    Also writes CSV if out_csv_path is provided.
    """
    # auto-detect kappa column if not provided (Algorithm2 uses Kappa instead of Sigma)
    if kappa_col is None:
        if 'Kappa' in df_runs.columns:
            kappa_col = 'Kappa'
        elif 'kappa' in df_runs.columns:
            kappa_col = 'kappa'
        else:
            kappa_col = None  # allowed; rhat will be NaN

    need_iter = 'Iter' if 'Iter' in df_runs.columns else None

    summaries = []
    for sim_id, g in df_runs.groupby('Simulation', sort=True):
        theta_list = []
        chains_kappa = []
        expected_len = None  # inferred from first parsed cost vector

        # per-chain: order rows, parse vectors, normalize
        # IMPORTANT: Sort chains by Chain ID to ensure consistent ordering for R-hat calculation
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

            # collect kappa series for PSRF if available (Algorithm2 uses Kappa)
            # Store as tuple (chain_id, kappa_array) to ensure proper ordering
            if kappa_col is not None and kappa_col in gc.columns:
                chains_kappa.append((ch_id, np.asarray(gc[kappa_col].values, dtype=float)))

        # stack all chains' theta samples
        theta_samples = np.vstack(theta_list)  # (n_chains * n_draws, p)

        # build cone & width
        cone = build_elliptical_cone(theta_samples, alpha=alpha, ridge=ridge, shrink=shrink)
        width_rad = cone_width(cone, theta_samples=theta_samples, mode=width_mode, angle_q=angle_q)
        width_deg = width_rad * 180.0 / np.pi

        # test vector for coverage
        if test_vector_u is not None:
            u = test_vector_u
        elif test_vector_map is not None:
            u = test_vector_map.get(sim_id, None)
        else:
            u = g[test_vector_col].iloc[0] if test_vector_col in g.columns else None

        coverage = None
        if u is not None:
            coverage = bool(in_elliptical_cone(u, cone))

        # Rhat
        rhat = np.nan
        if psrf_fn is not None and len(chains_kappa) > 0:
            chains_kappa = sorted(chains_kappa, key=lambda t: t[0])
            kappa_series = np.concatenate([ck for _, ck in chains_kappa])
            rhat = psrf_fn(pd.Series(kappa_series), len(chains_kappa))

        summaries.append({
            "Simulation": sim_id,
            "alpha": alpha,
            "width_mode": width_mode,
            "width_rad": width_rad,
            "width_deg": width_deg,
            "coverage": coverage,
            "rhat": rhat
        })

    out = pd.DataFrame(summaries)
    if out_csv_path:
        out.to_csv(out_csv_path, index=False)
    return out
# ============================================================================


def find_n_iterations_var(dim, n_constraints, n_points, c_actual, kappa_actual, n_chains_find, n_adaptive, n_iterations, main_columns, prior_density_log, chunck_iterations):
    """
    Finds suitable number of iterations for the general case. 

    Parameters:
    dim (int)
    n_constraints (int)
    n_points (int)
    c_actual (vector)
    kappa_actual (float)
    n_chains_find (int)
    n_adaptive (int)
    n_iterations (int)
    main_columns (list): list of column names for saving simulations
    prior_density_log (float)
    chunck_iterations (int)

    Returns:
    number of iterations
    """
    r_hat = 2
    data = generate_lp_data_var(dim, n_constraints, n_points, c_actual, kappa_actual)
    while r_hat > 1.1:
        c_optimals = solve_io_all(data)
        result = run_mcmc_var(0, c_optimals, dim, c_actual, kappa_actual, n_chains_find, n_adaptive, n_iterations, prior_density_log, main_columns)
        r_hat = result['Rhat']
        n_iterations += chunck_iterations
    return n_iterations
