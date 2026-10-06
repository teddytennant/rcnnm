"""Two-block ADMM for reflective and circular convolution nuclear norm minimization.

Section 5 of arXiv:2610.04979, model (22) and Algorithm 1. The updates are,
in this order, using the penalties from the start of the iteration:

    Q = A*(tau1 * Z - Y1) + P_Omega(tau2 * (M - S) - Y2)
    L_u = Q_u / (tau1 * d_u + tau2 * 1_{u in Omega})
    S = soft_{lambda/tau2}(P_Omega(M - L_new - Y2/tau2)), zero off Omega
    Z = SVT_{1/tau1}(A(L_new) + Y1/tau1)
    Y1 += tau1 * (A(L_new) - Z)
    Y2 += tau2 * P_Omega(L_new + S - M)
    tau <- min(rho * tau, tau_max) for both penalties

Singular-value thresholding uses the right Gram matrix (52):

    G = H^T H = V diag(xi) V^T,  sigma_i = sqrt(max(xi_i, 0))
    SVT_tau(H) = H V_tau (I - tau Sigma_tau^{-1}) V_tau^T

with columns kept only when ``sigma_i > tau``. The circular solver (CNNM)
uses the same updates with circular shifts and ``D = K * ones``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .lifting import adjoint, lift, multiplicity


def soft_threshold(x, eta):
    """Entrywise ``sgn(x) * max(|x| - eta, 0)``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    eta = jnp.asarray(eta, dtype=jnp.float64)
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - eta, 0.0)


def svt_gram(h, tau) -> jnp.ndarray:
    """Nuclear-norm proximal map through the smaller right Gram matrix.

    Matches ``U diag((sigma - tau)_+) V^T`` in exact arithmetic. ``tau = 0``
    reproduces ``H`` (null components of ``H^T H`` do not contribute).
    """
    h = jnp.asarray(h, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    gram = h.T @ h
    gram = 0.5 * (gram + gram.T)
    xi, v = jnp.linalg.eigh(gram)
    sigma = jnp.sqrt(jnp.maximum(xi, 0.0))
    # Avoid division by zero on dropped components. sigma > tau is the
    # paper's retained index set J_tau; equality contributes nothing.
    safe = jnp.where(sigma > tau, sigma, 1.0)
    scale = jnp.where(sigma > tau, 1.0 - tau / safe, 0.0)
    return h @ (v * scale) @ v.T


def svt_svd(h, tau) -> jnp.ndarray:
    """Direct SVD form (45), for comparison with ``svt_gram``."""
    h = jnp.asarray(h, dtype=jnp.float64)
    tau = jnp.asarray(tau, dtype=jnp.float64)
    u, sigma, vh = jnp.linalg.svd(h, full_matrices=False)
    shrink = jnp.maximum(sigma - tau, 0.0)
    return (u * shrink) @ vh


def _project(mask, values):
    return jnp.where(mask, values, 0.0)


def admm_step(
    L,
    S,
    Z,
    Y1,
    Y2,
    M,
    omega,
    kernel_shape,
    lam,
    tau1,
    tau2,
    *,
    rho=1.05,
    tau_max=100.0,
    boundary: str = "reflect",
):
    """One Algorithm 1 iteration. Penalties used here are the incoming ones.

    ``L`` is updated from the previous ``Z`` and ``S``. ``S`` and ``Z`` are
    then updated from ``L_new``. Multipliers use those new blocks and the
    incoming penalties. Penalties increase only after that.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    L = jnp.asarray(L, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    Y1 = jnp.asarray(Y1, dtype=jnp.float64)
    Y2 = jnp.asarray(Y2, dtype=jnp.float64)
    omega = jnp.asarray(omega, dtype=bool)
    tau1 = jnp.asarray(tau1, dtype=jnp.float64)
    tau2 = jnp.asarray(tau2, dtype=jnp.float64)
    lam = jnp.asarray(lam, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    tau_max = jnp.asarray(tau_max, dtype=jnp.float64)

    d = multiplicity(M.shape, kernel_shape, boundary)

    # (40)-(41): entrywise inverse of the diagonal operator tau1*D + tau2*P_Omega.
    q = adjoint(tau1 * Z - Y1, M.shape, kernel_shape, boundary)
    q = q + _project(omega, tau2 * (M - S) - Y2)
    denom = tau1 * d + jnp.where(omega, tau2, 0.0)
    l_new = q / denom

    # (43): soft-threshold on Omega only.
    s_arg = _project(omega, M - l_new - Y2 / tau2)
    s_new = _project(omega, soft_threshold(s_arg, lam / tau2))

    # (46)/(52): Gram SVT at level 1/tau1.
    h = lift(l_new, kernel_shape, boundary) + Y1 / tau1
    z_new = svt_gram(h, 1.0 / tau1)

    # (47)-(48), still with the incoming penalties.
    a_l = lift(l_new, kernel_shape, boundary)
    y1_new = Y1 + tau1 * (a_l - z_new)
    data_res = _project(omega, l_new + s_new - M)
    y2_new = Y2 + tau2 * data_res

    # (49)
    tau1_new = jnp.minimum(rho * tau1, tau_max)
    tau2_new = jnp.minimum(rho * tau2, tau_max)

    # (50). r1 is +inf when the lifting vanishes.
    num1 = jnp.linalg.norm(a_l - z_new)
    den1 = jnp.linalg.norm(a_l)
    r1 = jnp.where(den1 > 0.0, num1 / den1, jnp.inf)
    r2 = jnp.linalg.norm(data_res) / (1.0 + jnp.linalg.norm(M))

    return {
        "L": l_new,
        "S": s_new,
        "Z": z_new,
        "Y1": y1_new,
        "Y2": y2_new,
        "Q": q,
        "tau1": tau1_new,
        "tau2": tau2_new,
        "r1": r1,
        "r2": r2,
    }


def solve(
    M,
    omega,
    kernel_shape,
    lam,
    *,
    boundary: str = "reflect",
    tau1: float = 5e-3,
    tau2: float = 1e-1,
    rho: float = 1.05,
    tau_max: float = 100.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    L0=None,
    S0=None,
    Z0=None,
    Y10=None,
    Y20=None,
):
    """Run Algorithm 1 until ``max(r1, r2) <= tol`` or ``max_iter``.

    Default penalties, factor, cap, iteration budget, and tolerance are the
    paper's Section 6.3 values. ``M`` is zero-filled off ``Omega`` so the
    residual denominator is the observed tensor. Default initialization is
    ``L = P_Omega(M)``, ``S = 0``, ``Z = A(L)``, ``Y1 = Y2 = 0``, which
    satisfies ``P_Omega^c(S) = 0``.
    """
    omega = jnp.asarray(omega, dtype=bool)
    M = _project(omega, jnp.asarray(M, dtype=jnp.float64))
    n = M.size
    k_cols = 1
    for k in kernel_shape:
        k_cols *= int(k)

    L = M if L0 is None else jnp.asarray(L0, dtype=jnp.float64)
    S = jnp.zeros_like(M) if S0 is None else _project(omega, jnp.asarray(S0, dtype=jnp.float64))
    Z = lift(L, kernel_shape, boundary) if Z0 is None else jnp.asarray(Z0, dtype=jnp.float64)
    Y1 = jnp.zeros((n, k_cols), dtype=jnp.float64) if Y10 is None else jnp.asarray(Y10, dtype=jnp.float64)
    Y2 = jnp.zeros_like(M) if Y20 is None else jnp.asarray(Y20, dtype=jnp.float64)

    r1 = jnp.array(jnp.inf, dtype=jnp.float64)
    r2 = jnp.array(jnp.inf, dtype=jnp.float64)
    history = []
    iterations = 0
    for _ in range(int(max_iter)):
        if max(float(r1), float(r2)) <= tol:
            break
        step = admm_step(
            L, S, Z, Y1, Y2, M, omega, kernel_shape, lam, tau1, tau2,
            rho=rho, tau_max=tau_max, boundary=boundary,
        )
        L, S, Z = step["L"], step["S"], step["Z"]
        Y1, Y2 = step["Y1"], step["Y2"]
        tau1, tau2 = step["tau1"], step["tau2"]
        r1, r2 = step["r1"], step["r2"]
        iterations += 1
        history.append((float(r1), float(r2)))

    return {
        "L": L,
        "S": S,
        "Z": Z,
        "Y1": Y1,
        "Y2": Y2,
        "tau1": tau1,
        "tau2": tau2,
        "r1": r1,
        "r2": r2,
        "iterations": iterations,
        "history": history,
        "M": M,
    }


def solve_rcnnm(M, omega, kernel_shape, lam, **kwargs):
    """RCNNM: Algorithm 1 with reflective lifting."""
    kwargs.pop("boundary", None)
    return solve(M, omega, kernel_shape, lam, boundary="reflect", **kwargs)


def solve_cnnm(M, omega, kernel_shape, lam, **kwargs):
    """CNNM: the same ADMM with circular lifting and constant multiplicity ``K``."""
    kwargs.pop("boundary", None)
    return solve(M, omega, kernel_shape, lam, boundary="circular", **kwargs)
