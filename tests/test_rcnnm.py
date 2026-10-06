"""CPU checks for the reflective lifting and the two-block ADMM."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from rcnnm import admm_step, adjoint, apply_ata, lift, multiplicity, reflect_indices, solve_rcnnm, svt_gram


def test_multiplicity_m4_k3():
    # 1-indexed pairs in the brief count positions 1..4 as 1, 3, 5, 3.
    d = np.asarray(multiplicity((4,), (3,)))
    np.testing.assert_array_equal(d, np.array([1.0, 3.0, 5.0, 3.0]))


def test_reflection_does_not_repeat_endpoint():
    m = 5
    idx = np.asarray(reflect_indices(m))
    assert idx.shape == (2 * m - 1,)
    # 1-indexed endpoint m is 0-indexed m-1. It appears once, and the next
    # reflected index is m-2, not a wrap to 0.
    endpoint = m - 1
    positions = np.flatnonzero(idx == endpoint)
    assert positions.size == 1
    nxt = idx[positions[0] + 1]
    assert nxt == m - 2
    assert nxt != 0


@pytest.mark.parametrize("shape,kernel", [((6, 5), (3, 4)), ((4, 3, 5), (2, 3, 2))])
def test_weighted_gram_identity(shape, kernel):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape)
    d = np.asarray(multiplicity(shape, kernel))
    ata = np.asarray(apply_ata(x, kernel, boundary="reflect"))
    np.testing.assert_allclose(ata, d * x, atol=1e-10, rtol=0.0)
    assert np.all(d > 0)


@pytest.mark.parametrize("shape,kernel", [((6, 5), (3, 4)), ((4, 3, 5), (2, 3, 2))])
def test_circular_gram_identity(shape, kernel):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape)
    k = int(np.prod(kernel))
    ata = np.asarray(apply_ata(x, kernel, boundary="circular"))
    np.testing.assert_allclose(ata, k * x, atol=1e-10, rtol=0.0)
    d = np.asarray(multiplicity(shape, kernel, boundary="circular"))
    np.testing.assert_array_equal(d, np.full(shape, float(k)))


def test_gram_svt_matches_full_svd():
    rng = np.random.default_rng(2)
    left, _ = np.linalg.qr(rng.standard_normal((8, 5)))
    right, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    singular = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
    h = left @ np.diag(singular) @ right.T

    def svd_svt(tau):
        u, sigma, vh = np.linalg.svd(h, full_matrices=False)
        return (u * np.maximum(sigma - tau, 0.0)) @ vh

    np.testing.assert_allclose(np.asarray(svt_gram(h, 0.0)), h, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(np.asarray(svt_gram(h, 0.0)), svd_svt(0.0), atol=1e-10, rtol=0.0)
    # Drops the singular value 0.25 and keeps the rest.
    tau = 0.4
    got = np.asarray(svt_gram(h, tau))
    np.testing.assert_allclose(got, svd_svt(tau), atol=1e-10, rtol=0.0)
    assert np.linalg.matrix_rank(got, tol=1e-8) == 4


def test_one_admm_iteration_matches_closed_form():
    """Hand update on a 1D problem. Order is L, then S and Z, with incoming penalties."""
    kernel = (3,)
    omega = np.array([True, False, True, True])
    lam = 0.8
    tau1 = 1.0
    tau2 = 2.0
    m_obs = np.array([2.0, 0.0, -1.5, 1.2])
    l0 = np.array([0.5, 0.2, -0.4, 0.1])
    s0 = np.where(omega, np.array([0.4, 9.0, -0.3, 0.2]), 0.0)
    rng = np.random.default_rng(1)
    z0 = rng.normal(size=(4, 3))
    y1 = rng.normal(size=(4, 3))
    y2 = np.where(omega, np.array([0.3, 8.0, -0.2, 0.5]), 0.0)

    step = admm_step(
        l0, s0, z0, y1, y2, m_obs, omega, kernel, lam, tau1, tau2, boundary="reflect"
    )

    # Hardcoded multiplicity for m=4, k=3, not taken from the step.
    d = np.array([1.0, 3.0, 5.0, 3.0])
    q = np.asarray(adjoint(tau1 * z0 - y1, (4,), kernel))
    q = q + omega * (tau2 * (m_obs - s0) - y2)
    denom = tau1 * d + omega.astype(np.float64) * tau2
    l_exp = q / denom

    s_arg = omega * (m_obs - l_exp - y2 / tau2)
    s_exp = np.sign(s_arg) * np.maximum(np.abs(s_arg) - lam / tau2, 0.0)
    s_exp = omega * s_exp

    h = np.asarray(lift(l_exp, kernel)) + y1 / tau1
    u, sigma, vh = np.linalg.svd(h, full_matrices=False)
    z_exp = (u * np.maximum(sigma - 1.0 / tau1, 0.0)) @ vh

    np.testing.assert_allclose(np.asarray(step["Q"]), q, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(step["L"]), l_exp, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(step["S"]), s_exp, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(np.asarray(step["Z"]), z_exp, atol=1e-10, rtol=0.0)
    assert np.all(np.asarray(step["S"])[~omega] == 0.0)
    # Incoming penalties, not the updated ones, determine L, S, and Z.
    assert abs(float(step["tau1"]) - min(1.05 * tau1, 100.0)) < 1e-15
    assert abs(float(step["tau2"]) - min(1.05 * tau2, 100.0)) < 1e-15


def test_reflective_recovery_of_low_frequency_cosine():
    """Separable low-frequency cosine, partial samples, a few sparse corruptions.

    ``cos(pi * n / (m-1))`` is the lowest non-constant cosine compatible with
    endpoint-nonrepeating reflection: its reflective lifting has rank 2.
    Paper default penalties (Section 6.3).
    """
    rng = np.random.default_rng(1)
    m = 32
    kernel = (7,)
    n = np.arange(m, dtype=np.float64)
    clean = np.cos(np.pi * n / (m - 1))
    clean = clean / np.linalg.norm(clean)

    observed = rng.random(m) < 0.8
    observed[0] = False
    observed[1] = True
    corrupted = clean.copy()
    sites = rng.choice(np.flatnonzero(observed), size=3, replace=False)
    corrupted[sites] = corrupted[sites] + rng.choice([-1.0, 1.0], size=3)
    measured = np.where(observed, corrupted, 0.0)

    out = solve_rcnnm(measured, observed, kernel, lam=2.0)
    recovered = np.asarray(out["L"])
    rse = np.linalg.norm(recovered - clean) / np.linalg.norm(clean)
    observed_error = np.linalg.norm(measured - clean) / np.linalg.norm(clean)
    assert rse < 1e-2, f"RSE {rse}"
    assert rse < observed_error
    assert float(out["r2"]) < 1e-4


def test_ramp_reflective_rank_does_not_exceed_circular():
    m, k = 32, 5
    ramp = np.arange(m, dtype=np.float64)

    def effective_rank(boundary):
        singular = np.linalg.svd(np.asarray(lift(ramp, (k,), boundary)), compute_uv=False)
        return int(np.sum(singular > 0.01 * singular[0]))

    assert effective_rank("reflect") <= effective_rank("circular")
