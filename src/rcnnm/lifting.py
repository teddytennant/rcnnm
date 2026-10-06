"""Reflective and circular convolutional liftings.

Section 3 of arXiv:2610.04979. Indices in the paper are 1-based. This
module is 0-based. The endpoint-nonrepeating reflection on a mode of
length m is

    phi(j) = j                  if 0 <= j < m,
             2*m - 2 - j        if m <= j <= 2*m - 2.

The reflected index list is 0, 1, ..., m-1, m-2, ..., 0. The endpoint
m-1 appears once, and phi(m) = m-2 (paper: phi(m+1) = m-1, not m).

Kernel lengths satisfy 1 <= k_a <= m_a. Offsets along mode a lie in
{0, ..., k_a-1}. Columns of the lifting are ordered by C-order (last
offset axis fastest), matching ``ravel``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def reflect_indices(m: int) -> jnp.ndarray:
    """Endpoint-nonrepeating reflection indices of length ``2*m - 1``.

    Entry ``j`` is ``phi(j)`` for ``j = 0, ..., 2*m-2``.
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"mode length must be positive, got {m}")
    j = jnp.arange(2 * m - 1, dtype=jnp.int32)
    return jnp.where(j < m, j, jnp.int32(2 * m - 2) - j)


def _as_shape(shape) -> tuple[int, ...]:
    shape = tuple(int(s) for s in shape)
    if not shape:
        raise ValueError("tensor order must be at least 1")
    if any(s < 1 for s in shape):
        raise ValueError(f"mode lengths must be positive, got {shape}")
    return shape


def _check_kernel(shape: tuple[int, ...], kernel_shape: tuple[int, ...]) -> tuple[int, ...]:
    kernel_shape = tuple(int(k) for k in kernel_shape)
    if len(kernel_shape) != len(shape):
        raise ValueError(
            f"kernel order {len(kernel_shape)} does not match tensor order {len(shape)}"
        )
    for m, k in zip(shape, kernel_shape):
        if not (1 <= k <= m):
            raise ValueError(f"kernel length {k} must satisfy 1 <= k <= {m}")
    return kernel_shape


def _flat_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for m in reversed(shape):
        strides.append(stride)
        stride *= m
    return tuple(reversed(strides))


def shift_index_map(shape, kernel_shape, boundary: str = "reflect") -> jnp.ndarray:
    """Flat C-order indices of ``phi(i + t)``, shape ``(N, K)``.

    Row ``i`` is a vectorized spatial position and column ``t`` is a
    vectorized kernel offset. ``boundary`` is ``"reflect"`` or ``"circular"``.
    """
    shape = _as_shape(shape)
    kernel_shape = _check_kernel(shape, kernel_shape)
    if boundary not in ("reflect", "circular"):
        raise ValueError(f"unknown boundary {boundary!r}")

    pos_axes = [jnp.arange(m, dtype=jnp.int32) for m in shape]
    off_axes = [jnp.arange(k, dtype=jnp.int32) for k in kernel_shape]
    pos = jnp.stack([g.ravel() for g in jnp.meshgrid(*pos_axes, indexing="ij")], axis=1)
    off = jnp.stack([g.ravel() for g in jnp.meshgrid(*off_axes, indexing="ij")], axis=1)
    summed = pos[:, None, :] + off[None, :, :]

    mapped = []
    for a, m in enumerate(shape):
        ja = summed[:, :, a]
        if boundary == "reflect":
            mapped.append(jnp.where(ja < m, ja, jnp.int32(2 * m - 2) - ja))
        else:
            mapped.append(jnp.mod(ja, jnp.int32(m)))
    mapped = jnp.stack(mapped, axis=-1)

    flat = jnp.zeros(mapped.shape[:2], dtype=jnp.int32)
    for a, stride in enumerate(_flat_strides(shape)):
        flat = flat + mapped[:, :, a] * jnp.int32(stride)
    return flat


def multiplicity(shape, kernel_shape, boundary: str = "reflect") -> jnp.ndarray:
    """Multiplicity tensor ``D``.

    ``d_u`` counts pairs ``(i, t)`` with ``phi(i + t) = u``. Reflective
    ``D`` is entrywise positive. Circular ``D`` equals ``K`` everywhere.
    """
    shape = _as_shape(shape)
    idx = shift_index_map(shape, kernel_shape, boundary)
    n = 1
    for s in shape:
        n *= s
    counts = jnp.bincount(idx.ravel(), length=n)
    return counts.reshape(shape).astype(jnp.float64)


def lift(x, kernel_shape, boundary: str = "reflect") -> jnp.ndarray:
    """Convolutional lifting ``A(X)``, shape ``(N, K)``.

    Column ``t`` is ``vec(R_t(X))`` with ``R_t(X)_i = X_{phi(i+t)}``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    idx = shift_index_map(x.shape, kernel_shape, boundary)
    return x.ravel()[idx]


def adjoint(h, shape, kernel_shape, boundary: str = "reflect") -> jnp.ndarray:
    """Adjoint ``A*(H)``.

    ``[A*(H)]_u = sum_{i,t: phi(i+t)=u} H_{i,t}``.
    """
    shape = _as_shape(shape)
    h = jnp.asarray(h, dtype=jnp.float64)
    idx = shift_index_map(shape, kernel_shape, boundary)
    n = idx.shape[0]
    acc = jnp.zeros((n,), dtype=jnp.float64)
    acc = acc.at[idx.ravel()].add(h.reshape(-1))
    return acc.reshape(shape)


def apply_ata(x, kernel_shape, boundary: str = "reflect") -> jnp.ndarray:
    """``A*(A(X))``. Reflective case equals ``D odot X``; circular equals ``K X``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    return adjoint(lift(x, kernel_shape, boundary), x.shape, kernel_shape, boundary)
