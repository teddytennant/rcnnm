"""Reflective convolution nuclear norm minimization (arXiv:2610.04979)."""

import jax

jax.config.update("jax_enable_x64", True)

from .admm import (
    admm_step,
    soft_threshold,
    solve,
    solve_cnnm,
    solve_rcnnm,
    svt_gram,
    svt_svd,
)
from .lifting import adjoint, apply_ata, lift, multiplicity, reflect_indices, shift_index_map

__all__ = [
    "adjoint",
    "admm_step",
    "apply_ata",
    "lift",
    "multiplicity",
    "reflect_indices",
    "shift_index_map",
    "soft_threshold",
    "solve",
    "solve_cnnm",
    "solve_rcnnm",
    "svt_gram",
    "svt_svd",
]
