# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================
"""Monotonic episodic phase encoding via a bank of Gaussian basis functions.

Disney's "Design and Control of a Bipedal Robotic Character" (arXiv:2501.05204)
encodes the episodic phase (Eq. 4, Section V) with a *monotonically increasing*
phase variable phi in [0, 1] projected onto a bank of ``N`` Gaussian basis
functions (Appendix A, paragraph 2):

    b_k(phi) = exp( -(phi - phi_k)^2 / (2 * sigma^2) )

with the ``N`` centers ``phi_k`` equally spaced on ``[0, 1]``. The paper uses
``N = 50``. These bases are "highly local in time and allow the policy to form a
rich phase-dependent feed-forward signal" -- exactly what a *one-shot* clip needs
and what a periodic ``[cos, sin]`` pair cannot represent.

Width note: the paper writes the Gaussian denominator compactly; for the bases to
actually be *local* on ``[0, 1]`` with ``N`` equally-spaced centers, the width
must scale like the center spacing ``~1/N`` (a literal ``sigma = N`` would make
every basis essentially flat and destroy locality). We therefore parameterise the
width as ``sigma = 1 / N`` (roughly one center-spacing), which reproduces the
"highly local" property the paper describes. ``sigma`` is configurable.

Both a JAX implementation (used inside the training env) and a NumPy
implementation (used by the deployment/inference path) are provided so the two
stay bit-for-bit consistent.
"""

import numpy as np

try:
    import jax.numpy as jp

    _HAS_JAX = True
except Exception:  # pragma: no cover - numpy-only environments
    _HAS_JAX = False


DEFAULT_NUM_BASES = 50


def basis_centers(num_bases: int = DEFAULT_NUM_BASES) -> np.ndarray:
    """Return the ``num_bases`` equally-spaced Gaussian centers on ``[0, 1]``."""
    return np.linspace(0.0, 1.0, num_bases)


def default_sigma(num_bases: int = DEFAULT_NUM_BASES) -> float:
    """Return the default Gaussian width (~one center spacing)."""
    return 1.0 / float(num_bases)


def gaussian_phase_np(
    phi,
    num_bases: int = DEFAULT_NUM_BASES,
    sigma: float | None = None,
) -> np.ndarray:
    """NumPy Gaussian phase encoding. ``phi`` is a scalar in ``[0, 1]``."""
    if sigma is None:
        sigma = default_sigma(num_bases)
    centers = basis_centers(num_bases)
    phi = np.clip(np.asarray(phi, dtype=np.float32), 0.0, 1.0)
    return np.exp(-((phi - centers) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


if _HAS_JAX:

    def gaussian_phase_jax(phi, num_bases: int = DEFAULT_NUM_BASES, sigma=None):
        """JAX Gaussian phase encoding. ``phi`` is a scalar tracer in ``[0, 1]``."""
        if sigma is None:
            sigma = default_sigma(num_bases)
        centers = jp.linspace(0.0, 1.0, num_bases)
        phi = jp.clip(phi, 0.0, 1.0)
        return jp.exp(-((phi - centers) ** 2) / (2.0 * sigma * sigma))
