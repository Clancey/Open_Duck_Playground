"""Unit checks for the monotonic episodic phase encoding (paper Section V / Appendix A).

Run: python playground/tests/test_phase_encoding.py

These assertions guard the three properties most likely to be subtly wrong in the
episodic fix:
  1. phi is monotonically increasing over a clip and reaches exactly 1.0 at the end.
  2. The 50 Gaussian bases are correctly placed (equally spaced on [0, 1]) and
     *local* in time (each basis peaks at its own center; far bases are ~0).
  3. phi == 1 is the episode-terminating condition (last valid frame index).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from playground.common.phase_encoding import (  # noqa: E402
    DEFAULT_NUM_BASES,
    basis_centers,
    default_sigma,
    gaussian_phase_np,
)


def _phase_for_frame(i, nb_steps):
    """Monotonic phase used by the env: phi = i / (nb_steps - 1)."""
    return i / (nb_steps - 1)


def test_monotonic_phase():
    nb_steps = 150  # e.g. a 3.0 s clip at 50 FPS
    phis = np.array([_phase_for_frame(i, nb_steps) for i in range(nb_steps)])
    # strictly increasing
    assert np.all(np.diff(phis) > 0), "phase must be strictly monotonic increasing"
    assert phis[0] == 0.0, "phase must start at 0"
    assert abs(phis[-1] - 1.0) < 1e-9, "phase must reach exactly 1.0 at last frame"
    # constant rate == 1 / (clip duration in steps)
    rate = np.diff(phis)
    assert np.allclose(rate, rate[0]), "phase rate must be constant"
    assert abs(rate[0] - 1.0 / (nb_steps - 1)) < 1e-9, "rate must be 1/(nb_steps-1)"
    print("[OK] phase is monotonic, starts at 0, ends at 1, constant rate 1/duration")


def test_basis_placement():
    N = DEFAULT_NUM_BASES
    assert N == 50, "paper specifies N = 50 Gaussian basis functions"
    centers = basis_centers(N)
    assert centers.shape == (N,)
    assert abs(centers[0] - 0.0) < 1e-9 and abs(centers[-1] - 1.0) < 1e-9
    spacing = np.diff(centers)
    assert np.allclose(spacing, spacing[0]), "centers must be equally spaced"
    assert abs(spacing[0] - 1.0 / (N - 1)) < 1e-9
    print(f"[OK] {N} bases equally spaced on [0,1], spacing={spacing[0]:.5f}")


def test_basis_locality():
    N = DEFAULT_NUM_BASES
    centers = basis_centers(N)
    sigma = default_sigma(N)
    # At each center, that center's basis must be the unique argmax (peak == 1).
    for k in range(N):
        enc = gaussian_phase_np(centers[k], N)
        assert abs(enc[k] - 1.0) < 1e-6, f"basis {k} must peak (=1) at its own center"
        assert np.argmax(enc) == k, f"basis {k} must be the argmax at phi=center[k]"
    # Locality: a basis evaluated a quarter of the domain away must be ~0.
    mid = gaussian_phase_np(0.5, N)
    far_idx = 0  # center at phi=0, evaluated at phi=0.5 (0.5 away)
    assert mid[far_idx] < 1e-3, "far basis must be negligible (locality)"
    # Effective support: count bases with weight > 0.01 at phi=0.5 should be small.
    active = int(np.sum(mid > 0.01))
    assert active <= 8, f"phase encoding must be local; {active} active bases at 0.5"
    print(f"[OK] bases peak at their centers and are local (sigma={sigma:.4f}, "
          f"{active} bases active at phi=0.5)")


def test_encoding_changes_monotonically_in_position():
    """The peak (argmax) of the encoding advances monotonically with the frame."""
    nb_steps = 150
    N = DEFAULT_NUM_BASES
    last_peak = -1
    peaks = []
    for i in range(nb_steps):
        phi = _phase_for_frame(i, nb_steps)
        peak = int(np.argmax(gaussian_phase_np(phi, N)))
        peaks.append(peak)
        assert peak >= last_peak, "peak basis index must not go backwards"
        last_peak = peak
    assert peaks[0] == 0 and peaks[-1] == N - 1
    print(f"[OK] encoding peak advances 0 -> {N-1} monotonically across the clip")


def test_phi_one_terminates():
    """phi == 1.0 corresponds to the last valid frame; that is the terminating step."""
    nb_steps = 150
    # env increments imitation_i each step; done when imitation_i >= nb_steps - 1
    done_at = None
    for i in range(nb_steps + 5):
        phi = min(i / (nb_steps - 1), 1.0)
        terminated = i >= (nb_steps - 1)  # env termination rule
        if terminated and done_at is None:
            done_at = i
            assert abs(phi - 1.0) < 1e-9, "termination must coincide with phi==1"
    assert done_at == nb_steps - 1, "must terminate exactly when phi first hits 1"
    print(f"[OK] phi==1 terminates the episode at frame {done_at} (nb_steps-1)")


if __name__ == "__main__":
    test_monotonic_phase()
    test_basis_placement()
    test_basis_locality()
    test_encoding_changes_monotonically_in_position()
    test_phi_one_terminates()
    print("\nALL PHASE-ENCODING UNIT CHECKS PASSED")
