"""Smoke test: construct the Joystick env, JIT a reset + several steps under the
new leg/neck-split reward, and assert nothing NaNs, shapes broadcast, and the
observation size is unchanged (must stay 101 for deployment compatibility).

Run: JAX_PLATFORMS=cpu python tests/smoke_env_step.py
"""

import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.open_duck_mini_v2 import joystick

EXPECTED_OBS = 101
EXPECTED_ACT = 14


def main():
    task = "flat_terrain_backlash"  # the current sim2real win (plan constants.py)
    print(f"Constructing Joystick(task={task!r}) ...")
    env = joystick.Joystick(task=task)

    obs_size = int(env.observation_size["state"][0])
    act_size = int(env.action_size)
    split = env._use_leg_neck_split
    print(f"observation_size(state) = {obs_size}")
    print(f"action_size             = {act_size}")
    print(f"use_leg_neck_split      = {split}")
    assert obs_size == EXPECTED_OBS, f"obs size changed: {obs_size} != {EXPECTED_OBS}"
    assert act_size == EXPECTED_ACT, f"act size changed: {act_size} != {EXPECTED_ACT}"
    assert split is True, "branch default must be the NEW leg/neck split"

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    assert state.obs["state"].shape[-1] == EXPECTED_OBS

    n_steps = 8
    for i in range(n_steps):
        rng, akey = jax.random.split(rng)
        action = jax.random.uniform(akey, (act_size,), minval=-1.0, maxval=1.0)
        state = step(state, action)
        r = float(state.reward)
        assert np.isfinite(r), f"non-finite reward at step {i}: {r}"
        assert np.all(np.isfinite(np.asarray(state.obs["state"]))), (
            f"non-finite obs at step {i}"
        )
        keys = ("action_rate", "action_rate_leg", "action_rate_neck",
                "action_accel_leg", "action_accel_neck", "imitation")
        buckets = {}
        for k in keys:
            for pref in ("reward/", "cost/"):
                if pref + k in state.metrics:
                    buckets[k] = round(float(state.metrics[pref + k]), 4)
        print(f"step {i}: reward={r:.4f} done={int(state.done)} "
              f"buckets={buckets}")

    print("\nSMOKE OK: env constructs, reward JITs, shapes broadcast, no NaNs, "
          f"obs={obs_size}, act={act_size}.")


if __name__ == "__main__":
    main()
