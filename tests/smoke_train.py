"""Tiny CPU PPO smoke test (NOT a real training run).

Runs a handful of PPO iterations with a deliberately tiny config (few envs,
short episodes, a few thousand env steps) purely to confirm the env + the new
leg/neck-split reward compile and run under brax PPO's vmap/scan, that shapes
broadcast, and that nothing NaNs. This is a correctness smoke, not convergence.

Run: JAX_PLATFORMS=cpu python tests/smoke_train.py
"""

import os
import sys
import functools

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.common import randomize
from playground.open_duck_mini_v2 import joystick


def progress(num_steps, metrics):
    er = metrics.get("eval/episode_reward", float("nan"))
    print(f"  [progress] steps={num_steps} eval/episode_reward={float(er):.4f}")


def main():
    task = "flat_terrain_backlash"
    env = joystick.Joystick(task=task)
    eval_env = joystick.Joystick(task=task)
    print(f"use_leg_neck_split = {env._use_leg_neck_split}, "
          f"obs={int(env.observation_size['state'][0])}, act={int(env.action_size)}")

    ppo_params = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
        del training_params["network_factory"]
    else:
        network_factory = ppo_networks.make_ppo_networks

    # Shrink everything to a CPU-friendly smoke config.
    training_params.update(
        num_timesteps=2048,
        num_envs=16,
        batch_size=16,
        num_minibatches=1,
        unroll_length=10,
        num_updates_per_batch=1,
        episode_length=60,
        num_evals=1,
        num_eval_envs=8,
    )
    print(f"smoke PPO params: num_envs={training_params['num_envs']} "
          f"num_timesteps={training_params['num_timesteps']} "
          f"episode_length={training_params['episode_length']}")

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        randomization_fn=randomize.domain_randomize,
        progress_fn=progress,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env, eval_env=eval_env
    )

    er = float(metrics.get("eval/episode_reward", float("nan")))
    assert er == er, "eval/episode_reward is NaN -> reward/env broke under PPO"
    print(f"SMOKE TRAIN OK: PPO ran to completion, eval/episode_reward={er:.4f} "
          "(finite, non-NaN). Params tree built.")


if __name__ == "__main__":
    main()
