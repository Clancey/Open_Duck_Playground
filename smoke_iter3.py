"""Tiny CPU PPO smoke for iteration-3 (head passthrough).

Confirms: env constructs, PPO train loop runs under JIT with the passthrough
active, nothing NaNs, and the ONNX export produces obs[1,101]->[1,14]. NOT a
convergence run. Run:  JAX_PLATFORMS=cpu python smoke_iter3.py
"""

import functools
import os

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from playground.common import randomize
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from playground.open_duck_mini_v2 import joystick

os.makedirs(".smoke_out", exist_ok=True)

env = joystick.Joystick(task="flat_terrain")
eval_env = joystick.Joystick(task="flat_terrain")
obs_size = env.observation_size["state"][0]
action_size = env.action_size
print(f"obs_size={obs_size} action_size={action_size} "
      f"HEAD_PASSTHROUGH={joystick.HEAD_PASSTHROUGH}")
assert obs_size == 101, f"OBS SIZE CHANGED: {obs_size}"
assert action_size == 14, f"ACTION SIZE CHANGED: {action_size}"

randomizer = randomize.domain_randomize

# Use the REAL network factory so the ONNX export shapes match production.
ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
network_factory = functools.partial(
    ppo_networks.make_ppo_networks, **ppo_params.network_factory
)


def progress(step, metrics):
    r = metrics.get("eval/episode_reward", float("nan"))
    print(f"  step={step} eval_reward={float(r):.4f}")
    assert np.isfinite(float(r)), "NaN eval reward"


train_fn = functools.partial(
    ppo.train,
    num_timesteps=4096,
    num_evals=2,
    episode_length=60,
    num_envs=8,
    batch_size=8,
    num_minibatches=2,
    unroll_length=10,
    num_updates_per_batch=1,
    learning_rate=1e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    action_repeat=1,
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
)

make_policy, params, _ = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print("PPO smoke loop completed with no NaN.")

# ONNX export shape check.
from playground.common.export_onnx import export_onnx  # noqa: E402

onnx_path = ".smoke_out/iter3_smoke.onnx"
export_onnx(params, action_size, ppo_params, obs_size, output_path=onnx_path)

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(onnx_path)
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print(f"ONNX input {inp.name} {inp.shape} -> output {out.name} {out.shape}")
feed = {inp.name: np.zeros((1, 101), dtype=np.float32)}
res = sess.run(None, feed)
print(f"ONNX inference output shape: {res[0].shape}")
assert res[0].shape[-1] == 14, "ONNX action dim != 14"
print("SMOKE PASS: env=101/14, PPO ran, no NaN, ONNX 101->14.")
