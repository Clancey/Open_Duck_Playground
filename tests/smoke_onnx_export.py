"""ONNX export smoke test.

Trains a tiny PPO policy (same tiny config as smoke_train) then runs the repo's
export_onnx path and loads the result with onnxruntime to assert the deployed
contract is unchanged: input obs = 101, output action = 14. Cleans up its
artifact afterwards.

Run: JAX_PLATFORMS=cpu python tests/smoke_onnx_export.py
"""

import os
import sys
import functools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from playground.common import randomize
from playground.common.export_onnx import export_onnx
from playground.open_duck_mini_v2 import joystick

EXPECTED_OBS = 101
EXPECTED_ACT = 14
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".smoke_artifacts")
OUT = os.path.join(OUT_DIR, "smoke_policy.onnx")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    task = "flat_terrain_backlash"
    env = joystick.Joystick(task=task)
    eval_env = joystick.Joystick(task=task)
    obs_size = int(env.observation_size["state"][0])
    act_size = int(env.action_size)
    assert obs_size == EXPECTED_OBS and act_size == EXPECTED_ACT

    ppo_params = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
        del training_params["network_factory"]
    else:
        network_factory = ppo_networks.make_ppo_networks

    training_params.update(
        num_timesteps=1024, num_envs=16, batch_size=16, num_minibatches=1,
        unroll_length=10, num_updates_per_batch=1, episode_length=60,
        num_evals=1, num_eval_envs=8,
    )

    train_fn = functools.partial(
        ppo.train, **training_params, network_factory=network_factory,
        randomization_fn=randomize.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    _, params, _ = train_fn(environment=env, eval_env=eval_env)

    export_onnx(params, act_size, ppo_params, obs_size, output_path=OUT)
    assert os.path.exists(OUT), "ONNX file not written"

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"ONNX input : name={inp.name} shape={inp.shape}")
    print(f"ONNX output: name={out.name} shape={out.shape}")
    assert inp.shape[-1] == EXPECTED_OBS, f"ONNX obs {inp.shape[-1]} != {EXPECTED_OBS}"
    assert out.shape[-1] == EXPECTED_ACT, f"ONNX act {out.shape[-1]} != {EXPECTED_ACT}"

    y = sess.run(None, {inp.name: np.zeros((1, EXPECTED_OBS), dtype=np.float32)})[0]
    assert y.shape[-1] == EXPECTED_ACT and np.all(np.isfinite(y))
    print(f"ONNX inference OK: obs[1,{EXPECTED_OBS}] -> action[1,{y.shape[-1]}], finite.")

    os.remove(OUT)
    try:
        os.rmdir(OUT_DIR)
    except OSError:
        pass
    print("SMOKE ONNX OK: export path runs; deployed obs=101 / action=14 preserved.")


if __name__ == "__main__":
    main()
