"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:

        for metric_name, metric_value in metrics.items():
            # Convert to float, but watch out for 0-dim JAX arrays
            self.writer.add_scalar(metric_name, metric_value, num_steps)

        print("-----------")
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )
        print("-----------")

    def policy_params_fn(self, current_step, make_policy, params):
        # save checkpoints

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        d = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = f"{self.output_dir}/{d}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        onnx_export_path = f"{self.output_dir}/{d}_{current_step}.onnx"
        # In-loop ONNX export spins up a TensorFlow runtime that grabs several GB
        # of GPU memory per checkpoint and can destabilise a long run on a shared
        # GPU. It is best-effort anyway (the orbax checkpoint above is the source
        # of truth). Set DISABLE_INLOOP_ONNX=1 to skip it and export offline.
        if os.environ.get("DISABLE_INLOOP_ONNX") == "1":
            return
        try:
            export_onnx(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,  # may not work
                output_path=onnx_export_path
            )
        except Exception as e:  # noqa: BLE001
            # ONNX export is best-effort: the orbax checkpoint above is the
            # source of truth. A TF/tf2onnx version mismatch must not kill a
            # long training run; export can be re-run offline from checkpoints.
            print(f"[warn] ONNX export failed at step {current_step}: {e}")

    def train(self) -> None:
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )  # TODO
        self.ppo_training_params = dict(self.ppo_params)
        # self.ppo_training_params["num_timesteps"] = 150000000 * 20
        

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        # Optional footprint cap: on a shared GPU (e.g. a homelab with other
        # containers) the default num_envs can OOM under transient contention,
        # especially with the larger episodic observation. Allow an override.
        _num_envs_override = os.environ.get("PPO_NUM_ENVS")
        if _num_envs_override:
            self.ppo_training_params["num_envs"] = int(_num_envs_override)
        # Fewer evaluation points => fewer one-off eval-unroll XLA compilations,
        # each of which is a transient GPU-memory spike. On a shared GPU those
        # spikes can collide with another container's spike and get the process
        # SIGKILLed (exit 137) even though steady-state memory fits. Allow the
        # eval count (and thus spike frequency) to be tuned down.
        _num_evals_override = os.environ.get("PPO_NUM_EVALS")
        if _num_evals_override:
            self.ppo_training_params["num_evals"] = int(_num_evals_override)
        # Stabilization overrides for the episodic policy. The episodic optimal
        # policy is nearly static (stand + small head wiggle), so the stock
        # entropy bonus inflates the action std (observed 0.4 -> ~10 => the policy
        # goes random and the robot falls). A huge initial value loss also drives a
        # catastrophic first update. Allow tuning entropy_cost / learning_rate /
        # reward_scaling from the environment so the walking defaults stay intact.
        for _env_key, _param in (
            ("PPO_ENTROPY_COST", "entropy_cost"),
            ("PPO_LEARNING_RATE", "learning_rate"),
            ("PPO_REWARD_SCALING", "reward_scaling"),
            ("PPO_CLIPPING_EPSILON", "clipping_epsilon"),
        ):
            _v = os.environ.get(_env_key)
            if _v:
                self.ppo_training_params[_param] = float(_v)
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
