"""Offline ONNX exporter for the episodic policy.

Restores a saved orbax checkpoint and writes an ONNX file with the episodic
observation contract (obs [1, OBS_SIZE] -> continuous_actions [1, 14]).

The episodic observation size differs from the walking policy: the command is
removed and a scalar phase is replaced by 50 Gaussian phase bases, so a trained
standing-wiggle policy expects obs = 142 (vs 101 for walking). Both emit 14
continuous actions. The two policies are therefore NOT interchangeable at the
observation interface; a deployment must route the correct obs vector to each.

Usage:
    python playground/open_duck_mini_v2/export_episodic_onnx.py \
        --checkpoint /path/to/<step_dir> --obs_size 142 --output policy.onnx
"""

import argparse

import jax
import numpy as np
from orbax import checkpoint as ocp

from playground.common.export_onnx import export_onnx
from mujoco_playground.config import locomotion_params


def restore_params_as_numpy(checkpoint_path: str):
    """Restore an orbax checkpoint's parameters.

    Checkpoints are written on the GPU with a device sharding baked into the
    array metadata. A plain restore works when a GPU device is visible (the
    saved single-device sharding resolves). On a CPU-only host it raises
    "sharding ... Got None"; in that case we retry, forcing every leaf to a
    host numpy array via construct_restore_args so the export never needs the
    training GPU.
    """
    ckptr = ocp.PyTreeCheckpointer()
    try:
        return ckptr.restore(checkpoint_path)
    except ValueError:
        from orbax.checkpoint import checkpoint_utils

        structure = ckptr.metadata(checkpoint_path)
        restore_args = checkpoint_utils.construct_restore_args(structure)
        restore_args = jax.tree_util.tree_map(
            lambda ra: ocp.RestoreArgs(restore_type=np.ndarray),
            restore_args,
            is_leaf=lambda x: isinstance(x, ocp.RestoreArgs),
        )
        return ckptr.restore(checkpoint_path, restore_args=restore_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export episodic policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a saved orbax checkpoint step directory")
    parser.add_argument("--obs_size", type=int, default=142,
                        help="Episodic observation size (142 for standing wiggle)")
    parser.add_argument("--act_size", type=int, default=14)
    parser.add_argument("--output", type=str, default="episodic_policy.onnx")
    args = parser.parse_args()

    ppo_params = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )

    params = restore_params_as_numpy(args.checkpoint)
    print(f"Restored checkpoint: {args.checkpoint}")

    export_onnx(
        params,
        args.act_size,
        ppo_params,
        args.obs_size,
        output_path=args.output,
    )
    print(f"Wrote ONNX to {args.output} (obs [1,{args.obs_size}] -> "
          f"continuous_actions [1,{args.act_size}])")


if __name__ == "__main__":
    main()
