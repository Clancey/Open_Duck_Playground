"""Headless MuJoCo evaluation for the episodic standing-wiggle policy.

Rolls the policy through the one-shot clip once (monotonic phase phi: 0 -> 1),
with no viewer, and reports the metrics the task asks for:

  * upright:      does the robot stay standing (torso up-axis stays > 0)?
  * peak tilt:    worst tilt from vertical over the rollout (degrees)
  * tracking MAE: mean abs joint error vs the reference (rad), compared to
                  Disney's episodic MAE of 0.027-0.043 rad (Table III)
  * cheating:     ratio of the robot's joint motion range to the reference's
                  range -- a policy that "cheats" by barely moving (Appendix A)
                  shows a ratio well below 1.0
  * fell:         whether any termination condition tripped, and at which step

The observation is built identically to mujoco_infer_episodic.py (the 142-float
episodic contract: gyro3 + accel3 + 14*6 joint/action blocks + contact2 + 50
Gaussian phase bases -- NO command). Deterministic action = tanh(mean) from the
exported ONNX. An optional --zero_action baseline holds the home pose for
comparison.
"""

import argparse
import numpy as np

from playground.common.episodic_reference_motion_numpy import EpisodicReferenceMotion
from playground.common.phase_encoding import gaussian_phase_np, DEFAULT_NUM_BASES
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase
from playground.open_duck_mini_v2 import constants

USE_MOTOR_SPEED_LIMITS = True


class EpisodicEvaluator(MJInferBase):
    def __init__(self, model_path, reference_data, onnx_model_path, zero_action=False):
        super().__init__(model_path)
        self.zero_action = zero_action
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.max_motor_velocity = 5.24  # rad/s

        self.ERM = EpisodicReferenceMotion(reference_data)
        self.num_bases = DEFAULT_NUM_BASES

        if not zero_action:
            from playground.common.onnx_infer import OnnxInfer

            self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # Termination geometry (mirror episodic.py _get_termination).
        self._torso_body_id = self.model.body(constants.ROOT_BODY).id
        self._head_body_id = self.model.body("head_assembly").id
        self._torso_ground_height = 0.08
        self._head_ground_height = 0.06
        self._head_torso_min_dist = 0.10

    def _reference_joints_14(self, i):
        """Reference actuated-joint targets (14) aligned to actuator order.

        The reference stores 16 joint slots; episodic tracking (ignore_head=False)
        drops the two antenna slots (indices 9, 10), leaving the 14 actuated
        joints in actuator order.
        """
        frame = np.asarray(self.ERM.get_frame(i), dtype=np.float32)
        joints_pos = frame[0:16]
        return np.concatenate([joints_pos[:9], joints_pos[11:16]])

    def _get_obs(self, data, imitation_phase):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = np.array(self.get_feet_contacts(data), dtype=np.float32)
        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                imitation_phase,
            ]
        )
        return obs

    def run(self):
        import mujoco

        # Reset to the home keyframe.
        self.data.qpos[:] = self.model.keyframe("home").qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_actuator
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        mujoco.mj_forward(self.model, self.data)

        nb = self.ERM.nb_steps
        up_z_hist, tilt_hist = [], []
        actual_joints, ref_joints = [], []
        contact_hist = []
        base_xy0 = self.data.qpos[0:2].copy()
        fell_step = None

        for i in range(nb):
            phi = i / (nb - 1)
            imitation_phase = gaussian_phase_np(phi, self.num_bases)

            obs = self._get_obs(self.data, imitation_phase)
            if self.zero_action:
                action = np.zeros(self.num_dofs)
            else:
                action = self.policy.infer(obs)

            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()

            self.motor_targets = self.default_actuator + action * self.action_scale
            if USE_MOTOR_SPEED_LIMITS:
                dt = self.sim_dt * self.decimation
                self.motor_targets = np.clip(
                    self.motor_targets,
                    self.prev_motor_targets - self.max_motor_velocity * dt,
                    self.prev_motor_targets + self.max_motor_velocity * dt,
                )
                self.prev_motor_targets = self.motor_targets.copy()

            self.data.ctrl[:] = self.motor_targets
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, self.data)

            # Uprightness from the trunk rotation matrix: R[2,2] is the world-z
            # component of the body z-axis (+1 = perfectly upright, 0 = horizontal,
            # <0 = fallen). This is unambiguous; the CPU "gravity" sensor is not a
            # clean projected-gravity signal on this model.
            R = self.data.xmat[self._torso_body_id].reshape(3, 3)
            up_z = float(R[2, 2])
            torso_z = float(self.data.xpos[self._torso_body_id][2])
            head_pos = self.data.xpos[self._head_body_id]
            head_z = float(head_pos[2])
            head_torso_dist = float(
                np.linalg.norm(head_pos - self.data.xpos[self._torso_body_id])
            )

            up_z_hist.append(up_z)
            tilt_hist.append(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
            actual_joints.append(self.get_actuator_joints_qpos(self.data.qpos).copy())
            ref_joints.append(self._reference_joints_14(i))
            contact_hist.append(self.get_feet_contacts(self.data))

            terminated = (
                up_z < 0.0
                or torso_z < self._torso_ground_height
                or head_z < self._head_ground_height
                or head_torso_dist < self._head_torso_min_dist
                or bool(np.isnan(self.data.qpos).any())
            )
            if terminated and fell_step is None:
                fell_step = i
                break

        # ---- Metrics ----
        actual_joints = np.array(actual_joints)
        ref_joints = np.array(ref_joints)
        up_z_hist = np.array(up_z_hist)
        tilt_hist = np.array(tilt_hist)

        survived = fell_step is None
        n_eval = len(actual_joints)

        mae = float(np.mean(np.abs(actual_joints - ref_joints)))
        per_joint_mae = np.mean(np.abs(actual_joints - ref_joints), axis=0)

        # Cheating: joint motion range (robot) vs reference range, over the eval
        # window. Focus on joints the reference actually moves.
        ref_range = ref_joints.max(axis=0) - ref_joints.min(axis=0)
        act_range = actual_joints.max(axis=0) - actual_joints.min(axis=0)
        moving = ref_range > 0.02  # joints with meaningful reference motion
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(moving, act_range / np.maximum(ref_range, 1e-6), np.nan)
        motion_ratio = float(np.nanmean(ratio)) if moving.any() else float("nan")

        base_xy_drift = float(
            np.linalg.norm(self.data.qpos[0:2].copy() - base_xy0)
        )

        joint_names = self.actuator_names

        print("\n================ EPISODIC WIGGLE EVAL ================")
        print(f"mode:                {'ZERO-ACTION baseline' if self.zero_action else 'ONNX policy'}")
        print(f"clip length:         {nb} control steps ({nb * self.sim_dt * self.decimation:.2f}s)")
        print(f"steps evaluated:     {n_eval}")
        print(f"UPRIGHT / survived:  {survived}  (fell at step {fell_step})" if not survived
              else f"UPRIGHT / survived:  {survived} (full clip)")
        print(f"min up_z:            {float(up_z_hist.min()):.3f}  (1.0=vertical, 0=horizontal, <0=fallen)")
        print(f"peak tilt:           {float(tilt_hist.max()):.1f} deg")
        print(f"mean tilt:           {float(tilt_hist.mean()):.1f} deg")
        print(f"base xy drift:       {base_xy_drift*100:.1f} cm")
        print(f"\nTRACKING MAE:        {mae:.4f} rad   (Disney episodic: 0.027-0.043 rad)")
        print("per-joint MAE (rad):")
        for nm, m, rr, ar in zip(joint_names, per_joint_mae, ref_range, act_range):
            tag = "  <-- wiggle" if rr > 0.02 else ""
            print(f"   {nm:16s} MAE={m:.4f}  ref_range={rr:.3f}  act_range={ar:.3f}{tag}")
        print(f"\nCHEATING check (motion ratio, robot/ref on wiggle joints): {motion_ratio:.2f}")
        print("   ratio ~1.0 = matches reference amplitude; <<1.0 = under-moving (cheating)")
        contact_hist = np.array(contact_hist)
        print(f"\nfoot contacts: left held {float(contact_hist[:,0].mean()):.2f} of steps, "
              f"right held {float(contact_hist[:,1].mean()):.2f} of steps  (expect ~1.0/1.0)")
        print("======================================================\n")

        return {
            "survived": survived,
            "fell_step": fell_step,
            "min_up_z": float(up_z_hist.min()),
            "peak_tilt_deg": float(tilt_hist.max()),
            "tracking_mae": mae,
            "motion_ratio": motion_ratio,
            "base_xy_drift_cm": base_xy_drift * 100,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, default=None)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/standing_wiggle.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--zero_action", action="store_true", default=False,
                        help="Hold the home pose (baseline) instead of running the policy.")
    args = parser.parse_args()

    if not args.zero_action and not args.onnx_model_path:
        parser.error("--onnx_model_path is required unless --zero_action is set")

    ev = EpisodicEvaluator(
        args.model_path, args.reference_data, args.onnx_model_path, args.zero_action
    )
    ev.run()
