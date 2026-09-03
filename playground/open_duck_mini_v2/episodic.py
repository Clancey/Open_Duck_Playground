# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Episodic task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
# from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.episodic_reference_motion import EpisodicReferenceMotion
from playground.common.phase_encoding import gaussian_phase_jax, DEFAULT_NUM_BASES
from playground.common.rewards import (
    cost_torques,
    cost_action_rate,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True

# Which episodic clip to imitate. A monotonic (one-shot) full-body wiggle that the
# policy must reproduce while standing and balancing (paper Section V, Eq. 4).
EPISODIC_CLIP = "playground/open_duck_mini_v2/data/standing_wiggle.json"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=0.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(

                torques=-1.0e-3,
                action_rate=-0.5,  # was -1.5
                alive=20.0,
                imitation=1.0,
            ),
            tracking_sigma=0.01,  # was working at 0.01
            # Eq. 13 phase-windowed weight boost (Appendix A, paragraph 3). For the
            # "excited motion" (a full-body torso shake) Disney increases the weight
            # on angular-velocity tracking during the window where the robot rapidly
            # shakes its torso; without it the policy "cheats" by barely moving.
            # w_tilde(phi) = w0 + I[shake_start < phi < shake_end] * w_extra
            episodic_phase=config_dict.create(
                num_bases=DEFAULT_NUM_BASES,  # N = 50 Gaussian basis functions
            ),
            ang_vel_boost=config_dict.create(
                shake_start=0.25,  # phi window where the torso shake happens
                shake_end=0.85,
                w_extra=2.0,  # added to the nominal 0.5 ang-vel-tracking weight
            ),
        ),
        push_config=config_dict.create(
            enable=False,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],  # [-1.0, 1.0]
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,  # to make it easier
    )


class Episodic(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Imitate an episodic animation"""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.ERM = EpisodicReferenceMotion(EPISODIC_CLIP)
            # One-shot clip: phase advances monotonically 0 -> 1 across the clip.
            # Phase rate is 1 / clip-duration (paper Section V-A): one frame per
            # control step (clip FPS == 1/ctrl_dt == 50), so phi hits 1 after exactly
            # nb_steps steps, at which point the episode is terminated.
            self._clip_len = int(self.ERM.nb_steps)
            self._num_bases = int(
                self._config.reward_config.episodic_phase.num_bases
            )

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        # Bodies used by the episodic fall/self-collision termination (paper
        # Section V-B: terminate on "either head or torso in contact with the
        # ground" and on "self-collision between the head and torso"). This model
        # has no named collision geoms on the head/trunk, so we use body-world-frame
        # proxies: torso/head height above the floor and head<->trunk separation.
        self._head_body_id = self._mj_model.body("head_assembly").id
        # Nominal upright head z ~= 0.344 m, trunk z ~= 0.15 m, head-trunk
        # separation ~= 0.198 m (measured from the "home" keyframe).
        self._torso_ground_height = 0.08  # torso center this low => on the ground
        self._head_ground_height = 0.06  # head this low => head on the ground
        self._head_torso_min_dist = 0.10  # closer than this => head folded onto torso

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        # FIX #4 (episodic): an episodic policy replays ONE specific clip (paper
        # Eq. 4: x_t = f_epis(f_t, phi_t)) whose torso-orientation / world-velocity
        # references are defined in a fixed world frame. The walking env spawns at a
        # fully random yaw U(-pi, pi) because its command is expressed in the local
        # frame, but for episodic imitation a random yaw makes the (non-yaw-invariant)
        # orientation-tracking reward pure noise: the value function then sees wildly
        # different returns for the same phase -> v_loss explodes (observed ~4e4),
        # advantages become garbage and the policy erodes below its own init. Start
        # near the clip's nominal orientation with only a small yaw perturbation for
        # robustness.
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # multiply actual joints with noise (excluding floating base and backlash)
        # multiply actual joints with noise (excluding floating base and backlash).
        # Episodic imitation needs most resets to start near the stable home pose so
        # the policy can learn the stand+wiggle manifold instead of constantly
        # recovering from near-fallen starts. The walking env used U(0.5, 1.5)
        # (+/-50%, ~1/3 of seeds topple immediately); tighten to +/-10% here.
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.9, maxval=1.1
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        # Episodic policy (paper Eq. 4): x_t = f_epis(f_t, phi_t) -- there is NO
        # command argument. "There is no additional user input until the episodic
        # motion finishes" (Section VI-B). We keep a zero command in info only so
        # the shared reward signature is satisfied; it never enters the observation
        # and never influences the episodic reward.
        cmd = jp.zeros(7)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if USE_IMITATION_REWARD:
            current_reference_motion = self.ERM.get_frame(0)
        else:
            current_reference_motion = jp.zeros(0)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            # Monotonic phase projected onto N Gaussian basis functions (paper
            # Section V / Appendix A). Replaces the old cyclic [cos, sin] pair.
            "imitation_phase": gaussian_phase_jax(0.0, self._num_bases),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        if USE_IMITATION_REWARD:
            # FIX #1: phase is MONOTONIC, not cyclic. Advance the frame index by one
            # per control step (no modulo) so phi = i / (clip_len - 1) increases
            # monotonically 0 -> 1 across the one-shot clip (paper Section V). The
            # index is clipped to the last valid frame; the episode is terminated
            # once phi reaches 1 (handled below in _get_termination).
            state.info["imitation_i"] += 1
            idx = jp.clip(state.info["imitation_i"], 0, self._clip_len - 1)
            phi = idx / (self._clip_len - 1)
            # FIX #1: encode phi with N Gaussian basis functions (highly local in
            # time), not a single [cos, sin] pair which cannot represent a one-shot
            # clip's local structure.
            state.info["imitation_phase"] = gaussian_phase_jax(phi, self._num_bases)
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.ERM.get_frame(
                jp.clip(state.info["imitation_i"], 0, self._clip_len - 1)
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # Handle action delay
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[
            action_idx[0]
        ]  # action with delay

        # self.action_filter.push(action_w_delay)
        # action_w_delay = self.action_filter.get_filtered_action()

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )  # floating base x,y
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        ####

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]

            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets
                - self._config.max_motor_velocity * self.dt,  # control dt
                prev_motor_targets
                + self._config.max_motor_velocity * self.dt,  # control dt
            )

        # trying mix manual head control with policy ?
        # head_motor_targets = state.info["command"][3:] + motor_targets[5:9]
        # motor_targets.at[5:9].set(head_motor_targets)  # head joints

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        state.info["motor_targets"] = motor_targets

        contact = jp.array(
            [
                geoms_colliding(data, geom_id, self._floor_geom_id)
                for geom_id in self._feet_geom_id
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        # FIX #2: force a transition (episode end) once the one-shot clip finishes
        # (phi == 1, i.e. imitation_i has reached the last frame) -- paper Section V:
        # "Once the motion ends, a transition to a new motion is forced." Also
        # terminate on head/torso ground contact or head-torso self-collision.
        clip_done = state.info["imitation_i"] >= (self._clip_len - 1)
        done = self._get_termination(data) | clip_done

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        # FIXME
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        # TODO removed clipping
        # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        reward = sum(rewards.values()) * self.dt
        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action  # was
        # state.info["last_act"] = motor_targets  # became
        # FIX #3: no command is sampled during an episodic motion (paper Eq. 4 has no
        # command argument; Section VI-B: "There is no additional user input until
        # the episodic motion finishes"). The old rollout-length hack that resampled
        # a command and reset "step" at step > 500 is removed; the episode is now
        # bounded by the clip itself (clip_done above) or by a fall.
        state.info["step"] = jp.where(done, 0, state.info["step"])
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        # Paper Section V-B terminates on "either head or torso in contact with the
        # ground" and on "self-collision between the head and torso". This model has
        # no named collision geoms on the head/trunk, so we use robust world-frame
        # geometric proxies instead of geom contact pairs.
        # (a) tilt: torso up-axis points below horizontal (fallen over).
        fall_termination = self.get_gravity(data)[-1] < 0.0
        # (b) torso in contact with ground: torso center dropped near the floor.
        torso_z = data.xpos[self._torso_body_id][2]
        torso_on_ground = torso_z < self._torso_ground_height
        # (c) head in contact with ground: head body dropped near the floor.
        head_pos = data.xpos[self._head_body_id]
        head_on_ground = head_pos[2] < self._head_ground_height
        # (d) head-torso self-collision: head folded back onto the torso. Matters for
        # an aggressive standing wiggle (Section V-B).
        torso_pos = data.xpos[self._torso_body_id]
        head_torso_dist = jp.linalg.norm(head_pos - torso_pos)
        self_collision = head_torso_dist < self._head_torso_min_dist
        return (
            fall_termination
            | torso_on_ground
            | head_on_ground
            | self_collision
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        # accelerometer[0] += 1.3 # TODO testing
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                # FIX #3: NO command in the episodic observation (paper Eq. 4:
                # x_t = f_epis(f_t, phi_t)). The command is pure noise in the
                # advantage estimate for a one-shot clip, so it is removed entirely.
                noisy_joint_angles - self._default_actuator,  # 14
                noisy_joint_vel * self._config.dof_vel_scale,  # 14
                info["last_act"],  # 14
                info["last_last_act"],  # 14
                info["last_last_last_act"],  # 14
                info["motor_targets"],  # 14
                contact,  # 2
                # FIX #1: monotonic phase encoded with N=50 Gaussian bases replaces
                # the old scalar imitation_i / cyclic [cos, sin] pair.
                info["imitation_phase"],  # num_bases (50)
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 14
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_phase"],  # num_bases (50)
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        # Recover the monotonic phase from the frame index for the Eq. 13 window.
        phi = jp.clip(info["imitation_i"], 0, self._clip_len - 1) / (
            self._clip_len - 1
        )
        boost_cfg = self._config.reward_config.ang_vel_boost

        ret = {
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(
                action, info["last_act"]
            ),
            "alive": reward_alive(),
            "imitation": reward_imitation(  # FIXME, this reward is so adhoc...
                self.get_floating_base_qpos(data.qpos),  # floating base qpos
                self.get_floating_base_qvel(data.qvel),  # floating base qvel
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                contact,
                info["current_reference_motion"],
                info["command"],
                USE_IMITATION_REWARD,
                ignore_head=False,
                episodic=True,
                # Eq. 13 phase-windowed weight boost on angular-velocity tracking.
                phase=phi,
                ang_vel_shake_start=boost_cfg.shake_start,
                ang_vel_shake_end=boost_cfg.shake_end,
                ang_vel_extra_weight=boost_cfg.w_extra,
            ),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 8)

        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        neck_pitch = jax.random.uniform(
            rng5,
            minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
        )

        head_pitch = jax.random.uniform(
            rng6,
            minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
            maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
        )

        head_yaw = jax.random.uniform(
            rng7,
            minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
            maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
        )

        head_roll = jax.random.uniform(
            rng8,
            minval=self._config.head_roll_range[0] * self._config.head_range_factor,
            maxval=self._config.head_roll_range[1] * self._config.head_range_factor,
        )

        return jp.where(
            jax.random.bernoulli(rng4, p=1.0),
            jp.zeros(7),
            jp.hstack(
                [
                    lin_vel_x,
                    lin_vel_y,
                    ang_vel_yaw,
                    neck_pitch,
                    head_pitch,
                    head_yaw,
                    head_roll,
                ]
            ),
        )
