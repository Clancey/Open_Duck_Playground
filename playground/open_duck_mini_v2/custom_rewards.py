import jax
import jax.numpy as jp

def reward_imitation(
    base_qpos: jax.Array,
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
    use_leg_neck_split: bool = True,
    w_joint_pos_leg: float = 15.0,
    w_joint_pos_neck: float = 100.0,
    w_joint_vel_leg: float = 1.0e-3,
    w_joint_vel_neck: float = 1.0,
    neck_tracks_command: bool = True,
) -> jax.Array:
    """Imitation (reference-motion tracking) reward.

    Two modes, selected by ``use_leg_neck_split`` (a *static* Python bool so the
    branch is resolved at JIT-trace time, adding no runtime cost):

    * ``False`` — the original Open Duck behaviour: the neck/head joints are
      discarded and only the leg joints are tracked, with a single combined
      position weight ``w_joint_pos_leg`` and velocity weight ``w_joint_vel_leg``.
      This is the reward the currently deployed policy was trained with, and is
      why it does not follow head commands (spike S0.1).
    * ``True`` (default on this branch) — Disney BD-X Table I (arXiv:2501.05204)
      leg/neck split: legs and neck are tracked in *separate* buckets so the neck
      can be weighted much more heavily (``w_joint_pos_neck`` ~6.7x the leg
      weight).

    ITERATION 2 FIX (``neck_tracks_command``, default True): the neck bucket now
    tracks the sampled **head command** ``cmd[3:7]`` instead of the walking-clip's
    nominal neck pose. Iteration 1 weighted the neck 100x while its target came
    from ``get_reference_motion`` — which is indexed only by locomotion velocity
    and gait phase and NEVER sees the head command. That pinned the head to the
    clip's nominal pose and actively suppressed command-following (spike S0.1:
    DC gain ~=0). Retargeting the same heavily-weighted term to the command turns
    it into the head-command-tracking signal it was always meant to be. This term
    is only active while walking (the whole imitation reward is gated by
    ``cmd_norm > 0.01`` at the end); the standing regime is covered by a separate
    ``cost_head_command_tracking`` term in the env (see joystick.py).

    All weights are passed in from the env config (``reward_config.imitation_config``)
    so they can be swept without editing this file.
    """
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = jp.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_contact = 1.0

    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37

    angular_vel_slice_start = 37
    angular_vel_slice_end = 40

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    root_quat_slice_start = 3
    root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    ref_base_orientation_quat = reference_frame[
        root_quat_slice_start:root_quat_slice_end
    ]
    ref_base_orientation_quat = ref_base_orientation_quat / jp.linalg.norm(
        ref_base_orientation_quat
    )  # normalize the quat
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / jp.linalg.norm(
        base_orientation
    )  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]

    # Joint-ordering map (see playground/common/poly_reference_motion.py:6-22 and
    # open_duck_anim/joint_order.py, the tested single source of truth):
    #
    #   Reference frame is 16-DOF (authoring order):
    #     0-4  left leg | 5 neck_pitch 6 head_pitch 7 head_yaw 8 head_roll |
    #     9 left_antenna 10 right_antenna | 11-15 right leg
    #   Simulation qpos/qvel is 14-DOF (NO antennas), same order minus antennas:
    #     0-4  left leg | 5-8 head/neck | 9-13 right leg
    #
    # The antennas (reference indices 9,10) are NOT simulated joints -- they have
    # no qpos/actuator counterpart -- so they are excluded from EVERY imitation
    # bucket. They can never be tracked and including them would misalign the
    # 16<->14 mapping. Hence the asymmetric slices below: legs drop 5:11 from the
    # 16-DOF reference but 5:9 from the 14-DOF qpos.

    # --- Leg bucket (indices 0:5 and, after dropping head+antennas, the right leg)
    ref_leg_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])  # ref 16 -> 10 legs
    leg_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])  # qpos 14 -> 10 legs
    ref_leg_vel = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    leg_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # --- Neck/head bucket (neck_pitch, head_pitch, head_yaw, head_roll)
    # indices 5:9 in BOTH the 16-DOF reference and the 14-DOF qpos.
    #
    # ITERATION 2: the neck TARGET is the sampled head command (cmd[3:7]), NOT the
    # walking clip. cmd is [lin_x, lin_y, ang_yaw, neck_pitch, head_pitch,
    # head_yaw, head_roll]; cmd[3:7] are exactly the 4 head channels, aligned with
    # qpos[5:9] (open_duck_anim/joint_order.py: head block is indices 5..8 in both
    # the 14- and 16-DOF orders). The command is a static setpoint, so the neck
    # velocity target is zero (tracking the clip's neck velocity would fight the
    # command). Set neck_tracks_command=False to restore the iteration-1 behaviour
    # (track the clip) for A/B comparison.
    neck_pos = joints_qpos[5:9]
    neck_vel = joints_qvel[5:9]
    if neck_tracks_command:
        ref_neck_pos = cmd[3:7]
        ref_neck_vel = jp.zeros(4)
    else:
        ref_neck_pos = ref_joint_pos[5:9]
        ref_neck_vel = ref_joint_vels[5:9]

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = jp.exp(-200.0 * jp.sum(jp.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = jp.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation

    # TODO ignore yaw here, we just want xy orientation
    torso_orientation_rew = (
        jp.exp(-20.0 * jp.sum(jp.square(base_orientation - ref_base_orientation_quat)))
        * w_torso_orientation
    )

    lin_vel_xy_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    # Joint tracking: leg and neck buckets. Disney BD-X weights the neck position
    # error ~6.7x the leg error (Table I). When the split is disabled we fall back
    # to the original head-excluded behaviour (leg bucket only).
    leg_pos_rew = -jp.sum(jp.square(leg_pos - ref_leg_pos)) * w_joint_pos_leg
    leg_vel_rew = -jp.sum(jp.square(leg_vel - ref_leg_vel)) * w_joint_vel_leg
    neck_pos_rew = -jp.sum(jp.square(neck_pos - ref_neck_pos)) * w_joint_pos_neck
    neck_vel_rew = -jp.sum(jp.square(neck_vel - ref_neck_vel)) * w_joint_vel_neck

    if use_leg_neck_split:
        joint_pos_rew = leg_pos_rew + neck_pos_rew
        joint_vel_rew = leg_vel_rew + neck_vel_rew
    else:
        # Original behaviour: neck/head discarded entirely.
        joint_pos_rew = leg_pos_rew
        joint_vel_rew = leg_vel_rew

    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )
    contact_rew = jp.sum(contacts == ref_foot_contacts) * w_contact

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)