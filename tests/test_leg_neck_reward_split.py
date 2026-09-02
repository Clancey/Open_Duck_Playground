"""Unit checks for the leg/neck imitation reward split (Phase 5).

These tests prove the 16<->14 joint slicing in ``reward_imitation`` is correct:
an error injected into ONLY a head/neck channel moves ONLY the neck bucket, and
an error injected into ONLY a leg channel moves ONLY the leg bucket. They also
verify the ``use_leg_neck_split`` A/B toggle and the split action-rate /
action-acceleration penalties.

Run:  python tests/test_leg_neck_reward_split.py
(or)  pytest tests/test_leg_neck_reward_split.py
"""

import os
import sys

import numpy as np
import jax.numpy as jp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playground.open_duck_mini_v2.custom_rewards import reward_imitation
from playground.common.rewards import (
    cost_action_rate,
    cost_action_rate_indexed,
    cost_action_acceleration_indexed,
)

# Canonical index maps (mirrors open_duck_anim/joint_order.py).
# 16-DOF reference: 0-4 left leg | 5-8 head/neck | 9,10 antennas | 11-15 right leg
# 14-DOF qpos:      0-4 left leg | 5-8 head/neck | 9-13 right leg
ANTENNA_IDX_16 = [9, 10]
HEAD_YAW_14 = 7          # neck bucket
LEFT_HIP_YAW_14 = 0      # leg bucket

W_POS_LEG = 15.0
W_POS_NECK = 100.0
W_VEL_LEG = 1.0e-3
W_VEL_NECK = 1.0

DELTA = 0.3  # a known injected joint error (rad)


def build_baseline():
    """Build a self-consistent 40-dim reference frame + matching qpos/qvel.

    Zero joint error everywhere; base velocities and contacts match the frame,
    so every non-joint reward term is constant across perturbations and cancels
    in a delta comparison.
    """
    rng = np.random.default_rng(0)
    ref_joint_pos16 = rng.uniform(-0.5, 0.5, size=16)
    ref_joint_vel16 = rng.uniform(-0.5, 0.5, size=16)
    contacts = np.array([1.0, 1.0])
    lin_vel = rng.uniform(-0.1, 0.1, size=3)
    ang_vel = rng.uniform(-0.1, 0.1, size=3)

    frame = np.concatenate(
        [ref_joint_pos16, ref_joint_vel16, contacts, lin_vel, ang_vel]
    )
    assert frame.shape == (40,)

    # qpos/qvel are the 14-DOF versions (antennas dropped) that exactly match.
    joints_qpos = np.delete(ref_joint_pos16, ANTENNA_IDX_16)
    joints_qvel = np.delete(ref_joint_vel16, ANTENNA_IDX_16)
    assert joints_qpos.shape == (14,)

    # Floating base: [x y z qw qx qy qz] for qpos; [vx vy vz wx wy wz] for qvel.
    base_qpos = np.array([0.0, 0.0, 0.16, 1.0, 0.0, 0.0, 0.0])
    base_qvel = np.concatenate([lin_vel, ang_vel])

    cmd = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # cmd_norm>0.01 gate open
    return frame, base_qpos, base_qvel, joints_qpos, joints_qvel, contacts, cmd


def call(frame, base_qpos, base_qvel, qpos, qvel, contacts, cmd, split):
    return float(
        reward_imitation(
            jp.array(base_qpos),
            jp.array(base_qvel),
            jp.array(qpos),
            jp.array(qvel),
            jp.array(contacts),
            jp.array(frame),
            jp.array(cmd),
            True,  # use_imitation_reward
            split,  # use_leg_neck_split
            W_POS_LEG,
            W_POS_NECK,
            W_VEL_LEG,
            W_VEL_NECK,
        )
    )


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol * (1.0 + abs(b))


def test_head_yaw_error_moves_only_neck_bucket():
    frame, bq, bv, qpos, qvel, contacts, cmd = build_baseline()
    base = call(frame, bq, bv, qpos, qvel, contacts, cmd, split=True)

    # Inject a known error into ONLY head_yaw (a neck channel).
    qpos_h = qpos.copy()
    qpos_h[HEAD_YAW_14] += DELTA
    perturbed = call(frame, bq, bv, qpos_h, qvel, contacts, cmd, split=True)

    delta = base - perturbed  # reward drops (penalty), so base > perturbed
    expected = W_POS_NECK * DELTA**2  # only neck-position bucket should react
    assert approx(delta, expected), (
        f"head_yaw error should cost exactly w_neck*delta^2={expected:.4f}, "
        f"got {delta:.4f}. Slicing routes head_yaw into the WRONG bucket."
    )
    # Prove it is NOT being charged at the leg weight (off-by-one guard).
    assert not approx(delta, W_POS_LEG * DELTA**2), (
        "head_yaw error was charged at the LEG weight -> slicing bug."
    )
    print(f"[PASS] head_yaw error -> neck bucket only: dR={delta:.5f} == "
          f"w_neck*delta^2={expected:.5f}")


def test_leg_error_moves_only_leg_bucket():
    frame, bq, bv, qpos, qvel, contacts, cmd = build_baseline()
    base = call(frame, bq, bv, qpos, qvel, contacts, cmd, split=True)

    qpos_l = qpos.copy()
    qpos_l[LEFT_HIP_YAW_14] += DELTA
    perturbed = call(frame, bq, bv, qpos_l, qvel, contacts, cmd, split=True)

    delta = base - perturbed
    expected = W_POS_LEG * DELTA**2
    assert approx(delta, expected), (
        f"leg error should cost w_leg*delta^2={expected:.4f}, got {delta:.4f}"
    )
    assert not approx(delta, W_POS_NECK * DELTA**2), (
        "leg error was charged at the NECK weight -> slicing bug."
    )
    print(f"[PASS] left_hip_yaw error -> leg bucket only: dR={delta:.5f} == "
          f"w_leg*delta^2={expected:.5f}")


def test_split_off_ignores_head():
    """With the split disabled (original behaviour) head error is free."""
    frame, bq, bv, qpos, qvel, contacts, cmd = build_baseline()
    base = call(frame, bq, bv, qpos, qvel, contacts, cmd, split=False)

    qpos_h = qpos.copy()
    qpos_h[HEAD_YAW_14] += DELTA
    perturbed = call(frame, bq, bv, qpos_h, qvel, contacts, cmd, split=False)

    assert approx(base, perturbed), (
        "In split=OFF (head-excluded) mode a head_yaw error must NOT change the "
        f"reward, but reward moved by {base - perturbed:.5f}."
    )
    # And leg error must STILL be charged even with split off.
    qpos_l = qpos.copy()
    qpos_l[LEFT_HIP_YAW_14] += DELTA
    perturbed_leg = call(frame, bq, bv, qpos_l, qvel, contacts, cmd, split=False)
    assert approx(base - perturbed_leg, W_POS_LEG * DELTA**2)
    print("[PASS] split=OFF: head error is free, leg error still penalised "
          "(reproduces the deployed policy's training reward)")


def test_neck_velocity_bucket():
    frame, bq, bv, qpos, qvel, contacts, cmd = build_baseline()
    base = call(frame, bq, bv, qpos, qvel, contacts, cmd, split=True)
    qvel_h = qvel.copy()
    qvel_h[HEAD_YAW_14] += DELTA
    perturbed = call(frame, bq, bv, qpos, qvel_h, contacts, cmd, split=True)
    delta = base - perturbed
    expected = W_VEL_NECK * DELTA**2
    assert approx(delta, expected), (
        f"head_yaw VELOCITY error should cost w_vel_neck*delta^2={expected:.5f}, "
        f"got {delta:.5f}"
    )
    print(f"[PASS] head_yaw velocity error -> neck-vel bucket: dR={delta:.6f}")


def test_antennas_never_tracked():
    """Perturbing antenna reference values must not change the reward: they have
    no qpos counterpart and must be excluded from every bucket."""
    frame, bq, bv, qpos, qvel, contacts, cmd = build_baseline()
    base = call(frame, bq, bv, qpos, qvel, contacts, cmd, split=True)
    frame2 = frame.copy()
    frame2[ANTENNA_IDX_16] += 1.0          # antenna positions in ref frame
    frame2[[16 + i for i in ANTENNA_IDX_16]] += 1.0  # antenna velocities
    perturbed = call(frame2, bq, bv, qpos, qvel, contacts, cmd, split=True)
    assert approx(base, perturbed), (
        "Antenna reference values leaked into the reward -> 16<->14 misalignment."
    )
    print("[PASS] antenna reference channels are excluded from all buckets")


def test_action_rate_split_indices():
    """The split action-rate / accel penalties must sum to the combined one, and
    partition cleanly into leg (10) + neck (4) = 14 channels."""
    leg = jp.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
    neck = jp.array([5, 6, 7, 8])
    rng = np.random.default_rng(1)
    act = jp.array(rng.uniform(-1, 1, size=14))
    last = jp.array(rng.uniform(-1, 1, size=14))

    combined = float(cost_action_rate(act, last))
    leg_c = float(cost_action_rate_indexed(act, last, leg))
    neck_c = float(cost_action_rate_indexed(act, last, neck))
    assert approx(leg_c + neck_c, combined), (
        f"leg+neck action-rate {leg_c + neck_c:.5f} != combined {combined:.5f}"
    )

    # Action acceleration only reacts to a genuine 2nd difference. Put a pure
    # 2nd-difference bump on head_yaw and confirm it lands in the neck bucket.
    a = jp.zeros(14).at[7].set(DELTA)  # a_t - 2 a_{t-1} + a_{t-2} = DELTA at head_yaw
    lm = jp.zeros(14)
    lm2 = jp.zeros(14)
    accel_leg = float(cost_action_acceleration_indexed(a, lm, lm2, leg))
    accel_neck = float(cost_action_acceleration_indexed(a, lm, lm2, neck))
    assert approx(accel_leg, 0.0), "head_yaw accel leaked into leg bucket"
    assert approx(accel_neck, DELTA**2), "head_yaw accel missing from neck bucket"
    print(f"[PASS] action-rate leg+neck == combined ({combined:.5f}); "
          f"action-accel head_yaw -> neck bucket ({accel_neck:.5f})")


def main():
    tests = [
        test_head_yaw_error_moves_only_neck_bucket,
        test_leg_error_moves_only_leg_bucket,
        test_split_off_ignores_head,
        test_neck_velocity_bucket,
        test_antennas_never_tracked,
        test_action_rate_split_indices,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} leg/neck reward-split checks PASSED.")


if __name__ == "__main__":
    main()
