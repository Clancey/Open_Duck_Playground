# TRAINING — Phase 5 head-command tracking (Open Duck Mini v2)

This branch (`feature/head-command-tracking`) is **iteration 2** of the Phase 5
head-command fix. It builds on `feature/leg-neck-reward-split` (iteration 1).

> **Why iteration 2.** Iteration 1 split the imitation reward into leg/neck
> buckets and weighted the neck 100×, but retrained the policy still measured DC
> gain ≈0 on all four head channels (S0.1 FAIL). Root cause: the neck bucket's
> *target* came from `PolyReferenceMotion.get_reference_motion(dx,dy,dtheta,i)`,
> which is indexed only by locomotion velocity and gait phase and **never sees the
> head command**. Weighting a command-independent target 100× just pinned the head
> to the walking clip's nominal pose — actively *suppressing* command response.
> Iteration 2 makes the heavily-weighted term track the **command**.

This file is the launch runbook for a CUDA box (RTX 3090 / 4090).

---

## 0. What changed (so you know what you're training)

**Iteration 2 (this branch) — the head now tracks the command:**

- `custom_rewards.py` — `reward_imitation` gains `neck_tracks_command` (default
  `True`): the neck bucket's target is the sampled head command `cmd[3:7]`
  (neck-velocity target = 0), **not** the walking clip. This is active only while
  walking (the imitation reward is gated by `||cmd[:3]|| > 0.01`).
- `playground/common/rewards.py` — new `cost_head_command_tracking(qpos, cmd)`:
  tracks `cmd[3:7]` while **standing** (`||cmd[:3]|| < 0.01`), the regime where the
  imitation reward is gated off entirely. Mirrors the proven `standing.py`
  mechanism, restricted to standing so it never double-counts with the walking
  imitation term. Wired as `reward_config.scales.head_command = -3.0`.
- `joystick.py` — `sample_command` now **decouples** head zeroing from locomotion
  zeroing and injects explicit standing episodes (`stand_probability=0.2`,
  `head_zero_probability=0.1`), so "stand still + move head" is actually trained
  (it never was — the old 10%-zero-everything path zeroed head and locomotion
  together, and locomotion is otherwise ~always nonzero). The head command is also
  resampled every `head_command_resample_steps=100` env steps (~2 s) via
  `sample_head_command`, for a denser command→response signal than the 500-step
  full-command resample.
- Master A/B flag `HEAD_COMMAND_TRACKING` (module constant, default `True`) gates
  the whole iteration-2 package; set it `False` to reproduce the iteration-1
  (clip-tracking) behaviour.

**Inherited from iteration 1 (still present):**

- `reward_imitation` tracks **legs and neck in separate buckets** (`use_leg_neck_split`);
  antennas (reference indices 9,10) excluded (not simulated joints).
- Configurable weights under `reward_config.imitation_config`; split action-rate /
  action-acceleration penalties (`action_rate_leg/neck`, `action_accel_leg/neck`).
- Observation layout, `action_scale`, `ctrl_dt`, domain randomisation and the
  ONNX export path are **unchanged**. Deployed obs is **101** and action **14**.

Reward weights (`default_config()` in `joystick.py`):

| Term | Value | Config key |
|---|---|---|
| leg joint position | 15.0 | `reward_config.imitation_config.w_joint_pos_leg` |
| neck joint position | 100.0 | `reward_config.imitation_config.w_joint_pos_neck` |
| leg joint velocity | 1.0e-3 | `reward_config.imitation_config.w_joint_vel_leg` |
| neck joint velocity | 1.0 | `reward_config.imitation_config.w_joint_vel_neck` |
| neck tracks command | True | `reward_config.imitation_config.neck_tracks_command` |
| head_command (standing) | -3.0 | `reward_config.scales.head_command` |
| action rate leg / neck | -1.5 / -5.0 | `reward_config.scales.action_rate_leg/neck` |
| action accel leg / neck | -0.45 / -5.0 | `reward_config.scales.action_accel_leg/neck` |
| stand probability | 0.2 | `stand_probability` |
| head resample steps | 100 | `head_command_resample_steps` |

To train the **iteration-1 (failed) behaviour** for an A/B comparison, set
`HEAD_COMMAND_TRACKING = False` in `joystick.py` (top of file) before launching.

---

## 1. Environment setup on the CUDA box

The repo is `uv`-managed and its `pyproject.toml` already pins `jax[cuda12]`, so
on a Linux CUDA host the canonical path just works — **use `uv`, do not
hand-assemble a venv** (the macOS-CPU dance in `tests/` was only needed because
there are no CUDA/JAX wheels for macOS):

```bash
# from the repo root
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync                                            # installs jax[cuda12], brax, mjx, tf2onnx...
# sanity: confirm JAX sees the GPU
uv run python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

If `uv sync` resolves a JAX newer than the training stack tolerates, pin the
known-good CUDA line explicitly (this mirrors the CPU pins that worked here —
`brax` currently needs `jax < 0.11`):

```bash
uv pip install "jax[cuda12]==0.5.3" "jaxlib==0.5.3" "flax==0.10.4"
```

The imitation reward requires the reference-motion polynomials; they are already
committed at `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`
and `USE_IMITATION_REWARD = True` is already set.

---

## 2. Launch training

Train the **`flat_terrain_backlash`** task — the backlash variants are the
current sim2real win (see `constants.py:20-36` and the repo README "Current
win"). 300M steps matches the repo's proven config:

```bash
uv run playground/open_duck_mini_v2/runner.py \
    --env joystick \
    --task flat_terrain_backlash \
    --num_timesteps 300000000 \
    --output_dir checkpoints_head_cmd
```

Notes:
- `--output_dir checkpoints_head_cmd` keeps iteration-2 artefacts separate from
  the iteration-1 run — do not overwrite the previous checkpoints.
- `--env joystick` is the walking/standing policy that carries the head command
  channels. (`standing` is a separate dock/perpetual policy and is not this fix.)
- To resume: add `--restore_checkpoint_path <checkpoints_head_cmd/DATE_STEP>`.
- Watch it: `uv run tensorboard --logdir=checkpoints_head_cmd` — the key new
  curves are `reward/imitation` (now command-driven while walking) and
  `cost/head_command` (standing head-tracking error; should fall toward 0).

### Expected wall-clock (rough — no GPU was available to calibrate)

MJX PPO with `num_envs=8192`, 300M steps on this ~14-DOF biped:

| GPU | Estimate |
|---|---|
| RTX 4090 | ~3–6 h |
| RTX 3090 | ~5–9 h |

These are order-of-magnitude only; the first run establishes the real number.
Reduce `--num_timesteps` (e.g. 150M) for a faster first pass if you just want a
policy to re-measure S0.1 against.

---

## 3. Checkpoints & ONNX export

Checkpoints and ONNX are written by `policy_params_fn`
(`playground/common/runner.py:68-83`) into `--output_dir` (default
`checkpoints/`) on **every eval**, named `YYYY_MM_DD_HHMMSS_<step>`:

- `checkpoints/<DATE>_<STEP>/`      — orbax checkpoint (resume/restore)
- `checkpoints/<DATE>_<STEP>.onnx`  — exported policy, **opset 11**

So a training run exports ONNX automatically; no separate step is required.
Each ONNX is validated to have input `obs [1,101]` and output
`continuous_actions [1,14]`.

### Manual / one-off ONNX export

`export_onnx()` (`playground/common/export_onnx.py`) takes the brax params tree,
the action size, the PPO params and the obs size. The tiny end-to-end example in
`tests/smoke_onnx_export.py` shows the exact call and verifies the shapes with
onnxruntime.

### Sanity-check an exported policy

```bash
uv run python -c "
import onnxruntime as ort, numpy as np
s = ort.InferenceSession('checkpoints/<DATE>_<STEP>.onnx', providers=['CPUExecutionProvider'])
i, o = s.get_inputs()[0], s.get_outputs()[0]
print(i.name, i.shape, '->', o.name, o.shape)   # obs [1,101] -> continuous_actions [1,14]
print(s.run(None, {i.name: np.zeros((1,101), np.float32)})[0].shape)
"
```

Then run it in the MuJoCo loop:

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o checkpoints/<DATE>_<STEP>.onnx
```

---

## 4. Acceptance gate (from the plan, Phase 5)

Re-run spike **S0.1** against the retrained ONNX on the same harness
(`experiments/animation/spike_s01_head_response.py` in the design repo) and
require, standing and walking:

- per-channel DC gain **≥ 0.6** (was ≈0 on the current checkpoint),
- cross-coupling **≤ 0.2**,
- in-sim RMS head-tracking error reduced **≥ 2×** vs the current checkpoint.

Only once that passes do you delete the additive head lines
(`Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py:310-311`) and switch head
injection to the command channel — **in the same change** as shipping this ONNX.

---

## 5. Local correctness checks (already run on CPU, no GPU needed)

```bash
# venv used here (outside the repo; CPU-only JAX):
python -m venv ~/.oduck_train_venv   # or: uv venv ~/.oduck_train_venv
source ~/.oduck_train_venv/bin/activate

JAX_PLATFORMS=cpu python tests/test_leg_neck_reward_split.py   # 9 checks; proves slicing + command drives the neck target
JAX_PLATFORMS=cpu python tests/smoke_env_step.py              # env constructs, obs=101, head_command bucket present, no NaN
JAX_PLATFORMS=cpu python tests/smoke_train.py                 # PPO runs a few tiny iters, no NaN
JAX_PLATFORMS=cpu python tests/smoke_onnx_export.py           # export -> obs[1,101]/action[1,14]
```
