# TRAINING — Phase 5 leg/neck reward split (Open Duck Mini v2)

This branch (`feature/leg-neck-reward-split`) restores Disney BD-X's leg/neck
imitation-reward split so the policy actually tracks the four head command
channels (`commands[3:7]`). It is ready to train the moment GPU access lands.
This file is the launch runbook for a CUDA box (RTX 3090 / 4090).

> Context: spike S0.1 measured the deployed policy's head-command DC gain at
> ≈0 because the old imitation reward discarded the neck/head joints. See
> `docs/animation_system_plan.md` §7 Phase 5 and Appendix C in the design repo.

---

## 0. What changed (so you know what you're training)

- `playground/open_duck_mini_v2/custom_rewards.py` — `reward_imitation` now
  tracks **legs and neck in separate buckets** instead of discarding the neck.
  Antennas (reference indices 9,10) are still excluded (they are not simulated
  joints). Controlled by `use_leg_neck_split`.
- `playground/open_duck_mini_v2/joystick.py` — new configurable weights under
  `reward_config.imitation_config` and split action-rate / action-acceleration
  penalties (`action_rate_leg/neck`, `action_accel_leg/neck`). A/B toggle
  `USE_LEG_NECK_SPLIT` (module constant → `imitation_config.use_leg_neck_split`),
  **defaulted to the NEW behaviour on this branch**.
- `playground/common/rewards.py` — `cost_action_rate_indexed`,
  `cost_action_acceleration_indexed` helpers for the leg/neck smoothness split.
- Observation layout, `action_scale`, `ctrl_dt`, domain randomisation and the
  ONNX export path are **unchanged**. Deployed obs is **101** and action **14**.

Reward weights (Disney BD-X Table I; `default_config()` in `joystick.py`):

| Term | Value | Config key |
|---|---|---|
| leg joint position | 15.0 | `reward_config.imitation_config.w_joint_pos_leg` |
| neck joint position | 100.0 | `reward_config.imitation_config.w_joint_pos_neck` |
| leg joint velocity | 1.0e-3 | `reward_config.imitation_config.w_joint_vel_leg` |
| neck joint velocity | 1.0 | `reward_config.imitation_config.w_joint_vel_neck` |
| action rate leg / neck | -1.5 / -5.0 | `reward_config.scales.action_rate_leg/neck` |
| action accel leg / neck | -0.45 / -5.0 | `reward_config.scales.action_accel_leg/neck` |

To train the **old (head-excluded) baseline** for an A/B comparison, set
`USE_LEG_NECK_SPLIT = False` in `joystick.py` (top of file) before launching.

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
    --output_dir checkpoints
```

Notes:
- `--env joystick` is the walking/standing policy that carries the head command
  channels. (`standing` is a separate dock/perpetual policy and is not this fix.)
- To resume: add `--restore_checkpoint_path <checkpoints/DATE_STEP>`.
- Watch it: `uv run tensorboard --logdir=checkpoints` — the key new curves are
  `reward/imitation`, `cost/action_rate_neck`, `cost/action_accel_neck`.

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

JAX_PLATFORMS=cpu python tests/test_leg_neck_reward_split.py   # proves the leg/neck slicing
JAX_PLATFORMS=cpu python tests/smoke_env_step.py              # env constructs, obs=101, no NaN
JAX_PLATFORMS=cpu python tests/smoke_train.py                 # PPO runs a few tiny iters, no NaN
JAX_PLATFORMS=cpu python tests/smoke_onnx_export.py           # export -> obs[1,101]/action[1,14]
```
