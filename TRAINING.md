# TRAINING — Phase 5 head passthrough (Open Duck Mini v2)

This branch (`feature/head-passthrough-training`) is **iteration 3** of the Phase 5
head work. It builds on `feature/head-command-tracking` (iteration 2) and
`feature/leg-neck-reward-split` (iteration 1).

> **Why iteration 3 — the design premise was wrong.** Iterations 1 and 2 tried to
> make the *walking policy learn* to follow the head command via reward shaping
> (leg/neck imitation split; then redirecting the neck target to the command).
> Both retrained and both measured DC gain ≈0 on all four head channels (S0.1
> FAIL) even though the reward was implemented correctly and active. Iteration 2's
> post-mortem probed the trained ONNX directly: the policy's head *actions*
> respond to a unit head command by only ~0.05 (~80× too weak) — a credit-
> assignment / signal-to-noise failure, not a weight-tuning problem. The head's
> marginal reward is buried under locomotion/DR/push variance once PPO normalises
> advantages.
>
> The evidence converges on a different conclusion: **the head was never meant to
> be learned by the walking policy — the deployed runtime drives it *additively*.**
> `v2_rl_walk_mujoco.py:310-311` does `motor_targets[5:9] = command[3:7] +
> motor_targets[5:9]`, `joystick.py:504` has that exact line commented out, and
> `standing.py`'s head reward only works with locomotion off. The additive path is
> the **correct architecture**, not a defect.
>
> So iteration 3 stops trying to *learn* head tracking. It enables the additive
> head passthrough **during training** so the legs learn to balance while the head
> is driven through its full randomised command range. **Goal = sim2real fidelity
> and a wider safe head envelope**, NOT S0.1 gain (which is trivially ≈1.0 with a
> passthrough and proves nothing about learning).

This file is the launch runbook for a CUDA box (RTX 3090 / 4090).

---

## 0. What changed (so you know what you're training)

**Iteration 3 (this branch) — additive head passthrough during training:**

- `joystick.py` `step()` — the 4 head joints (`motor_targets[5:9]`) are now driven
  additively from the sampled head command: `motor_targets[5:9] = command[3:7] +
  motor_targets[5:9]`, applied **after** the motor-speed clip, **exactly** mirroring
  the deployed runtime (`v2_rl_walk_mujoco.py:310-311`). The policy's own head
  action still rides on top (additive, not override) — same semantics as the
  runtime. The command is added raw (not rate-limited) so fast command transitions
  are a genuine dynamic disturbance the legs must reject.
- **Two motor-target values are tracked** to preserve runtime parity: the speed
  clip tracks the **pre-additive** target (`info["motor_targets_preadd"]`, like the
  runtime's `prev_motor_targets` which is saved before the head add), while
  `info["motor_targets"]` holds the **command-inclusive** target that feeds the
  observation (what the runtime actually observes). When passthrough is off the two
  are identical, so the A/B is exact.
- **Reward reverted to the iteration-1/baseline leg-only imitation.**
  `USE_LEG_NECK_SPLIT = False` restores leg-only imitation (neck discarded, single
  `action_rate=-0.5`); `HEAD_COMMAND_TRACKING = False` and
  `reward_config.scales.head_command = 0.0` disable iteration-2's neck-redirect and
  standing head cost. Rewarding the head is now pointless (and was actively
  suppressing head motion), so all head reward shaping is off. Legs are trained
  exactly as the original deployed policy was.
- `cost_stand_still(..., ignore_head=self._head_passthrough)` — while standing, the
  head is driven externally, so its deviation from the default pose must **not** be
  penalised as "not standing still", or the stand-still term would fight the
  commanded head motion. Only the legs are held still.
- **Disturbance-rich head sampling stays on.** `sample_command` decouples head
  zeroing from locomotion zeroing (`stand_probability=0.2`,
  `head_zero_probability=0.1`) and the head command is resampled every
  `head_command_resample_steps=100` env steps (~2 s) across its full range, so the
  legs experience both static offsets and fast head transitions. Gated by
  `HEAD_DISTURBANCE_SAMPLING = HEAD_COMMAND_TRACKING or HEAD_PASSTHROUGH`.
- Master A/B flag `HEAD_PASSTHROUGH` (module constant, default `True`) gates the
  whole iteration-3 package. Set it `False` (with the other flags already `False`)
  to reproduce the original head-excluded baseline exactly.

**Unchanged (deployed contract — non-negotiable):**

- Observation layout is **101**, action is **14**, `action_scale`, `ctrl_dt`,
  domain randomisation and the ONNX export path are untouched. Both sizes are
  asserted in the env smoke and unit tests.

Flag state on this branch (top of `joystick.py`):

| Flag | Value | Meaning |
|---|---|---|
| `HEAD_PASSTHROUGH` | `True` | additive head passthrough during training (iteration 3) |
| `HEAD_COMMAND_TRACKING` | `False` | iteration-2 neck-redirect + standing head cost OFF |
| `USE_LEG_NECK_SPLIT` | `False` | leg-only baseline imitation (neck discarded) |
| `HEAD_DISTURBANCE_SAMPLING` | `True` | derived: full-range head resampling every ~2 s |

To train the **original head-excluded baseline** for an A/B comparison, set
`HEAD_PASSTHROUGH = False` in `joystick.py` before launching (the other two feature
flags are already `False`).

---

## 1. Environment setup on the CUDA box

**Do NOT use plain `uv sync`** — it resolves `playground==0.2.0`, which breaks on
`mujoco_playground._src.collision`. Pin the proven stack explicitly:

```bash
# from the repo root, inside the uv container image ghcr.io/astral-sh/uv:python3.11-bookworm
uv pip install "playground==0.0.3" "jax[cuda12]==0.5.3" "jaxlib==0.5.3" "flax==0.10.4"
# sanity: confirm JAX sees the GPU
uv run --no-sync python -c "import jax; print(jax.devices())"   # -> [CudaDevice(id=0)]
```

Always launch the runner with `uv run --no-sync` so uv does not re-resolve to the
broken `0.2.0`. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the run coexists with
the homelab's other GPU containers.

The imitation reward requires the reference-motion polynomials; they are already
committed at `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`
and `USE_IMITATION_REWARD = True` is already set.

---

## 2. Launch training

Train the **`flat_terrain_backlash`** task — the backlash variants are the
current sim2real win (see `constants.py:20-36`). 300M steps matches the proven
config. Use a **new** container name and a **new** output dir so iteration-1/2
artefacts are not overwritten:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --no-sync playground/open_duck_mini_v2/runner.py \
    --env joystick \
    --task flat_terrain_backlash \
    --num_timesteps 300000000 \
    --output_dir checkpoints_head_passthrough
```

Notes:
- `--output_dir checkpoints_head_passthrough` keeps iteration-3 artefacts separate.
- `--env joystick` is the walking/standing policy that carries the head command
  channels. (`standing` is a separate policy and is not this fix.)
- Watch it: `uv run --no-sync tensorboard --logdir=checkpoints_head_passthrough`.
  With the head reward off, the key curve is just `eval/episode_reward` (should
  climb and stay healthy) — head tracking is now handled by the passthrough, not a
  reward, so there is no head-reward curve to watch.

### Expected wall-clock (measured on this homelab's RTX 3090)

~68k steps/sec, **300M steps in ~1.2 h**. Checkpoints + ONNX export every ~5 min.

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

## 4. Acceptance gate (iteration 3 — envelope, NOT S0.1 gain)

Iteration 3's acceptance criteria **replace** the S0.1 gain gate. With a
passthrough, `policy_only` S0.1 gain is ≈0 (the policy alone barely moves the
head) and `additive` gain is ≈1.0 by construction — neither proves learning. The
real objective is a **wider safe head operating envelope**.

1. **Locomotion not regressed.** Stands and walks, no falls. Compare walk tilt to
   baseline ~3°, iter-1 ~9°, iter-2 ~4.1°. A large tilt regression is a fail.
2. **The safe head envelope widens (primary metric).** Re-run
   `experiments/animation/envelope_sweep.py` (main repo, branch
   `clancey-blender-animation-sim2real`) in `--mode additive` (matches how the
   head is now driven) on the new checkpoint, stand and walk. Compare the
   per-channel deflection limits and combined L2 budget to the current committed
   constants in `open_duck_anim/envelope.py`:
   `neck_pitch [-0.16,+0.31]`, `head_pitch [-0.78,+0.78]`,
   `head_yaw [-0.29,+1.50]`, `head_roll [-0.50,+0.50]`, `COMBINED_L2_BUDGET=0.55`.
   The harness derives limits by a fine first-onset outward sweep with ≥5 s holds
   (instability is non-monotonic and time-dependent) — do not substitute bisection
   or a coarse grid.
3. **Head tracking gain in the driven (additive) mode ≈1.0** — sanity only, that
   the passthrough is wired correctly. Not a learning claim.
4. **No fall at full commanded deflection.** Re-test the cases that toppled the old
   policy: step inputs at the extremes of `neck_pitch` and `head_yaw`, stand and
   walk. Surviving those is the headline result.

If the envelope does **not** widen materially: **STOP — do not attempt iteration
4.** The conclusion is that the additive path plus the currently-measured
conservative envelope is what ships (an acceptable outcome; the animation feature
already works via that path). Report and stop.

If it **passes**: update the constants in `open_duck_anim/envelope.py` to the new
measured limits. The additive lines at `v2_rl_walk_mujoco.py:310-311` stay (they
are now the intended architecture, matched in training) — they do **not** become a
double-count, because iteration 3 does not add a competing command channel.

---

## 5. Local correctness checks (already run on CPU, no GPU needed)

```bash
# venv used here (outside the repo; CPU-only JAX):
python -m venv ~/.oduck_train_venv   # or: uv venv ~/.oduck_train_venv
source ~/.oduck_train_venv/bin/activate

# 10 checks: leg/neck slicing + command-drives-target + iteration-3 end-to-end
# passthrough (zero policy action -> head driven to command; obs=101/action=14):
JAX_PLATFORMS=cpu python tests/test_leg_neck_reward_split.py

# tiny PPO smoke: env constructs, passthrough active, PPO loop runs, no NaN,
# ONNX export -> obs[1,101] / continuous_actions[1,14]:
JAX_PLATFORMS=cpu python smoke_iter3.py
```
