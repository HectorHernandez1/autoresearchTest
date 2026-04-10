# autoresearch program

## Overview

You are an autonomous ML research agent. Your job is to iteratively improve a small GPT-style
language model by modifying `train.py`, running experiments, and keeping only the changes that
improve validation performance. You are a completely independent researcher — do not wait for
human input once the session has started.

The metric you are optimizing is **val_bpb** (validation bits per byte). Lower is better.
This metric is vocabulary-size-independent, so architectural changes are fairly compared.

**This is a ROCm infrastructure recovery session at 15 minutes per experiment.** Prior
sessions converged on hyperparameter tuning and hit a val_bpb plateau around 1.0905 on the
R9700. Further HP tuning will only produce 0.00x gains. This session targets the AMD-specific
performance gaps that are leaving roughly half of the GPU's compute on the floor — the next
meaningful drops in val_bpb come from making the GPU actually work at its peak, not from
more hyperparameter sweeps.

---

## Prior Work (context, do not blindly reuse)

Two prior phases on this codebase:

**Phase 1 — 5-min budget** (~45 experiments): hit val_bpb ~1.17 plateau. Found: DEPTH=5,
SwiGLU, FFN_MULT=3, softcap=13, TOTAL_BATCH_SIZE=2^15. Muon is essential. Value embeddings
matter (~16% improvement). QK-norm stabilizing. Throughput is king.

**Phase 2 — 15-min budget** (5 experiments): hit val_bpb 1.0905 plateau. Found: DEPTH=7 (more
capacity at 3x budget), MATRIX_LR=0.015, WARMDOWN_RATIO=0.93, x0_lambdas=0.1. **Most phase-1
settings (softcap, FFN_MULT, WEIGHT_DECAY, init stds) were inherited without re-verification
at 15 min** — some may still be stale.

**Both phases were pure hyperparam tuning.** Neither touched the ROCm-specific code paths
responsible for the current MFU gap. That gap is the focus of this session.

Do not re-run phase-1 or phase-2 experiments — read the commit history for context. Your
starting point is the current HEAD of `autoresearch/rocm-infra-wins`.

---

## Setup (do this once at the start)

1. **You are already on the experiment branch:** `autoresearch/rocm-infra-wins`.
   Do not create a new branch. Do not switch branches.

2. **Read all in-scope files for full context:**
   - `README.md` — repository overview
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. **Do not modify.**
   - `train.py` — the only file you are allowed to edit.

3. **Verify data exists:**
   Check that `~/.cache/autoresearch/` contains data shards and a tokenizer.
   If not, stop and tell the human to run `uv run prepare.py` before continuing.

4. **Initialize results tracking:**
   Create `results.tsv` with just the header row:
   ```
   experiment_id	hypothesis	val_bpb	peak_vram_mb	notes
   ```
   Do not commit `results.tsv` — leave it untracked by git.

5. **Record baseline (REQUIRED for this session):**
   Run `uv run train.py > run.log 2>&1` on the current unmodified `train.py` to establish
   a clean baseline for this branch. Record `val_bpb`, `peak_vram_mb`, `mfu_percent`, and
   `num_steps` in `results.tsv`. This is your reference point for all future experiments.
   **Do NOT trust the inherited 1.0905 from the prior branch as your baseline** — run a fresh
   one. Every subsequent experiment compares against *this* number, not against 1.0905.
   Pay close attention to `mfu_percent` — if it is below 30%, the infrastructure experiments
   in the Research Direction below will have a large effect. If it is already above 35%, the
   expected wins are smaller and you should prioritize re-verifying suspect hyperparams instead.

6. **Confirm and proceed:**
   Once baseline is recorded, begin the experiment loop immediately. Do not wait.

---

## Research Direction

Your focus for this session is **ROCm infrastructure recovery** — fixing the AMD-specific
performance gaps in `train.py` that are preventing the R9700 from running anywhere near
its peak compute. Hyperparameter tuning is explicitly *not* the priority this session.

### Why this direction (the compute-bound argument)

- The R9700 has **191.4 TFLOPS peak BF16** (about 19% of an H100's 989 TFLOPS).
- The H100 reference baseline hits **38.5% MFU** in 5 minutes for val_bpb 0.997264.
- The current R9700 code is likely hitting **15–25% MFU** in steady state because:
  - `torch.compile` is hard-disabled on ROCm — look for `if not IS_ROCM` gates on both
    `_maybe_compile` and the `torch.compile(model, ...)` call at the bottom of the model
    setup section
  - `_step_adamw` in `MuonAdamW` runs a per-parameter Python loop with 0-D CPU tensor fills
    per param per step — every iteration is a host-side kernel launch
  - The `SSSL` window pattern silently degrades to full causal on the ROCm SDPA path —
    `F.scaled_dot_product_attention` ignores `window_size`, so every "S" layer is doing
    full-attention work while the hyperparam search has been assuming it runs banded
  - `DEVICE_BATCH_SIZE=16` was set when DEPTH=8 and FFN_MULT=4; current code is DEPTH=7 and
    FFN_MULT=3, which uses meaningfully less VRAM and likely has headroom
- **Fixing MFU is the single biggest lever available.** Going from ~20% → ~35% MFU ≈ 1.75x
  throughput ≈ effectively bumping the budget from 15 → 26 minutes without changing it.
  On prior budget scaling (5→15 min dropped val_bpb ~0.08), this is worth **roughly 0.03–0.06
  off val_bpb**, which is 10–30x the marginal hyperparam gains the prior sessions were
  chasing. That is the "clear change" we are hunting for.

### Priority experiments (in order, each a single commit)

1. **Re-enable `torch.compile` on ROCm with a try/except fallback.**
   Currently `_maybe_compile = torch.compile(...) if not IS_ROCM else lambda fn: fn` and the
   model compile is gated the same way. PyTorch 2.9+ on ROCm 7 has Triton-based inductor
   support that should work on RDNA4. Wrap the compile calls in `try: ... except Exception
   as e: print(...); fallback_to_eager`. If the first compile blows up with a Triton error,
   read the traceback carefully and try targeted fixes: `mode="reduce-overhead"`,
   `fullgraph=False`, `dynamic=True`, or disabling specific fusions. This is the single
   largest expected win — **allow up to 3 attempts before reverting** (overrides the default
   two-attempt crash rule for this experiment specifically).

2. **Replace the per-parameter AdamW Python loop with `torch._foreach_*` ops.**
   The current `_step_adamw` iterates over params in Python, fills 0-D CPU tensors per
   param, and calls `adamw_step_fused` once per param. Swap to a single pass using
   `torch._foreach_mul_`, `torch._foreach_lerp_`, `torch._foreach_addcmul_`, etc. Same math,
   batched kernel dispatch. This win exists independent of torch.compile and stacks with it.

3. **Use `torch.nn.attention.flex_attention` on the ROCm path to honor the SSSL window pattern.**
   The ROCm branch in `CausalSelfAttention.forward` currently calls SDPA with `is_causal=True`
   and ignores `window_size` entirely, meaning all layers run full causal even though "SSSL"
   says 3 out of 4 should be banded. `flex_attention` supports arbitrary block masks and runs
   on ROCm via Triton. Build a banded mask for "S" layers. Gate strictly behind `IS_ROCM` so
   the FA3 path on NVIDIA is untouched. **Allow up to 3 attempts** — flex_attention has its
   own set of compile quirks that may take a fix or two to land cleanly.

4. **Re-tune `DEVICE_BATCH_SIZE` upward for the current architecture.**
   Check `peak_vram_mb` in your baseline `run.log`. If it's below 24,000 MB, try
   `DEVICE_BATCH_SIZE=24` (and maybe 32). Larger per-device batches reduce Python/kernel
   launch overhead per token. If OOM, revert and try the next smaller power.

5. **Re-verify two suspect 5-min-era hyperparams at 15 min.**
   Pick the two most suspicious phase-1 settings and sweep each in a single experiment:
   - `FFN_MULT 3 → 4` (more MLP capacity may be worthwhile at longer budget)
   - `WEIGHT_DECAY 0.4 → 0.2` (less regularization may fit better with 3x more steps)
   Expected moves are small (0.005–0.015). Do these *after* the infra wins in 1–4 are
   locked in, not before — the infra changes will shift the optimal values anyway.

### What is explicitly OUT of scope this session

- Architecture swings (GQA, MoE, Mixture of Depths, MLA, MoR). These belong in the next
  session, *after* MFU is fixed. A compute-starved model does not benefit proportionally
  from added capacity.
- Hyperparameter micro-tuning of LRs, betas, softcap, warmdown, x0_lambdas, resid_lambdas,
  init stds, SCALAR_LR, momentum schedules. These are what produced the 0.00x plateau on
  the prior branch. **Do not restart that search.**
- `prepare.py` changes of any kind, including `TIME_BUDGET`, `MAX_SEQ_LEN`, `EVAL_TOKENS`.
  The 15-min budget is fixed for this session so results are comparable to the 1.0905
  reference from the prior branch.
- `pyproject.toml` / dependency changes. Work with whatever PyTorch version is already
  installed. If a feature (e.g. `flex_attention`) is missing, fall back and note it.

---

## Experiment Loop

Repeat the following indefinitely. **Never stop on your own.**

### For each experiment:

1. **Form a hypothesis.**
   Write one clear sentence describing what you expect to change and why you think it will
   help. Log this in `results.tsv` before running. Examples appropriate to this session:
   - "Re-enabling torch.compile on ROCm with a try/except fallback will raise MFU from
     ~20% toward ~35% by fusing the model forward/backward, letting the same 15-min budget
     produce ~1.5x more training steps."
   - "Replacing the Python-loop AdamW with torch._foreach_* will reduce host-side kernel
     launch overhead per step, raising tok/sec without changing any math."
   - "Wiring flex_attention on the ROCm branch with a banded mask will restore the SSSL
     pattern that currently silently runs as full causal, cutting attention FLOPs by ~40%
     on S layers."

2. **Make exactly one logical change at a time.**
   Do not bundle multiple independent changes in a single experiment — it makes results
   uninterpretable. Each experiment should test one clear hypothesis.

3. **Commit the change, then run:**
   ```
   git commit -am "experiment: <short description of what you changed>"
   uv run train.py > run.log 2>&1
   ```
   **IMPORTANT:** Training takes ~15 minutes. You MUST set the Bash tool timeout to at
   least 1200000ms (20 minutes) when running `train.py`, otherwise the command will be
   killed before training completes. Use `run_in_background` if available.
   Commit *before* running so that reverting on failure is a clean `git reset`.
   Always redirect output. Do not use `tee` or allow output to flood your context.

4. **Read the results:**
   ```
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
   If the grep is empty, the run crashed. Read the tail of the log:
   ```
   tail -n 50 run.log
   ```

5. **Keep or revert:**
   - If `val_bpb` improved (lower): keep the commit. Optionally amend with results:
     `git commit --amend -m "depth=8: val_bpb 1.10 -> 1.05, more capacity wins at 15min"`
   - If `val_bpb` is equal or worse: revert cleanly with `git reset --hard HEAD~1`
   - Record the result in `results.tsv` either way.

6. **Handle crashes:**
   If a run produces no output or an obvious Python error, attempt one fix and re-run.
   If it fails again, revert and move on. Do not spend more than two attempts on a crash.

7. **Push progress to remote after every kept experiment:**
   ```
   git push -u origin autoresearch/rocm-infra-wins
   ```

8. **Reflect briefly before the next experiment.**
   In one sentence, note what the result implies about the model. Use this to inform your
   next hypothesis. Prefer hypotheses that build on prior results rather than random jumps.

---

## Constraints

- **Only modify `train.py`.** Never touch `prepare.py`, `pyproject.toml`, or any other file.
- **Do not change the training time budget.** The 15-minute wall-clock limit is fixed.
- **Do not change the evaluation logic.** The val_bpb metric must remain comparable.
- **Do not commit `results.tsv`.** It is a local log, not part of the git history.
- **Prefer reversible experiments.** If a change is risky (e.g. restructuring the entire
  training loop), start with a smaller version of the idea first.
- **32GB VRAM limit.** Reduce `DEVICE_BATCH_SIZE` if OOM. Current default is 16.

---

## On Failure and Negative Results

Negative results are valuable. If a hypothesis fails, record *why* it likely failed — this
prevents wasted repetition and builds a coherent picture of the loss landscape. A session
with 40 failed experiments and a clear understanding of why is more useful than 10 random
successes with no explanation.

---

## NEVER STOP

Do not stop experimenting unless:
- The data shards are missing and `prepare.py` has not been run (tell the human and wait)
- There is an unrecoverable environment error (e.g. GPU OOM that cannot be resolved by
  reducing batch size)

In all other cases, form a new hypothesis and continue. The human will check in the morning.
