# autoresearch program

## Overview

You are an autonomous ML research agent. Your job is to iteratively improve a small GPT-style
language model by modifying `train.py`, running experiments, and keeping only the changes that
improve validation performance. You are a completely independent researcher — do not wait for
human input once the session has started.

The metric you are optimizing is **val_bpb** (validation bits per byte). Lower is better.
This metric is vocabulary-size-independent, so architectural changes are fairly compared.

---

## Setup (do this once at the start)

1. **Create the experiment branch:**
   ```
   git checkout -b autoresearch/rtx5090-exploration
   git push -u origin autoresearch/rtx5090-exploration
   ```

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
   experiment_id\thypothesis\tval_bpb\tpeak_vram_mb\tnotes
   ```
   Do not commit `results.tsv` — leave it untracked by git.

5. **Record baseline:**
   Run `uv run train.py > run.log 2>&1` on the unmodified `train.py` to establish a baseline
   val_bpb. Record it in `results.tsv`. This is your reference point for all future experiments.

   **RTX 5090 baseline (established, do not re-run unless train.py was modified):**
   ```
   val_bpb:          1.100640
   peak_vram_mb:     22805.5
   mfu_percent:      35.93
   tok_per_sec:      ~646K
   depth:            8
   device_batch:     64
   ```
   The target is to beat **val_bpb 1.100640**.

   **Platform notes for this machine (RTX 5090, Blackwell sm_120):**
   - `kernels-community/flash-attn3` lacks sm_120 kernels — train.py auto-falls back to PyTorch SDPA
   - `torch.compile` is enabled and working (~36% MFU)
   - Keep `DEVICE_BATCH_SIZE = 64` (whisper.cpp server occupies ~4 GB VRAM on this machine)
   - `TOTAL_BATCH_SIZE` must be divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN` = 64 * 2048 = 131072

6. **Confirm and proceed:**
   Once baseline is recorded, begin the experiment loop immediately. Do not wait.

---

## Research Direction

Your focus for this session is **architecture exploration** — understanding how the shape and
structure of the transformer model affects val_bpb within the fixed 5-minute compute budget.
The goal is not just to find improvements, but to build a picture of *why* each change works
or fails. Prefer interpretable changes over opaque ones.

### Suggested experiments (start here, in order):

These are concrete hypotheses to test. Each is one change. Run them in order — results from
earlier ones should inform whether later ones are worth trying.

1. **Deeper model** — `DEPTH 8 → 12`
   More layers = more capacity. At the same time budget, fewer steps but more expressive
   per-step computation. Expect higher VRAM usage; check OOM before committing.
   Hypothesis: deeper model learns better representations within the 5-min budget.

2. **Larger total batch** — `TOTAL_BATCH_SIZE 524288 → 1048576`
   More tokens per optimizer step = smoother gradient estimates. Must remain divisible by
   131072 (DEVICE_BATCH_SIZE=64 × MAX_SEQ_LEN=2048). 1048576 / 131072 = 8 grad accum steps.
   Hypothesis: larger batch reduces gradient noise and improves final val_bpb.

3. **Longer warmdown** — `WARMDOWN_RATIO 0.5 → 0.7`
   Spending more of the budget cooling the LR down gives the model more time at low LR to
   converge. Simple change, no memory impact.
   Hypothesis: more warmdown time improves final val_bpb.

4. **Wider FFN** — `FFN_MULT 3 → 4` (already a configurable param in the hyperparameter block)
   Increases FFN capacity without changing depth or attention. Adds parameters, so slightly
   fewer steps in the time budget.
   Hypothesis: wider FFN improves val_bpb more than the lost steps cost.

5. **Higher matrix LR** — `MATRIX_LR 0.04 → 0.06`
   Muon learning rate for 2D matrix params. Small LR increase may speed up convergence
   within the fixed budget. Easy to revert.
   Hypothesis: higher Muon LR allows faster convergence without instability.

6. **Attention window patterns** — `WINDOW_PATTERN "SSSL" → "L"`
   All-full attention vs. the default mix of banded + full layers. More expressive but
   uses more memory per step. Check if it fits at DEVICE_BATCH_SIZE=64.
   Hypothesis: full attention in all layers improves val_bpb at the cost of throughput.

7. **Shallower model** — `DEPTH 8 → 6`
   Fewer layers = more steps within the time budget (faster per-step). May trade model
   capacity for more gradient updates.
   Hypothesis: more steps at smaller depth beats fewer steps at larger depth.

### Phase 2: torch.compile Kernel Optimization (after optimizer plateau)

**Status:** Optimizer hyperparameter tuning has saturated (best val_bpb 1.089838, -0.98% from baseline).
Pivot to kernel-level optimizations via torch.compile modes and Inductor flags.

**Torch.compile experiments (in order):**

1. **Reduce-overhead mode** — `torch.compile(..., mode="reduce-overhead")`
   Minimizes Python-to-C++ overhead, trades some fusion for faster dispatch.
   Hypothesis: overhead reduction yields measurable throughput gain and faster convergence.

2. **Max-autotune mode** — `torch.compile(..., mode="max-autotune")`
   Aggressive kernel fusion and memory layout optimization. Slower compilation but better kernel perf.
   Hypothesis: better-fused kernels reduce memory traffic and improve val_bpb.

3. **Inductor coordinate descent** — Add `torch._inductor.config.coordinate_descent_tuning = True`
   Aggressive kernel autotuning during compilation.
   Hypothesis: coordinate descent finds better tile/loop orderings for this GPU.

4. **Aggressive fusion** — Add `torch._inductor.config.aggressive_fusion = True`
   Fuses more operations into single kernels, trades compilation time for runtime.
   Hypothesis: reduced memory roundtrips improve convergence within time budget.

5. **Async compilation** — `torch.compile(..., mode="reduce-overhead", fullgraph=False)`
   Allows dynamic shapes and async fallback to eager mode on unsupported ops.
   Hypothesis: reduces stalls if any ops can't be compiled, maintains steady throughput.

6. **Combine best flags** — Once individual effects are clear, combine top 2-3 winners.
   Hypothesis: complementary optimizations stack to yield largest speedup.

**How to modify train.py:**
Locate the model compilation line (~line 475):
```python
model = torch.compile(model, ...)
```
Change only the mode string or add torch._inductor.config flags. One flag per experiment.

---

## Experiment Loop

Repeat the following indefinitely. **Never stop on your own.**

### For each experiment:

1. **Form a hypothesis.**
   Write one clear sentence describing what you expect to change and why you think it will
   help. Log this in `results.tsv` before running. Examples:
   - "Reducing DEPTH from 8 to 6 will lower val_bpb by trading model capacity for faster
     iterations and more gradient steps within the time budget."
   - "Switching WINDOW_PATTERN from SSSL to L will improve val_bpb by allowing all layers
     to attend globally, at the cost of higher memory usage."

2. **Make exactly one logical change at a time.**
   Do not bundle multiple independent changes in a single experiment — it makes results
   uninterpretable. Each experiment should test one clear hypothesis.

3. **Commit the change, then run:**
   ```
   git commit -am "experiment: <short description of what you changed>"
   uv run train.py > run.log 2>&1
   ```
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
     `git commit --amend -m "depth=6: val_bpb 0.961 -> 0.948, faster iterations within budget"`
   - If `val_bpb` is equal or worse: revert cleanly with `git reset --hard HEAD~1`
   - Record the result in `results.tsv` either way.

6. **Handle crashes:**
   If a run produces no output or an obvious Python error, attempt one fix and re-run.
   If it fails again, revert and move on. Do not spend more than two attempts on a crash.

7. **Push progress to remote after every kept experiment:**
   ```
   git push -u origin autoresearch/rtx5090-exploration
   ```
   This lets the human monitor results remotely. Only push after keeping a change, not after reverts.

8. **Reflect briefly before the next experiment.**
   In one sentence, note what the result implies about the model. Use this to inform your
   next hypothesis. Prefer hypotheses that build on prior results rather than random jumps.

---

## Constraints

- **Only modify `train.py`.** Never touch `prepare.py`, `pyproject.toml`, or any other file.
- **Do not change the training time budget.** The 5-minute wall-clock limit is fixed by design.
- **Do not change the evaluation logic.** The val_bpb metric must remain comparable.
- **Do not commit `results.tsv`.** It is a local log, not part of the git history.
- **Prefer reversible experiments.** If a change is risky (e.g. restructuring the entire
  training loop), start with a smaller version of the idea first.

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