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
   val_bpb:          1.100640 (5-minute budget)
   peak_vram_mb:     22805.5
   mfu_percent:      35.93
   tok_per_sec:      ~646K
   depth:            8
   device_batch:     64
   ```
   The target is to beat **val_bpb 1.100640**.
   
   **TIME_BUDGET UPDATED to 15 minutes (900s):** As of this session, all new experiments run with 3x the original
   budget. This aligns with AMD GPU experiments and enables deeper exploration. Previous 5-minute results are
   NOT directly comparable to new 15-minute runs.

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

### Phase 3: Architectural and Regularization Pivots (after kernel optimization plateau)

**Status:** Hyperparameter tuning and kernel optimization have plateaued at val_bpb ~1.0734
(-2.46% from baseline). Pivot to architectural and regularization changes that may unlock
further gains. These require more careful code edits than the previous phases — test each
in isolation, revert cleanly on regression. Run **3 seeds per experiment** and compare
averages, since gains in this regime (~0.001-0.005) are near the noise floor.

**Pivot experiments (in order, highest expected value first):**

1. **QK Normalization** — Add RMSNorm to queries and keys before attention computation
   Stabilizes attention at higher learning rates, prevents logit drift, often unlocks
   0.5-2% gains in modern transformer recipes. Apply after Q/K projection, before
   the attention dot-product.
   Hypothesis: normalized Q/K enable cleaner attention gradients and allow more
   aggressive optimization without instability.

2. **Z-loss regularization** — Add a small penalty on logit magnitude
   `loss += 1e-4 * (logsumexp(logits, dim=-1) ** 2).mean()`
   Penalizes logit drift, improves training stability without hurting capacity.
   Hypothesis: regularizing logit magnitude complements logit softcapping and allows
   sharper but better-calibrated output distributions.

3. **Untie embeddings + lm_head** — Currently weight-tied; untie them
   Adds parameters (~10M for vocab=50257, dim=512) but allows input/output to learn
   separate representations. Will cost some steps within the time budget.
   Hypothesis: decoupled embeddings improve final loss despite the step-count cost.

4. **Gradient clipping** — Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
   after backward(), before optimizer.step(). Currently no clipping is applied.
   Hypothesis: at the tightened optimum, clipping prevents occasional outlier gradient
   steps that hurt convergence.

5. **Cosine warmdown schedule** — Replace linear LR warmdown with cosine decay
   Holds higher LR longer, then drops sharply at the end. Modify the lr_multiplier
   function to use `0.5 * (1 + cos(pi * progress_in_warmdown))` shape during warmdown.
   Hypothesis: more time at high LR improves convergence; sharp drop locks in details.

6. **Attention temperature** — Change attention scale from `1/sqrt(d)` to learnable
   per-layer scalar (init to `1/sqrt(d)`). Allows each layer to choose its sharpness.
   Hypothesis: learnable temperature lets early/late layers diverge in attention
   sharpness based on their role.

7. **RoPE base frequency sweep** — Currently uses default RoPE base (10000)
   Sweep base ∈ {1000, 10000, 50000, 500000}. Different frequencies change positional
   encoding granularity at the 2048 context length.
   Hypothesis: higher base frequency improves long-range attention precision at our
   sequence length.

8. **Logit softcap value** — Currently softcap=15; try 10, 20, 30
   Changes how aggressively output logits are clamped before loss computation.
   Hypothesis: tighter softcap regularizes more; looser allows sharper distributions
   — sweep to find the optimum at the new tighter baseline.

**How to modify train.py:**
Architectural changes require deeper code edits than hyperparameter sweeps:
- QK Norm: modify the attention forward pass to apply RMSNorm to q/k tensors
- Z-loss: add to the loss computation block (search for `F.cross_entropy`)
- Untie embeddings: find where `lm_head.weight = wte.weight` or similar tying occurs
- Grad clipping: insert `torch.nn.utils.clip_grad_norm_` before `optimizer.step()`
- Cosine warmdown: modify the `lr_multiplier()` function's warmdown branch
- Attention temperature: replace `scale=1.0/math.sqrt(head_dim)` with learnable param
- RoPE base: search for `theta=10000` or similar; update once at construction
- Softcap: search for `softcap` or `15.0`; change the constant

Each experiment changes exactly one of the above. If a change requires multiple lines
to implement correctly (e.g. QK Norm needs both q and k normalized), that still counts
as ONE logical change. Run each in isolation and revert on regression.

### Phase 4: Depth Reduction (HIGH PRIORITY — pursue aggressively)

**Status:** Phase 3 architectural pivots (QK Norm, Z-loss, grad clip, cosine warmdown, attention
temperature, softcap tuning) were all reverted — they made things worse or were neutral.
Optimizer tuning beat architecture at the 5-minute budget. With 15-minute budget now active,
pivot to depth reduction. AMD results showed depth 5 works well; current RTX 5090 uses depth 8.

**Why depth reduction is critical:**
- Current model (depth=8) uses fixed time budget → step-count limited
- Shallower model = more steps within budget = better convergence
- AMD achieved 39% improvement partly via depth 5 vs. depth 8
- With 3x time budget (15 min), can explore deeper models, but should first test shallow

**Depth experiments (in order):**

1. **DEPTH 8 → 7** — one layer shallower
   Expect +10% more steps (~660 steps vs 594 at 5min).
   At 15-min budget, may reach ~2000 steps.
   Hypothesis: extra 10-15% steps at smaller depth beats one layer's capacity loss.

2. **DEPTH 7 → 6** — if exp 1 works, go shallower
   Another ~10% step gain. Depth 6 is half the original.
   Hypothesis: 6 layers + 20% more steps beats 8 layers at baseline steps.

3. **DEPTH 6 → 5** — match AMD's winning depth
   Further step increase, but now model capacity is significantly reduced.
   Hypothesis: at full 15-min budget, depth 5 with massive step count converges better.

**Re-tuning after depth change:**
After each depth reduction, the optimal hyperparameters shift:
- Momentum warmup steps may need to shorten (model trains faster with more steps)
- SCALAR_LR may change (different layer structure)
- x0_lambdas initialization may shift
- Run 3 seeds per experiment and re-tune the top 2-3 hyperparams at new depth

**How to modify train.py:**
Locate the hyperparameter block (~line 450-470) and find:
```python
DEPTH = 8
```
Change to 7, 6, 5 per experiment. This auto-scales model_dim and all downstream sizing.

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