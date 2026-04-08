# autoresearch program

## Overview

You are an autonomous ML research agent. Your job is to iteratively improve a small GPT-style
language model by modifying `train.py`, running experiments, and keeping only the changes that
improve validation performance. You are a completely independent researcher — do not wait for
human input once the session has started.

The metric you are optimizing is **val_bpb** (validation bits per byte). Lower is better.
This metric is vocabulary-size-independent, so architectural changes are fairly compared.

**This is a fresh architecture exploration session at 15 minutes per experiment.** All
hyperparameters are open for re-tuning. Prior sessions were optimized for a 5-min budget —
those results do NOT carry over. The optimal depth, width, batch size, and LRs will be
different at 15 min.

---

## Prior Work (context, do not blindly reuse)

Prior sessions at 5-min budget found:
- DEPTH=5, SwiGLU MLP, FFN_MULT=3, softcap=13, TOTAL_BATCH_SIZE=2^15 was optimal at 5 min
- Muon is essential — AdamW and Lion were significantly worse
- Value embeddings matter (~16% of val_bpb improvement)
- QK-norm is stabilizing
- **Throughput is king** — any change costing even 10% per-step speed gets punished

**IMPORTANT:** These were tuned for 5 min. At 15 min you have 3x the steps, so:
- Deeper models (DEPTH=7-10) may now converge where they couldn't before
- Larger batch sizes may work (better gradient quality matters more with enough steps)
- LR schedules need re-tuning for the longer horizon
- Capacity vs throughput tradeoff shifts toward capacity

Start from the defaults in `train.py` and explore systematically. Do not assume the
5-min optimum is the 15-min optimum.

---

## Setup (do this once at the start)

1. **Create the experiment branch:**
   ```
   git checkout -b autoresearch/arch-exploration-amd-15min
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
   experiment_id	hypothesis	val_bpb	peak_vram_mb	notes
   ```
   Do not commit `results.tsv` — leave it untracked by git.

5. **Record baseline:**
   Run `uv run train.py > run.log 2>&1` on the current `train.py` to establish a baseline
   val_bpb. Record it in `results.tsv`. This is your reference point for all future experiments.
   The baseline will reflect the inherited 5-min config running for 15 min — expect ~1.05-1.10.

6. **Confirm and proceed:**
   Once baseline is recorded, begin the experiment loop immediately. Do not wait.

---

## Research Direction

Your focus is **full architecture and hyperparameter exploration at the 15-min budget.**
Everything is open. The goal is to find the best possible val_bpb for this GPU
(AMD Radeon AI PRO R9700, 32GB VRAM, ~10% MFU) in 15 minutes of training.

### Priority areas to explore (in rough order):

1. **Model depth vs. width tradeoffs**
   The current `DEPTH` is 5 (optimized for 5 min). With 3x the budget, deeper models have
   more time to converge. Try DEPTH 6, 7, 8, 10. Note that many dimensions (hidden size,
   heads, FFN width) are derived from DEPTH, so changing it has broad downstream effects.
   Understand these dependencies before experimenting.

2. **Batch size and gradient accumulation**
   Current `TOTAL_BATCH_SIZE` is 2^15 (optimized for max steps at 5 min). With 15 min, you
   can afford larger batches (2^16, 2^17) for better gradient quality while still getting
   enough steps. Sweep this early — it affects everything downstream.

3. **Learning rates**
   All LRs (`MATRIX_LR`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `SCALAR_LR`) were tuned for
   5-min DEPTH=5. Re-tune them for whatever depth/width you settle on. The LR scaling
   `1/sqrt(model_dim/768)` should help, but the base rates may need adjustment.

4. **Warmdown and schedule**
   `WARMDOWN_RATIO=0.85` was optimal at 5 min. At 15 min, the model has more time at peak
   LR, so the optimal warmdown fraction may shift. Also re-examine `FINAL_LR_FRAC`.

5. **MLP and attention variants**
   SwiGLU with FFN_MULT=3 was optimal at 5 min. At larger model sizes enabled by the longer
   budget, FFN_MULT=4 or different activations may win. Also try different `WINDOW_PATTERN`
   values — though note SDPA on ROCm ignores window_size, so this only affects the window
   size metadata, not actual computation.

6. **Weight decay, softcap, init**
   These interact with model size and LR. Re-tune after settling on depth/width/LR.

7. **Optimizer balance**
   Muon is confirmed essential. But the Muon HPs (momentum, ns_steps, beta2) and the
   AdamW-vs-Muon split may have different optima at the new scale.

---

## Experiment Loop

Repeat the following indefinitely. **Never stop on your own.**

### For each experiment:

1. **Form a hypothesis.**
   Write one clear sentence describing what you expect to change and why you think it will
   help. Log this in `results.tsv` before running. Examples:
   - "Increasing DEPTH from 5 to 8 will lower val_bpb because the 15-min budget provides
     enough steps for a larger model to converge."
   - "Increasing TOTAL_BATCH_SIZE from 2^15 to 2^17 will improve gradient quality enough
     to offset the reduced step count at 15 min."

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
   git push -u origin autoresearch/arch-exploration-amd-15min
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
