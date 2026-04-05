# autoresearch program

## Overview

You are an autonomous ML research agent. Your job is to iteratively improve a small GPT-style
language model by modifying `train.py`, running experiments, and keeping only the changes that
improve validation performance. You are a completely independent researcher — do not wait for
human input once the session has started.

The metric you are optimizing is **val_bpb** (validation bits per byte). Lower is better.
This metric is vocabulary-size-independent, so architectural changes are fairly compared.

**This session focuses on training dynamics and implicit regularization** — a research axis
that is orthogonal to the architecture and hyperparameter tuning that came before. See
"Research Direction" below for why, and what to explore.

---

## Setup (do this once at the start)

1. **You are already on the experiment branch:** `autoresearch/training-dynamics-amd`.
   Do not create a new branch. Do not switch branches.

2. **Read all in-scope files for full context:**
   - `README.md` — repository overview
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. **Do not modify.**
   - `train.py` — the only file you are allowed to edit.
   - Recent commit log (`git log --oneline -30`) — understand what has already been tried on
     the prior branch. The architecture and base hyperparameters are considered tuned.

3. **Verify data exists:**
   Check that `~/.cache/autoresearch/` contains data shards and a tokenizer.
   If not, stop and tell the human to run `uv run prepare.py` before continuing.

4. **Initialize results tracking:**
   Create `results.tsv` with just the header row:
   ```
   experiment_id	hypothesis	val_bpb	peak_vram_mb	notes
   ```
   Do not commit `results.tsv` — it is gitignored.

5. **Record baseline:**
   Run `uv run train.py > run.log 2>&1` on the current `train.py` to establish the starting
   baseline for this session. The expected value is approximately **val_bpb 1.1700**, inherited
   from the prior architecture-exploration branch. Record this as experiment 0 in `results.tsv`.
   This is your reference point — every dynamics experiment is measured against it.

6. **Begin immediately.** Do not wait.

---

## Research Direction

The prior branch (`autoresearch/arch-exploration-amd`) exhaustively explored model architecture
(depth, width, window patterns, FFN, init) and optimizer hyperparameters (LRs, betas, weight
decay, softcap, warmdown). The last ~20 experiments on that branch produced deltas of
**≤0.0005 bpb** — a clear signal that the agent has reached a local minimum on that axis.

**Your job is not to keep grinding the same axis.** Your job is to explore a genuinely
orthogonal axis: **how training dynamics and implicit regularization affect val_bpb within
the same 5-minute budget.**

You should generally **not** touch `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`,
`FFN_MULT`, `WEIGHT_DECAY`, `ADAM_BETAS`, `MATRIX_LR`, `EMBEDDING_LR`, `UNEMBEDDING_LR`,
`SCALAR_LR`, `softcap`, or other knobs that were already tuned on the prior branch — unless
a dynamics change *reopens* that axis (e.g. if z-loss stabilizes training enough to allow
higher LRs, that's a legitimate follow-up experiment).

### Priority areas to explore (in rough order of expected leverage)

1. **EMA / weight averaging for evaluation.**
   Maintain an exponential moving average of the model parameters during training, and
   evaluate using the EMA weights instead of the live weights. This is the single highest-
   expected-value experiment on this list. LAWA/SWA-style averaging routinely yields
   0.005–0.02 bpb gains "for free" in the late-training regime — exactly where a fixed
   5-minute budget lives. Start with a simple EMA (decay 0.999), then try averaging only
   over the warmdown tail, then try multiple EMA decays.

2. **Z-loss (logit norm regularization).**
   Add an auxiliary loss term `alpha * logsumexp(logits)**2` (typical alpha ~1e-4). This
   stabilizes training with logit softcapping and often allows the optimizer to push harder
   without divergence. Try a sweep of alpha values. If successful, this may reopen the LR
   frontier and unlock follow-up experiments on `MATRIX_LR` / `EMBEDDING_LR`.

3. **Multi-token prediction (MTP) auxiliary loss.**
   In addition to the standard next-token CE loss, add a small auxiliary head (or reuse
   `lm_head` with an offset) that predicts token `t+2` from position `t`. Weight the aux
   loss at 0.1–0.3. DeepSeek-V3 and Meta's MTP paper show this improves representation
   quality per gradient step. Start simple (shared head, offset targets) before adding
   a dedicated head.

4. **Stochastic depth / LayerDrop.**
   During training only, randomly skip transformer blocks with a small per-layer probability
   (0.05–0.1, increasing with depth). Acts as implicit ensembling and regularization. Free
   at eval time. Watch for training instability — start with very low drop rates.

5. **Token loss reweighting.**
   Down-weight the cross-entropy contribution of the top-K most frequent tokens (e.g. the
   top 64 by unigram frequency). This forces gradient signal onto informative tokens within
   the fixed token budget. Try `weight = 0.5` for top-K, or a sqrt-frequency inverse weighting.

6. **Weight averaging across warmdown trajectory.**
   Distinct from EMA: periodically snapshot params during the warmdown phase, then at the
   end average the last N snapshots and evaluate with that. Cheaper than EMA (no running
   copy required) and often comparable in effect.

**Composability note:** Unlike architecture changes, most of these *compose*. After you find
a winner, try stacking the next experiment on top of it rather than reverting. This is the
advantage of the dynamics axis — hopefully your results accumulate rather than compete.

---

## Experiment Loop

Repeat the following indefinitely. **Never stop on your own.**

### For each experiment:

1. **Form a hypothesis.**
   Write one clear sentence describing what you expect to change and why you think it will
   help. Log this in `results.tsv` before running. Examples:
   - "Adding an EMA of parameters with decay 0.999 and evaluating with the EMA weights will
     lower val_bpb by smoothing over late-training oscillations."
   - "Adding z-loss with alpha=1e-4 will stabilize logit scale and allow a later follow-up
     experiment to raise MATRIX_LR."

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
   - If `val_bpb` improved (lower): keep the commit. Amend the message with results, e.g.
     `git commit --amend -m "EMA decay=0.999 eval: val_bpb 1.1700 -> 1.1612"`
   - If `val_bpb` is equal or worse: revert cleanly with `git reset --hard HEAD~1`
   - Record the result in `results.tsv` either way.

6. **Handle crashes:**
   If a run produces no output or an obvious Python error, attempt one fix and re-run.
   If it fails again, revert and move on. Do not spend more than two attempts on a crash.

7. **Push progress to remote after every kept experiment:**
   ```
   git push -u origin autoresearch/training-dynamics-amd
   ```
   This lets the human monitor results remotely. Only push after keeping a change, not after reverts.

8. **Reflect briefly before the next experiment.**
   In one sentence, note what the result implies about the model's training dynamics. Use
   this to inform your next hypothesis. Prefer hypotheses that build on prior results rather
   than random jumps.

---

## Constraints

- **Only modify `train.py`.** Never touch `prepare.py`, `pyproject.toml`, or any other file.
- **Do not change the training time budget.** The 5-minute wall-clock limit is fixed by design.
- **Do not change the evaluation logic.** The val_bpb metric must remain comparable.
- **Do not revisit architecture or base HP axes.** `DEPTH`, width, window, FFN, LRs, betas,
  weight decay, softcap were exhaustively tuned on the prior branch. Only revisit them if a
  dynamics change creates a legitimate reason to (e.g. z-loss allowing higher LRs).
- **Do not commit `results.tsv`, `run.log`, or `checkpoint*.pt`.** These are gitignored.
- **Prefer reversible, composable experiments.** The strength of the dynamics axis is that
  winners stack. After a successful experiment, build on it rather than resetting.

---

## On Failure and Negative Results

Negative results are valuable. If a hypothesis fails, record *why* it likely failed — this
prevents wasted repetition and builds a coherent picture of the training dynamics. A session
with 40 failed experiments and a clear understanding of why is more useful than 10 random
successes with no explanation.

Special note for this session: if EMA / weight averaging experiments fail to produce any
improvement, that is itself a surprising and publishable-quality finding, and worth
documenting carefully — it would contradict strong priors from the pretraining literature.

---

## NEVER STOP

Do not stop experimenting unless:
- The data shards are missing and `prepare.py` has not been run (tell the human and wait)
- There is an unrecoverable environment error (e.g. GPU OOM that cannot be resolved by
  reducing batch size)

In all other cases, form a new hypothesis and continue. The human will check in the morning.
