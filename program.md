# autoresearch program

## Overview

You are an autonomous ML research agent. Your job is to iteratively improve a small GPT-style
language model by modifying `train.py`, running experiments, and keeping only the changes that
improve validation performance. You are a completely independent researcher — do not wait for
human input once the session has started.

The metric you are optimizing is **val_bpb** (validation bits per byte). Lower is better.
This metric is vocabulary-size-independent, so architectural changes are fairly compared.

**This session focuses on structural and optimizer shifts.** Prior sessions closed all the
tuning slack. The only remaining leverage comes from changing the *shape* of the model or
the *shape* of optimization. See "Research Direction" below.

---

## Prior Work (context)

Two branches have already been exhaustively explored:

1. **`autoresearch/arch-exploration-amd`** — 231 experiments, 35 kept.
   Tuned DEPTH, width, window pattern, FFN, init, LRs, betas, WD, softcap, warmdown.
   **Result: val_bpb 1.711 → 1.170 (31% improvement).** Tail experiments ≤0.0005 bpb delta.

2. **`autoresearch/training-dynamics-amd`** — 25 experiments, 0 kept.
   Tested EMA, SWA, z-loss, MTP, LayerDrop, label smoothing, dropout, gradient noise,
   Lookahead, token reweighting, depth supervision, stochastic residual scaling, inverted WD.
   **Result:** every single dynamics perturbation was net negative. Cross-seed noise floor ≈ 0.005.
   Finding: in this tight 10-min / DEPTH=5 / MuonAdamW regime, training never plateaus, so
   weight averaging cannot help. Gradient noise was catastrophic (Muon normalizes).

**Conclusion:** HP tuning is done. Dynamics tricks don't work here. The only remaining
axes are **structural**. Do not waste experiments on tiny HP nudges or dynamics variants —
they are all within noise.

---

## Setup (do this once at the start)

1. **You are already on the experiment branch:** `autoresearch/structural-shifts-amd`.
   Do not create a new branch. Do not switch branches.

2. **Read all in-scope files for full context:**
   - `README.md` — repository overview
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. **Do not modify.**
   - `train.py` — the only file you are allowed to edit.
   - Recent commit history of prior branches:
     ```
     git log --oneline -50 autoresearch/arch-exploration-amd
     git log --oneline -50 autoresearch/training-dynamics-amd
     ```

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
   Run `uv run train.py > run.log 2>&1` on the inherited config to establish this session's
   baseline. The prior config was tuned at 5 min (val_bpb ~1.170). With the new 10-min budget,
   the baseline will be lower — record whatever you get as experiment 0. Every structural
   experiment is measured against this new 10-min baseline.

6. **Begin immediately.** Do not wait.

---

## Research Direction

**Your job is to make structural moves that shift the optimization landscape.** HP tuning
and dynamics tricks are exhausted. The only remaining leverage comes from changes that
replace core components of the model or optimizer. Expect a higher crash rate and more
aggressive reverts — this is expected and fine.

Each of the five priority areas below is a *category* of experiments, not a single
experiment. Work through them roughly in order, but jump between categories freely if a
lead looks promising or a direction is clearly dead.

### Priority 1: Optimizer swaps

Muon+AdamW has been tuned to a fine point. A genuinely different optimizer may find a
different local minimum. Each swap requires re-tuning `MATRIX_LR` and `EMBEDDING_LR` for
the new optimizer — this is the **only** HP re-tuning allowed in this session.

1. **Full AdamW** — Drop Muon entirely. Use AdamW for *all* 2D matrix params. Sanity
   baseline for whether Muon is load-bearing in this regime.
2. **Lion** (signSGD-family) — `update = sign(beta1*m + (1-beta1)*g) * lr`. Completely
   different update geometry. Smaller memory footprint. Known to favor small batches.
3. **Sophia-G** — Hessian-free second-order. Replaces AdamW's second moment with a
   diagonal Hessian estimate. Reported 20–30% wall-clock savings on small pretraining.
4. **SOAP** — Shampoo in Adam's eigenbasis. Second-order preconditioner with AdamW-like
   numerical behavior.
5. **Full Shampoo** — Block-diagonal second-order. Most aggressive, most expensive.
   Only worth trying if Sophia/SOAP show signs of life.

### Priority 2: Alternative block structures

The current Block is strictly serial: `x + attn(norm(x))` → `x + mlp(norm(x))`. Alternative
topologies change the gradient geometry and throughput tradeoff.

1. **Parallel Transformer Block** (GPT-J / PaLM) —
   `x + attn(norm1(x)) + mlp(norm2(x))`. Both branches computed from the same input.
   Faster (enables kernel fusion) and sometimes improves quality at small scale.
2. **NormFormer** — Extra LayerNorm inside the attention output and inside the MLP.
   Known to stabilize deep/narrow models.
3. **ReZero** — Replace layer norms with a learnable scalar gate (init 0). Removes norm
   overhead; may unlock higher LRs.
4. **Tied input/output embeddings** — Share weights between `wte` and `lm_head`. Halves
   embedding params, frees budget for other parts. Common in small models.
5. **Differential Attention** (MSFT, 2024) — Two softmax attention maps subtracted to
   cancel attention noise. Reported gains on small-scale LM perplexity.

### Priority 3: Data ordering / curriculum

`prepare.py` is frozen, but *what* you feed the model in what order is controlled by
`train.py`. Don't touch the eval split.

1. **Hard-example mining** — Track per-batch loss; revisit the top-K hardest recent
   batches with extra weight. Requires a small batch-loss buffer.
2. **Sequence length curriculum (v2)** — 1024→2048 was tried. Try finer schedules
   (3-stage: 512→1024→2048) or invert it (long→short).
3. **Bucketed batching** — Group sequences by effective length or entropy.
4. **Anti-curriculum** — Hard-to-easy. Works for some regimes.

### Priority 4: Compute reallocation

Reshape how fixed compute (32GB VRAM, 10 min) is spent.

1. **Gradient checkpointing** — Trade activation memory for recomputation. Enables a
   bigger model (DEPTH=8? wider MLP?) in the same VRAM. Cost: ~25% slower per step. If
   the model-size win outweighs the step-count loss, this is structural leverage.
2. **Selective recomputation** — Checkpoint only the MLP, not attention. Usually the
   best perf/memory tradeoff for small models.
3. **FP8 weights / activations** — Only if R9700 gfx1201 supports it natively; gate
   behind a capability check. If yes, halves memory and may speed up matmul.
4. **Grad accum rebalance** — The current `TOTAL_BATCH_SIZE / DEVICE_BATCH_SIZE` split
   was chosen for throughput. Explore unusual splits only if a larger model requires a
   smaller DEVICE_BATCH_SIZE.

### Priority 5: The wild ones

Bigger structural changes. High variance, high expected value on a good day.

1. **Mixture of Experts (MoE)** — Replace the single MLP per block with 4 experts and a
   top-2 router. Doubles total params but keeps active params roughly constant. Known
   to improve val loss per active FLOP.
2. **Mixture of Depths (MoD)** — A router lets each token skip layers. Each token visits
   K-of-N transformer blocks. Frees compute for hard examples.
3. **State Space block (Mamba/S4)** — Replace attention with a linear-recurrent block in
   some layers. SDPA's limitations on ROCm make pure-attention variants hard to beat in
   throughput; a hybrid may sidestep that.
4. **Hybrid Attention+SSM** — Keep attention in a few layers, SSM in the rest.
5. **Early exit / token dropping** — Router terminates easy tokens early and propagates
   hard tokens deeper. Frees compute for hard examples.

**Composability note:** Wins from different categories may stack. After any kept
experiment, try building on top of it rather than starting fresh.

---

## Experiment Loop

Repeat the following indefinitely. **Never stop on your own** except per the plateau rule below.

### For each experiment:

1. **Form a hypothesis.**
   One clear sentence. Log in `results.tsv` before running. Examples:
   - "Replacing the serial Block with a parallel attn+mlp block will lower val_bpb by
     improving throughput and allowing more steps in the 10-min budget."
   - "Swapping MuonAdamW for Lion with re-tuned LR will reach a different local minimum
     because sign-based updates have different curvature behavior."

2. **Make exactly one logical change at a time.**
   Do not bundle.

3. **Commit the change, then run:**
   ```
   git commit -am "experiment: <short description>"
   uv run train.py > run.log 2>&1
   ```

4. **Read the results:**
   ```
   grep "^val_bpb:\|^peak_vram_mb:" run.log
   ```
   If empty, the run crashed. Read `tail -n 50 run.log`.

5. **Keep or revert:**
   - If `val_bpb` improved: amend the commit message with the result and keep.
   - Otherwise: `git reset --hard HEAD~1`.
   - Record in `results.tsv` either way.

6. **Handle crashes:**
   One fix attempt max. If it fails again, revert.

7. **Push after every kept experiment:**
   ```
   git push -u origin autoresearch/structural-shifts-amd
   ```

8. **Reflect briefly.** What did this tell you structurally? Inform the next hypothesis.

---

## Experimental Discipline

- **Composability first.** Wins from different categories may stack. Build on top.
- **Expect crashes.** Structural changes touch hot paths; dtype / shape / autograd bugs
  are common. Two-attempt rule still applies.
- **Track VRAM.** New architectures may blow the 32GB budget. If a promising experiment
  OOMs, reduce `DEVICE_BATCH_SIZE` and raise grad accum steps to keep `TOTAL_BATCH_SIZE` fixed.
- **One change per experiment.** Always.
- **Cross-seed noise floor ≈ 0.005.** Do not chase improvements smaller than this without
  multi-seed verification.
- **Plateau rule.** If 15 consecutive experiments produce no kept change across multiple
  categories, stop and write a final summary — the structural axis may also be saturated.

---

## Constraints

- **Only modify `train.py`.** Never touch `prepare.py`, `pyproject.toml`, or any other file.
- **Do not change the training time budget.** 10 minutes is fixed.
- **Do not change the evaluation logic.** val_bpb must remain comparable.
- **Do not revisit HP tuning on the existing optimizer.** LRs, betas, WD, softcap,
  warmdown, init — these are tuned. You may re-tune `MATRIX_LR`/`EMBEDDING_LR` **only**
  as part of an optimizer swap.
- **Do not commit `results.tsv`, `run.log`, or `checkpoint*.pt`.** Gitignored.

---

## On Failure and Negative Results

Negative results remain valuable. If an entire category (e.g. all optimizer swaps) fails,
write a one-paragraph finding explaining *why* the category failed structurally — this
prevents repetition in future sessions and is the real product of autoresearch at this
stage.

---

## NEVER STOP

Do not stop experimenting unless:
- The data shards are missing and `prepare.py` has not been run (tell the human and wait)
- There is an unrecoverable environment error (e.g. OOM that cannot be resolved by reducing
  DEVICE_BATCH_SIZE)
- You hit the plateau rule above (15 consecutive failures across multiple categories)

In all other cases, form a new hypothesis and continue.
