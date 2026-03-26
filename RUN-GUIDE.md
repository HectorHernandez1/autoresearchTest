# Autoresearch Run Guide — AMD Radeon AI PRO R9700

## Prerequisites

- ROCm 7.2+ installed and working (`rocm-smi` shows your GPU)
- On branch `test/amd-r9700`
- uv installed

## Step 1: Install dependencies

```bash
uv sync
```

This pulls PyTorch 2.9.1 (ROCm 6.3 wheel) and all other deps.

## Step 2: Prepare data (one-time, ~2 min)

```bash
uv run prepare.py
```

Downloads data shards and trains a BPE tokenizer to `~/.cache/autoresearch/`.

## Step 3: (Optional) Generate "before" samples

```bash
uv run generate.py
```

Saves a timestamped `samples_*.txt` file with text from the untrained model. This is your "before" snapshot.

## Step 4: Run overnight with Claude Code

Open a tmux session so it survives if your terminal closes:

```bash
tmux new -s research
```

Then start Claude Code:

```bash
claude --dangerously-skip-permissions
```

Tell it:

> Read program.md and follow it. Run autonomously overnight — do not stop to ask me anything.

It will:
1. Create branch `autoresearch/arch-exploration-amd`
2. Run baseline (~5 min)
3. Loop experiments: hypothesis → edit train.py → commit → train → keep/revert → repeat
4. Push after every improvement
5. ~12 experiments/hour, ~100 overnight

### tmux controls

- **Detach** (leave running): `Ctrl+b` then `d`
- **Reattach** (check progress): `tmux attach -t research`
- **Kill session**: `tmux kill-session -t research`

## Step 5: Review in the morning

Check what happened:

```bash
# See the experiment log
cat results.tsv

# See the latest val_bpb
grep "^val_bpb:" run.log

# See git history of improvements
git log --oneline autoresearch/arch-exploration-amd
```

## Step 6: Generate "after" samples

```bash
uv run generate.py
```

Compare the new `samples_*.txt` with the "before" file to see the quality difference.

## Troubleshooting

- **OOM**: Reduce `DEVICE_BATCH_SIZE` from 128 to 64 in `train.py`
- **BF16 issues**: RDNA 4 BF16 support should work but if you see NaN losses, this may need investigation
- **torch.compile**: Disabled on ROCm for now. May work with future PyTorch releases.
