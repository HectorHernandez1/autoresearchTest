# Autoresearch Run Guide — AMD Radeon AI PRO R9700

## Prerequisites

- ROCm 7.2+ installed and working (`rocm-smi` shows your GPU)
- On branch `autoresearch/arch-exploration-amd`
- uv installed

## First-time setup

```bash
# Install dependencies (pulls PyTorch 2.9.1 ROCm wheel, ~5GB)
uv sync

# Download data + train tokenizer (one-time, ~2 min)
uv run prepare.py

# (Optional) Generate "before" samples from untrained model
uv run generate.py
```

## Start the agent

```bash
cd ~/repo/autoresearchTest
claude --dangerously-skip-permissions
```

Tell it:

> Read program.md and follow it. Run autonomously overnight — do not stop to ask me anything.

It will loop experiments: hypothesis → edit train.py → commit → train → keep/revert → repeat. Each experiment takes ~5 minutes.

## Resume the agent (after stopping)

Same thing — just start Claude Code again:

```bash
cd ~/repo/autoresearchTest
claude --dangerously-skip-permissions
```

Tell it:

> Read program.md and follow it. Continue from where the last session left off — check results.tsv and git log for what's been tried. Do not stop to ask me anything.

All improvements are committed to git, so it picks up where it left off.

## Stop the agent

Just Ctrl+C in the terminal. Nothing is lost — all kept experiments are already committed.

## Check progress (while it's running or after)

```bash
# See the experiment log
cat results.tsv

# See the latest val_bpb
grep "^val_bpb:" run.log

# See git history of improvements
git log --oneline
```

## Generate "after" samples

After stopping the agent:

```bash
# Train once on the final best code (5 min) — saves good checkpoint
uv run train.py

# Generate text from the trained model
uv run generate.py
```

Compare the new `samples_trained_*.txt` with the earlier `samples_untrained_*.txt`.

## Troubleshooting

- **OOM**: Reduce `DEVICE_BATCH_SIZE` in `train.py` (currently 64 for 32GB VRAM)
- **BF16 issues**: RDNA 4 BF16 support should work but if you see NaN losses, this may need investigation
- **torch.compile**: Disabled on ROCm for now. May work with future PyTorch releases.
