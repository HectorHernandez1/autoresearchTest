# RunPod Setup Guide

## Prerequisites

- RunPod H100 (or A100) on-demand pod
- Wait for pod initialization to complete (2/3 → 3/3) before starting
- Claude Pro or Max subscription (Max recommended for heavy overnight usage)

## Important: Use /workspace

RunPod pods only persist `/workspace` (network volume). Everything else is lost on terminate. Clone and work from there.

## Setup

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Clone repo into /workspace
cd /workspace
git clone https://github.com/HectorHernandez1/autoresearchTest.git
cd autoresearchTest

# 3. Install system dependencies (needed by Triton for torch.compile)
apt-get update && apt-get install -y python3.10-dev tmux

# 4. Install Python dependencies
uv sync

# 5. Verify GPU
nvidia-smi

# 6. Persist data cache across pod restarts (optional but recommended)
mkdir -p /workspace/.cache/autoresearch
ln -s /workspace/.cache/autoresearch ~/.cache/autoresearch

# 7. Download data + train tokenizer (~2 min)
uv run prepare.py
```

## Install Claude Code

```bash
# Install Node.js (if not already available)
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs

# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

## Configure git

```bash
git config --global user.email "researcher@autoresearch.dev"
git config --global user.name "AutoResearch Agent"
```

## Create a non-root user

RunPod runs as root, but Claude Code refuses `--dangerously-skip-permissions` under root. Create a non-root user:

```bash
useradd -m -s /bin/bash researcher
cp -r /root/.claude* /home/researcher/ 2>/dev/null
chown -R researcher:researcher /home/researcher/
chown -R researcher:researcher /workspace/autoresearchTest
```

## Run the experiment

Use tmux so the experiment survives SSH disconnects:

```bash
tmux new -s research
su - researcher
cd /workspace/autoresearchTest
claude --dangerously-skip-permissions
```

The `--dangerously-skip-permissions` flag is required for autonomous operation — without it, Claude Code will prompt for approval on every file edit, git command, and training run.

On first launch, Claude Code will give you a URL to open in your browser to authenticate with your Anthropic subscription. After that, tell it:

> Please read program.md carefully and follow it. Our H100 baseline is val_bpb 0.997264 (38.5% MFU, 50.3M params, depth 8). Your goal is to beat it. Run autonomously overnight — do not stop to ask me anything. Good luck!

The agent will run autonomously from there (~12 experiments/hour, 5 min each).

### tmux controls

- **Detach** (leave running in background): `Ctrl+b` then `d`
- **Reattach** (check on progress): `tmux attach -t research`
- **Kill session**: `tmux kill-session -t research`

## Resuming after pod restart

If you terminate and recreate the pod, your `/workspace` data survives. Just:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
cd /workspace/autoresearchTest

# Re-link cache if you used the symlink approach
ln -s /workspace/.cache/autoresearch ~/.cache/autoresearch

apt-get update && apt-get install -y python3.10-dev tmux
uv sync

# Reinstall Claude Code
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Ready to go — no need to re-run prepare.py if cache was persisted
```
