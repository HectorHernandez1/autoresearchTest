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

# 3. Install dependencies
uv sync

# 4. Verify GPU
nvidia-smi

# 5. Persist data cache across pod restarts (optional but recommended)
mkdir -p /workspace/.cache/autoresearch
ln -s /workspace/.cache/autoresearch ~/.cache/autoresearch

# 6. Download data + train tokenizer (~2 min)
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

## Run the experiment

```bash
cd /workspace/autoresearchTest
claude
```

On first launch, Claude Code will give you a URL to open in your browser to authenticate with your Anthropic subscription. After that, tell it:

> Have a look at program.md and let's kick off a new experiment!

The agent will run autonomously from there (~12 experiments/hour, 5 min each).

## Resuming after pod restart

If you terminate and recreate the pod, your `/workspace` data survives. Just:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
cd /workspace/autoresearchTest

# Re-link cache if you used the symlink approach
ln -s /workspace/.cache/autoresearch ~/.cache/autoresearch

uv sync

# Reinstall Claude Code
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Ready to go — no need to re-run prepare.py if cache was persisted
```
