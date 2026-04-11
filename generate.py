"""
Generate text samples from the current model and save to a timestamped file.
Run before training for a "before" snapshot, and after training for "after".

Usage: uv run generate.py
"""

import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from prepare import MAX_SEQ_LEN, Tokenizer

# --- Settings (edit these directly) ---
PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy far away,",
    "The president of the United States announced today",
]
MAX_TOKENS = 200
TEMPERATURE = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.3  # >1 discourages repeating recent tokens
REPETITION_WINDOW = 64    # only penalize tokens seen in the last N positions
DEPTH = 7
ASPECT_RATIO = 64
HEAD_DIM = 128
FFN_MULT = 3

# ---- Model definition (mirrors train.py) ----

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    ffn_mult: int = 4


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 0.1 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = norm(self.c_proj(y))  # NormFormer: norm after output proj
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn_dim = config.ffn_mult * config.n_embd
        self.w_gate = nn.Linear(config.n_embd, ffn_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, ffn_dim, bias=False)
        self.c_proj = nn.Linear(ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        return norm(self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x)))  # NormFormer


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        x = norm(x)

        softcap = 13
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits


# ---- Generation ----

def build_config(depth, tokenizer):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=tokenizer.get_vocab_size(),
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern="SSSL", ffn_mult=FFN_MULT,
    )


def generate(model, tokenizer, prompt, max_tokens, temperature, top_k, device,
             repetition_penalty=1.0, repetition_window=64):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_tokens):
            x_cond = x[:, -model.config.sequence_len:]
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty: divide logits of recently-seen tokens by penalty
            # (positive logits shrink, negative logits grow more negative)
            if repetition_penalty != 1.0:
                recent = x[0, -repetition_window:].tolist()
                for tok in set(recent):
                    if logits[0, tok] > 0:
                        logits[0, tok] /= repetition_penalty
                    else:
                        logits[0, tok] *= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    backend = "ROCm (HIP)" if is_rocm else "CUDA"
    print(f"Using: {gpu_name} via {backend}")

    tokenizer = Tokenizer.from_directory()
    config = build_config(DEPTH, tokenizer)

    # Load checkpoint if it exists, otherwise use random weights
    ckpt_path = "checkpoint.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading trained model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_config = GPTConfig(**ckpt["config"])
        with torch.device("meta"):
            model = GPT(ckpt_config)
        model.to_empty(device=device)
        # Strip "_orig_mod." prefix added by torch.compile when the model was saved
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        label = "trained"
    else:
        print("No checkpoint found — using random (untrained) weights")
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        # Random init (same as train.py)
        torch.manual_seed(42)
        n_embd = config.n_embd
        s = 3**0.5 * n_embd**-0.5
        nn.init.normal_(model.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.001)
        for block in model.transformer.h:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.w_gate.weight, -s, s)
            nn.init.uniform_(block.mlp.w_up.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
        with torch.no_grad():
            model.resid_lambdas.fill_(1.0)
            model.x0_lambdas.fill_(0.1)
        for ve in model.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        model.transformer.wte.to(dtype=torch.bfloat16)
        for ve in model.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        label = "untrained"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"samples_{label}_{timestamp}.txt"

    lines = []
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {label}")
    lines.append(f"GPU: {gpu_name} ({backend})")
    lines.append(f"depth: {config.n_layer}, n_embd: {config.n_embd}, params: {sum(p.numel() for p in model.parameters()):,}")
    lines.append(f"temperature: {TEMPERATURE}, top_k: {TOP_K}, max_tokens: {MAX_TOKENS}, rep_penalty: {REPETITION_PENALTY}")
    lines.append("=" * 60)

    for i, prompt in enumerate(PROMPTS):
        text = generate(model, tokenizer, prompt,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        device=device,
                        repetition_penalty=REPETITION_PENALTY,
                        repetition_window=REPETITION_WINDOW)
        lines.append(f"\n--- Prompt {i+1}: {prompt!r} ---")
        lines.append(text)

    lines.append("\n" + "=" * 60)

    output = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(output)

    print(output)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
