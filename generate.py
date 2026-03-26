"""
Generate text samples from the current model and save to a timestamped file.
Run before training for a "before" snapshot, and after training for "after".

Usage: uv run generate.py
"""

from datetime import datetime

import torch
import torch.nn.functional as F

from train import GPT, build_model_config, DEPTH, DEVICE_BATCH_SIZE, IS_ROCM
from prepare import Tokenizer, evaluate_bpb

# --- Settings (edit these directly) ---
PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy far away,",
    "The president of the United States announced today",
]
MAX_TOKENS = 200
TEMPERATURE = 1.0
TOP_K = 40


def generate(model, tokenizer, prompt, max_tokens, temperature, top_k, device):
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_tokens):
            x_cond = x[:, -model.config.sequence_len:]
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    # PyTorch uses "cuda" for both NVIDIA and AMD — on ROCm it maps to HIP automatically
    device = torch.device("cuda")
    backend = "ROCm (HIP)" if IS_ROCM else "CUDA"
    print(f"Using: {torch.cuda.get_device_name(0)} via {backend}")
    tokenizer = Tokenizer.from_directory()

    config = build_model_config(DEPTH)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"samples_{timestamp}.txt"

    lines = []
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"val_bpb: {val_bpb:.6f}")
    lines.append(f"depth: {DEPTH}")
    lines.append(f"temperature: {TEMPERATURE}, top_k: {TOP_K}, max_tokens: {MAX_TOKENS}")
    lines.append("=" * 60)

    for i, prompt in enumerate(PROMPTS):
        text = generate(model, tokenizer, prompt,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        device=device)
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
