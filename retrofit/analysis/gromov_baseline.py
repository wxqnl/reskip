"""Gromov et al. 2024 baseline: Block Influence via cosine similarity.

Measures for each layer n:  I(n) = 1 - cos(h_n, h_{n+1})
(low I → layer barely changes representation → skippable).

This is a training-free static importance estimator. Our α-guided
method should match or beat it.
"""
import argparse
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL = "/home/user01/Minko/models/Qwen3-VL-2B"


def build_text_stream(tok, seq_len=2048, seed=0):
    """Yield 1-D token tensors of length seq_len from LAMBADA test text.

    Self-contained replacement for the (stale) ``train_qwen3vl.build_text_stream``
    import. Gromov needs only a handful of sequences for cosine-sim averaging,
    so plain LAMBADA prose is sufficient.
    """
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").shuffle(seed=seed)
    buf: list[int] = []
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= seq_len:
            seq, buf = buf[:seq_len], buf[seq_len:]
            yield torch.tensor(seq, dtype=torch.long)


@torch.no_grad()
def compute_block_influence(base, tok, device, num_seqs=8, seq_len=2048):
    """Measure cos-sim change per layer."""
    base.eval()
    text_model = base.model.language_model
    num_layers = len(text_model.layers)

    sim_accum = [0.0] * num_layers
    count = 0

    stream = build_text_stream(tok, seq_len=seq_len, seed=0)
    for i, seq in enumerate(stream):
        if i >= num_seqs:
            break
        ids = seq.unsqueeze(0).to(device)
        h = text_model.embed_tokens(ids)
        B, T = ids.shape
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        pos_emb = text_model.rotary_emb(h, position_ids)

        prev = h
        for lidx, layer in enumerate(text_model.layers):
            out = layer(prev, attention_mask=None, position_embeddings=pos_emb)
            if isinstance(out, tuple):
                cur = out[0]
            else:
                cur = out
            # Cosine similarity per-token, averaged over seq
            prev_flat = prev.reshape(-1, prev.shape[-1]).float()
            cur_flat = cur.reshape(-1, cur.shape[-1]).float()
            cos = F.cosine_similarity(prev_flat, cur_flat, dim=-1).mean().item()
            sim_accum[lidx] += cos
            prev = cur
        count += 1

    avg_sim = [s / count for s in sim_accum]
    influence = [1.0 - s for s in avg_sim]  # higher = bigger change = important
    return influence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="If set, save sorted-by-influence layer ranking as JSON.")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)

    influence = compute_block_influence(base, tok, device, num_seqs=args.num_seqs)
    print(f"\nQwen3-VL-2B Block Influence (Gromov et al. 2024 baseline)")
    print(f"=" * 60)
    print(f"{'layer':>6s} {'influence':>11s} {'skip candidate?'}")
    ranked = sorted(range(len(influence)), key=lambda i: influence[i])
    for l in range(len(influence)):
        mark = ""
        if l in ranked[:8]:
            mark = f"← rank {ranked.index(l)+1} lowest"
        print(f"  {l:3d}   {influence[l]:.4f}   {mark}")

    print(f"\nLowest-influence layers (suggest for pruning, lowest first):")
    for rank, lidx in enumerate(ranked[:12]):
        print(f"  #{rank+1}: layer {lidx} (influence={influence[lidx]:.4f})")

    if args.output:
        import json
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": MODEL,
            "num_layers": len(influence),
            "influence": influence,
            "ranked_low_to_high": ranked,
        }, indent=2))
        print(f"\n[gromov] ranking saved to {out_path}")


if __name__ == "__main__":
    main()
