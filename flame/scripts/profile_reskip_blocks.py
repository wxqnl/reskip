from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch

import custom_models  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_texts(data_glob: str, text_key: str) -> Iterable[str]:
    for path in sorted(glob.glob(data_glob)):
        if path.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(columns=[text_key], batch_size=128):
                for value in batch.column(0).to_pylist():
                    if isinstance(value, str) and value.strip():
                        yield value
        elif path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                for line in f:
                    row = json.loads(line)
                    value = row.get(text_key)
                    if isinstance(value, str) and value.strip():
                        yield value
        elif path.endswith(".txt"):
            with open(path) as f:
                for line in f:
                    value = line.strip()
                    if value:
                        yield value


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile executed block ratios for a local ReSkip checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--enable-skipping", action="store_true")
    parser.add_argument("--skip-threshold", type=float, default=0.0)
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    if hasattr(model, "set_inference_config"):
        model.set_inference_config(
            enable_skipping=args.enable_skipping,
            skip_threshold=args.skip_threshold,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ratios = []
    blocks = []
    total_blocks = None
    for idx, text in enumerate(iter_texts(args.data_glob, args.text_key)):
        if idx >= args.max_samples:
            break
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.seq_len,
            return_tensors="pt",
        )
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        with torch.no_grad():
            _ = model(**encoded)
        stats = model.get_last_inference_stats() if hasattr(model, "get_last_inference_stats") else None
        if not stats:
            continue
        ratios.append(float(stats["effective_block_ratio"]))
        blocks.append(int(stats["num_blocks_executed"]))
        total_blocks = int(stats["total_blocks"])

    result = {
        "model_path": args.model_path,
        "enable_skipping": args.enable_skipping,
        "skip_threshold": args.skip_threshold,
        "num_profiled_samples": len(ratios),
        "avg_effective_block_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "avg_blocks_executed": (sum(blocks) / len(blocks)) if blocks else None,
        "total_blocks": total_blocks,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Saved block profile to {output_path}")


if __name__ == "__main__":
    main()
