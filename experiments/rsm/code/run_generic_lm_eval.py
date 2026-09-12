"""Exact zero-shot lm-eval runner for generic VLM AttnRes/RSM checkpoints."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer

from generic_vlm_attnres_retrofit import load_generic_retrofit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print(f"[runner] label={args.label}", flush=True)
    print(f"[runner] model={args.model_path}", flush=True)
    print(f"[runner] checkpoint={args.checkpoint_path or 'none'}", flush=True)
    print(f"[runner] device={device} dtype={dtype} batch_size={args.batch_size}", flush=True)
    print(f"[runner] zero_shot=true chat_template=false tasks={tasks}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    retrofit = None
    checkpoint_config = None
    if args.checkpoint_path:
        retrofit = load_generic_retrofit(model, args.checkpoint_path)
        checkpoint_config = torch.load(
            args.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )["config"]
        model.eval()
        print(
            "[runner] retrofit loaded "
            f"family={retrofit.family} num_blocks={retrofit.num_blocks} "
            f"adapter_rank={retrofit.adapter_rank} "
            f"residual_strength_gate={retrofit.residual_strength_gate} "
            f"train_seed={checkpoint_config.get('seed')}",
            flush=True,
        )

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=str(device),
        dtype=dtype,
        trust_remote_code=True,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        log_samples=False,
        apply_chat_template=False,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        verbosity="INFO",
        metadata={
            "experiment": "reskip_llm_benchmark_20260911_extension_001",
            "label": args.label,
        },
    )
    if not isinstance(results, dict):
        raise TypeError(f"unexpected lm-eval result type: {type(results)!r}")

    summary = {
        "protocol": {
            "label": args.label,
            "model_path": args.model_path,
            "checkpoint_path": args.checkpoint_path or None,
            "checkpoint_config": checkpoint_config,
            "tasks": tasks,
            "num_fewshot": 0,
            "apply_chat_template": False,
            "batch_size": args.batch_size,
            "dtype": str(dtype),
            "gpu": args.gpu,
            "seed": seed,
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "lm_eval": getattr(lm_eval, "__version__", "unknown"),
        },
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
        "higher_is_better": results.get("higher_is_better", {}),
        "config": results.get("config", {}),
        "retrofit": (
            {
                "family": retrofit.family,
                "num_blocks": retrofit.num_blocks,
                "adapter_rank": retrofit.adapter_rank,
                "residual_strength_gate": retrofit.residual_strength_gate,
                "gamma": retrofit.gamma.detach().float().cpu().tolist(),
            }
            if retrofit is not None
            else None
        ),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[runner] summary={path}", flush=True)
    print(f"[runner] peak_gpu_memory_bytes={summary['peak_gpu_memory_bytes']}", flush=True)
    for name, metrics in summary["results"].items():
        if isinstance(metrics, dict):
            print(f"[result] {name} {metrics}", flush=True)


if __name__ == "__main__":
    main()
