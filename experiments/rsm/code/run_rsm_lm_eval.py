"""Exact zero-shot lm-eval runner for the Qwen3-VL AttnRes/RSM checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration


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
    summary_path = output_dir / "summary.json"
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

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    wrapper = None
    checkpoint_config = None
    if args.checkpoint_path:
        from checkpoint_utils import load_role_retrofit

        wrapper, checkpoint = load_role_retrofit(
            model,
            args.checkpoint_path,
            device=device,
            dtype=dtype,
        )
        checkpoint_config = checkpoint.get("config", {})
        print(
            "[runner] retrofit loaded "
            f"method_id={checkpoint_config.get('method_id')} "
            f"num_blocks={checkpoint_config.get('num_blocks')} "
            f"router_variant={checkpoint_config.get('router_variant')} "
            f"residual_strength_gate={checkpoint_config.get('residual_strength_gate', False)}",
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
            "experiment": "reskip_llm_benchmark_20260911",
            "label": args.label,
        },
    )
    if not isinstance(results, dict):
        raise TypeError(f"unexpected lm-eval result type: {type(results)!r}")

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
            "finished_at": finished_at,
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
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    if wrapper is not None:
        summary["retrofit_skip_stats"] = wrapper.get_skip_stats()
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[runner] summary={summary_path}", flush=True)
    print(f"[runner] peak_gpu_memory_bytes={summary['peak_gpu_memory_bytes']}", flush=True)
    for name, metrics in summary["results"].items():
        if isinstance(metrics, dict):
            print(f"[result] {name} {metrics}", flush=True)


if __name__ == "__main__":
    main()
