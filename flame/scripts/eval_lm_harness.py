from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import custom_models  # noqa: F401
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_metric(task_result: dict[str, Any]) -> tuple[str, float] | tuple[None, None]:
    for key in ("acc_norm,none", "acc,none", "exact_match,none", "em,none"):
        if key in task_result:
            return key, float(task_result[key])
    for key, value in task_result.items():
        if isinstance(value, (float, int)):
            return key, float(value)
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local flame/HF checkpoint with lm_eval.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--tasks", default="lambada_openai,hellaswag,arc_easy,arc_challenge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
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
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(model, "set_inference_config"):
        model.set_inference_config(
            enable_skipping=args.enable_skipping,
            skip_threshold=args.skip_threshold,
        )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )

    compact = {
        "scenario": args.scenario,
        "model_path": args.model_path,
        "tasks": tasks,
        "enable_skipping": args.enable_skipping,
        "skip_threshold": args.skip_threshold,
        "results": results.get("results", {}),
        "primary_metrics": {},
        "aggregate_mean": None,
    }
    scores = []
    for task, task_result in compact["results"].items():
        metric_name, metric_value = select_metric(task_result)
        if metric_name is not None:
            compact["primary_metrics"][task] = {
                "metric": metric_name,
                "value": metric_value,
            }
            scores.append(metric_value)
    if scores:
        compact["aggregate_mean"] = sum(scores) / len(scores)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(compact, f, indent=2)
        f.write("\n")
    print(f"Saved lm_eval results to {output_path}")


if __name__ == "__main__":
    main()
