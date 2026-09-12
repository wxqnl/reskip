#!/usr/bin/env python3
"""Benchmark natural-generation timing for the frozen RSM-ReSkip policy.

The harness builds the official lmms-eval requests, selects deterministic
quantile-spaced documents from each task, and times four runtime variants in a
single loaded process. Image loading and processor work are outside the timed
region; vision encoding, language prefill, and natural autoregressive decode
are inside it. Each timed row contains a one-token TTFT measurement followed by
the task-native natural generation measurement with a reset StaticCache.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.cache_utils import StaticCache


TASKS = (
    "ai2d",
    "mmbench_en_dev_static",
    "mmmu_val",
    "mmstar",
    "ocrbench",
    "realworldqa",
)
VARIANTS = ("base", "full_rsm", "device_graph_full", "rsm_reskip")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--method-code-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--device-config", type=Path, required=True)
    parser.add_argument("--force-full-config", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--fixed-new-tokens", type=int)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--max-cache-len", type=int, default=8192)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--include-host-equivalence", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path):
    payload = json.loads(path.read_text())
    return payload.get("action_risk_skip_config", payload)


def geometric_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        return None
    return float(np.exp(np.mean(np.log(values))))


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def bootstrap_geometric_mean(values, repeats, rng):
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = values[rng.integers(0, len(values), len(values))]
        draws[index] = math.exp(float(np.mean(np.log(sampled))))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def quantile_spaced_indices(population, count):
    if count <= 0:
        raise ValueError("samples-per-task must be positive")
    if population < count:
        raise ValueError(f"requested {count} samples from population {population}")
    raw = np.floor((np.arange(count, dtype=np.float64) + 0.5) * population / count).astype(int)
    selected = raw.tolist()
    if len(set(selected)) != count:
        raise RuntimeError("quantile-spaced selection produced duplicate indices")
    return selected


def build_task_requests(lm, task_names, samples_per_task):
    from lmms_eval.evaluator_utils import get_task_list
    from lmms_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(verbosity="INFO", model_name="qwen3_vl")
    task_dict = get_task_dict(list(task_names), manager, "simple")
    outputs = {item.task_name: item for item in get_task_list(task_dict)}
    selected = {}
    populations = {}
    lm.task_dict = {}

    for task_name in task_names:
        if task_name not in outputs:
            raise KeyError(f"task manager did not return {task_name}: {sorted(outputs)}")
        task = outputs[task_name].task
        task.args = SimpleNamespace()
        lm.task_dict[task_name] = task.dataset
        task.build_all_requests(
            limit=None,
            offset=0,
            rank=0,
            world_size=1,
            cache_requests=False,
            rewrite_requests_cache=False,
            system_instruction=None,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            chat_template=None,
            tokenizer_name="",
        )
        instances = [
            instance
            for instance in task.instances
            if instance.request_type == "generate_until"
        ]
        if not instances:
            raise RuntimeError(f"{task_name} has no generate_until requests")
        populations[task_name] = len(instances)
        positions = quantile_spaced_indices(len(instances), samples_per_task)
        selected[task_name] = [instances[position] for position in positions]
    return selected, populations


def main():
    args = parse_args()
    if args.repeat < 3:
        raise ValueError("repeat must be at least three")
    args.runtime_root = args.runtime_root.resolve()
    args.eval_root = args.eval_root.resolve()
    args.method_code_root = args.method_code_root.resolve()
    args.state_path = args.state_path.resolve()
    args.device_config = args.device_config.resolve()
    args.force_full_config = args.force_full_config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    rows_path = args.out_dir / "raw_rows.jsonl"
    metadata_path = args.out_dir / "run_metadata.json"
    summary_path = args.out_dir / "BENCHMARK_NATURAL_RUNTIME.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    variants = VARIANTS + (("host_rsm_reskip",) if args.include_host_equivalence else ())
    task_names = tuple(name.strip() for name in args.tasks.split(",") if name.strip())
    if not task_names or not set(task_names).issubset(TASKS):
        raise ValueError(f"tasks must be a non-empty subset of {TASKS}")
    if args.fixed_new_tokens is not None and args.fixed_new_tokens <= 0:
        raise ValueError("fixed-new-tokens must be positive")

    # The lmms-eval plugin imports the retrofit wrapper by module name. Load
    # the device implementation first, then use the formal RSM checkpoint
    # constructor from this experiment.
    sys.path.insert(0, str(args.runtime_root))
    import qwen3vl_attnres_retrofit  # noqa: F401
    import device_conditional_reskip  # noqa: F401

    sys.path.insert(0, str(args.method_code_root))
    from checkpoint_utils import load_role_retrofit

    sys.path.insert(0, str(args.eval_root))
    sys.path.insert(0, str(args.eval_root.parent))
    import lmms_eval_retrofit as plugin
    sys.path.insert(0, str(args.runtime_root))

    def _load_rsm_checkpoint(self, state_path, num_blocks, adapter_rank):
        del num_blocks, adapter_rank
        base_model = self._model
        dtype = next(base_model.parameters()).dtype
        checkpoint_device = next(base_model.parameters()).device
        wrapper, _ = load_role_retrofit(
            base_model,
            state_path,
            device=checkpoint_device,
            dtype=dtype,
        )
        return wrapper

    plugin.Qwen3_VL_Retrofit._load_retrofit = _load_rsm_checkpoint
    Qwen3_VL_Retrofit = plugin.Qwen3_VL_Retrofit

    lm = Qwen3_VL_Retrofit(
        pretrained=args.model_path,
        retrofit_state_path=str(args.state_path),
        compile_mode="off",
        device=str(device),
        device_map=str(device),
        batch_size=1,
        use_cache=True,
        max_pixels=1605632,
        min_pixels=200704,
    )
    retrofit = lm._retrofit
    language_model = retrofit.base_model.model.language_model
    patched_forward = language_model.forward
    base_forward = language_model._original_forward

    force_full_config = load_config(args.force_full_config)
    device_config = load_config(args.device_config)
    host_config = dict(device_config)
    host_config["device_conditional_runtime"] = False
    host_config["device_fused_native_features"] = False
    retrofit.configure_native_token_skip(force_full_config)
    retrofit.configure_native_token_skip(device_config)
    retrofit.configure_native_token_skip(None)

    def configure(variant):
        if variant == "base":
            retrofit.configure_native_token_skip(None)
            language_model.forward = base_forward
        elif variant == "full_rsm":
            language_model.forward = patched_forward
            retrofit.configure_native_token_skip(None)
        elif variant == "device_graph_full":
            language_model.forward = patched_forward
            retrofit.configure_native_token_skip(force_full_config)
        elif variant == "rsm_reskip":
            language_model.forward = patched_forward
            retrofit.configure_native_token_skip(device_config)
        elif variant == "host_rsm_reskip":
            language_model.forward = patched_forward
            retrofit.configure_native_token_skip(host_config)
        else:
            raise ValueError(variant)
        retrofit.reset_skip_stats()

    selected, populations = build_task_requests(lm, task_names, args.samples_per_task)
    metadata = {
        "status": "running",
        "tasks": list(task_names),
        "variants": list(variants),
        "samples_per_task": args.samples_per_task,
        "repeat": args.repeat,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "max_cache_len": args.max_cache_len,
        "fixed_new_tokens": args.fixed_new_tokens,
        "population_requests": populations,
        "selection": "deterministic quantile-spaced positions over official lmms-eval requests",
        "timed_region": (
            "vision encoding + language prefill + fixed-length decode"
            if args.fixed_new_tokens is not None
            else "vision encoding + language prefill + task-native natural decode"
        ),
        "excluded": ["dataset access", "image/processor preprocessing", "model loading", "device graph construction", "cache reset"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    cache = StaticCache(
        config=retrofit.base_model.config.get_text_config(),
        max_batch_size=1,
        max_cache_len=args.max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )

    def run_generate(
        variant,
        inputs,
        generate_kwargs,
        forced_first_token=False,
        fixed_new_tokens=None,
    ):
        configure(variant)
        kwargs = dict(generate_kwargs)
        kwargs["past_key_values"] = cache
        kwargs["disable_compile"] = True
        kwargs["use_cache"] = True
        if forced_first_token:
            kwargs["max_new_tokens"] = 1
            kwargs["min_new_tokens"] = 1
        elif fixed_new_tokens is not None:
            kwargs["max_new_tokens"] = int(fixed_new_tokens)
            kwargs["min_new_tokens"] = int(fixed_new_tokens)
        else:
            kwargs.pop("min_new_tokens", None)
        with torch.inference_mode():
            cache.reset()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = lm.model.generate(**inputs, **kwargs)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_ids = generated[0, prompt_tokens:].detach().cpu().tolist()
        stats = retrofit.get_skip_stats()
        decode_forwards = int(stats.get("decode_forwards", 0))
        block_equivalents = None
        if decode_forwards > 0:
            block_equivalents = (
                float(stats.get("partial_decoder_layer_skip_events", 0))
                / decode_forwards
                / 4.0
            )
        return {
            "elapsed_seconds": elapsed,
            "generated_token_ids": generated_ids,
            "generated_tokens": len(generated_ids),
            "skip_stats": stats,
            "block_equivalents_per_decode_token": block_equivalents,
        }

    rows = []
    selected_doc_ids = {}
    with rows_path.open("x") as row_handle:
        global_position = 0
        for task_position, task_name in enumerate(task_names):
            requests = selected[task_name]
            selected_doc_ids[task_name] = [request.doc_id for request in requests]
            first = requests[0]
            warm_inputs, _, warm_gen_kwargs, _ = lm._preprocess_chunk([first.args])
            warm_inputs = warm_inputs.to(device)
            warm_kwargs = lm._build_generate_kwargs(warm_gen_kwargs)
            warm_kwargs["max_new_tokens"] = min(8, int(warm_kwargs["max_new_tokens"]))
            for variant in variants:
                run_generate(variant, warm_inputs, warm_kwargs, forced_first_token=False)

            for sample_position, request in enumerate(requests):
                inputs, contexts, gen_kwargs, until = lm._preprocess_chunk([request.args])
                inputs = inputs.to(device)
                generate_kwargs = lm._build_generate_kwargs(gen_kwargs)
                prompt_tokens = int(inputs["input_ids"].shape[1])
                maximum_new_tokens = int(generate_kwargs["max_new_tokens"])
                if prompt_tokens + maximum_new_tokens + 8 > args.max_cache_len:
                    raise RuntimeError(
                        f"cache too small for {task_name}/{request.doc_id}: "
                        f"prompt={prompt_tokens}, max_new={maximum_new_tokens}"
                    )

                for repetition in range(args.repeat):
                    shift = (task_position + sample_position + repetition) % len(variants)
                    order = variants[shift:] + variants[:shift]
                    if repetition % 2:
                        order = tuple(reversed(order))
                    for order_position, variant in enumerate(order):
                        first_token = run_generate(
                            variant, inputs, generate_kwargs, forced_first_token=True
                        )
                        natural = run_generate(
                            variant,
                            inputs,
                            generate_kwargs,
                            forced_first_token=False,
                            fixed_new_tokens=args.fixed_new_tokens,
                        )
                        decoded = lm.processor.decode(
                            natural["generated_token_ids"],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        for term in until:
                            if term:
                                decoded = decoded.split(term)[0]
                        row = {
                            "task": task_name,
                            "task_position": task_position,
                            "sample_position": sample_position,
                            "global_position": global_position,
                            "doc_id": request.doc_id,
                            "repetition": repetition,
                            "order_position": order_position,
                            "variant": variant,
                            "prompt_tokens": prompt_tokens,
                            "task_max_new_tokens": maximum_new_tokens,
                            "ttft_seconds": first_token["elapsed_seconds"],
                            "natural_seconds": natural["elapsed_seconds"],
                            "generated_tokens": natural["generated_tokens"],
                            "generated_token_ids": natural["generated_token_ids"],
                            "decoded_response": decoded,
                            "block_equivalents_per_decode_token": natural[
                                "block_equivalents_per_decode_token"
                            ],
                            "skip_stats": natural["skip_stats"],
                        }
                        rows.append(row)
                        row_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        row_handle.flush()
                        print(
                            json.dumps(
                                {
                                    "task": task_name,
                                    "doc_id": request.doc_id,
                                    "repetition": repetition,
                                    "variant": variant,
                                    "ttft_ms": 1000.0 * row["ttft_seconds"],
                                    "natural_ms": 1000.0 * row["natural_seconds"],
                                    "generated_tokens": row["generated_tokens"],
                                    "block_eq": row[
                                        "block_equivalents_per_decode_token"
                                    ],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                global_position += 1

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], str(row["doc_id"]), row["variant"])].append(row)

    per_sample = defaultdict(dict)
    for (task, doc_id, variant), sample_rows in grouped.items():
        token_counts = [int(row["generated_tokens"]) for row in sample_rows]
        if len(set(token_counts)) != 1:
            raise RuntimeError(
                f"non-deterministic output length for {task}/{doc_id}/{variant}: {token_counts}"
            )
        natural = [float(row["natural_seconds"]) for row in sample_rows]
        ttft = [float(row["ttft_seconds"]) for row in sample_rows]
        block_eq = [
            float(row["block_equivalents_per_decode_token"])
            for row in sample_rows
            if row["block_equivalents_per_decode_token"] is not None
        ]
        decode_forwards = [
            int(row["skip_stats"].get("decode_forwards", 0))
            for row in sample_rows
        ]
        layer_requests = [
            int(row["skip_stats"].get("partial_decoder_layer_skip_events", 0))
            for row in sample_rows
        ]
        per_sample[(task, doc_id)][variant] = {
            "median_natural_seconds": statistics.median(natural),
            "median_ttft_seconds": statistics.median(ttft),
            "stdev_natural_seconds": statistics.stdev(natural),
            "generated_tokens": token_counts[0],
            "median_block_equivalents_per_decode_token": (
                statistics.median(block_eq) if block_eq else 0.0
            ),
            "median_decode_forwards": statistics.median(decode_forwards),
            "median_decoder_layer_skip_events": statistics.median(
                layer_requests
            ),
        }

    rng = np.random.default_rng(args.seed)
    comparisons = {
        "rsm_reskip_vs_base": "base",
        "rsm_reskip_vs_full_rsm": "full_rsm",
        "rsm_reskip_vs_device_graph_full": "device_graph_full",
    }
    task_summary = {}
    task_ratios = defaultdict(dict)
    task_matched_ratios = defaultdict(dict)
    for task_name in task_names:
        sample_keys = [key for key in per_sample if key[0] == task_name]
        systems = {}
        for variant in variants:
            sample_values = [per_sample[key][variant] for key in sample_keys]
            natural_ms = [1000.0 * item["median_natural_seconds"] for item in sample_values]
            ttft_ms = [1000.0 * item["median_ttft_seconds"] for item in sample_values]
            output_tokens = [item["generated_tokens"] for item in sample_values]
            total_seconds = sum(item["median_natural_seconds"] for item in sample_values)
            total_decode_forwards = sum(
                item["median_decode_forwards"] for item in sample_values
            )
            total_layer_requests = sum(
                item["median_decoder_layer_skip_events"] for item in sample_values
            )
            systems[variant] = {
                "p50_natural_ms": statistics.median(natural_ms),
                "p90_natural_ms": percentile(natural_ms, 0.90),
                "p95_natural_ms": percentile(natural_ms, 0.95),
                "p50_ttft_ms": statistics.median(ttft_ms),
                "p90_ttft_ms": percentile(ttft_ms, 0.90),
                "sequential_samples_per_second": len(sample_values) / total_seconds,
                "end_to_end_output_tokens_per_second": sum(output_tokens) / total_seconds,
                "mean_generated_tokens": statistics.mean(output_tokens),
                "median_generated_tokens": statistics.median(output_tokens),
                "median_block_equivalents_per_decode_token": statistics.median(
                    item["median_block_equivalents_per_decode_token"]
                    for item in sample_values
                ),
                "pooled_block_equivalents_per_decode_token": (
                    total_layer_requests / total_decode_forwards / 4.0
                    if total_decode_forwards > 0
                    else 0.0
                ),
                "pooled_decode_forwards": total_decode_forwards,
                "pooled_decoder_layer_skip_events": total_layer_requests,
            }

        speedups = {}
        for label, baseline in comparisons.items():
            ratios = np.asarray(
                [
                    per_sample[key][baseline]["median_natural_seconds"]
                    / per_sample[key]["rsm_reskip"]["median_natural_seconds"]
                    for key in sample_keys
                ],
                dtype=np.float64,
            )
            matched = np.asarray(
                [
                    per_sample[key][baseline]["generated_tokens"]
                    == per_sample[key]["rsm_reskip"]["generated_tokens"]
                    for key in sample_keys
                ],
                dtype=bool,
            )
            task_ratios[label][task_name] = ratios.tolist()
            matched_values = ratios[matched]
            task_matched_ratios[label][task_name] = matched_values.tolist()
            speedups[label] = {
                "geometric_mean": geometric_mean(ratios),
                "sample_bootstrap_95ci": bootstrap_geometric_mean(
                    ratios, args.bootstrap, rng
                ),
                "positive_samples": int(np.sum(ratios > 1.0)),
                "sample_count": len(ratios),
                "output_length_match_samples": int(np.sum(matched)),
                "output_length_match_rate_percent": 100.0 * float(np.mean(matched)),
                "matched_length_geometric_mean": geometric_mean(matched_values),
                "matched_length_sample_bootstrap_95ci": (
                    bootstrap_geometric_mean(matched_values, args.bootstrap, rng)
                    if len(matched_values)
                    else None
                ),
            }
        task_summary[task_name] = {"systems": systems, "speedups": speedups}

    macro = {}
    for label in comparisons:
        per_task_gmean = {
            task: geometric_mean(task_ratios[label][task]) for task in task_names
        }
        pooled = [value for task in task_names for value in task_ratios[label][task]]
        matched_per_task_gmean = {
            task: geometric_mean(task_matched_ratios[label][task]) for task in task_names
        }
        matched_task_values = [
            value for value in matched_per_task_gmean.values() if value is not None
        ]
        matched_pooled = [
            value for task in task_names for value in task_matched_ratios[label][task]
        ]
        macro[label] = {
            "task_equal_geometric_mean": geometric_mean(list(per_task_gmean.values())),
            "task_geometric_means": per_task_gmean,
            "pooled_sample_geometric_mean": geometric_mean(pooled),
            "pooled_sample_bootstrap_95ci": bootstrap_geometric_mean(
                pooled, args.bootstrap, rng
            ),
            "positive_samples": int(np.sum(np.asarray(pooled) > 1.0)),
            "sample_count": len(pooled),
            "matched_length_task_equal_geometric_mean": geometric_mean(
                matched_task_values
            ),
            "matched_length_task_geometric_means": matched_per_task_gmean,
            "matched_length_task_count": len(matched_task_values),
            "matched_length_pooled_sample_geometric_mean": geometric_mean(
                matched_pooled
            ),
            "matched_length_pooled_sample_bootstrap_95ci": (
                bootstrap_geometric_mean(matched_pooled, args.bootstrap, rng)
                if matched_pooled
                else None
            ),
            "matched_length_sample_count": len(matched_pooled),
        }

    metadata["status"] = "completed"
    metadata["selected_doc_ids"] = selected_doc_ids
    summary = {
        "status": "completed",
        "protocol": metadata,
        "selected_doc_ids": selected_doc_ids,
        "task_summary": task_summary,
        "macro": macro,
        "per_sample": {
            f"{task}:{doc_id}": values
            for (task, doc_id), values in sorted(per_sample.items())
        },
        "material_passport": {
            "official_lmms_eval_requests": "VERIFIED",
            "raw_timing_rows": "RECORDED",
            "order_rotated_repetitions": "ANALYZED",
            "independent_process_replication": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary_path": str(summary_path), "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
