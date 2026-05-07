"""Build the ReSkip adaptive-depth visualization for the paper.

The figure connects three quantities measured on VLM multiple-choice samples:

1. routing traces for representative easy / hard inputs;
2. average skipped blocks as full-depth uncertainty changes;
3. skip-induced score perturbation versus the full-depth prediction margin.

The margin-preservation plot uses the standard sufficient condition in the
paper: if every candidate score changes by at most epsilon, then the top-1
prediction is unchanged whenever 2 * epsilon <= Delta.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib.patches import Rectangle
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
RETROFIT_DIR = ROOT / "retrofit"
sys.path.insert(0, str(RETROFIT_DIR))

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit  # noqa: E402


DEFAULT_MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
DEFAULT_STATE_PATH = "retrofit/outputs/H_2B_r256_10k_L4_v3/retrofit_attnres_state.pt"
DEFAULT_DYN_CONFIG = (
    "retrofit/outputs/related_qwen2b/dyn_skip_configs/"
    "vlm_ai2d_mmmu_mmstar_limit64_q050_p4.json"
)


def as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_dynamic_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = json.loads(path.read_text())
    thresholds = {
        int(k): float(v)
        for k, v in (raw.get("thresholds") or {}).items()
        if as_float(v) is not None
    }
    eligible_raw = raw.get("eligible_blocks")
    eligible = None
    if eligible_raw is not None:
        eligible = {int(x) for x in eligible_raw}
    cfg = {
        "thresholds": thresholds,
        "eligible_blocks": eligible,
        "max_skips": raw.get("max_skips"),
    }
    if cfg["max_skips"] is not None:
        cfg["max_skips"] = int(cfg["max_skips"])
    return cfg, raw


def load_model(
    model_path: str,
    state_path: Path,
    device: str,
    dtype: torch.dtype,
    num_blocks: int | None,
) -> tuple[Qwen3VLAttnResRetrofit, AutoProcessor, dict[str, Any]]:
    processor = AutoProcessor.from_pretrained(model_path)
    base = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype).to(device)
    ckpt = torch.load(state_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})
    kwargs: dict[str, Any] = {
        "num_blocks": int(ckpt_cfg.get("num_blocks", num_blocks or 14)),
    }
    if "adapter_rank" in ckpt_cfg:
        kwargs["adapter_rank"] = int(ckpt_cfg["adapter_rank"])
    if "no_adapter" in ckpt_cfg:
        kwargs["no_adapter"] = bool(ckpt_cfg["no_adapter"])

    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    model.router.load_state_dict(
        {k: v.to(device=device, dtype=dtype) for k, v in ckpt["router"].items()}
    )
    model.adapters.load_state_dict(
        {k: v.to(device=device, dtype=dtype) for k, v in ckpt["adapters"].items()}
    )
    model.gamma.data.copy_(ckpt["gamma"].to(device=device, dtype=dtype))
    model.eval()
    return model, processor, {"checkpoint_config": ckpt_cfg, "model_kwargs": kwargs}


def load_task_dataset(task: str, split: str, n: int, offset: int):
    if task == "mmbench":
        try:
            ds = load_dataset("lmms-lab/MMBench_EN", split=split)
            dataset_name = "lmms-lab/MMBench_EN"
        except Exception:
            ds = load_dataset("lmms-lab/MMBench", "dev", split=split)
            dataset_name = "lmms-lab/MMBench"
    elif task == "mmstar":
        ds = load_dataset("Lin-Chen/MMStar", split=split)
        dataset_name = "Lin-Chen/MMStar"
    else:
        raise ValueError(f"unsupported task: {task}")
    end = min(len(ds), offset + n)
    if offset > 0 or end < len(ds):
        ds = ds.select(range(offset, end))
    return ds, dataset_name


def build_mmbench_prompt(example: dict[str, Any], idx: int):
    image = example.get("image")
    if image is None:
        return None
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    options: list[tuple[str, str]] = []
    for letter in "ABCD":
        value = example.get(letter)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
            options.append((letter, value.strip()))
    if len(options) < 2:
        return None

    question = str(example.get("question", "")).strip()
    hint = str(example.get("hint") or "").strip()
    prompt = f"{hint}\n{question}" if hint else question
    option_text = "\n".join(f"{letter}. {value}" for letter, value in options)
    user = [
        {"type": "image", "image": image},
        {
            "type": "text",
            "text": f"{prompt}\n\n{option_text}\n\nAnswer with the correct letter.",
        },
    ]
    gold = str(example.get("answer", "")).strip().upper()
    sample_id = example.get("index", example.get("id", idx))
    return {
        "sample_id": str(sample_id),
        "messages": [{"role": "user", "content": user}],
        "images": [image],
        "options": options,
        "gold": gold[:1] if gold else "",
        "question": question[:300],
    }


def parse_inline_options(question: str):
    matches = list(
        re.finditer(
            r"(?:^|\n|\s)([A-F])[\.\):\-]\s*(.*?)(?=(?:\n|\s)[A-F][\.\):\-]\s*|$)",
            question,
            flags=re.S,
        )
    )
    options: list[tuple[str, str]] = []
    for match in matches:
        letter = match.group(1)
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        if value:
            options.append((letter, value))
    seen = set()
    deduped = []
    for letter, value in options:
        if letter in seen:
            continue
        seen.add(letter)
        deduped.append((letter, value))
    return deduped


def build_mmstar_prompt(example: dict[str, Any], idx: int):
    image = example.get("image")
    if image is None:
        return None
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    question = str(example.get("question", "")).strip()
    options = parse_inline_options(question)
    if len(options) < 2:
        return None
    gold = str(example.get("answer", "")).strip().upper()
    user = [
        {"type": "image", "image": image},
        {
            "type": "text",
            "text": f"{question}\n\nAnswer with the correct letter.",
        },
    ]
    sample_id = example.get("index", example.get("id", idx))
    return {
        "sample_id": str(sample_id),
        "messages": [{"role": "user", "content": user}],
        "images": [image],
        "options": options,
        "gold": gold[:1] if gold else "",
        "question": question[:300],
    }


def build_prompt(task: str, example: dict[str, Any], idx: int):
    if task == "mmbench":
        return build_mmbench_prompt(example, idx)
    if task == "mmstar":
        return build_mmstar_prompt(example, idx)
    raise ValueError(f"unsupported task: {task}")


def softmax_np(scores: list[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, 1.0 / max(len(arr), 1))
    floor = np.min(arr[finite]) - 50.0
    arr = np.where(finite, arr, floor)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def option_margin(scores: list[float]) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    top2 = np.sort(arr)[-2:]
    return float(top2[-1] - top2[-2])


def normalized_entropy(scores: list[float]) -> float:
    probs = softmax_np(scores)
    probs = np.clip(probs, 1e-12, 1.0)
    ent = float(-(probs * np.log(probs)).sum())
    return ent / math.log(max(len(probs), 2))


def aggregate_trace_to_vectors(trace: list[dict[str, Any]] | None, num_blocks: int):
    per_block_w = [[] for _ in range(num_blocks)]
    skipped: list[int] = []
    requested: list[int] = []
    for item in trace or []:
        block_idx = int(item.get("block_idx", -1))
        if not (0 <= block_idx < num_blocks):
            continue
        w_recent = as_float(item.get("w_recent"))
        if w_recent is not None:
            per_block_w[block_idx].append(w_recent)
        if item.get("skipped"):
            if block_idx not in skipped:
                skipped.append(block_idx)
        if item.get("dynamic_skip_requested"):
            if block_idx not in requested:
                requested.append(block_idx)
    w = [
        float(np.mean(values)) if values else None
        for values in per_block_w
    ]
    return w, skipped, requested


@torch.no_grad()
def raw_retrofit_base_forward(
    model: Qwen3VLAttnResRetrofit,
    inputs: dict[str, torch.Tensor | Any],
    dynamic_cfg: dict[str, Any] | None,
    return_trace: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    """Call the patched HF base model while controlling ReSkip internals.

    The public retrofit wrapper does not expose past_key_values in its dataclass
    output. For cached scoring we call `base_model` directly; its language-model
    forward is already monkey-patched by Qwen3VLAttnResRetrofit.
    """
    model._fwd_alpha_list = None
    model._fwd_skip_trace = None
    model._fwd_block_inputs = None
    model._fwd_block_outputs = None
    model._fwd_surrogate_outputs = None
    model._fwd_entropy = None
    model._active_skip_blocks = set()
    model._dynamic_skip_config = dynamic_cfg
    model._random_skip_config = None
    model._collect_block_states = False
    model._return_alpha_flag = return_trace
    call_inputs = dict(inputs)
    call_inputs["use_cache"] = True
    out = model.base_model(**call_inputs)
    return out, list(model._fwd_skip_trace or [])


@torch.no_grad()
def score_option_prefill(
    model: Qwen3VLAttnResRetrofit,
    processor: AutoProcessor,
    device: str,
    prefix: str,
    prefix_len: int,
    images: list[Any],
    option_text: str,
    dynamic_cfg: dict[str, Any] | None,
    return_trace: bool,
) -> tuple[float, list[dict[str, Any]], int, int]:
    full = prefix + option_text
    inputs = processor(text=[full], images=images or None, return_tensors="pt")
    full_len = int(inputs["input_ids"].shape[1])
    if full_len <= prefix_len:
        return -1e9, [], 0, 0
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(
        **inputs,
        dynamic_skip_config=dynamic_cfg,
        return_alpha=return_trace,
        use_cache=False,
    )
    logits = out.logits
    target = inputs["input_ids"][0, prefix_len:]
    pred = logits[0, prefix_len - 1 : prefix_len - 1 + len(target), :]
    log_probs = F.log_softmax(pred.float(), dim=-1)
    ll = log_probs.gather(1, target.unsqueeze(-1)).sum().item()
    score = ll / max(int(len(target)), 1)
    trace = list(out.skip_trace or [])
    skipped_events = sum(1 for item in trace if item.get("skipped"))
    return float(score), trace, 1, skipped_events


@torch.no_grad()
def score_option_cached(
    model: Qwen3VLAttnResRetrofit,
    processor: AutoProcessor,
    device: str,
    prefix: str,
    prefix_inputs: dict[str, torch.Tensor],
    prefix_len: int,
    images: list[Any],
    option_text: str,
    dynamic_cfg: dict[str, Any] | None,
    return_trace: bool,
    max_option_tokens: int,
) -> tuple[float, list[dict[str, Any]], int, int]:
    full = prefix + option_text
    full_inputs = processor(text=[full], images=images or None, return_tensors="pt")
    full_len = int(full_inputs["input_ids"].shape[1])
    if full_len <= prefix_len:
        return -1e9, [], 0, 0
    target = full_inputs["input_ids"][0, prefix_len:]
    if max_option_tokens > 0:
        target = target[:max_option_tokens]
    if target.numel() == 0:
        return -1e9, [], 0, 0
    target = target.to(device)

    out, trace = raw_retrofit_base_forward(
        model=model,
        inputs=prefix_inputs,
        dynamic_cfg=dynamic_cfg,
        return_trace=return_trace,
    )
    traces = list(trace)
    n_forwards = 1
    skipped_events = sum(1 for item in trace if item.get("skipped"))

    log_probs = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    ll = log_probs[target[0]].item()
    past = out.past_key_values
    prev = target[0].view(1, 1)

    for pos in range(1, int(target.numel())):
        step_inputs = {
            "input_ids": prev,
            "past_key_values": past,
        }
        out, trace = raw_retrofit_base_forward(
            model=model,
            inputs=step_inputs,
            dynamic_cfg=dynamic_cfg,
            return_trace=return_trace,
        )
        n_forwards += 1
        traces.extend(trace)
        skipped_events += sum(1 for item in trace if item.get("skipped"))
        log_probs = F.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        ll += log_probs[target[pos]].item()
        past = out.past_key_values
        prev = target[pos].view(1, 1)

    score = ll / max(int(target.numel()), 1)
    return float(score), traces, n_forwards, skipped_events


@torch.no_grad()
def collect_records(
    model: Qwen3VLAttnResRetrofit,
    processor: AutoProcessor,
    ds,
    device: str,
    dynamic_cfg: dict[str, Any],
    num_blocks: int,
    progress_every: int,
    scoring_mode: str,
    max_option_tokens: int,
    task: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped_examples = 0
    t0 = time.time()
    for i, example in enumerate(ds):
        item = build_prompt(task, example, i)
        if item is None:
            skipped_examples += 1
            continue

        prefix = processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
        prefix_inputs = processor(
            text=[prefix], images=item["images"] or None, return_tensors="pt"
        )
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        prefix_inputs = {k: v.to(device) for k, v in prefix_inputs.items()}

        full_scores: list[float] = []
        skip_scores: list[float] = []
        skip_traces: list[list[dict[str, Any]]] = []
        skip_forwards: list[int] = []
        skip_events: list[int] = []
        for letter, value in item["options"]:
            option_text = f" {letter}. {value}"
            if scoring_mode == "cached":
                full_score, _, _, _ = score_option_cached(
                    model=model,
                    processor=processor,
                    device=device,
                    prefix=prefix,
                    prefix_inputs=prefix_inputs,
                    prefix_len=prefix_len,
                    images=item["images"],
                    option_text=option_text,
                    dynamic_cfg=None,
                    return_trace=False,
                    max_option_tokens=max_option_tokens,
                )
                skip_score, trace, n_fwds, n_skips = score_option_cached(
                    model=model,
                    processor=processor,
                    device=device,
                    prefix=prefix,
                    prefix_inputs=prefix_inputs,
                    prefix_len=prefix_len,
                    images=item["images"],
                    option_text=option_text,
                    dynamic_cfg=dynamic_cfg,
                    return_trace=True,
                    max_option_tokens=max_option_tokens,
                )
            else:
                full_score, _, _, _ = score_option_prefill(
                    model=model,
                    processor=processor,
                    device=device,
                    prefix=prefix,
                    prefix_len=prefix_len,
                    images=item["images"],
                    option_text=option_text,
                    dynamic_cfg=None,
                    return_trace=False,
                )
                skip_score, trace, n_fwds, n_skips = score_option_prefill(
                    model=model,
                    processor=processor,
                    device=device,
                    prefix=prefix,
                    prefix_len=prefix_len,
                    images=item["images"],
                    option_text=option_text,
                    dynamic_cfg=dynamic_cfg,
                    return_trace=True,
                )
            full_scores.append(full_score)
            skip_scores.append(skip_score)
            skip_traces.append(trace)
            skip_forwards.append(n_fwds)
            skip_events.append(n_skips)

        letters = [x[0] for x in item["options"]]
        full_pred_idx = int(np.argmax(np.asarray(full_scores)))
        skip_pred_idx = int(np.argmax(np.asarray(skip_scores)))
        full_pred = letters[full_pred_idx]
        skip_pred = letters[skip_pred_idx]
        margin = option_margin(full_scores)
        diffs = [
            abs(float(a) - float(b))
            for a, b in zip(skip_scores, full_scores)
            if math.isfinite(float(a)) and math.isfinite(float(b))
        ]
        epsilon = float(max(diffs)) if diffs else 0.0
        req_counts = [
            sum(1 for tr in trace if tr.get("dynamic_skip_requested"))
            for trace in skip_traces
        ]
        pred_trace = skip_traces[full_pred_idx] if skip_traces else []
        pred_w, pred_skipped, pred_requested = aggregate_trace_to_vectors(pred_trace, num_blocks)
        total_forwards = int(sum(skip_forwards))
        total_skip_events = int(sum(skip_events))

        record = {
            "idx": len(records),
            "dataset_row": i,
            "sample_id": item["sample_id"],
            "gold": item["gold"],
            "full_pred": full_pred,
            "skip_pred": skip_pred,
            "full_correct": bool(full_pred == item["gold"]),
            "skip_correct": bool(skip_pred == item["gold"]),
            "prediction_preserved": bool(full_pred == skip_pred),
            "margin_delta": margin,
            "epsilon": epsilon,
            "two_epsilon": 2.0 * epsilon,
            "margin_preservation_condition": bool(2.0 * epsilon <= margin),
            "uncertainty_entropy": normalized_entropy(full_scores),
            "top_probability": float(softmax_np(full_scores).max()),
            "options": letters,
            "full_scores": full_scores,
            "skip_scores": skip_scores,
            "avg_skipped_blocks": float(total_skip_events / max(total_forwards, 1)),
            "total_skip_events": total_skip_events,
            "total_scoring_forwards": total_forwards,
            "max_skipped_blocks": int(max(skip_events)) if skip_events else 0,
            "avg_dynamic_requests": float(np.mean(req_counts)) if req_counts else 0.0,
            "pred_trace_w_recent": pred_w,
            "pred_trace_skipped": pred_skipped,
            "pred_trace_dynamic_requested": pred_requested,
            "question_preview": item["question"],
        }
        records.append(record)

        if progress_every and (len(records) % progress_every == 0):
            elapsed = time.time() - t0
            preserved = np.mean([r["prediction_preserved"] for r in records])
            mean_skip = np.mean([r["avg_skipped_blocks"] for r in records])
            print(
                f"[collect] {len(records):4d}/{len(ds)} usable, "
                f"preserved={preserved:.3f}, skip={mean_skip:.3f}, "
                f"skipped_examples={skipped_examples}, {elapsed:.0f}s",
                flush=True,
            )
    return records


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else None


def bin_by_uncertainty(records: list[dict[str, Any]]):
    uncertainty = np.asarray([r["uncertainty_entropy"] for r in records], dtype=float)
    skips = np.asarray([r["avg_skipped_blocks"] for r in records], dtype=float)
    q1, q2 = np.quantile(uncertainty, [1 / 3, 2 / 3])
    groups = [
        ("Low\n(easy)", uncertainty <= q1),
        ("Medium", (uncertainty > q1) & (uncertainty <= q2)),
        ("High\n(hard)", uncertainty > q2),
    ]
    stats = []
    for label, mask in groups:
        vals = skips[mask]
        stats.append(
            {
                "label": label.replace("\n", " "),
                "n": int(mask.sum()),
                "mean_skips": float(vals.mean()) if len(vals) else 0.0,
                "sem_skips": float(vals.std(ddof=1) / math.sqrt(len(vals)))
                if len(vals) > 1
                else 0.0,
                "mean_uncertainty": float(uncertainty[mask].mean()) if mask.any() else 0.0,
            }
        )
    return stats, (float(q1), float(q2))


def select_heatmap_rows(records: list[dict[str, Any]], rows_per_group: int):
    preserved = [r for r in records if r["prediction_preserved"]]
    usable = preserved or records
    easy_candidates = sorted(
        usable,
        key=lambda r: (
            r["avg_skipped_blocks"] <= 0,
            -float(r["margin_delta"]),
            r["idx"],
        ),
    )
    hard_candidates = sorted(
        usable,
        key=lambda r: (
            float(r["margin_delta"]),
            float(r["avg_skipped_blocks"]),
            r["idx"],
        ),
    )
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for r in easy_candidates:
        if len(selected) >= rows_per_group:
            break
        selected.append(r)
        seen.add(int(r["idx"]))
    for r in hard_candidates:
        if len(selected) >= rows_per_group * 2:
            break
        if int(r["idx"]) in seen:
            continue
        selected.append(r)
        seen.add(int(r["idx"]))
    return selected


def plot_figure(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    fig_path: Path,
    png_path: Path,
    rows_per_group: int,
):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.5, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1.15])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    num_blocks = int(summary["num_blocks"])
    heat_rows = select_heatmap_rows(records, rows_per_group)
    heat = np.full((len(heat_rows), num_blocks), np.nan, dtype=float)
    for row_idx, record in enumerate(heat_rows):
        for block_idx, value in enumerate(record["pred_trace_w_recent"]):
            if value is not None:
                heat[row_idx, block_idx] = float(value)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax0.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    for row_idx, record in enumerate(heat_rows):
        for block_idx in record["pred_trace_skipped"]:
            ax0.add_patch(
                Rectangle(
                    (block_idx - 0.5, row_idx - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#d62728",
                    linewidth=1.8,
                )
            )
    ax0.set_xticks(range(num_blocks))
    ax0.set_xticklabels([f"B{i}" for i in range(num_blocks)])
    ylabels = []
    for i, record in enumerate(heat_rows):
        prefix = "easy" if i < rows_per_group else "hard"
        ylabels.append(
            f"{prefix} #{record['sample_id']}\n"
            f"Delta={record['margin_delta']:.2f}, skip={record['avg_skipped_blocks']:.1f}"
        )
    ax0.set_yticks(range(len(heat_rows)))
    ax0.set_yticklabels(ylabels)
    ax0.set_title("(a) Input-dependent routing")
    ax0.set_xlabel("Block")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label("recent-source weight")

    bin_stats = summary["uncertainty_bins"]
    xs = np.arange(len(bin_stats))
    means = [x["mean_skips"] for x in bin_stats]
    sems = [x["sem_skips"] for x in bin_stats]
    labels = [x["label"].replace(" ", "\n", 1) if x["label"].startswith("Low") else x["label"] for x in bin_stats]
    ax1.bar(xs, means, yerr=sems, capsize=3, color=["#4c78a8", "#72b7b2", "#f58518"], alpha=0.9)
    ax1.plot(xs, means, color="#222222", linewidth=1.0, marker="o", markersize=3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Avg. skipped blocks / forward")
    rho = summary.get("spearman_uncertainty_skip")
    rho_text = "n/a" if rho is None else f"{rho:.2f}"
    ax1.set_title("(b) Depth vs. full-depth uncertainty")
    ax1.text(
        0.02,
        0.95,
        f"Spearman rho={rho_text}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )
    ax1.grid(axis="y", alpha=0.25)

    margins = np.asarray([r["margin_delta"] for r in records], dtype=float)
    two_eps = np.asarray([r["two_epsilon"] for r in records], dtype=float)
    preserved = np.asarray([r["prediction_preserved"] for r in records], dtype=bool)
    cond = np.asarray([r["margin_preservation_condition"] for r in records], dtype=bool)
    finite = np.isfinite(margins) & np.isfinite(two_eps)
    margins = margins[finite]
    two_eps = two_eps[finite]
    preserved = preserved[finite]
    cond = cond[finite]
    lim = float(max(np.max(margins), np.max(two_eps), 1e-3) * 1.08)
    line = np.linspace(0.0, lim, 200)
    ax2.fill_between(line, 0.0, line, color="#59a14f", alpha=0.11, linewidth=0)
    ax2.scatter(
        margins[preserved],
        two_eps[preserved],
        s=20,
        color="#1f77b4",
        alpha=0.72,
        label="prediction preserved",
        edgecolors="none",
    )
    if (~preserved).any():
        ax2.scatter(
            margins[~preserved],
            two_eps[~preserved],
            s=26,
            color="#d62728",
            alpha=0.86,
            label="prediction changed",
            marker="x",
        )
    ax2.plot(line, line, "--", color="#222222", linewidth=1.0, label=r"$2\epsilon=\Delta$")
    ax2.set_xlim(0.0, lim)
    ax2.set_ylim(0.0, lim)
    ax2.set_xlabel(r"Full-depth margin $\Delta(x)$")
    ax2.set_ylabel(r"Skip perturbation $2\epsilon(x)$")
    ax2.set_title("(c) Margin-preservation check")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)
    ax2.text(
        0.98,
        0.04,
        f"{100.0 * cond.mean():.1f}% below boundary\n"
        f"{100.0 * preserved.mean():.1f}% predictions preserved",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    fig.suptitle(
        "Figure X: ReSkip routing and margin-preservation diagnostics",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize(records: list[dict[str, Any]], args, model_meta, dyn_raw, dataset_name: str):
    if not records:
        raise RuntimeError("No usable records collected.")
    full_correct = np.asarray([r["full_correct"] for r in records], dtype=float)
    skip_correct = np.asarray([r["skip_correct"] for r in records], dtype=float)
    preserved = np.asarray([r["prediction_preserved"] for r in records], dtype=float)
    cond = np.asarray([r["margin_preservation_condition"] for r in records], dtype=float)
    skips = np.asarray([r["avg_skipped_blocks"] for r in records], dtype=float)
    margins = np.asarray([r["margin_delta"] for r in records], dtype=float)
    uncertainty = np.asarray([r["uncertainty_entropy"] for r in records], dtype=float)
    two_eps = np.asarray([r["two_epsilon"] for r in records], dtype=float)
    bin_stats, thresholds = bin_by_uncertainty(records)
    return {
        "task": args.task,
        "dataset": dataset_name,
        "split": args.split,
        "n_requested": int(args.n),
        "n_used": int(len(records)),
        "offset": int(args.offset),
        "model_path": args.model_path,
        "state_path": str(args.state_path),
        "dynamic_config_path": str(args.dynamic_config),
        "dynamic_config": dyn_raw,
        "scoring_mode": args.scoring_mode,
        "max_option_tokens": int(args.max_option_tokens),
        "checkpoint_config": model_meta["checkpoint_config"],
        "num_blocks": int(model_meta["model_kwargs"]["num_blocks"]),
        "full_accuracy": float(full_correct.mean()),
        "reskip_accuracy": float(skip_correct.mean()),
        "prediction_preserved_rate": float(preserved.mean()),
        "margin_condition_rate": float(cond.mean()),
        "mean_avg_skipped_blocks": float(skips.mean()),
        "mean_margin_delta": float(margins.mean()),
        "median_margin_delta": float(np.median(margins)),
        "mean_two_epsilon": float(two_eps.mean()),
        "median_two_epsilon": float(np.median(two_eps)),
        "uncertainty_bins": bin_stats,
        "uncertainty_terciles": thresholds,
        "spearman_uncertainty_skip": spearman(uncertainty, skips),
        "spearman_margin_skip": spearman(margins, skips),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--state-path", type=Path, default=Path(DEFAULT_STATE_PATH))
    parser.add_argument("--dynamic-config", type=Path, default=Path(DEFAULT_DYN_CONFIG))
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--task", choices=["mmbench", "mmstar"], default="mmbench")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("retrofit/outputs/fig_reskip_adaptive_depth/qwen2b_mmbench_q050"))
    parser.add_argument("--fig-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--rows-per-group", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--scoring-mode",
        choices=["cached", "prefill"],
        default="cached",
        help="cached scores option tokens incrementally, matching the decode regime where dynamic skip triggers.",
    )
    parser.add_argument(
        "--max-option-tokens",
        type=int,
        default=12,
        help="For cached scoring, truncate each answer option to this many target tokens; <=0 means no truncation.",
    )
    args = parser.parse_args()

    args.state_path = args.state_path.resolve()
    args.dynamic_config = args.dynamic_config.resolve()
    args.out_dir = args.out_dir.resolve()
    args.fig_dir = args.fig_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"[setup] device={device} dtype={dtype}", flush=True)
    dynamic_cfg, dyn_raw = load_dynamic_config(args.dynamic_config)
    model, processor, model_meta = load_model(
        args.model_path, args.state_path, device, dtype, args.num_blocks
    )
    num_blocks = int(model_meta["model_kwargs"]["num_blocks"])
    print(
        f"[setup] num_blocks={num_blocks}, dynamic eligible="
        f"{sorted(dynamic_cfg['eligible_blocks']) if dynamic_cfg['eligible_blocks'] else 'all'}, "
        f"max_skips={dynamic_cfg.get('max_skips')}",
        flush=True,
    )
    if args.task == "mmstar" and args.split == "dev":
        args.split = "val"
    ds, dataset_name = load_task_dataset(args.task, args.split, args.n, args.offset)
    print(f"[data] {dataset_name}/{args.split}, rows={len(ds)}", flush=True)
    print(
        f"[collect] scoring_mode={args.scoring_mode}, "
        f"max_option_tokens={args.max_option_tokens}",
        flush=True,
    )

    records = collect_records(
        model=model,
        processor=processor,
        ds=ds,
        device=device,
        dynamic_cfg=dynamic_cfg,
        num_blocks=num_blocks,
        progress_every=args.progress_every,
        scoring_mode=args.scoring_mode,
        max_option_tokens=args.max_option_tokens,
        task=args.task,
    )
    summary = summarize(records, args, model_meta, dyn_raw, dataset_name)

    records_path = args.out_dir / "records.jsonl"
    with records_path.open("w") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    fig_path = args.fig_dir / "reskip_adaptive_depth_vlm.pdf"
    png_path = args.fig_dir / "reskip_adaptive_depth_vlm.png"
    plot_figure(records, summary, fig_path, png_path, args.rows_per_group)
    print(f"[done] records: {records_path}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)
    print(f"[done] figure:  {fig_path}", flush=True)
    print(f"[done] figure:  {png_path}", flush=True)
    print(
        "[summary] "
        f"n={summary['n_used']} full_acc={summary['full_accuracy']:.4f} "
        f"reskip_acc={summary['reskip_accuracy']:.4f} "
        f"preserved={summary['prediction_preserved_rate']:.4f} "
        f"condition={summary['margin_condition_rate']:.4f} "
        f"skip={summary['mean_avg_skipped_blocks']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
