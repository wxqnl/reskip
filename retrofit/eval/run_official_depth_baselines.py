"""Run official-style depth baseline evals for Qwen3-VL-2B.

The training jobs are launched separately because they run for a while. This
script assumes the trained artifacts exist under:

  retrofit/outputs/related_qwen2b/official_depth/{layerskip,calm,mod}_v3_10k

It then runs representative LLM/VLM benchmarks plus 512->4096 decode speed for
LayerSkip, CALM, and MoD method-faithful baselines, and writes one appendix
summary JSON/Markdown table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


PROJECT = Path("/home/user01/Minko/reskip2/reskip")
PY = Path("/home/user01/Minko/reskip2/.venv/bin/python")
MODEL = Path("/home/user01/Minko/models/Qwen3-VL-2B")
OUT = PROJECT / "retrofit/outputs/related_qwen2b/official_depth"


def make_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = (
        str(PROJECT / "retrofit")
        + ":"
        + str(PROJECT / "retrofit/eval")
        + ":"
        + env.get("PYTHONPATH", "")
    )
    return env


def run(cmd: list[str], log_path: Path, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[official-depth] GPU{gpu} -> {' '.join(cmd)}")
    with open(log_path, "w") as f:
        p = subprocess.Popen(cmd, cwd=PROJECT, env=make_env(gpu), stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def run_parallel(jobs: list[tuple[str, list[str], Path]], gpus: list[int]) -> list[tuple[str, int, str]]:
    """Run at most one subprocess per listed GPU."""
    failures: list[tuple[str, int, str]] = []
    pending = list(jobs)
    active = []
    free_gpus = list(gpus)
    while pending or active:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            label, cmd, log_path = pending.pop(0)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = open(log_path, "w")
            print(f"[official-depth] START {label} on GPU{gpu} -> {log_path}")
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT,
                env=make_env(gpu),
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {
                    "label": label,
                    "cmd": cmd,
                    "log": log_path,
                    "gpu": gpu,
                    "proc": proc,
                    "log_f": log_f,
                    "t0": time.time(),
                }
            )

        time.sleep(5)
        still_active = []
        for item in active:
            rc = item["proc"].poll()
            if rc is None:
                still_active.append(item)
                continue
            item["log_f"].close()
            elapsed = time.time() - item["t0"]
            print(
                f"[official-depth] DONE {item['label']} rc={rc} "
                f"elapsed={elapsed:.1f}s GPU{item['gpu']} log={item['log']}"
            )
            free_gpus.append(item["gpu"])
            if rc != 0:
                failures.append((item["label"], int(rc), str(item["log"])))
        active = still_active
        if failures:
            for item in active:
                item["proc"].terminate()
                item["log_f"].close()
            break
    return failures


def require_artifacts() -> dict[str, dict[str, Path]]:
    jobs = {
        "layerskip": OUT / "layerskip_v3_10k",
        "calm": OUT / "calm_v3_10k",
        "mod": OUT / "mod_v3_10k",
    }
    paths: dict[str, dict[str, Path]] = {}
    missing = []
    for name, root in jobs.items():
        adapter = root / "lora_adapter"
        if not (adapter / "adapter_config.json").exists():
            missing.append(str(adapter))
        entry = {"root": root, "adapter": adapter}
        if name == "mod":
            router = root / "mod_routers.pt"
            if not router.exists():
                missing.append(str(router))
            entry["router"] = router
        paths[name] = entry
    if missing:
        raise FileNotFoundError("missing trained artifacts:\n" + "\n".join(missing))
    return paths


def lm_cmd(name: str, paths: dict[str, Path], limit: int) -> list[str]:
    out_dir = OUT / f"eval_lm_{name}"
    cmd = [
        str(PY),
        "retrofit/eval/run_lm_eval.py",
        "--model-path",
        str(MODEL),
        "--lora-adapter-path",
        str(paths["adapter"]),
        "--tasks",
        "lambada_openai,hellaswag",
        "--limit",
        str(limit),
        "--batch-size",
        "8",
        "--gpu",
        "0",
        "--output",
        str(out_dir),
        "--label",
        f"official_{name}_v3_10k",
    ]
    if name in {"layerskip", "calm"}:
        cmd += ["--early-exit-layer", "24"]
    else:
        cmd += [
            "--mod-layers",
            "12|13|14|15",
            "--mod-router-path",
            str(paths["router"]),
            "--mod-keep-ratio",
            "0.8",
        ]
    return cmd


def vlm_cmd(name: str, paths: dict[str, Path]) -> list[str]:
    out_dir = OUT / f"eval_vlm_{name}"
    if name in {"layerskip", "calm"}:
        model_args = (
            f"pretrained={MODEL},"
            f"lora_adapter_path={paths['adapter']},"
            "skip_layers=24|25|26|27,"
            "max_pixels=1605632,min_pixels=200704"
        )
    else:
        model_args = (
            f"pretrained={MODEL},"
            f"lora_adapter_path={paths['adapter']},"
            "mod_layers=12|13|14|15,"
            f"mod_router_path={paths['router']},"
            "mod_keep_ratio=0.8,"
            "max_pixels=1605632,min_pixels=200704"
        )
    code = (
        "import sys; "
        f"sys.path.insert(0, '{PROJECT / 'retrofit'}'); "
        f"sys.path.insert(0, '{PROJECT / 'retrofit/eval'}'); "
        "import lmms_eval_retrofit; "
        "import lmms_eval.__main__ as m; "
        "sys.argv=['lmms_eval','--model','qwen3_vl_pruned',"
        f"'--model_args','{model_args}',"
        "'--tasks','ai2d,mmmu_val,mmstar',"
        "'--batch_size','1',"
        f"'--output_path','{out_dir}']; "
        "m.cli_evaluate()"
    )
    return [str(PY), "-c", code]


def speed_cmd(name: str, paths: dict[str, Path] | None = None) -> list[str]:
    out_path = OUT / "decode_512_4096" / f"{name}.json"
    variant = {
        "base": "base",
        "layerskip": "official_layerskip",
        "calm": "official_calm",
        "mod": "official_mod",
    }[name]
    cmd = [
        str(PY),
        "retrofit/bench/bench_decode_512_4096_variants.py",
        "--variant",
        variant,
        "--model-path",
        str(MODEL),
        "--prompt-len",
        "512",
        "--decode-tokens",
        "4096",
        "--warmup-decode-tokens",
        "32",
        "--gpu",
        "0",
        "--out",
        str(out_path),
    ]
    if paths is not None:
        cmd += ["--lora-adapter-path", str(paths["adapter"])]
    if name == "mod" and paths is not None:
        cmd += ["--mod-router-path", str(paths["router"]), "--mod-keep-ratio", "0.8"]
    return cmd


def latest_result_json(root: Path) -> Path | None:
    files = sorted(root.glob("**/*_results.json"))
    return files[-1] if files else None


def metric(res: dict, task: str, key: str):
    try:
        return res["results"][task][key]
    except KeyError:
        return None


def aggregate() -> dict:
    payload: dict[str, dict] = {"llm": {}, "vlm": {}, "speed": {}}
    for name in ("layerskip", "calm", "mod"):
        summary = OUT / f"eval_lm_{name}" / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            payload["llm"][name] = {
                "lambada_acc": metric(data, "lambada_openai", "acc,none"),
                "lambada_ppl": metric(data, "lambada_openai", "perplexity,none"),
                "hellaswag_acc_norm": metric(data, "hellaswag", "acc_norm,none"),
                "details": str(summary),
                "mod_stats": data.get("mod_proxy_stats"),
            }
        result_json = latest_result_json(OUT / f"eval_vlm_{name}")
        if result_json is not None:
            data = json.loads(result_json.read_text())
            payload["vlm"][name] = {
                "ai2d_exact": metric(data, "ai2d", "exact_match,flexible-extract"),
                "mmmu_acc": metric(data, "mmmu_val", "mmmu_acc,none"),
                "mmstar_avg": metric(data, "mmstar", "average,none"),
                "details": str(result_json),
            }
        speed = OUT / "decode_512_4096" / f"{name}.json"
        if speed.exists():
            data = json.loads(speed.read_text())
            payload["speed"][name] = {
                "decode_ms_per_token": data["best"]["decode_ms_per_token"],
                "decode_tokens_per_second": data["best"]["decode_tokens_per_second"],
                "prefill_ms": data["best"]["prefill_ms"],
                "details": str(speed),
                "aux_stats": data.get("aux_stats"),
            }
    base_speed = OUT / "decode_512_4096/base.json"
    if base_speed.exists():
        data = json.loads(base_speed.read_text())
        payload["speed"]["base"] = {
            "decode_ms_per_token": data["best"]["decode_ms_per_token"],
            "decode_tokens_per_second": data["best"]["decode_tokens_per_second"],
            "prefill_ms": data["best"]["prefill_ms"],
            "details": str(base_speed),
        }
    return payload


def write_markdown(payload: dict):
    md = OUT / "official_depth_summary.md"
    lines = [
        "# Official-depth Qwen3-VL-2B Baselines",
        "",
        "Training data: same v3 retrofit mix.",
        "LLM tasks: LAMBADA OpenAI and HellaSwag, limit=2000.",
        "VLM tasks: AI2D, MMMU validation, MMStar.",
        "Speed: input 512 tokens, decode 4096 tokens, StaticCache.",
        "",
        "## LLM",
        "",
        "| method | lambada acc | lambada ppl | hellaswag acc_norm |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload.get("llm", {}).items():
        lines.append(
            f"| {name} | {fmt(row.get('lambada_acc'))} | "
            f"{fmt(row.get('lambada_ppl'))} | {fmt(row.get('hellaswag_acc_norm'))} |"
        )
    lines += [
        "",
        "## VLM",
        "",
        "| method | AI2D | MMMU | MMStar |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload.get("vlm", {}).items():
        lines.append(
            f"| {name} | {fmt(row.get('ai2d_exact'))} | "
            f"{fmt(row.get('mmmu_acc'))} | {fmt(row.get('mmstar_avg'))} |"
        )
    lines += [
        "",
        "## 512-to-4096 Decode Speed",
        "",
        "| method | ms/token | tok/s | prefill ms |",
        "|---|---:|---:|---:|",
    ]
    order = ["base", "layerskip", "calm", "mod"]
    for name in order:
        row = payload.get("speed", {}).get(name)
        if row:
            lines.append(
                f"| {name} | {fmt(row.get('decode_ms_per_token'))} | "
                f"{fmt(row.get('decode_tokens_per_second'))} | {fmt(row.get('prefill_ms'))} |"
            )
    md.write_text("\n".join(lines) + "\n")
    return md


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eval", "speed", "aggregate", "all"], default="all")
    ap.add_argument("--lm-limit", type=int, default=2000)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    paths = require_artifacts() if args.stage != "aggregate" else {}

    jobs: list[tuple[str, list[str], Path]] = []
    if args.stage in {"eval", "all"}:
        for name in ("layerskip", "calm", "mod"):
            out = OUT / f"eval_lm_{name}" / "summary.json"
            if not (args.skip_existing and out.exists()):
                jobs.append((f"lm_{name}", lm_cmd(name, paths[name], args.lm_limit), OUT / "logs" / f"eval_lm_{name}.log"))
        for name in ("layerskip", "calm", "mod"):
            if not (args.skip_existing and latest_result_json(OUT / f"eval_vlm_{name}")):
                jobs.append((f"vlm_{name}", vlm_cmd(name, paths[name]), OUT / "logs" / f"eval_vlm_{name}.log"))
    if args.stage in {"speed", "all"}:
        speed_names = ("base", "layerskip", "calm", "mod")
        for name in speed_names:
            out = OUT / "decode_512_4096" / f"{name}.json"
            if args.skip_existing and out.exists():
                continue
            cmd = speed_cmd(name, None if name == "base" else paths[name])
            jobs.append((f"speed_{name}", cmd, OUT / "logs" / f"speed_{name}.log"))

    failures = run_parallel(jobs, gpus)
    payload = aggregate()
    (OUT / "official_depth_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    md = write_markdown(payload)
    print(f"[official-depth] summary -> {md}")
    if failures:
        raise SystemExit(f"failures: {failures}")


if __name__ == "__main__":
    main()
