"""lmms-eval model plugin for the AR-Retrofit on top of Qwen3-VL-2B.

Extends the stock ``Qwen3_VL`` plugin: after the base model is loaded the
retrofit wrapper monkey-patches ``base.model.language_model.forward`` in
place, so ``self.model.generate(...)`` (called by lmms-eval) executes through
our retrofit. Usage::

    python -m retrofit.lmms_eval_retrofit \
        --retrofit_state outputs/H_r256_5k/retrofit_attnres_state.pt \
        --model_args "pretrained=/home/user01/Minko/models/Qwen3-VL-2B,retrofit_state_path=..." \
        --tasks mmbench_en_dev,mmstar,mmmu_val,ai2d,ocrbench,mathvista_testmini,realworldqa,hallusion_bench \
        ...

Dynamic-skip support: pass ``dynamic_skip=1,quantile=0.95,max_skips=1,positions=4,6,11``
in ``--model_args`` to turn on phase-1 skip during generation. Thresholds are
calibrated on 32 LAMBADA prefixes once at load time.
"""
from __future__ import annotations

import sys
import os
import atexit
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # retrofit/eval (this plugin)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # retrofit/ (core module)

import torch
import torch.nn as nn
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
from lmms_eval.api.registry import register_model
from lmms_eval.models.registry_v2 import ModelManifest
from lmms_eval.models import MODEL_REGISTRY_V2

from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit
from compile_utils import wrap_compile
from peft import PeftModel


def _parse_ints(value: str):
    if not value:
        return []
    sep = "|" if "|" in str(value) else ","
    return [int(x) for x in str(value).split(sep) if str(x).strip()]


def _install_mod_proxy(model, layers_to_patch, keep_ratio, router_state_path=None):
    import math
    import types

    layers = model.model.language_model.layers
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    hidden_size = model.config.text_config.hidden_size
    routers = nn.ModuleDict()
    stats = {
        "layers": sorted(layers_to_patch),
        "keep_ratio": float(keep_ratio),
        "calls": 0,
        "tokens": 0,
        "tokens_bypassed": 0,
        "per_layer_tokens_bypassed": {i: 0 for i in sorted(layers_to_patch)},
    }
    for i in sorted(layers_to_patch):
        if i < 0 or i >= len(layers):
            raise ValueError(f"mod_layer {i} out of range [0, {len(layers)})")
        router = nn.Linear(hidden_size, 1, bias=False).to(device=device, dtype=dtype)
        nn.init.normal_(router.weight, std=0.02)
        routers[str(i)] = router
        original_forward = layers[i].forward
        def _mod_forward(self, hidden_states, *args, _idx=i, _orig=original_forward, **kwargs):
            out = _orig(hidden_states, *args, **kwargs)
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3:
                return out
            bsz, seqlen, _ = h.shape
            n_tok = bsz * seqlen
            keep = max(1, min(seqlen, int(math.ceil(seqlen * float(keep_ratio)))))
            if router_state_path:
                scores = routers[str(_idx)](hidden_states).squeeze(-1)
            else:
                scores = hidden_states.float().pow(2).mean(dim=-1)
            top_idx = scores.topk(k=keep, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            mixed = torch.where(mask.unsqueeze(-1), h, hidden_states)
            bypassed = int((~mask).sum().item())
            stats["calls"] += 1
            stats["tokens"] += n_tok
            stats["tokens_bypassed"] += bypassed
            stats["per_layer_tokens_bypassed"][_idx] += bypassed
            if isinstance(out, tuple):
                return (mixed,) + out[1:]
            return mixed
        layers[i].forward = types.MethodType(_mod_forward, layers[i])
    if router_state_path:
        routers.load_state_dict(torch.load(router_state_path, map_location=device))
    model.mod_routers = routers
    return stats


@register_model("qwen3_vl_retrofit")
class Qwen3_VL_Retrofit(Qwen3_VL):
    """Qwen3-VL base with AR-Retrofit state loaded; monkey-patches forward."""

    def __init__(
        self,
        retrofit_state_path: str = None,
        num_blocks: int = 14,
        adapter_rank: int = 256,
        dynamic_skip: bool = False,
        dyn_quantile: float = 0.95,
        dyn_max_skips: int = 1,
        dyn_positions: str = "4,6,11",
        dynamic_skip_config_path: str = None,
        static_skip_blocks: str = "",
        random_skip_blocks: str = "",
        random_skip_p: float = 0.0,
        random_skip_max_skips: int = 1,
        random_seed: int = 1234,
        skip_stats_path: str = None,
        compile_mode: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if retrofit_state_path is None:
            raise ValueError("retrofit_state_path is required for qwen3_vl_retrofit")
        self._retrofit = self._load_retrofit(
            retrofit_state_path, num_blocks, adapter_rank
        )
        if static_skip_blocks:
            blocks = set(_parse_ints(static_skip_blocks))
            self._retrofit._active_skip_blocks = blocks
            self._retrofit.reset_skip_stats()
            print(f"[retrofit] static block skip armed: {sorted(blocks)}", flush=True)
        if random_skip_blocks:
            blocks = set(_parse_ints(random_skip_blocks))
            self._retrofit._random_skip_config = {
                "eligible_blocks": blocks,
                "p": float(random_skip_p),
                "max_skips": int(random_skip_max_skips),
                "seed": int(random_seed),
                "rng": random.Random(int(random_seed)),
            }
            self._retrofit.reset_skip_stats()
            print(
                f"[retrofit] random block skip armed: blocks={sorted(blocks)} "
                f"p={random_skip_p} M={random_skip_max_skips} seed={random_seed}",
                flush=True,
            )
        if dynamic_skip_config_path:
            cfg = json.load(open(dynamic_skip_config_path))
            thresholds = {int(k): float(v) for k, v in (cfg.get("thresholds") or {}).items()}
            eligible = cfg.get("eligible_blocks")
            self._retrofit._dynamic_skip_config = dict(
                strategy=cfg.get("strategy", "recent_weight_gt"),
                thresholds=thresholds,
                eligible_blocks=set(int(x) for x in eligible) if eligible is not None else None,
                max_skips=int(cfg.get("max_skips", dyn_max_skips)),
            )
            self._retrofit.reset_skip_stats()
            print(
                f"[retrofit] dynamic skip armed from {dynamic_skip_config_path}: "
                f"{self._retrofit._dynamic_skip_config}",
                flush=True,
            )
        elif dynamic_skip:
            self._configure_dynamic_skip(
                quantile=float(dyn_quantile),
                max_skips=int(dyn_max_skips),
                positions=tuple(_parse_ints(dyn_positions)),
            )
        if static_skip_blocks or random_skip_blocks or dynamic_skip or dynamic_skip_config_path:
            self._register_skip_stats_dump(skip_stats_path)
        # Iso-cost paper claim: compile the underlying HF model so lmms-eval's
        # ``self._model.generate(...)`` (which calls ``__call__``) goes through
        # the retrofit-monkey-patched forward under torch.compile. Pass
        # ``compile_mode=off`` in --model_args to bypass for accuracy
        # reproduction. Dyn-skip path graph-breaks on the .item() guard but
        # other layer kernels still compile, so we leave compile on by default
        # there too.
        self._model = wrap_compile(
            self._model,
            mode=compile_mode,
            label="lmms_eval_retrofit",
        )

    def _load_retrofit(self, state_path, num_blocks, adapter_rank):
        ck = torch.load(state_path, map_location="cpu")
        cfg = ck.get("config", {})
        kwargs = {"num_blocks": int(cfg.get("num_blocks", num_blocks))}
        ar = cfg.get("adapter_rank", adapter_rank)
        kwargs["adapter_rank"] = int(ar)
        no_adapter = cfg.get("no_adapter", not ck.get("adapters"))
        if no_adapter:
            kwargs["no_adapter"] = True
            kwargs.pop("adapter_rank", None)

        model = self._model
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        wrapper = Qwen3VLAttnResRetrofit(model, **kwargs).to(device=device, dtype=dtype)
        wrapper.router.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()}
        )
        if not no_adapter:
            wrapper.adapters.load_state_dict(
                {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()}
            )
        wrapper.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
        wrapper.eval()
        gmax = float(wrapper.gamma.detach().abs().max())
        gmean = float(wrapper.gamma.detach().mean())
        print(
            f"[retrofit] state loaded: num_blocks={kwargs['num_blocks']}, "
            f"adapter_rank={kwargs.get('adapter_rank', 'Identity')}, "
            f"γ_max={gmax:.3f}, γ_mean={gmean:+.3f}",
            flush=True,
        )
        return wrapper

    def _configure_dynamic_skip(self, quantile, max_skips, positions):
        """Calibrate per-block thresholds on 32 LAMBADA prefixes, then enable."""
        from datasets import load_dataset
        from collections import defaultdict

        retro = self._retrofit
        tok = self._tokenizer
        device = retro.gamma.device
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").select(
            range(32)
        )
        samples = defaultdict(list)
        with torch.no_grad():
            for ex in ds:
                ids = tok.encode(
                    ex["text"].strip(), add_special_tokens=False
                )[:512]
                inp = torch.tensor([ids], device=device)
                out = retro(input_ids=inp, return_alpha=True)
                for bidx, trace in enumerate(out.skip_trace or []):
                    w = trace.get("w_recent")
                    if w is not None:
                        samples[bidx].append(w)
        thresholds = {}
        for b, vals in samples.items():
            if not vals:
                continue
            vs = sorted(vals)
            thresholds[b] = vs[int(quantile * (len(vs) - 1))]
        retro._dynamic_skip_config = dict(
            strategy="recent_weight_gt",
            thresholds=thresholds,
            eligible_blocks=set(positions) if positions else None,
            max_skips=max_skips,
        )
        retro.reset_skip_stats()
        print(
            f"[retrofit] dynamic skip armed: q={quantile}, M={max_skips}, "
            f"P={positions}, calibrated thresholds={thresholds}",
            flush=True,
        )

    def _register_skip_stats_dump(self, skip_stats_path=None):
        retro = self._retrofit
        def _dump():
            stats = retro.get_skip_stats()
            if stats is None:
                return
            payload = json.dumps(stats, sort_keys=True)
            print(f"[retrofit] skip_stats_json={payload}", flush=True)
            if skip_stats_path:
                with open(skip_stats_path, "w") as f:
                    json.dump(stats, f, indent=2, sort_keys=True)
        atexit.register(_dump)


# Register the model in MODEL_REGISTRY_V2 so lmms-eval resolves it by name.
MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="qwen3_vl_retrofit",
        simple_class_path="lmms_eval_retrofit.Qwen3_VL_Retrofit",
    ),
    overwrite=True,
)


# --------------------------------------------------------------------------
# POPE metric patch: lmms-eval pope.utils.pope_process_results does strict
# `pred == "yes"`/`"no"`. Qwen3-VL output drifts by 1 token under retrofit
# (`"Yes."` vs `"yes"`), giving correct answers a 0 score. Patch to extract
# the first yes/no word so semantically-correct predictions count.
# --------------------------------------------------------------------------
import re as _re
import lmms_eval.tasks.pope.utils as _pope_utils

def _pope_process_results_fuzzy(doc, results):
    raw = results[0].lower().strip()
    m = _re.search(r"\b(yes|no)\b", raw)
    pred = m.group(1) if m else raw
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        k: {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans}
        for k in ("pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_yes_ratio")
    }

_pope_utils.pope_process_results = _pope_process_results_fuzzy
print("[retrofit] patched lmms_eval.tasks.pope.utils.pope_process_results -> fuzzy yes/no matcher", flush=True)


# --------------------------------------------------------------------------
# Static-pruning baseline (Gromov / ShortGPT style): drop N decoder layers
# from Qwen3-VL-2B's text decoder by overriding their forward to identity.
# --------------------------------------------------------------------------


@register_model("qwen3_vl_pruned")
class Qwen3_VL_Pruned(Qwen3_VL):
    """Qwen3-VL base with specific text-decoder layers replaced by identity.

    Use ``model_args`` like:
        pretrained=/path/to/Qwen3-VL-2B,skip_layers=14|18|20|24,...
    """

    def __init__(
        self,
        lora_adapter_path: str = "",
        skip_layers: str = "",
        random_skip_layers: str = "",
        random_skip_p: float = 0.0,
        random_seed: int = 1234,
        mod_layers: str = "",
        mod_keep_ratio: float = 0.8,
        mod_router_path: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if lora_adapter_path:
            self._model = PeftModel.from_pretrained(self._model, lora_adapter_path).eval()
            print(f"[pruned] LoRA adapter loaded: {lora_adapter_path}", flush=True)
        if mod_layers:
            mod_set = set(_parse_ints(mod_layers))
            stats = _install_mod_proxy(
                self._model,
                mod_set,
                mod_keep_ratio,
                router_state_path=mod_router_path or None,
            )
            print(
                f"[pruned] MoD {'trained' if mod_router_path else 'proxy'} armed: "
                f"layers={sorted(mod_set)} keep_ratio={mod_keep_ratio}",
                flush=True,
            )
            atexit.register(lambda: print(f"[pruned] mod_proxy_stats_json={json.dumps(stats, sort_keys=True)}", flush=True))
            return
        if random_skip_layers:
            rng = random.Random(int(random_seed))
            random_set = set(_parse_ints(random_skip_layers))
            layers = self._model.model.language_model.layers
            stats = {"calls": 0, "skips": 0, "per_layer": {i: 0 for i in sorted(random_set)}}
            for i in sorted(random_set):
                if i < 0 or i >= len(layers):
                    raise ValueError(f"random_skip_layer {i} out of range [0, {len(layers)})")
                original_forward = layers[i].forward
                def _maybe_identity(self, hidden_states, *args, _idx=i, _orig=original_forward, **kwargs):
                    stats["calls"] += 1
                    if rng.random() < float(random_skip_p):
                        stats["skips"] += 1
                        stats["per_layer"][_idx] += 1
                        return hidden_states
                    return _orig(hidden_states, *args, **kwargs)
                import types
                layers[i].forward = types.MethodType(_maybe_identity, layers[i])
            print(
                f"[pruned] random layer skip armed: layers={sorted(random_set)} "
                f"p={random_skip_p} seed={random_seed}",
                flush=True,
            )
            atexit.register(lambda: print(f"[pruned] random_skip_stats_json={json.dumps(stats, sort_keys=True)}", flush=True))
            return
        if not skip_layers:
            print("[pruned] WARNING: skip_layers empty — running base model")
            return
        # Pipe '|' separator (',' is reserved by lmms-eval model_args parsing).
        skip_set = sorted(_parse_ints(skip_layers))
        layers = self._model.model.language_model.layers
        for i in skip_set:
            if i < 0 or i >= len(layers):
                raise ValueError(f"skip_layer {i} out of range [0, {len(layers)})")
            # Qwen3VLTextDecoderLayer.forward returns torch.Tensor (single).
            # Identity passes hidden_states through unchanged.
            def _identity(self, hidden_states, *args, **kwargs):
                return hidden_states
            import types
            layers[i].forward = types.MethodType(_identity, layers[i])
        print(
            f"[pruned] dropped {len(skip_set)}/{len(layers)} layers: {skip_set}",
            flush=True,
        )


MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="qwen3_vl_pruned",
        simple_class_path="lmms_eval_retrofit.Qwen3_VL_Pruned",
    ),
    overwrite=True,
)


if __name__ == "__main__":
    # Re-enter lmms-eval's CLI with our plugin registered.
    import lmms_eval.__main__ as m
    m.cli_evaluate()
