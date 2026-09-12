"""Architecture-light RSM-AttnRes retrofit for Hugging Face VLM decoders.

This wrapper is used only for the non-Qwen-family generalization extension.
It leaves the vision tower and decoder weights frozen and injects the same
cross-depth routed correction at decoder block boundaries through PyTorch
forward hooks. The hook design avoids copying model-specific attention code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fp = x.float()
    return (fp * torch.rsqrt(fp.square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


class CrossDepthRouter(nn.Module):
    def __init__(self, hidden_size: int, num_sources: int, initializer_range: float = 0.02):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_sources = int(num_sources)
        self.w_query = nn.Parameter(torch.empty(num_sources, hidden_size))
        self.key_pos_bias = nn.Parameter(torch.empty(num_sources, hidden_size))
        nn.init.normal_(self.w_query, std=initializer_range)
        nn.init.normal_(self.key_pos_bias, std=initializer_range)

    def route(
        self,
        position: int,
        completed_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.stack(completed_outputs, dim=0)
        keys = _rms_norm(values)
        keys = keys + self.key_pos_bias[:position, None, None, :].to(keys.dtype)
        query = self.w_query[position].to(keys.dtype)
        scores = torch.einsum("h,nbth->nbt", query, keys)
        scores = scores / math.sqrt(values.shape[-1])
        alpha = torch.softmax(scores.float(), dim=0).to(values.dtype)
        routed = torch.einsum("nbt,nbth->bth", alpha, values)
        return routed, alpha.permute(1, 2, 0)


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


@dataclass
class GenericRetrofitOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    entropy_penalty: torch.Tensor | None
    alpha_list: list[torch.Tensor] | None


def _extract_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = hidden
        return output
    raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")


def locate_text_decoder(base_model: nn.Module) -> tuple[nn.Module, nn.ModuleList, str]:
    """Return (decoder module, layers, stable family label)."""
    candidates = [
        ("model.language_model", ("model", "language_model")),
        ("model.text_model", ("model", "text_model")),
        ("model.language_model.model", ("model", "language_model", "model")),
    ]
    for family, path in candidates:
        obj: Any = base_model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and hasattr(obj, "layers"):
            return obj, obj.layers, family
    raise AttributeError(
        "Could not locate a standard decoder layer list; expected "
        "base.model.language_model.layers or base.model.text_model.layers"
    )


class GenericVLMAttnResRetrofit(nn.Module):
    """Inject cross-depth corrections at groups of frozen decoder layers."""

    def __init__(
        self,
        base_model: nn.Module,
        num_blocks: int,
        adapter_rank: int,
        initializer_range: float = 0.02,
        residual_strength_gate: bool = True,
        gate_scale: float = 0.5,
    ):
        super().__init__()
        # The frozen backbone is owned by the caller.  Keep non-registering
        # references here: registering both the full VLM and its nested text
        # decoder/layers as children creates a cyclic/duplicated module graph
        # for some remote-code model families, so wrapper.eval() recurses.
        text_model, text_layers, family = locate_text_decoder(base_model)
        object.__setattr__(self, "base_model", base_model)
        object.__setattr__(self, "text_model", text_model)
        object.__setattr__(self, "text_layers", text_layers)
        self.family = family
        self.num_layers = len(self.text_layers)
        if self.num_layers % num_blocks:
            raise ValueError(
                f"num_hidden_layers={self.num_layers} is not divisible by "
                f"num_blocks={num_blocks}"
            )
        self.num_blocks = int(num_blocks)
        self.layers_per_block = self.num_layers // self.num_blocks
        self.hidden_size = int(base_model.config.text_config.hidden_size)
        self.adapter_rank = int(adapter_rank)
        self.residual_strength_gate = bool(residual_strength_gate)
        self.gate_scale = float(gate_scale)
        if self.residual_strength_gate and not 0.0 < self.gate_scale < 1.0:
            raise ValueError("gate_scale must lie strictly between zero and one")

        self.router = CrossDepthRouter(
            hidden_size=self.hidden_size,
            num_sources=self.num_blocks + 1,
            initializer_range=initializer_range,
        )
        self.adapters = nn.ModuleList(
            [
                ResidualAdapter(self.hidden_size, self.adapter_rank)
                for _ in range(self.num_blocks)
            ]
        )
        self.gamma = nn.Parameter(torch.zeros(self.num_blocks))
        self.residual_strength_gates = nn.ModuleDict()
        if self.residual_strength_gate:
            # Block 0 only sees the embedding source, so its routed delta is
            # identically zero and a gate there would be an unused parameter.
            for block_idx in range(1, self.num_blocks):
                predictor = nn.Linear(self.hidden_size, 1, bias=True)
                nn.init.zeros_(predictor.weight)
                nn.init.zeros_(predictor.bias)
                self.residual_strength_gates[str(block_idx)] = predictor

        self._active_skip_blocks: set[int] = set()
        self._collect_alpha = False
        self._completed: list[torch.Tensor] = []
        self._alpha_list: list[torch.Tensor] = []
        self._entropy_terms: list[torch.Tensor] = []
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._gate_values: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._install_hooks()

    def _install_hooks(self) -> None:
        for layer_idx, layer in enumerate(self.text_layers):
            pre = layer.register_forward_pre_hook(
                self._make_pre_hook(layer_idx),
                with_kwargs=True,
            )
            post = layer.register_forward_hook(
                self._make_post_hook(layer_idx),
                with_kwargs=True,
            )
            self._handles.extend([pre, post])

    def _make_pre_hook(self, layer_idx: int):
        block_idx = layer_idx // self.layers_per_block
        first_in_block = layer_idx % self.layers_per_block == 0

        def hook(module, args, kwargs):
            del module
            if args:
                hidden = args[0]
            else:
                hidden = kwargs.get("hidden_states")
            if not torch.is_tensor(hidden):
                raise TypeError(
                    f"Decoder layer {layer_idx} did not receive tensor hidden_states"
                )

            if layer_idx == 0:
                self._completed = [hidden]
                self._alpha_list = []
                self._entropy_terms = []
                self._layer_inputs = {}
                self._gate_values = {}

            if first_in_block:
                expected = block_idx + 1
                if len(self._completed) != expected:
                    raise RuntimeError(
                        f"AttnRes hook state mismatch at block {block_idx}: "
                        f"expected {expected} completed states, got {len(self._completed)}"
                    )
                routed, alpha = self.router.route(block_idx + 1, self._completed)
                delta = routed - hidden
                update = self.gamma[block_idx].to(hidden.dtype) * self.adapters[
                    block_idx
                ](delta)
                if self.residual_strength_gate and block_idx > 0:
                    predictor = self.residual_strength_gates[str(block_idx)]
                    raw = predictor(_rms_norm(update)) / math.sqrt(
                        float(self.hidden_size)
                    )
                    factor = 1.0 + self.gate_scale * torch.tanh(
                        raw.float()
                    ).to(update.dtype)
                    update = factor * update
                    self._gate_values[block_idx] = factor.detach()
                hidden = hidden + update
                if self._collect_alpha:
                    self._alpha_list.append(alpha)
                if self.training:
                    entropy = -(
                        alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()
                    ).sum(dim=-1).mean()
                    self._entropy_terms.append(entropy)

            self._layer_inputs[layer_idx] = hidden
            if args:
                return (hidden,) + args[1:], kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = hidden
            return args, new_kwargs

        return hook

    def _make_post_hook(self, layer_idx: int):
        block_idx = layer_idx // self.layers_per_block
        last_in_block = layer_idx % self.layers_per_block == self.layers_per_block - 1

        def hook(module, args, kwargs, output):
            del module, args, kwargs
            hidden = _extract_hidden(output)
            if block_idx in self._active_skip_blocks:
                hidden = self._layer_inputs[layer_idx]
                output = _replace_hidden(output, hidden)
            if last_in_block:
                self._completed.append(hidden)
            return output

        return hook

    def freeze_base(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def move_retrofit(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "GenericVLMAttnResRetrofit":
        """Move only newly introduced modules.

        Calling ``wrapper.to(dtype)`` would also recast model-specific fp32
        buffers inside the already-loaded backbone, violating identity at
        initialization on some VLM families.
        """
        self.router.to(device=device, dtype=dtype)
        self.adapters.to(device=device, dtype=dtype)
        self.residual_strength_gates.to(device=device, dtype=dtype)
        self.gamma.data = self.gamma.data.to(device=device, dtype=dtype)
        return self

    def retrofit_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.router.parameters())
            + list(self.adapters.parameters())
            + list(self.residual_strength_gates.parameters())
            + [self.gamma]
        )

    def residual_strength_statistics(self) -> dict[str, dict[str, float | int]]:
        result = {}
        for block_idx, value in sorted(self._gate_values.items()):
            flat = value.detach().float().reshape(-1)
            result[str(block_idx)] = {
                "count": int(flat.numel()),
                "mean": float(flat.mean()),
                "std": float(flat.std(unbiased=False)),
                "min": float(flat.min()),
                "max": float(flat.max()),
            }
        return result

    def forward(
        self,
        labels: torch.Tensor | None = None,
        return_alpha: bool = False,
        skip_block_indices: Iterable[int] | None = None,
        **inputs,
    ) -> GenericRetrofitOutput:
        self._active_skip_blocks = {
            int(index) for index in (skip_block_indices or ())
        }
        self._collect_alpha = bool(return_alpha)
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        outputs = self.base_model(**model_inputs, labels=None)
        logits = outputs.logits
        loss = None
        if labels is not None:
            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)],
                dim=1,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                shifted.reshape(-1),
                ignore_index=-100,
            )
        entropy = (
            torch.stack(self._entropy_terms).mean()
            if self._entropy_terms
            else None
        )
        return GenericRetrofitOutput(
            loss=loss,
            logits=logits,
            entropy_penalty=entropy,
            alpha_list=list(self._alpha_list) if return_alpha else None,
        )

    def state_payload(self, *, extra_config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = {
            "family": self.family,
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "layers_per_block": self.layers_per_block,
            "hidden_size": self.hidden_size,
            "adapter_rank": self.adapter_rank,
            "residual_strength_gate": self.residual_strength_gate,
            "gate_scale": self.gate_scale,
        }
        if extra_config:
            config.update(extra_config)
        return {
            "router": self.router.state_dict(),
            "adapters": self.adapters.state_dict(),
            "gamma": self.gamma.detach().cpu(),
            "residual_strength_gates": self.residual_strength_gates.state_dict(),
            "config": config,
        }

    def save_state(
        self,
        path: str | Path,
        *,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        torch.save(self.state_payload(extra_config=extra_config), Path(path))


def load_generic_retrofit(
    base_model: nn.Module,
    state_path: str | Path,
) -> GenericVLMAttnResRetrofit:
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    wrapper = GenericVLMAttnResRetrofit(
        base_model,
        num_blocks=int(config["num_blocks"]),
        adapter_rank=int(config["adapter_rank"]),
        residual_strength_gate=bool(config.get("residual_strength_gate", False)),
        gate_scale=float(config.get("gate_scale", 0.5)),
    )
    parameter = next(base_model.parameters())
    wrapper.move_retrofit(device=parameter.device, dtype=parameter.dtype)
    wrapper.router.load_state_dict(
        {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            for key, value in checkpoint["router"].items()
        }
    )
    wrapper.adapters.load_state_dict(
        {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            for key, value in checkpoint["adapters"].items()
        }
    )
    wrapper.gamma.data.copy_(
        checkpoint["gamma"].to(device=parameter.device, dtype=parameter.dtype)
    )
    if wrapper.residual_strength_gate:
        wrapper.residual_strength_gates.load_state_dict(
            {
                key: value.to(device=parameter.device, dtype=parameter.dtype)
                for key, value in checkpoint["residual_strength_gates"].items()
            }
        )
    wrapper.eval()
    return wrapper
