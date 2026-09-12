from __future__ import annotations

import math
import json
import os
import types
import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


NATIVE_TOKEN_SKIP_FEATURES = (
    "w_recent",
    "source_dispersion",
    "route_novelty",
    "correction_ratio",
    "delta_path_ratio",
    "delta_cancellation_ratio",
    "recent_delta_ratio",
    "delta_trend_ratio",
    "recent_delta_cosine",
    "residual_strength_factor",
)


@dataclass
class Qwen3VLAttnResRetrofitOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    alpha_list: list[torch.Tensor] | None = None
    skip_trace: list[dict] | None = None
    block_inputs: list[torch.Tensor] | None = None
    block_outputs: list[torch.Tensor] | None = None
    surrogate_outputs: list[torch.Tensor | None] | None = None
    entropy_penalty: torch.Tensor | None = None
    correction_ratios: list[torch.Tensor] | None = None
    native_features_by_block: (
        list[dict[str, torch.Tensor] | None] | None
    ) = None
    action_risks_by_block: (
        list[dict[str, torch.Tensor] | None] | None
    ) = None
    joint_action_risks_by_block: (
        list[dict[str, torch.Tensor] | None] | None
    ) = None


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fp = x.float()
    inv = torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (fp * inv).to(x.dtype)


def routing_confidence_score(
    alpha: torch.Tensor,
    margin_weight: float = 0.5,
    entropy_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Score routing concentration without adding a selector head.

    The first term is recent-source evidence against the uniform-routing null,
    the second is its log-margin over the strongest competing source, and the
    third rewards low routing entropy.  The score is exactly zero at uniform
    routing and requires no selector parameters.  Higher is safer to skip.
    """
    if alpha.shape[-1] < 2:
        return torch.full_like(alpha[..., 0].float(), -torch.inf)
    probs = alpha.float().clamp_min(eps)
    recent = probs[..., -1].clamp(max=1.0 - eps)
    competitor = probs[..., :-1].amax(dim=-1).clamp_min(eps)
    num_sources = alpha.shape[-1]
    normalized_entropy = -(
        probs * probs.log()
    ).sum(dim=-1) / math.log(num_sources)
    recent_evidence = (
        math.log(num_sources - 1) + recent.log() - torch.log1p(-recent)
    )
    recent_margin = recent.log() - competitor.log()
    concentration = 1.0 - normalized_entropy
    return (
        recent_evidence
        + float(margin_weight) * recent_margin
        + float(entropy_weight) * concentration
    )


def native_alpha_feature_vector(alpha: torch.Tensor) -> torch.Tensor:
    """Deployment feature vector used by the cross-fit linear risk selector."""
    probability = alpha.reshape(-1, alpha.shape[-1])[0].float().clamp_min(1e-8)
    log_probability = probability.log()
    source_count = probability.numel()
    age_scale = max(source_count - 1, 1)
    age = torch.arange(
        source_count - 1,
        -1,
        -1,
        device=probability.device,
        dtype=probability.dtype,
    )
    normalized_age = age / float(age_scale)
    expected_age = (probability * normalized_age).sum()
    age_variance = (
        probability * (normalized_age - expected_age).pow(2)
    ).sum()
    recent = probability[-1]
    competitor = probability[:-1].amax()
    normalized_entropy = -(
        probability * log_probability
    ).sum() / math.log(source_count)
    scalar_features = torch.stack(
        [
            recent - competitor,
            recent.log() - competitor.log(),
            1.0 - normalized_entropy,
            expected_age,
            age_variance,
            probability.max(),
            probability[:-1].sum(),
        ]
    )
    return torch.cat([probability, log_probability, scalar_features])


def native_action_feature_tensor(
    alpha: torch.Tensor,
    native_features: dict[str, torch.Tensor],
    cache_position: torch.Tensor,
) -> torch.Tensor:
    """Fixed-width causal AttnRes features for a per-action risk surrogate."""
    probability = alpha.float().clamp_min(1e-8)
    recent = probability[..., -1]
    competitor = probability[..., :-1].amax(dim=-1)
    log_margin = recent.log() - competitor.log()
    source_count = probability.shape[-1]
    normalized_entropy = -(
        probability * probability.log()
    ).sum(dim=-1) / math.log(source_count)
    age_scale = max(source_count - 1, 1)
    age = torch.arange(
        source_count - 1,
        -1,
        -1,
        device=probability.device,
        dtype=probability.dtype,
    ) / float(age_scale)
    expected_age = (probability * age).sum(dim=-1)
    age_variance = (
        probability * (age - expected_age[..., None]).pow(2)
    ).sum(dim=-1)
    position = cache_position.to(
        device=probability.device,
        dtype=torch.float32,
    ).reshape(1, -1)
    if position.shape[-1] != probability.shape[-2]:
        raise ValueError(
            "cache position length does not match the AttnRes token dimension"
        )
    position = position.expand(probability.shape[0], -1)
    log_position = torch.log1p(position) / math.log1p(32768.0)
    return torch.stack(
        [
            recent,
            competitor,
            log_margin,
            1.0 - normalized_entropy,
            expected_age,
            age_variance,
            native_features["source_dispersion"].float(),
            native_features["route_novelty"].float(),
            native_features["correction_ratio"].float(),
            log_position,
        ],
        dim=-1,
    )


class AttnResActionRiskSurrogate(nn.Module):
    """Small block-local predictor for decoder-layer actions or groups."""

    feature_dim = 10

    def __init__(
        self,
        actions_by_block: dict[int, Iterable[int]],
        hidden_dim: int = 32,
        flip_weight: float = 0.25,
        hidden_size: int | None = None,
        semantic_dim: int = 0,
    ):
        super().__init__()
        self.actions_by_block = {
            int(block): tuple(sorted(set(int(offset) for offset in offsets)))
            for block, offsets in actions_by_block.items()
        }
        if not self.actions_by_block or any(
            not offsets for offsets in self.actions_by_block.values()
        ):
            raise ValueError("risk surrogate requires at least one action per block")
        self.hidden_dim = int(hidden_dim)
        self.flip_weight = float(flip_weight)
        self.hidden_size = int(hidden_size) if hidden_size is not None else None
        self.semantic_dim = int(semantic_dim)
        if (
            self.hidden_dim <= 0
            or self.semantic_dim < 0
            or not math.isfinite(self.flip_weight)
        ):
            raise ValueError("invalid risk surrogate width or flip weight")
        if self.semantic_dim > 0 and (
            self.hidden_size is None or self.hidden_size <= 0
        ):
            raise ValueError("semantic risk features require a positive hidden size")
        self.semantic_projection = (
            nn.Linear(self.hidden_size, self.semantic_dim, bias=False)
            if self.semantic_dim > 0
            else None
        )
        if self.semantic_projection is not None:
            nn.init.normal_(
                self.semantic_projection.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.hidden_size)),
            )
        self.heads = nn.ModuleDict()
        for block, offsets in self.actions_by_block.items():
            head = nn.Sequential(
                nn.Linear(
                    self.feature_dim + self.semantic_dim,
                    self.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 2 * len(offsets)),
            )
            nn.init.normal_(head[0].weight, mean=0.0, std=0.02)
            nn.init.zeros_(head[0].bias)
            nn.init.normal_(head[2].weight, mean=0.0, std=0.002)
            with torch.no_grad():
                head[2].bias.view(len(offsets), 2)[:, 0].fill_(-1.6)
                head[2].bias.view(len(offsets), 2)[:, 1].fill_(-1.7)
            self.heads[str(block)] = head
        self._compiled_inference_predictors: dict[int, object] = {}

    def configure_compiled_inference(self, enabled: bool):
        """Compile the score-only decode subgraph once per block shape."""
        if not enabled or self._compiled_inference_predictors:
            return
        for block_idx in self.actions_by_block:
            head = self.heads[str(block_idx)]
            action_count = len(self.actions_by_block[block_idx])
            semantic_projection = self.semantic_projection
            flip_weight = self.flip_weight

            def make_predictor(
                bound_head,
                bound_action_count,
                bound_semantic_projection,
                bound_flip_weight,
            ):
                def predictor(
                    alpha,
                    source_dispersion,
                    route_novelty,
                    correction_ratio,
                    cache_position,
                    block_input,
                ):
                    features = native_action_feature_tensor(
                        alpha,
                        {
                            "source_dispersion": source_dispersion,
                            "route_novelty": route_novelty,
                            "correction_ratio": correction_ratio,
                        },
                        cache_position,
                    ).detach()
                    if bound_semantic_projection is not None:
                        semantic_input = block_input.detach().float()
                        semantic_input = semantic_input / semantic_input.pow(
                            2
                        ).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-6)
                        semantic_features = F.silu(
                            bound_semantic_projection(semantic_input)
                        )
                        features = torch.cat(
                            [features, semantic_features], dim=-1
                        )
                    raw = bound_head(features).view(
                        *features.shape[:-1],
                        bound_action_count,
                        2,
                    )
                    predicted_log_kl = F.softplus(raw[..., 0])
                    return torch.expm1(predicted_log_kl) + (
                        bound_flip_weight * torch.sigmoid(raw[..., 1])
                    )

                return predictor

            predictor = make_predictor(
                head,
                action_count,
                semantic_projection,
                flip_weight,
            )
            self._compiled_inference_predictors[block_idx] = torch.compile(
                predictor,
                fullgraph=True,
                mode="reduce-overhead",
            )

    def predict_block_risk_score(
        self,
        block_idx: int,
        alpha: torch.Tensor,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self._compiled_inference_predictors.get(int(block_idx))
        if predictor is None:
            prediction = self.predict_block(
                block_idx,
                alpha,
                native_features,
                cache_position,
                block_input=block_input,
                detach_features=True,
            )
            return prediction["risk_score"]
        return predictor(
            alpha,
            native_features["source_dispersion"],
            native_features["route_novelty"],
            native_features["correction_ratio"],
            cache_position,
            block_input,
        )

    def predict_block(
        self,
        block_idx: int,
        alpha: torch.Tensor,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
        block_input: torch.Tensor | None = None,
        detach_features: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        block_idx = int(block_idx)
        if block_idx not in self.actions_by_block:
            return None
        features = native_action_feature_tensor(
            alpha,
            native_features,
            cache_position,
        )
        if detach_features:
            features = features.detach()
        if self.semantic_projection is not None:
            if block_input is None:
                raise ValueError("semantic risk surrogate requires block_input")
            semantic_input = block_input.detach().float()
            semantic_input = semantic_input / semantic_input.pow(2).mean(
                dim=-1, keepdim=True
            ).sqrt().clamp_min(1.0e-6)
            semantic_features = F.silu(
                self.semantic_projection(semantic_input)
            )
            features = torch.cat([features, semantic_features], dim=-1)
        raw = self.heads[str(block_idx)](features).view(
            *features.shape[:-1],
            len(self.actions_by_block[block_idx]),
            2,
        )
        predicted_log_kl = F.softplus(raw[..., 0])
        flip_logit = raw[..., 1]
        risk_score = torch.expm1(predicted_log_kl) + self.flip_weight * torch.sigmoid(
            flip_logit
        )
        return {
            "predicted_log1p_kl": predicted_log_kl,
            "flip_logit": flip_logit,
            "risk_score": risk_score,
        }

    def predict_all_from_gate(
        self,
        alpha: torch.Tensor,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
        detach_features: bool = True,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Predict every target action from one causal AttnRes gate state."""
        features = native_action_feature_tensor(
            alpha,
            native_features,
            cache_position,
        )
        if detach_features:
            features = features.detach()
        if self.semantic_projection is not None:
            semantic_input = block_input.detach().float()
            semantic_input = semantic_input / semantic_input.pow(
                2
            ).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-6)
            semantic_features = F.silu(
                self.semantic_projection(semantic_input)
            )
            features = torch.cat([features, semantic_features], dim=-1)

        predictions: dict[int, dict[str, torch.Tensor]] = {}
        for block_idx, actions in self.actions_by_block.items():
            raw = self.heads[str(block_idx)](features).view(
                *features.shape[:-1],
                len(actions),
                2,
            )
            predicted_log_kl = F.softplus(raw[..., 0])
            flip_logit = raw[..., 1]
            risk_score = torch.expm1(predicted_log_kl) + (
                self.flip_weight * torch.sigmoid(flip_logit)
            )
            predictions[block_idx] = {
                "predicted_log1p_kl": predicted_log_kl,
                "flip_logit": flip_logit,
                "risk_score": risk_score,
            }
        return predictions


class BlockAttnResRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        temperature: float = 1.0,
        use_positional_bias: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.base_temperature = temperature
        self.w_query = nn.Parameter(torch.empty(num_sources, hidden_size))
        if use_positional_bias:
            self.key_pos_bias = nn.Parameter(torch.empty(num_sources, hidden_size))
        else:
            self.register_parameter("key_pos_bias", None)
        nn.init.normal_(self.w_query, std=initializer_range)
        if self.key_pos_bias is not None:
            nn.init.normal_(self.key_pos_bias, std=initializer_range)

    def _route_impl(
        self,
        position: int,
        completed_outputs: list[torch.Tensor],
        compute_source_dispersion: bool,
    ):
        values = torch.stack(completed_outputs, dim=0)  # [N, B, T, H]
        keys = _rms_norm(values)
        if self.key_pos_bias is not None:
            bias = self.key_pos_bias[:position].to(keys.dtype)
            keys = keys + bias[:, None, None, :]
        query = self.w_query[position].to(keys.dtype)
        scale = math.sqrt(values.shape[-1]) * self.base_temperature
        scores = torch.einsum("h,nbth->nbt", query, keys) / scale
        alpha = torch.softmax(scores.float(), dim=0).to(values.dtype)
        routed = torch.einsum("nbt,nbth->bth", alpha, values)
        native_delta_features = None
        if compute_source_dispersion:
            values_fp = values.float()
            routed_fp = routed.float()
            alpha_fp = alpha.float()
            dispersion = (
                alpha_fp[..., None]
                * (values_fp - routed_fp[None, ...]).pow(2)
            ).sum(dim=0).mean(dim=-1).sqrt()
            routed_scale = routed_fp.pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)
            source_dispersion = dispersion / routed_scale
            source_deltas = values_fp[1:] - values_fp[:-1]
            latest_scale = (
                values_fp[-1].pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)
            )
            route_delta = routed_fp - values_fp[-1]
            route_delta_rms = route_delta.pow(2).mean(dim=-1).sqrt()
            if source_deltas.shape[0] == 0:
                zero = torch.zeros_like(source_dispersion)
                path_rms = zero
                recent_delta_rms = zero
                delta_trend_ratio = torch.ones_like(source_dispersion)
                recent_delta_cosine = zero
            else:
                source_delta_rms = source_deltas.pow(2).mean(dim=-1).sqrt()
                prefix_mass = alpha_fp.cumsum(dim=0)[:-1]
                path_rms = (prefix_mass * source_delta_rms).sum(dim=0)
                recent_delta = source_deltas[-1]
                recent_delta_rms = source_delta_rms[-1]
                if source_delta_rms.shape[0] > 1:
                    older_delta_rms = source_delta_rms[:-1].mean(dim=0)
                else:
                    older_delta_rms = recent_delta_rms
                delta_trend_ratio = (
                    recent_delta_rms / older_delta_rms.clamp_min(1e-8)
                )
                recent_delta_cosine = (
                    (route_delta * recent_delta).sum(dim=-1)
                    / (
                        route_delta.norm(dim=-1).clamp_min(1e-8)
                        * recent_delta.norm(dim=-1).clamp_min(1e-8)
                    )
                )
            native_delta_features = {
                "source_dispersion": source_dispersion,
                "delta_path_ratio": path_rms / latest_scale,
                "delta_cancellation_ratio": (
                    route_delta_rms / path_rms.clamp_min(1e-8)
                ),
                "recent_delta_ratio": recent_delta_rms / latest_scale,
                "delta_trend_ratio": delta_trend_ratio,
                "recent_delta_cosine": recent_delta_cosine,
            }
        return routed, alpha.permute(1, 2, 0), native_delta_features, values

    def route(self, position: int, completed_outputs: list[torch.Tensor]):
        routed, alpha, _, _ = self._route_impl(
            position, completed_outputs, compute_source_dispersion=False
        )
        return routed, alpha

    def route_with_native_stats(
        self, position: int, completed_outputs: list[torch.Tensor]
    ):
        routed, alpha, native_features, _ = self._route_impl(
            position, completed_outputs, compute_source_dispersion=True
        )
        return routed, alpha, native_features

    def route_with_source_values(
        self, position: int, completed_outputs: list[torch.Tensor]
    ):
        routed, alpha, _, values = self._route_impl(
            position, completed_outputs, compute_source_dispersion=False
        )
        return routed, alpha, values


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        adapter_rank: int = 128,
        zero_init_up: bool = False,
    ):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_rank, bias=False)
        self.up = nn.Linear(adapter_rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        # Small random (not zero) — combined with γ=0 we still get identity at
        # init (x_n = prev_block + 0 * adapter = prev_block), but gradients
        # can flow through adapter(delta) into γ. Pure zero-init on both γ AND
        # up.weight causes gradient deadlock — γ stuck at 0 forever.
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.normal_(self.up.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


class Qwen3VLAttnResRetrofit(nn.Module):
    """Retrofit Qwen3-VL so AttnRes enters the normal forward path safely.

    For selected blocks (all blocks by default), ``bridge_mode`` chooses how
    routed AttnRes states enter the frozen decoder:

      adapter:      x_n = h + gamma * A(r - h)
      pure:         x_n = h + gamma * (r - h)
      exact_compat: x_n = h + gamma * (r - h)
                          + 4 * compat_peak * gamma * (1-gamma) * A(r - h)

    ``exact_compat`` is exactly the base model at gamma=0 and exactly the
    direct AttnRes architecture at gamma=1.  Its adapter is only a trainable
    scaffold along the homotopy between the two endpoints.

    Full path:
      h_n = Block_n(x_n)

    Skip path:
      h_n = x_n

    `gamma_n` is zero-initialized so the wrapper starts as the original model.
    """

    def __init__(
        self,
        base_model,
        num_blocks: int = 14,
        skippable_blocks: Iterable[int] | None = None,
        adapter_rank: int = 128,
        initializer_range: float = 0.02,
        no_adapter: bool = False,
        bridge_mode: str | None = None,
        compat_peak: float = 1.0,
        identity_anchored_source_calibration: bool = False,
        recovery_profiles: Iterable[Iterable[int]] | None = None,
        recovery_rank: int = 256,
        recovery_mode: str = "profile",
        risk_surrogate_actions: dict[int, Iterable[int]] | None = None,
        risk_surrogate_groups: (
            dict[int, dict[int, Iterable[int]]] | None
        ) = None,
        risk_surrogate_hidden_dim: int = 32,
        risk_surrogate_flip_weight: float = 0.25,
        risk_surrogate_semantic_dim: int = 0,
        joint_risk_gate_block: int | None = None,
        joint_risk_gate_by_block: dict[int, int] | None = None,
        joint_risk_surrogate_hidden_dim: int | None = None,
        joint_risk_surrogate_semantic_dim: int | None = None,
    ):
        # no_adapter=True → pure Route A: x_n = (1-γ)h_{n-1} + γ·r_n
        #   (matches Part 1 AttnRes structurally when γ=1)
        # no_adapter=False (default) → x_n = h_{n-1} + γ·Adapter(r_n - h_{n-1})
        super().__init__()
        if identity_anchored_source_calibration:
            raise ValueError(
                "the frozen RSM checkpoint does not use source calibration"
            )
        if bridge_mode is None:
            bridge_mode = "pure" if no_adapter else "adapter"
        bridge_mode = str(bridge_mode).lower()
        if bridge_mode not in {"adapter", "pure", "exact_compat"}:
            raise ValueError(
                "bridge_mode must be one of: adapter, pure, exact_compat; "
                f"got {bridge_mode!r}"
            )
        if no_adapter and bridge_mode != "pure":
            raise ValueError("no_adapter=True is only compatible with bridge_mode='pure'")
        if compat_peak < 0:
            raise ValueError("compat_peak must be non-negative")
        self.base_model = base_model
        cfg = base_model.config.text_config
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.num_layers = cfg.num_hidden_layers
        if self.num_layers % num_blocks != 0:
            raise ValueError(
                f"num_hidden_layers={self.num_layers} must be divisible by num_blocks={num_blocks}"
            )
        self.num_blocks = num_blocks
        self.layers_per_block = self.num_layers // num_blocks
        if skippable_blocks is None:
            # Apply AttnRes injection to ALL blocks so the retrofit produces a
            # genuine AttnRes model (not a hybrid where only late blocks went AttnRes).
            # γ=0 zero-init already guarantees identity-at-init, so restricting
            # to late blocks is unnecessary conservatism and weakens the Part-2 claim.
            skippable_blocks = list(range(num_blocks))
        self.skippable_blocks = tuple(sorted(set(int(x) for x in skippable_blocks)))
        self.skippable_block_set = set(self.skippable_blocks)

        self.router = BlockAttnResRouter(
            hidden_size=self.hidden_size,
            num_sources=num_blocks + 1,
            use_positional_bias=True,
            initializer_range=initializer_range,
        )
        self.bridge_mode = bridge_mode
        self.compat_peak = float(compat_peak)
        self.no_adapter = bridge_mode == "pure"
        if self.no_adapter:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        else:
            self.adapters = nn.ModuleList(
                [
                    ResidualAdapter(
                        self.hidden_size,
                        adapter_rank=adapter_rank,
                        zero_init_up=(bridge_mode == "exact_compat"),
                    )
                    for _ in range(num_blocks)
                ]
            )
        # Identity-at-init comes from gamma == 0. The adapter up-projection is
        # small-random rather than zero so gradients flow once the curriculum
        # moves gamma above zero.
        self.gamma = nn.Parameter(torch.zeros(num_blocks))

        normalized_recovery_profiles = tuple(
            tuple(sorted(set(int(block) for block in profile)))
            for profile in (recovery_profiles or ())
        )
        if any(not profile for profile in normalized_recovery_profiles):
            raise ValueError("recovery profiles cannot be empty")
        if len(set(normalized_recovery_profiles)) != len(normalized_recovery_profiles):
            raise ValueError("recovery profiles must be unique")
        if any(
            not set(profile).issubset(self.skippable_block_set)
            for profile in normalized_recovery_profiles
        ):
            raise ValueError("recovery profile contains a non-skippable block")
        if recovery_rank <= 0:
            raise ValueError("recovery_rank must be positive")
        recovery_mode = str(recovery_mode).lower()
        if recovery_mode not in {"profile", "block", "surrogate"}:
            raise ValueError(
                "recovery_mode must be 'profile', 'block', or 'surrogate'"
            )
        self.recovery_profiles = normalized_recovery_profiles
        self.recovery_rank = int(recovery_rank)
        self.recovery_mode = recovery_mode
        self.recovery_blocks = tuple(
            sorted(set().union(*(set(profile) for profile in self.recovery_profiles)))
        ) if self.recovery_profiles else ()
        recovery_keys = (
            [self._recovery_profile_key(profile) for profile in self.recovery_profiles]
            if self.recovery_mode == "profile"
            else [self._recovery_block_key(block) for block in self.recovery_blocks]
        )
        self.recovery_adapters = nn.ModuleDict(
            {
                key: ResidualAdapter(
                    self.hidden_size,
                    adapter_rank=self.recovery_rank,
                    zero_init_up=True,
                )
                for key in recovery_keys
            }
        )
        self.recovery_router = (
            BlockAttnResRouter(
                hidden_size=self.hidden_size,
                num_sources=num_blocks + 1,
                use_positional_bias=True,
                initializer_range=initializer_range,
            )
            if self.recovery_mode == "surrogate"
            else None
        )
        normalized_risk_groups: dict[int, dict[int, tuple[int, ...]]] = {}
        for raw_block, raw_groups in (risk_surrogate_groups or {}).items():
            block = int(raw_block)
            groups: dict[int, tuple[int, ...]] = {}
            for raw_action, raw_offsets in raw_groups.items():
                action = int(raw_action)
                offsets = tuple(sorted(set(int(offset) for offset in raw_offsets)))
                if not offsets:
                    raise ValueError("risk surrogate groups cannot be empty")
                groups[action] = offsets
            if not groups:
                raise ValueError("risk surrogate block requires a group")
            normalized_risk_groups[block] = groups
        if normalized_risk_groups:
            normalized_risk_actions = {
                block: tuple(sorted(groups))
                for block, groups in normalized_risk_groups.items()
            }
            supplied_actions = {
                int(block): tuple(sorted(set(int(action) for action in actions)))
                for block, actions in (risk_surrogate_actions or {}).items()
            }
            if supplied_actions and supplied_actions != normalized_risk_actions:
                raise ValueError(
                    "risk surrogate actions do not match the supplied groups"
                )
        else:
            normalized_risk_actions = {
                int(block): tuple(sorted(set(int(offset) for offset in offsets)))
                for block, offsets in (risk_surrogate_actions or {}).items()
            }
            normalized_risk_groups = {
                block: {offset: (offset,) for offset in offsets}
                for block, offsets in normalized_risk_actions.items()
            }
        if any(
            block not in self.skippable_block_set
            for block in normalized_risk_actions
        ):
            raise ValueError("risk surrogate targets a non-skippable block")
        if any(
            offset < 0 or offset >= self.layers_per_block
            for groups in normalized_risk_groups.values()
            for offsets in groups.values()
            for offset in offsets
        ):
            raise ValueError("risk surrogate decoder-layer offset is out of range")
        self.risk_surrogate_actions = normalized_risk_actions
        self.risk_surrogate_groups = normalized_risk_groups
        self.risk_surrogate_hidden_dim = int(risk_surrogate_hidden_dim)
        self.risk_surrogate_flip_weight = float(risk_surrogate_flip_weight)
        self.risk_surrogate_semantic_dim = int(risk_surrogate_semantic_dim)
        self.risk_surrogate = (
            AttnResActionRiskSurrogate(
                normalized_risk_actions,
                hidden_dim=self.risk_surrogate_hidden_dim,
                flip_weight=self.risk_surrogate_flip_weight,
                hidden_size=self.hidden_size,
                semantic_dim=self.risk_surrogate_semantic_dim,
            )
            if normalized_risk_actions
            else None
        )
        normalized_gate_by_block = {
            int(block): int(gate)
            for block, gate in (joint_risk_gate_by_block or {}).items()
        }
        if joint_risk_gate_block is not None:
            if normalized_gate_by_block:
                raise ValueError(
                    "specify either one joint gate or a segmented gate map"
                )
            normalized_gate_by_block = {
                block: int(joint_risk_gate_block)
                for block in normalized_risk_actions
            }
        if normalized_gate_by_block:
            if set(normalized_gate_by_block) != set(normalized_risk_actions):
                raise ValueError(
                    "joint risk gate map must cover every risk target exactly"
                )
            for target_block, gate_block in normalized_gate_by_block.items():
                if gate_block < 0 or gate_block >= self.num_blocks:
                    raise ValueError("joint risk gate block is out of range")
                if gate_block > target_block:
                    raise ValueError(
                        "joint risk gate cannot follow its target block"
                    )
        self.joint_risk_gate_by_block = normalized_gate_by_block
        self.joint_risk_gate_blocks = tuple(
            sorted(set(normalized_gate_by_block.values()))
        )
        self.joint_risk_surrogate_hidden_dim = int(
            self.risk_surrogate_hidden_dim
            if joint_risk_surrogate_hidden_dim is None
            else joint_risk_surrogate_hidden_dim
        )
        self.joint_risk_surrogate_semantic_dim = int(
            self.risk_surrogate_semantic_dim
            if joint_risk_surrogate_semantic_dim is None
            else joint_risk_surrogate_semantic_dim
        )
        if self.joint_risk_surrogate_hidden_dim <= 0:
            raise ValueError("joint risk surrogate width must be positive")
        if self.joint_risk_surrogate_semantic_dim < 0:
            raise ValueError(
                "joint risk surrogate semantic width must be non-negative"
            )
        actions_by_gate: dict[int, dict[int, tuple[int, ...]]] = {}
        for target_block, gate_block in normalized_gate_by_block.items():
            actions_by_gate.setdefault(gate_block, {})[target_block] = (
                normalized_risk_actions[target_block]
            )
        self.joint_risk_surrogates = nn.ModuleDict(
            {
                str(gate_block): AttnResActionRiskSurrogate(
                    target_actions,
                    hidden_dim=self.joint_risk_surrogate_hidden_dim,
                    flip_weight=self.risk_surrogate_flip_weight,
                    hidden_size=self.hidden_size,
                    semantic_dim=self.joint_risk_surrogate_semantic_dim,
                )
                for gate_block, target_actions in actions_by_gate.items()
            }
        )

        self._fwd_alpha_list: list[torch.Tensor] | None = None
        self._fwd_skip_trace: list[dict] | None = None
        self._fwd_block_inputs: list[torch.Tensor] | None = None
        self._fwd_block_outputs: list[torch.Tensor] | None = None
        self._fwd_surrogate_outputs: list[torch.Tensor | None] | None = None
        self._fwd_entropy: torch.Tensor | None = None
        self._fwd_correction_ratios: list[torch.Tensor] | None = None
        self._fwd_native_features_by_block: (
            list[dict[str, torch.Tensor] | None] | None
        ) = None
        self._fwd_native_vectors_by_block: (
            list[dict[str, torch.Tensor] | None] | None
        ) = None
        self._fwd_layer_vectors_by_block: (
            list[list[dict[str, torch.Tensor]] | None] | None
        ) = None
        self._fwd_action_risks_by_block: (
            list[dict[str, torch.Tensor] | None] | None
        ) = None
        self._fwd_joint_action_risks_by_block: (
            list[dict[str, torch.Tensor] | None] | None
        ) = None
        self._active_skip_blocks: set[int] = set()
        self._dynamic_skip_config: dict | None = None
        self._native_token_skip_config: dict | None = None
        self._action_risk_skip_config: dict | None = None
        self._action_risk_single_action_specs: dict[
            int, tuple[int, int, float, tuple[int, ...]]
        ] = {}
        self._triton_joint_risk_selectors: dict[int, object] = {}
        self._device_conditional_runtime: object | None = None
        self._device_conditional_runtime_enabled = False
        self._native_device_conditional_runtime: object | None = None
        self._native_device_conditional_runtime_enabled = False
        self._native_device_conditional_runtime_fingerprint: str | None = None
        self._native_device_conditional_runtimes: dict[str, object] = {}
        self.early_joint_risk_surrogate: AttnResActionRiskSurrogate | None = None
        self._triton_early_joint_risk_selector: object | None = None
        self._risk_profile_skip_config: dict | None = None
        self._risk_profile_decode_steps = 0
        self._token_skip_masks: dict[int, torch.Tensor] | None = None
        self._token_layer_skip_masks: (
            dict[int, dict[int, torch.Tensor]] | None
        ) = None
        self._token_recovery_masks: dict[tuple[int, ...], torch.Tensor] | None = None
        self._random_skip_config: dict | None = None
        self._sticky_skip_config: dict | None = None
        self._sticky_active_blocks: set[int] = set()
        self._sticky_profile_ready = False
        self._sticky_observation_sums: dict[int, float] = {}
        self._sticky_observation_count = 0
        self._sticky_observations_remaining = 0
        self._skip_stats = None
        self._routing_dump_path = os.environ.get("ROUTING_DUMP_PATH")
        self._routing_dump_limit = int(os.environ.get("ROUTING_DUMP_LIMIT", "0") or "0")
        self._routing_dump_count = 0
        self._collect_block_states = False
        self._return_alpha_flag = False
        self._return_correction_ratios_flag = False
        self._return_native_features_flag = False
        self._return_action_risks_flag = False
        self._return_joint_action_risks_flag = False
        self._eval_gamma_cache_version = -1
        self._eval_gamma_is_zero = False
        self._last_forward_used_exact_base = False
        self._linear_selector_tensor_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

        self._install_retrofit_forward()

    @staticmethod
    def _recovery_profile_key(profile: Iterable[int]) -> str:
        return "blocks_" + "_".join(str(int(block)) for block in profile)

    @staticmethod
    def _recovery_block_key(block: int) -> str:
        return f"block_{int(block)}"

    def reset_skip_stats(self):
        self._skip_stats = {
            "forwards": 0,
            "decode_forwards": 0,
            "total_positions": 0,
            "skip_events": 0,
            "partial_mlp_skip_events": 0,
            "partial_mlp_layer_skip_events": 0,
            "partial_layer_skip_events": 0,
            "partial_decoder_layer_skip_events": 0,
            "dynamic_skip_requests": 0,
            "native_token_skip_requests": 0,
            "action_risk_skip_requests": 0,
            "action_risk_layer_requests": 0,
            "joint_gate_score_transfers": 0,
            "early_gate_candidate_evaluations": 0,
            "early_gate_bypass_forwards": 0,
            "early_gate_fallback_forwards": 0,
            "early_gate_predicted_actions": 0,
            "ratio_gate_candidate_evaluations": 0,
            "ratio_gate_bypass_forwards": 0,
            "ratio_gate_fallback_forwards": 0,
            "ratio_gate_predicted_actions": 0,
            "risk_profile_skip_requests": 0,
            "risk_profile_selections": 0,
            "risk_profile_warmup_forwards": 0,
            "static_skip_requests": 0,
            "random_skip_requests": 0,
            "sticky_skip_requests": 0,
            "sticky_profile_selections": 0,
            "sticky_kv_projections_avoided": 0,
            "per_block_skips": [0 for _ in range(self.num_blocks)],
            "per_block_partial_mlp_skips": [0 for _ in range(self.num_blocks)],
            "per_block_partial_mlp_layer_skips": [
                0 for _ in range(self.num_blocks)
            ],
            "per_block_partial_layer_skips": [0 for _ in range(self.num_blocks)],
            "per_block_partial_decoder_layer_skips": [
                0 for _ in range(self.num_blocks)
            ],
            "per_block_dynamic_requests": [0 for _ in range(self.num_blocks)],
            "per_block_native_token_requests": [0 for _ in range(self.num_blocks)],
            "per_block_action_risk_requests": [0 for _ in range(self.num_blocks)],
            "per_block_action_risk_layer_requests": [
                0 for _ in range(self.num_blocks)
            ],
            "per_block_risk_profile_requests": [0 for _ in range(self.num_blocks)],
            "per_block_static_requests": [0 for _ in range(self.num_blocks)],
            "per_block_random_requests": [0 for _ in range(self.num_blocks)],
            "per_block_sticky_requests": [0 for _ in range(self.num_blocks)],
            "per_block_sticky_selections": [0 for _ in range(self.num_blocks)],
        }
        if (
            self._device_conditional_runtime_enabled
            and self._device_conditional_runtime is not None
        ):
            self._device_conditional_runtime.reset_history()
        if (
            self._native_device_conditional_runtime_enabled
            and self._native_device_conditional_runtime is not None
        ):
            self._native_device_conditional_runtime.reset_history()

    def configure_native_token_skip(self, config: dict | None):
        """Configure the training-free, per-decode-token AttnRes selector.

        ``feature_upper_bounds`` and ``feature_lower_bounds`` map each eligible
        block to bounds for source dispersion, route novelty, and correction
        ratio. A block action is taken only when every supplied native
        consistency bound holds for the current token. ``action_by_block``
        selects either a full ``block`` bypass or an ``mlp`` bypass that keeps
        self-attention and its exact KV update. No decision is held across
        tokens.
        """
        if config is None:
            self._native_token_skip_config = None
            self._native_device_conditional_runtime_enabled = False
            return
        cfg = dict(config)
        required = set(NATIVE_TOKEN_SKIP_FEATURES)
        raw_upper = cfg.get("feature_upper_bounds") or {}
        raw_lower = cfg.get("feature_lower_bounds") or {}
        configured_blocks = {int(block) for block in raw_upper}.union(
            int(block) for block in raw_lower
        )
        if not configured_blocks:
            raise ValueError(
                "native token selector requires feature_upper_bounds or "
                "feature_lower_bounds"
            )
        normalized_upper: dict[int, dict[str, float]] = {}
        normalized_lower: dict[int, dict[str, float]] = {}
        active_feature_names_by_block: dict[int, frozenset[str]] = {}
        for block_idx in configured_blocks:
            upper_values = {
                str(key): float(value)
                for key, value in raw_upper.get(block_idx, raw_upper.get(str(block_idx), {})).items()
            }
            lower_values = {
                str(key): float(value)
                for key, value in raw_lower.get(block_idx, raw_lower.get(str(block_idx), {})).items()
            }
            unknown = set(upper_values).union(lower_values).difference(required)
            if unknown:
                raise ValueError(
                    f"unknown native token selector features: {sorted(unknown)}"
                )
            if not all(
                math.isfinite(value)
                for value in (*upper_values.values(), *lower_values.values())
            ):
                raise ValueError("native token selector bounds must be finite")
            upper = {name: upper_values.get(name, math.inf) for name in required}
            lower = {name: lower_values.get(name, -math.inf) for name in required}
            if any(lower[name] >= upper[name] for name in required):
                raise ValueError(
                    f"native token selector lower bound must be below upper "
                    f"bound for block {block_idx}"
                )
            normalized_upper[block_idx] = upper
            normalized_lower[block_idx] = lower
            active_feature_names_by_block[block_idx] = frozenset(
                set(upper_values).union(lower_values)
            )
        eligible = cfg.get("eligible_blocks")
        cfg["eligible_blocks"] = (
            set(int(value) for value in eligible)
            if eligible is not None
            else set(configured_blocks)
        )
        if not cfg["eligible_blocks"].issubset(self.skippable_block_set):
            raise ValueError("native token selector includes a non-skippable block")
        raw_actions = cfg.get("action_by_block") or {}
        action_by_block = {
            int(block): str(action).lower()
            for block, action in raw_actions.items()
        }
        unknown_action_blocks = set(action_by_block).difference(configured_blocks)
        if unknown_action_blocks:
            raise ValueError(
                "native token actions target blocks without feature bounds: "
                f"{sorted(unknown_action_blocks)}"
            )
        invalid_actions = {
            action for action in action_by_block.values()
            if action not in {"block", "mlp", "layer"}
        }
        if invalid_actions:
            raise ValueError(
                "native token action must be 'block', 'mlp', or 'layer'; got "
                f"{sorted(invalid_actions)}"
            )
        raw_mlp_offsets = cfg.get("mlp_layer_offsets_by_block") or {}
        mlp_offsets_by_block: dict[int, tuple[int, ...]] = {}
        for block_idx in configured_blocks:
            raw_offsets = raw_mlp_offsets.get(
                block_idx, raw_mlp_offsets.get(str(block_idx))
            )
            if raw_offsets is None:
                raw_offsets = range(self.layers_per_block)
            offsets = tuple(sorted(set(int(offset) for offset in raw_offsets)))
            if not offsets:
                raise ValueError(
                    f"native MLP action for block {block_idx} has no layer offsets"
                )
            if min(offsets) < 0 or max(offsets) >= self.layers_per_block:
                raise ValueError(
                    f"native MLP layer offset outside [0, "
                    f"{self.layers_per_block - 1}] for block {block_idx}"
                )
            mlp_offsets_by_block[block_idx] = offsets
        raw_layer_offsets = cfg.get("layer_offsets_by_block") or {}
        layer_offsets_by_block: dict[int, tuple[int, ...]] = {}
        for block_idx in configured_blocks:
            raw_offsets = raw_layer_offsets.get(
                block_idx, raw_layer_offsets.get(str(block_idx))
            )
            if raw_offsets is None:
                raw_offsets = range(self.layers_per_block)
            offsets = tuple(sorted(set(int(offset) for offset in raw_offsets)))
            if not offsets:
                raise ValueError(
                    f"native layer action for block {block_idx} has no offsets"
                )
            if min(offsets) < 0 or max(offsets) >= self.layers_per_block:
                raise ValueError(
                    f"native decoder-layer offset outside [0, "
                    f"{self.layers_per_block - 1}] for block {block_idx}"
                )
            layer_offsets_by_block[block_idx] = offsets
        cfg["feature_upper_bounds"] = normalized_upper
        cfg["feature_lower_bounds"] = normalized_lower
        cfg["active_feature_names_by_block"] = active_feature_names_by_block
        cfg["action_by_block"] = {
            block: action_by_block.get(block, "block")
            for block in configured_blocks
        }
        cfg["mlp_layer_offsets_by_block"] = mlp_offsets_by_block
        cfg["layer_offsets_by_block"] = layer_offsets_by_block
        cfg["max_skips"] = int(cfg.get("max_skips", 1))
        cfg["decode_only"] = bool(cfg.get("decode_only", True))
        cfg["device_conditional_runtime"] = bool(
            cfg.get("device_conditional_runtime", False)
        )
        cfg["device_fused_native_features"] = bool(
            cfg.get("device_fused_native_features", False)
        )
        cfg["device_conditional_max_positions"] = int(
            cfg.get("device_conditional_max_positions", 8192)
        )
        if cfg["max_skips"] < 0:
            raise ValueError("native token selector max_skips must be non-negative")
        if (
            cfg["device_fused_native_features"]
            and not cfg["device_conditional_runtime"]
        ):
            raise ValueError(
                "fused native features require the native device runtime"
            )
        if cfg["device_conditional_runtime"]:
            if not cfg["decode_only"]:
                raise ValueError("native device runtime requires decode_only")
            if cfg["device_conditional_max_positions"] <= 0:
                raise ValueError(
                    "native device_conditional_max_positions must be positive"
                )
            if cfg["max_skips"] < len(cfg["eligible_blocks"]):
                raise ValueError(
                    "native device runtime requires an inactive global skip cap"
                )
            if any(
                cfg["action_by_block"][block] != "layer"
                for block in cfg["eligible_blocks"]
            ):
                raise ValueError(
                    "native device runtime requires decoder-layer actions"
                )
        self._native_token_skip_config = cfg
        self._native_device_conditional_runtime_enabled = bool(
            cfg["device_conditional_runtime"]
        )
        if self._native_device_conditional_runtime_enabled:
            fingerprint = json.dumps(
                {
                    "eligible_blocks": sorted(cfg["eligible_blocks"]),
                    "feature_upper_bounds": cfg["feature_upper_bounds"],
                    "feature_lower_bounds": cfg["feature_lower_bounds"],
                    "action_by_block": cfg["action_by_block"],
                    "layer_offsets_by_block": cfg["layer_offsets_by_block"],
                    "device_conditional_max_positions": cfg[
                        "device_conditional_max_positions"
                    ],
                    "device_fused_native_features": cfg[
                        "device_fused_native_features"
                    ],
                },
                sort_keys=True,
            )
            runtime = self._native_device_conditional_runtimes.get(fingerprint)
            if runtime is None:
                try:
                    from .device_conditional_reskip import (
                        DeviceConditionalNativeReskipRuntime,
                    )
                except ImportError:
                    from device_conditional_reskip import (
                        DeviceConditionalNativeReskipRuntime,
                    )
                runtime = (
                    DeviceConditionalNativeReskipRuntime(cfg, self.gamma.device)
                )
                self._native_device_conditional_runtimes[fingerprint] = runtime
            self._native_device_conditional_runtime = runtime
            self._native_device_conditional_runtime_fingerprint = fingerprint

    def load_early_joint_risk_surrogate(
        self,
        state_dict: dict[str, torch.Tensor],
        target_blocks: Iterable[int],
    ):
        """Attach only a frozen early-risk head; never load its parent model."""
        targets = tuple(sorted(set(int(block) for block in target_blocks)))
        if not targets:
            raise ValueError("early joint-risk surrogate needs target blocks")
        unknown = set(targets).difference(self.risk_surrogate_actions)
        if unknown:
            raise ValueError(
                f"early joint-risk surrogate targets unknown blocks {sorted(unknown)}"
            )
        module = AttnResActionRiskSurrogate(
            {
                block: self.risk_surrogate_actions[block]
                for block in targets
            },
            hidden_dim=self.joint_risk_surrogate_hidden_dim,
            flip_weight=self.risk_surrogate_flip_weight,
            hidden_size=self.hidden_size,
            semantic_dim=self.joint_risk_surrogate_semantic_dim,
        )
        required = {"semantic_projection.weight"}
        for block in targets:
            required.update(
                {
                    f"heads.{block}.0.weight",
                    f"heads.{block}.0.bias",
                    f"heads.{block}.2.weight",
                    f"heads.{block}.2.bias",
                }
            )
        missing = required.difference(state_dict)
        if missing:
            raise ValueError(
                f"early joint-risk checkpoint lacks {sorted(missing)}"
            )
        module.load_state_dict(
            {name: state_dict[name] for name in required},
            strict=True,
        )
        module.to(device=self.gamma.device, dtype=torch.float32)
        module.eval()
        self.early_joint_risk_surrogate = module
        self._triton_early_joint_risk_selector = None

    def configure_action_risk_skip(self, config: dict | None):
        """Configure causal per-layer actions from the learned AttnRes risk head."""
        if config is None:
            self._action_risk_skip_config = None
            self._action_risk_single_action_specs = {}
            self._device_conditional_runtime_enabled = False
            return
        if self.risk_surrogate is None:
            raise ValueError("action-risk skipping requires a configured risk surrogate")
        cfg = dict(config)
        selector_mode = str(cfg.get("selector_mode", "block_local"))
        if selector_mode not in {"block_local", "joint_gate"}:
            raise ValueError(
                "action-risk selector_mode must be block_local or joint_gate"
            )
        if selector_mode == "joint_gate" and not self.joint_risk_surrogates:
            raise ValueError(
                "joint-gate action-risk skipping requires joint surrogate weights"
            )
        raw_thresholds = cfg.get("thresholds_by_block") or {}
        thresholds: dict[int, dict[int, float]] = {}
        for raw_block, raw_offsets in raw_thresholds.items():
            block = int(raw_block)
            if block not in self.risk_surrogate_actions:
                raise ValueError(f"action-risk threshold targets unknown block {block}")
            values = {
                int(offset): float(threshold)
                for offset, threshold in raw_offsets.items()
            }
            unknown = set(values).difference(self.risk_surrogate_actions[block])
            if unknown:
                raise ValueError(
                    f"action-risk thresholds target unknown offsets at block "
                    f"{block}: {sorted(unknown)}"
                )
            if not values or not all(math.isfinite(value) for value in values.values()):
                raise ValueError("action-risk thresholds must be non-empty and finite")
            thresholds[block] = values
        if not thresholds:
            raise ValueError("action-risk skipping requires thresholds_by_block")
        eligible = cfg.get("eligible_blocks")
        eligible_blocks = (
            set(int(block) for block in eligible)
            if eligible is not None
            else set(thresholds)
        )
        if not eligible_blocks.issubset(thresholds):
            raise ValueError("eligible action-risk blocks lack thresholds")
        max_layer_skips = int(cfg.get("max_layer_skips", 0))
        if max_layer_skips < 0:
            raise ValueError("max_layer_skips must be non-negative")
        cfg["thresholds_by_block"] = thresholds
        cfg["eligible_blocks"] = eligible_blocks
        cfg["max_layer_skips"] = max_layer_skips
        cfg["decode_only"] = bool(cfg.get("decode_only", True))
        cfg["selector_mode"] = selector_mode
        cfg["gate_blocks"] = (
            self.joint_risk_gate_blocks
            if selector_mode == "joint_gate"
            else ()
        )
        cfg["gate_by_block"] = (
            dict(self.joint_risk_gate_by_block)
            if selector_mode == "joint_gate"
            else {}
        )
        # Normal inference consumes aggregate skip counters, not the expensive
        # Python-valued per-block trace. Calibration/debug callers can opt in.
        cfg["collect_trace"] = bool(cfg.get("collect_trace", False))
        cfg["compile_selector"] = bool(
            cfg.get(
                "compile_selector",
                os.environ.get("RESKIP_COMPILE_ACTION_RISK", "0") == "1",
            )
        )
        cfg["triton_fused_selector"] = bool(
            cfg.get("triton_fused_selector", False)
        )
        cfg["triton_fused_native_features"] = bool(
            cfg.get("triton_fused_native_features", False)
        )
        cfg["triton_packed_host_scores"] = bool(
            cfg.get("triton_packed_host_scores", False)
        )
        cfg["kv_only_key_rope"] = bool(
            cfg.get("kv_only_key_rope", False)
        )
        cfg["triton_fused_k_norm_rope"] = bool(
            cfg.get("triton_fused_k_norm_rope", False)
        )
        cfg["device_conditional_runtime"] = bool(
            cfg.get("device_conditional_runtime", False)
        )
        cfg["device_conditional_max_positions"] = int(
            cfg.get("device_conditional_max_positions", 8192)
        )
        if (
            cfg["triton_fused_native_features"]
            and not cfg["triton_fused_selector"]
        ):
            raise ValueError(
                "triton_fused_native_features requires the Triton selector"
            )
        if (
            cfg["triton_packed_host_scores"]
            and not cfg["triton_fused_selector"]
        ):
            raise ValueError(
                "triton_packed_host_scores requires the Triton selector"
            )
        if (
            cfg["triton_fused_k_norm_rope"]
            and not cfg["kv_only_key_rope"]
        ):
            raise ValueError(
                "triton_fused_k_norm_rope requires kv_only_key_rope"
            )
        if cfg["device_conditional_runtime"]:
            if not (
                selector_mode == "joint_gate"
                and cfg["triton_fused_selector"]
                and cfg["triton_fused_native_features"]
                and cfg["triton_packed_host_scores"]
                and cfg["kv_only_key_rope"]
                and not cfg["triton_fused_k_norm_rope"]
            ):
                raise ValueError(
                    "device conditional runtime requires the frozen packed "
                    "joint selector and native key-only RoPE"
                )
            if cfg["device_conditional_max_positions"] <= 0:
                raise ValueError(
                    "device_conditional_max_positions must be positive"
                )
        if cfg["triton_fused_selector"]:
            if selector_mode != "joint_gate":
                raise ValueError(
                    "triton_fused_selector requires joint_gate mode"
                )
            try:
                from .triton_joint_risk_selector import (
                    TritonJointRiskSelector,
                )
            except ImportError:
                from triton_joint_risk_selector import (
                    TritonJointRiskSelector,
                )
            for gate_block in self.joint_risk_gate_blocks:
                if gate_block not in self._triton_joint_risk_selectors:
                    self._triton_joint_risk_selectors[gate_block] = (
                        TritonJointRiskSelector(
                            self.joint_risk_surrogates[str(gate_block)]
                        )
                    )
        raw_early_fallback = cfg.get("early_gate_fallback")
        if raw_early_fallback is not None:
            if selector_mode != "joint_gate":
                raise ValueError("early gate fallback requires joint_gate mode")
            if self.early_joint_risk_surrogate is None:
                raise ValueError(
                    "early gate fallback requires a loaded frozen early surrogate"
                )
            early_fallback = dict(raw_early_fallback)
            early_gate = int(early_fallback.get("gate_block", -1))
            targets = tuple(
                sorted(
                    set(
                        int(block)
                        for block in early_fallback.get("target_blocks", ())
                    )
                )
            )
            if early_gate not in self.joint_risk_gate_blocks:
                raise ValueError(
                    "early fallback gate must already be an operational joint gate"
                )
            if set(targets) != set(
                self.early_joint_risk_surrogate.actions_by_block
            ):
                raise ValueError(
                    "early fallback targets do not match the loaded surrogate"
                )
            if any(
                self.joint_risk_gate_by_block[target] <= early_gate
                for target in targets
            ):
                raise ValueError(
                    "early fallback can only replace strictly later gates"
                )
            raw_skip = early_fallback.get("skip_thresholds") or {}
            raw_no_skip = early_fallback.get("no_skip_thresholds") or {}
            skip_thresholds = {
                block: float(raw_skip.get(str(block), raw_skip.get(block)))
                for block in targets
            }
            no_skip_thresholds = {
                block: float(raw_no_skip.get(str(block), raw_no_skip.get(block)))
                for block in targets
            }
            if not all(
                math.isfinite(skip_thresholds[block])
                and math.isfinite(no_skip_thresholds[block])
                and skip_thresholds[block] < no_skip_thresholds[block]
                for block in targets
            ):
                raise ValueError(
                    "early fallback needs finite ordered skip/no-skip thresholds"
                )
            early_fallback["gate_block"] = early_gate
            early_fallback["target_blocks"] = targets
            early_fallback["skip_thresholds"] = skip_thresholds
            early_fallback["no_skip_thresholds"] = no_skip_thresholds
            early_fallback["triton_fused_selector"] = bool(
                early_fallback.get(
                    "triton_fused_selector", cfg["triton_fused_selector"]
                )
            )
            if early_fallback["triton_fused_selector"]:
                try:
                    from .triton_joint_risk_selector import (
                        TritonJointRiskSelector,
                    )
                except ImportError:
                    from triton_joint_risk_selector import (
                        TritonJointRiskSelector,
                    )
                self._triton_early_joint_risk_selector = (
                    TritonJointRiskSelector(self.early_joint_risk_surrogate)
                )
            cfg["early_gate_fallback"] = early_fallback
        raw_ratio_fallback = cfg.get("gate1_ratio_fallback")
        if raw_ratio_fallback is not None:
            if selector_mode != "joint_gate":
                raise ValueError("gate-1 ratio fallback requires joint_gate mode")
            if raw_early_fallback is not None:
                raise ValueError(
                    "gate-1 ratio fallback and early surrogate fallback "
                    "cannot be enabled together"
                )
            ratio_fallback = dict(raw_ratio_fallback)
            ratio_gate = int(ratio_fallback.get("gate_block", -1))
            source_blocks = tuple(
                sorted(
                    set(
                        int(block)
                        for block in ratio_fallback.get("source_blocks", ())
                    )
                )
            )
            target_blocks = tuple(
                sorted(
                    set(
                        int(block)
                        for block in ratio_fallback.get("target_blocks", ())
                    )
                )
            )
            if ratio_gate not in self.joint_risk_gate_blocks:
                raise ValueError(
                    "ratio fallback gate must already be an operational joint gate"
                )
            if not source_blocks or not target_blocks:
                raise ValueError(
                    "ratio fallback requires source_blocks and target_blocks"
                )
            if any(
                self.joint_risk_gate_by_block.get(block) != ratio_gate
                for block in source_blocks
            ):
                raise ValueError(
                    "ratio fallback sources must be scored by its gate"
                )
            if any(
                self.joint_risk_gate_by_block.get(block, -1) <= ratio_gate
                for block in target_blocks
            ):
                raise ValueError(
                    "ratio fallback can only replace strictly later gates"
                )
            configured_blocks = set(thresholds)
            if not set(source_blocks + target_blocks).issubset(
                configured_blocks
            ):
                raise ValueError(
                    "ratio fallback blocks require action-risk thresholds"
                )
            if not set(source_blocks + target_blocks).issubset(
                eligible_blocks
            ):
                raise ValueError(
                    "ratio fallback blocks must be action-risk eligible"
                )
            if any(
                len(self.risk_surrogate_actions[block]) != 1
                for block in source_blocks + target_blocks
            ):
                raise ValueError(
                    "ratio fallback currently requires one action per block"
                )
            all_skip_upper = float(ratio_fallback["all_skip_upper"])
            all_compute_lower = float(
                ratio_fallback["all_compute_lower"]
            )
            if not (
                math.isfinite(all_skip_upper)
                and math.isfinite(all_compute_lower)
                and 0.0 < all_skip_upper < 1.0 < all_compute_lower
            ):
                raise ValueError(
                    "ratio fallback needs finite skip<1<compute bounds"
                )
            ratio_fallback["gate_block"] = ratio_gate
            ratio_fallback["source_blocks"] = source_blocks
            ratio_fallback["target_blocks"] = target_blocks
            ratio_fallback["all_skip_upper"] = all_skip_upper
            ratio_fallback["all_compute_lower"] = all_compute_lower
            cfg["gate1_ratio_fallback"] = ratio_fallback
        if selector_mode == "block_local":
            self.risk_surrogate.configure_compiled_inference(
                cfg["compile_selector"]
            )
        single_action_specs = {}
        for block, values in thresholds.items():
            if len(values) != 1:
                continue
            action, threshold = next(iter(values.items()))
            action_position = self.risk_surrogate_actions[block].index(action)
            single_action_specs[block] = (
                action_position,
                action,
                threshold,
                self.risk_surrogate_groups[block][action],
            )
        self._action_risk_single_action_specs = single_action_specs
        self._action_risk_skip_config = cfg
        self._device_conditional_runtime_enabled = bool(
            cfg["device_conditional_runtime"]
        )
        if self._device_conditional_runtime_enabled:
            if self._device_conditional_runtime is None:
                try:
                    from .device_conditional_reskip import (
                        DeviceConditionalReskipRuntime,
                    )
                except ImportError:
                    from device_conditional_reskip import (
                        DeviceConditionalReskipRuntime,
                    )
                self._device_conditional_runtime = (
                    DeviceConditionalReskipRuntime(
                        cfg,
                        self.risk_surrogate_groups,
                        self.gamma.device,
                    )
                )

    def _select_joint_action_risk_actions(
        self,
        score_values: Iterable[float],
        score_specs: Iterable[tuple[int, int, float]],
        *,
        max_layer_skips: int,
        initial_layer_count: int,
    ) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
        candidates = sorted(
            (
                (
                    float(score) / max(threshold, 1.0e-12),
                    float(score),
                    target_block,
                    action,
                )
                for score, (target_block, action, threshold) in zip(
                    score_values, score_specs
                )
                if float(score) <= threshold
            ),
            key=lambda item: (item[0], item[2], item[3]),
        )
        selected_offsets: dict[int, set[int]] = {}
        selected_actions: dict[int, list[int]] = {}
        selected_layer_count = int(initial_layer_count)
        for _, _, target_block, action in candidates:
            group_offsets = set(
                self.risk_surrogate_groups[target_block][action]
            )
            block_offsets = selected_offsets.setdefault(target_block, set())
            new_offsets = group_offsets.difference(block_offsets)
            if (
                max_layer_skips > 0
                and selected_layer_count + len(new_offsets) > max_layer_skips
            ):
                continue
            selected_actions.setdefault(target_block, []).append(action)
            block_offsets.update(group_offsets)
            selected_layer_count += len(new_offsets)
        action_ids_by_block = {
            target_block: tuple(sorted(actions))
            for target_block, actions in selected_actions.items()
        }
        layer_offsets_by_block = {
            target_block: tuple(sorted(offsets))
            for target_block, offsets in selected_offsets.items()
            if offsets
        }
        return action_ids_by_block, layer_offsets_by_block

    def configure_risk_profile_skip(self, config: dict | None):
        """Configure one-decision, risk-ordered, per-token ReSkip.

        A legacy config contains one ``profile_blocks`` / ``score_threshold``
        pair. A ladder config contains nested ``profile_levels`` ordered from
        the least to the most aggressive action. The routing score is copied
        to the host exactly once at ``gate_block`` and the chosen action lives
        only for the current decode token.
        """
        if config is None:
            self._risk_profile_skip_config = None
            self._risk_profile_decode_steps = 0
            return
        cfg = dict(config)
        gate_block = int(cfg["gate_block"])
        selector_mode = str(cfg.get("selector_mode", "shared"))
        if selector_mode not in {"shared", "linear_native"}:
            raise ValueError("risk profile selector_mode must be shared or linear_native")
        raw_levels = cfg.get("profile_levels")
        if raw_levels is None:
            raw_levels = [
                {
                    "profile_blocks": cfg["profile_blocks"],
                    "score_threshold": cfg["score_threshold"],
                }
            ]
        levels = []
        for raw_level in raw_levels:
            blocks = tuple(
                sorted(set(int(x) for x in raw_level["profile_blocks"]))
            )
            if not blocks:
                raise ValueError("risk profile level requires at least one block")
            if min(blocks) < gate_block:
                raise ValueError("risk profile cannot skip a block before gate_block")
            if not set(blocks).issubset(self.skippable_block_set):
                raise ValueError("risk profile includes a non-skippable block")
            level = {
                "profile_blocks": blocks,
                "score_threshold": float(raw_level["score_threshold"]),
            }
            if selector_mode == "linear_native":
                raw_model = dict(raw_level["linear_model"])
                mean = tuple(float(value) for value in raw_model["mean"])
                scale = tuple(float(value) for value in raw_model["scale"])
                coefficient = tuple(
                    float(value) for value in raw_model["coefficient"]
                )
                feature_names = tuple(str(value) for value in raw_model["feature_names"])
                if not mean or not (
                    len(mean) == len(scale) == len(coefficient) == len(feature_names)
                ):
                    raise ValueError("linear native selector feature dimensions mismatch")
                if any(value <= 0.0 for value in scale):
                    raise ValueError("linear native selector scales must be positive")
                level["linear_model"] = {
                    "mean": mean,
                    "scale": scale,
                    "coefficient": coefficient,
                    "intercept": float(raw_model["intercept"]),
                    "feature_names": feature_names,
                }
            levels.append(level)
        if selector_mode == "shared":
            levels.sort(key=lambda level: level["score_threshold"])
        for previous, current in zip(levels, levels[1:]):
            if not set(previous["profile_blocks"]).issubset(
                current["profile_blocks"]
            ):
                raise ValueError("risk profile levels must be nested by threshold")
            if (
                selector_mode == "shared"
                and not current["score_threshold"] > previous["score_threshold"]
            ):
                raise ValueError("risk profile thresholds must be strictly increasing")
        margin_weight = float(cfg.get("margin_weight", 0.5))
        entropy_weight = float(cfg.get("entropy_weight", 0.5))
        score_mode = str(cfg.get("score_mode", "confidence"))
        if score_mode not in {"confidence", "recent"}:
            raise ValueError("risk profile score_mode must be confidence or recent")
        numeric_values = [
            margin_weight,
            entropy_weight,
            *(level["score_threshold"] for level in levels),
        ]
        if selector_mode == "linear_native":
            for level in levels:
                linear_model = level["linear_model"]
                numeric_values.extend(linear_model["mean"])
                numeric_values.extend(linear_model["scale"])
                numeric_values.extend(linear_model["coefficient"])
                numeric_values.append(linear_model["intercept"])
        if not all(math.isfinite(x) for x in numeric_values):
            raise ValueError("risk profile score parameters must be finite")
        union_blocks = tuple(
            sorted(set().union(*(set(level["profile_blocks"]) for level in levels)))
        )
        cfg.update(
            gate_block=gate_block,
            profile_blocks=union_blocks,
            profile_levels=tuple(levels),
            score_threshold=levels[0]["score_threshold"],
            margin_weight=margin_weight,
            entropy_weight=entropy_weight,
            score_mode=score_mode,
            selector_mode=selector_mode,
            decode_only=bool(cfg.get("decode_only", True)),
            warmup_decode_tokens=int(cfg.get("warmup_decode_tokens", 0)),
        )
        if cfg["warmup_decode_tokens"] < 0:
            raise ValueError("risk profile warmup_decode_tokens must be non-negative")
        self._risk_profile_skip_config = cfg
        self._risk_profile_decode_steps = 0

    def configure_sticky_skip(self, config: dict | None):
        """Configure request-level ReSkip and reset its per-request state.

        A request is observed at full depth during prefill and, optionally,
        during a fixed number of early decode steps.  The selected profile is
        then held for the rest of that request.  Permanently skipped blocks do
        not update K/V because they cannot be re-enabled later in the request.
        """
        if config is None:
            self._sticky_skip_config = None
            self._reset_sticky_request_state()
            return
        cfg = dict(config)
        cfg["thresholds"] = {
            int(k): float(v) for k, v in (cfg.get("thresholds") or {}).items()
        }
        eligible = cfg.get("eligible_blocks")
        cfg["eligible_blocks"] = (
            set(int(x) for x in eligible)
            if eligible is not None
            else set(self.skippable_blocks)
        )
        cfg["max_skips"] = int(cfg.get("max_skips", 1))
        cfg["observe_decode_tokens"] = int(cfg.get("observe_decode_tokens", 0))
        if cfg["max_skips"] < 0 or cfg["observe_decode_tokens"] < 0:
            raise ValueError("max_skips and observe_decode_tokens must be non-negative")
        profiles = cfg.get("profiles")
        if profiles is not None:
            cfg["profiles"] = [tuple(sorted(set(int(x) for x in p))) for p in profiles]
        self._sticky_skip_config = cfg
        self._reset_sticky_request_state()

    def _reset_sticky_request_state(self):
        self._sticky_active_blocks = set()
        self._sticky_profile_ready = False
        self._sticky_observation_sums = {}
        self._sticky_observation_count = 0
        cfg = self._sticky_skip_config or {}
        self._sticky_observations_remaining = 1 + int(cfg.get("observe_decode_tokens", 0))

    def _finish_sticky_observation(self, observations: list[tuple[int, torch.Tensor]]):
        cfg = self._sticky_skip_config
        if cfg is None or self._sticky_profile_ready or not observations:
            return
        block_ids = [int(block_idx) for block_idx, _ in observations]
        values = torch.stack([value for _, value in observations]).detach().float().cpu().tolist()
        for block_idx, value in zip(block_ids, values):
            self._sticky_observation_sums[block_idx] = (
                self._sticky_observation_sums.get(block_idx, 0.0) + float(value)
            )
        self._sticky_observation_count += 1
        self._sticky_observations_remaining -= 1
        if self._sticky_observations_remaining > 0:
            return

        count = max(self._sticky_observation_count, 1)
        means = {b: total / count for b, total in self._sticky_observation_sums.items()}
        thresholds = cfg.get("thresholds", {})
        eligible = set(cfg.get("eligible_blocks") or self.skippable_blocks)
        max_skips = int(cfg.get("max_skips", 1))
        margins = {
            b: means[b] - float(thresholds[b])
            for b in eligible
            if b in means and b in thresholds
        }
        passing = {b for b, margin in margins.items() if margin > 0.0}

        profiles = cfg.get("profiles")
        if profiles is not None:
            valid = []
            for profile in profiles:
                selected = set(profile)
                if (
                    len(selected) <= max_skips
                    and selected.issubset(eligible)
                    and selected.issubset(passing)
                ):
                    score = sum(margins[b] for b in selected)
                    valid.append((len(selected), score, tuple(sorted(selected))))
            chosen = max(valid, default=(0, 0.0, ()), key=lambda row: (row[0], row[1]))[2]
        else:
            chosen = tuple(
                b for b, _ in sorted(margins.items(), key=lambda row: row[1], reverse=True)
                if b in passing
            )[:max_skips]

        self._sticky_active_blocks = set(chosen)
        self._sticky_profile_ready = True
        if self._skip_stats is not None:
            self._skip_stats["sticky_profile_selections"] += 1
            for block_idx in self._sticky_active_blocks:
                self._skip_stats["per_block_sticky_selections"][block_idx] += 1

    def get_skip_stats(self):
        if self._skip_stats is None:
            return None
        stats = dict(self._skip_stats)
        if (
            self._device_conditional_runtime_enabled
            and self._device_conditional_runtime is not None
        ):
            device_stats = self._device_conditional_runtime.summarize()
            block_actions = device_stats["per_block_actions"]
            block_layers = device_stats["per_block_layers"]
            action_count = sum(block_actions)
            layer_count = sum(block_layers)
            # Host fallback remains active for DynamicCache and unsupported
            # shapes.  Merge device history into those counters instead of
            # replacing valid host-path statistics with an empty history.
            stats["action_risk_skip_requests"] += action_count
            stats["action_risk_layer_requests"] += layer_count
            stats["partial_layer_skip_events"] += action_count
            stats["partial_decoder_layer_skip_events"] += layer_count
            stats["per_block_action_risk_requests"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_action_risk_requests"], block_actions
                )
            ]
            stats["per_block_action_risk_layer_requests"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_action_risk_layer_requests"], block_layers
                )
            ]
            stats["per_block_partial_layer_skips"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_partial_layer_skips"], block_actions
                )
            ]
            stats["per_block_partial_decoder_layer_skips"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_partial_decoder_layer_skips"], block_layers
                )
            ]
            visited = device_stats["visited_tokens"]
            fallback = device_stats["ratio_fallback_tokens"]
            all_skip = device_stats["ratio_all_skip_tokens"]
            all_compute = device_stats["ratio_all_compute_tokens"]
            stats["ratio_gate_candidate_evaluations"] += visited
            stats["ratio_gate_fallback_forwards"] += fallback
            stats["ratio_gate_bypass_forwards"] += all_skip + all_compute
            stats["ratio_gate_predicted_actions"] += 3 * all_skip
            stats["device_conditional_decode_tokens"] = visited
            stats["device_conditional_graph_count"] = (
                len(self._device_conditional_runtime.layer_graphs)
                + len(self._device_conditional_runtime.recovery_graphs)
            )
        if (
            self._native_device_conditional_runtime_enabled
            and self._native_device_conditional_runtime is not None
        ):
            native_device_stats = (
                self._native_device_conditional_runtime.summarize()
            )
            block_actions = native_device_stats["per_block_actions"]
            block_layers = native_device_stats["per_block_layers"]
            action_count = sum(block_actions)
            layer_count = sum(block_layers)
            stats["native_token_skip_requests"] += action_count
            stats["partial_layer_skip_events"] += action_count
            stats["partial_decoder_layer_skip_events"] += layer_count
            stats["per_block_native_token_requests"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_native_token_requests"], block_actions
                )
            ]
            stats["per_block_partial_layer_skips"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_partial_layer_skips"], block_actions
                )
            ]
            stats["per_block_partial_decoder_layer_skips"] = [
                int(host) + int(device)
                for host, device in zip(
                    stats["per_block_partial_decoder_layer_skips"], block_layers
                )
            ]
            stats["native_device_conditional_decode_tokens"] = (
                native_device_stats["visited_tokens"]
            )
            stats["native_device_conditional_graph_count"] = len(
                self._native_device_conditional_runtime.layer_graphs
            )
        forwards = max(int(stats["forwards"]), 1)
        decode_forwards = max(int(stats["decode_forwards"]), 1)
        total_positions = max(int(stats["total_positions"]), 1)
        stats["avg_skips_per_forward"] = stats["skip_events"] / forwards
        stats["avg_skips_per_decode_forward"] = (
            stats["skip_events"] / decode_forwards
        )
        stats["avg_partial_mlp_skips_per_forward"] = (
            stats["partial_mlp_skip_events"] / forwards
        )
        stats["avg_partial_mlp_skips_per_decode_forward"] = (
            stats["partial_mlp_skip_events"] / decode_forwards
        )
        stats["avg_partial_mlp_layer_skips_per_forward"] = (
            stats["partial_mlp_layer_skip_events"] / forwards
        )
        stats["avg_partial_mlp_layer_skips_per_decode_forward"] = (
            stats["partial_mlp_layer_skip_events"] / decode_forwards
        )
        stats["avg_partial_decoder_layer_skips_per_forward"] = (
            stats["partial_decoder_layer_skip_events"] / forwards
        )
        stats["avg_partial_decoder_layer_skips_per_decode_forward"] = (
            stats["partial_decoder_layer_skip_events"] / decode_forwards
        )
        stats["block_skip_rate"] = stats["skip_events"] / total_positions
        stats["forward_skip_rate_upper_bound"] = stats["skip_events"] / forwards
        return stats

    @property
    def text_layers(self):
        return self.base_model.model.language_model.layers

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def _install_retrofit_forward(self):
        retrofit = self
        lm = self.base_model.model.language_model
        if not hasattr(lm, "_original_forward"):
            lm._original_forward = lm.forward

        def patched_forward(
            self_lm,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            return retrofit._text_model_forward(
                self_lm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        lm.forward = types.MethodType(patched_forward, lm)

    def _block_contains_deepstack_layers(self, block_idx: int, deepstack_visual_embeds) -> bool:
        if deepstack_visual_embeds is None:
            return False
        if not hasattr(deepstack_visual_embeds, "__len__"):
            return False
        if len(deepstack_visual_embeds) == 0:
            return False
        start = block_idx * self.layers_per_block
        end = start + self.layers_per_block
        return start < len(deepstack_visual_embeds) and end > 0

    def _compute_block_input(
        self,
        block_idx: int,
        prev_block: torch.Tensor,
        completed: list[torch.Tensor],
        collect_alpha: bool,
        compute_entropy: bool = False,
        compute_correction_ratio: bool = False,
        compute_native_features: bool = False,
        return_fused_native_inputs: bool = False,
        minimal_native_features: bool = False,
    ):
        if block_idx not in self.skippable_block_set:
            return prev_block, None, None, None, None, None, None
        source_values = None
        if return_fused_native_inputs:
            routed, alpha, source_values = (
                self.router.route_with_source_values(
                    block_idx + 1, completed
                )
            )
            source_dispersion = None
        elif compute_native_features and not minimal_native_features:
            routed, alpha, router_native_features = self.router.route_with_native_stats(
                block_idx + 1, completed
            )
            source_dispersion = router_native_features["source_dispersion"]
        else:
            routed, alpha = self.router.route(block_idx + 1, completed)
            source_dispersion = None
        delta = routed - prev_block
        gamma = self.gamma[block_idx].to(prev_block.dtype)
        if self.bridge_mode == "adapter":
            update = gamma * self.adapters[block_idx](delta)
        elif self.bridge_mode == "pure":
            update = gamma * delta
        else:
            compat_gate = self.compat_peak * 4.0 * gamma * (1.0 - gamma)
            update = gamma * delta + compat_gate * self.adapters[block_idx](delta)
        corrected = prev_block + update
        native_features = None
        if compute_native_features:
            w_recent = alpha[..., -1].float()
            if minimal_native_features:
                # The frozen RSM policy only reads w_recent and the RSM
                # strength factor. Reuse one finite scalar for inactive
                # dimensions whose bounds are +/-inf, avoiding all dense
                # native-statistics reductions on the decode path.
                native_features = {
                    name: w_recent for name in NATIVE_TOKEN_SKIP_FEATURES
                }
                native_features["residual_strength_factor"] = torch.ones_like(
                    w_recent
                )
                native_features["_minimal_native_features"] = True
            else:
                previous_scale = (
                    prev_block.float().pow(2).mean(dim=-1).sqrt().clamp_min(1e-8)
                )
                native_features = {
                    **router_native_features,
                    "w_recent": w_recent,
                    "route_novelty": (
                        delta.float().pow(2).mean(dim=-1).sqrt() / previous_scale
                    ),
                    "correction_ratio": (
                        update.float().pow(2).mean(dim=-1).sqrt() / previous_scale
                    ),
                    # Identity default for ordinary AttnRes checkpoints. RSM
                    # replaces this with its learned token-conditioned factor.
                    "residual_strength_factor": torch.ones_like(
                        w_recent
                    ),
                }
        correction_ratio = None
        if compute_correction_ratio:
            update_rms = update.float().pow(2).mean().sqrt()
            base_rms = prev_block.float().pow(2).mean().sqrt().clamp_min(1e-8)
            correction_ratio = (update_rms / base_rms).detach()
        # Entropy is only consumed as a training regulariser. Computing it on
        # every inference forward wastes ~14 kernel launches with no effect
        # on outputs; gate behind compute_entropy.
        entropy = None
        if compute_entropy:
            entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
        if (
            collect_alpha
            or compute_native_features
            or return_fused_native_inputs
        ):
            return (
                corrected,
                alpha,
                routed,
                entropy,
                correction_ratio,
                native_features,
                (
                    (source_values, routed, prev_block)
                    if return_fused_native_inputs
                    else None
                ),
            )
        return corrected, None, routed, entropy, correction_ratio, None, None

    @staticmethod
    def _update_layer_kv_only(
        layer,
        hidden_states,
        past_key_values,
        position_embeddings,
        cache_position,
        *,
        key_only_rope: bool = False,
        triton_fused_k_norm_rope: bool = False,
    ):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            rotate_half,
        )

        attn = layer.self_attn
        normed = layer.input_layernorm(hidden_states)
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        key_projection = attn.k_proj(normed)
        value_projection = attn.v_proj(normed)
        value_states = value_projection.view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if triton_fused_k_norm_rope:
            try:
                from .triton_kv_only import fused_key_norm_rope
            except ImportError:
                from triton_kv_only import fused_key_norm_rope
            key_states = fused_key_norm_rope(
                key_projection,
                attn.k_norm.weight,
                attn.k_norm.variance_epsilon,
                cos,
                sin,
                attn.head_dim,
            )
        else:
            key_states = attn.k_norm(
                key_projection.view(hidden_shape)
            ).transpose(1, 2)
        if key_only_rope and not triton_fused_k_norm_rope:
            broadcast_cos = cos.unsqueeze(1)
            broadcast_sin = sin.unsqueeze(1)
            key_states = (key_states * broadcast_cos) + (
                rotate_half(key_states) * broadcast_sin
            )
        elif not triton_fused_k_norm_rope:
            _, key_states = apply_rotary_pos_emb(
                key_states, key_states, cos, sin
            )
        past_key_values.update(
            key_states,
            value_states,
            attn.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    def _text_model_forward(
        self,
        text_model,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        from transformers.cache_utils import DynamicCache, StaticCache
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            create_causal_mask,
        )

        # Preserve the pretrained model's exact inference path at the base
        # endpoint.  Although adding a numerical zero is algebraically an
        # identity, allocating a new BF16 hidden-state tensor can select a
        # slightly different fused-kernel path and accumulate rounding jitter.
        # Cache the scalar check by Parameter version so trained inference pays
        # at most one device synchronisation after loading a checkpoint.
        exact_base_candidate = (
            not self.training
            and not self._active_skip_blocks
            and self._dynamic_skip_config is None
            and self._native_token_skip_config is None
            and self._action_risk_skip_config is None
            and self._risk_profile_skip_config is None
            and not self._token_skip_masks
            and not self._token_layer_skip_masks
            and self._random_skip_config is None
            and self._sticky_skip_config is None
            and not self._collect_block_states
            and not self._return_alpha_flag
            and not self._return_correction_ratios_flag
            and not self._return_native_features_flag
            and not self._return_action_risks_flag
            and not self._routing_dump_path
        )
        exact_base_endpoint = False
        self._last_forward_used_exact_base = False
        if exact_base_candidate:
            gamma_version = self.gamma._version
            if gamma_version != self._eval_gamma_cache_version:
                self._eval_gamma_is_zero = bool(
                    torch.count_nonzero(self.gamma.detach()).item() == 0
                )
                self._eval_gamma_cache_version = gamma_version
            exact_base_endpoint = self._eval_gamma_is_zero
        if exact_base_endpoint:
            self._last_forward_used_exact_base = True
            return text_model._original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        incoming_cache_len = (
            int(past_key_values.get_seq_length())
            if past_key_values is not None
            else 0
        )
        is_single_decode = incoming_cache_len > 0
        if self._sticky_skip_config is not None and incoming_cache_len == 0:
            self._reset_sticky_request_state()
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=text_model.config)
        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)
        is_single_decode = is_single_decode and int(inputs_embeds.shape[1]) == 1
        if self._risk_profile_skip_config is not None and not is_single_decode:
            # Every prefill begins a new request.  The counter is deliberately
            # local to the request; it never turns token-level ReSkip into a
            # sticky/permanent profile.
            self._risk_profile_decode_steps = 0
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)

        # Fast-path gate for the Python-valued diagnostic trace. Returning the
        # differentiable alpha tensors must not implicitly trigger per-block
        # GPU-to-CPU synchronisations during selector-alignment training.
        collect_trace = (
            self._collect_block_states
            or (self._dynamic_skip_config is not None)
            or (
                self._action_risk_skip_config is not None
                and bool(
                    self._action_risk_skip_config.get("collect_trace", False)
                )
            )
            or (self._random_skip_config is not None)
            or bool(self._routing_dump_path)
        )
        compute_entropy_flag = self.training
        sticky_collecting = (
            self._sticky_skip_config is not None and not self._sticky_profile_ready
        )
        sticky_observations: list[tuple[int, torch.Tensor]] = []
        sticky_active_blocks = (
            set(self._sticky_active_blocks) if self._sticky_profile_ready else set()
        )

        completed: list[torch.Tensor] = [inputs_embeds]
        prev_block = inputs_embeds
        alpha_list: list[torch.Tensor] = []
        skip_trace: list[dict] = []
        block_inputs: list[torch.Tensor] = []
        block_outputs: list[torch.Tensor] = []
        surrogate_outputs: list[torch.Tensor | None] = []
        correction_ratios: list[torch.Tensor] = []
        native_features_by_block: list[
            dict[str, torch.Tensor] | None
        ] = []
        native_vectors_by_block: list[
            dict[str, torch.Tensor] | None
        ] = []
        layer_vectors_by_block: list[
            list[dict[str, torch.Tensor]] | None
        ] = []
        action_risks_by_block: list[
            dict[str, torch.Tensor] | None
        ] = []
        joint_action_risks_by_block: list[
            dict[str, torch.Tensor] | None
        ] = [None] * self.num_blocks
        entropy_accum: torch.Tensor | None = None
        layer_counter = 0
        gamma_cpu: list[float] | None = None  # lazy fill only if collect_trace

        dynamic_cfg = self._dynamic_skip_config  # None or dict
        dyn_skipped_count = 0
        native_cfg = self._native_token_skip_config
        native_skipped_count = 0
        native_profile_active_blocks: set[int] = set()
        action_risk_cfg = self._action_risk_skip_config
        action_risk_single_action_specs = self._action_risk_single_action_specs
        action_risk_layer_count = 0
        action_risk_selector_mode = str(
            (action_risk_cfg or {}).get("selector_mode", "block_local")
        )
        joint_gate_blocks = (
            self.joint_risk_gate_blocks
            if (
                action_risk_selector_mode == "joint_gate"
                or self._return_joint_action_risks_flag
            )
            else ()
        )
        joint_action_risk_predictions: dict[
            int, dict[str, torch.Tensor]
        ] = {}
        joint_action_ids_by_block: dict[int, tuple[int, ...]] = {}
        joint_layer_offsets_by_block: dict[int, tuple[int, ...]] = {}
        profile_cfg = self._risk_profile_skip_config
        profile_active_blocks: set[int] = set()
        static_recovery_profile = tuple(sorted(self._active_skip_blocks))
        random_cfg = self._random_skip_config
        random_skipped_count = 0
        if self._skip_stats is not None:
            self._skip_stats["forwards"] += 1
            if is_single_decode:
                self._skip_stats["decode_forwards"] += 1
            self._skip_stats["total_positions"] += self.num_blocks
        # Precompute Python-side dyn-skip config once per forward.
        dyn_thr_map = (dynamic_cfg or {}).get("thresholds", {}) or {}
        dyn_eligible = (dynamic_cfg or {}).get("eligible_blocks")
        dyn_max_skips = (dynamic_cfg or {}).get("max_skips")
        native_bounds_map = (native_cfg or {}).get("feature_upper_bounds", {}) or {}
        native_lower_bounds_map = (
            (native_cfg or {}).get("feature_lower_bounds", {}) or {}
        )
        native_active_feature_names_by_block = (
            (native_cfg or {}).get("active_feature_names_by_block", {}) or {}
        )
        native_action_by_block = (
            (native_cfg or {}).get("action_by_block", {}) or {}
        )
        native_mlp_offsets_by_block = (
            (native_cfg or {}).get("mlp_layer_offsets_by_block", {}) or {}
        )
        native_layer_offsets_by_block = (
            (native_cfg or {}).get("layer_offsets_by_block", {}) or {}
        )
        native_eligible = (native_cfg or {}).get("eligible_blocks")
        native_max_skips = (native_cfg or {}).get("max_skips")
        native_decode_only = bool((native_cfg or {}).get("decode_only", True))
        action_risk_thresholds = (
            (action_risk_cfg or {}).get("thresholds_by_block", {}) or {}
        )
        action_risk_eligible = (action_risk_cfg or {}).get("eligible_blocks")
        action_risk_max_layers = int(
            (action_risk_cfg or {}).get("max_layer_skips", 0)
        )
        action_risk_decode_only = bool(
            (action_risk_cfg or {}).get("decode_only", True)
        )
        triton_packed_host_scores = bool(
            (action_risk_cfg or {}).get(
                "triton_packed_host_scores", False
            )
        )
        kv_only_key_rope = bool(
            (action_risk_cfg or {}).get("kv_only_key_rope", False)
        )
        triton_fused_k_norm_rope = bool(
            (action_risk_cfg or {}).get(
                "triton_fused_k_norm_rope", False
            )
        ) and not self.training
        device_conditional_request = bool(
            self._device_conditional_runtime_enabled
            and self._device_conditional_runtime is not None
            and (action_risk_cfg or {}).get(
                "device_conditional_runtime", False
            )
            and is_single_decode
            and isinstance(past_key_values, StaticCache)
            and deepstack_visual_embeds is None
            and not self.training
            and not collect_trace
        )
        native_device_conditional_request = bool(
            self._native_device_conditional_runtime_enabled
            and self._native_device_conditional_runtime is not None
            and (native_cfg or {}).get("device_conditional_runtime", False)
            and is_single_decode
            and isinstance(past_key_values, StaticCache)
            and deepstack_visual_embeds is None
            and not self.training
            and not collect_trace
        )
        if device_conditional_request and native_device_conditional_request:
            raise RuntimeError(
                "action-risk and native device runtimes cannot be active together"
            )
        early_fallback_cfg = (action_risk_cfg or {}).get(
            "early_gate_fallback"
        )
        early_fallback_gate = (
            int(early_fallback_cfg["gate_block"])
            if early_fallback_cfg is not None
            else -1
        )
        early_fallback_targets = (
            tuple(early_fallback_cfg["target_blocks"])
            if early_fallback_cfg is not None
            else ()
        )
        early_replaced_gate_blocks = {
            self.joint_risk_gate_by_block[target]
            for target in early_fallback_targets
        }
        early_fallback_decided = False
        early_fallback_bypass = False
        ratio_fallback_cfg = (action_risk_cfg or {}).get(
            "gate1_ratio_fallback"
        )
        ratio_fallback_gate = (
            int(ratio_fallback_cfg["gate_block"])
            if ratio_fallback_cfg is not None
            else -1
        )
        ratio_fallback_sources = (
            tuple(ratio_fallback_cfg["source_blocks"])
            if ratio_fallback_cfg is not None
            else ()
        )
        ratio_fallback_targets = (
            tuple(ratio_fallback_cfg["target_blocks"])
            if ratio_fallback_cfg is not None
            else ()
        )
        ratio_replaced_gate_blocks = {
            self.joint_risk_gate_by_block[target]
            for target in ratio_fallback_targets
        }
        ratio_fallback_decided = False
        ratio_fallback_bypass = False
        profile_gate = (profile_cfg or {}).get("gate_block")
        profile_levels = (profile_cfg or {}).get("profile_levels", ())
        profile_decode_only = bool((profile_cfg or {}).get("decode_only", True))
        profile_warmup_decode_tokens = int(
            (profile_cfg or {}).get("warmup_decode_tokens", 0)
        )
        profile_warmup_active = (
            profile_cfg is not None
            and is_single_decode
            and self._risk_profile_decode_steps < profile_warmup_decode_tokens
        )
        if profile_warmup_active and self._skip_stats is not None:
            self._skip_stats["risk_profile_warmup_forwards"] += 1
        rnd_eligible = (random_cfg or {}).get("eligible_blocks")
        rnd_p = float((random_cfg or {}).get("p", 0.0) or 0.0)
        rnd_max_skips = (random_cfg or {}).get("max_skips")
        rnd_rng = (random_cfg or {}).get("rng")
        if random_cfg is not None and rnd_rng is None:
            rnd_rng = random.Random(int((random_cfg or {}).get("seed", 1234)))
            random_cfg["rng"] = rnd_rng
        for block_idx in range(self.num_blocks):
            block_layer_vectors: list[dict[str, torch.Tensor]] | None = (
                [] if collect_trace else None
            )
            # alpha is needed only for trace or dyn-skip; routed+corrected
            # always needed for the forward math.
            inference_action_risk_request = (
                action_risk_cfg is not None
                and action_risk_selector_mode == "block_local"
                and self.risk_surrogate is not None
                and block_idx in action_risk_thresholds
                and (
                    action_risk_eligible is None
                    or block_idx in action_risk_eligible
                )
                and (
                    action_risk_max_layers <= 0
                    or action_risk_layer_count < action_risk_max_layers
                )
                and (
                    not action_risk_decode_only
                    or incoming_cache_len > 0
                )
                and int(inputs_embeds.shape[1]) == 1
            )
            training_action_risk_request = (
                self._return_action_risks_flag
                and self.risk_surrogate is not None
                and block_idx in self.risk_surrogate_actions
            )
            inference_joint_action_risk_request = (
                action_risk_cfg is not None
                and action_risk_selector_mode == "joint_gate"
                and bool(self.joint_risk_surrogates)
                and block_idx in joint_gate_blocks
                and not (
                    (
                        early_fallback_decided
                        and early_fallback_bypass
                        and block_idx in early_replaced_gate_blocks
                    )
                    or (
                        ratio_fallback_decided
                        and ratio_fallback_bypass
                        and block_idx in ratio_replaced_gate_blocks
                    )
                )
                and (
                    not action_risk_decode_only
                    or incoming_cache_len > 0
                )
                and int(inputs_embeds.shape[1]) == 1
            )
            training_joint_action_risk_request = (
                self._return_joint_action_risks_flag
                and bool(self.joint_risk_surrogates)
                and block_idx in joint_gate_blocks
            )
            inference_early_fallback_request = (
                early_fallback_cfg is not None
                and self.early_joint_risk_surrogate is not None
                and block_idx == early_fallback_gate
                and (
                    not action_risk_decode_only
                    or incoming_cache_len > 0
                )
                and int(inputs_embeds.shape[1]) == 1
            )
            joint_action_risk_feature_request = (
                inference_joint_action_risk_request
                or training_joint_action_risk_request
                or inference_early_fallback_request
            )
            local_action_risk_feature_request = (
                inference_action_risk_request or training_action_risk_request
            )
            action_risk_feature_request = (
                local_action_risk_feature_request
                or joint_action_risk_feature_request
            )
            need_alpha = (
                self._return_alpha_flag
                or collect_trace
                or (dynamic_cfg is not None)
                or sticky_collecting
                or (profile_cfg is not None and block_idx == profile_gate)
                or action_risk_feature_request
            )
            inference_native_feature_request = (
                native_cfg is not None
                and block_idx in native_bounds_map
                and (native_eligible is None or block_idx in native_eligible)
                and (
                    native_max_skips is None
                    or native_skipped_count < native_max_skips
                )
                and (not native_decode_only or incoming_cache_len > 0)
                and int(inputs_embeds.shape[1]) == 1
            )
            fused_joint_native_request = (
                inference_joint_action_risk_request
                and bool(
                    (action_risk_cfg or {}).get(
                        "triton_fused_native_features", False
                    )
                )
                and not training_joint_action_risk_request
                and not inference_early_fallback_request
                and not self._return_native_features_flag
                and not inference_native_feature_request
                and not local_action_risk_feature_request
            )
            fused_native_device_request = (
                inference_native_feature_request
                and native_device_conditional_request
                and bool(
                    (native_cfg or {}).get(
                        "device_fused_native_features", False
                    )
                )
                and not self._return_native_features_flag
                and not local_action_risk_feature_request
                and not training_joint_action_risk_request
                and not inference_early_fallback_request
            )
            native_feature_request = (
                self._return_native_features_flag
                or (
                    inference_native_feature_request
                    and not fused_native_device_request
                )
                or local_action_risk_feature_request
                or training_joint_action_risk_request
                or inference_early_fallback_request
                or (
                    inference_joint_action_risk_request
                    and not fused_joint_native_request
                )
            )
            minimal_native_features = bool(
                inference_native_feature_request
                and not self._return_native_features_flag
                and not local_action_risk_feature_request
                and not training_joint_action_risk_request
                and not inference_early_fallback_request
                and not inference_joint_action_risk_request
                and set(
                    native_active_feature_names_by_block.get(block_idx, ())
                ).issubset({"w_recent", "residual_strength_factor"})
            )
            if (
                (
                    inference_native_feature_request
                    or inference_action_risk_request
                    or inference_joint_action_risk_request
                    or inference_early_fallback_request
                )
                and int(inputs_embeds.shape[0]) != 1
            ):
                raise ValueError(
                    "native token-level ReSkip currently requires batch size 1"
                )
            (
                block_input,
                alpha,
                routed,
                entropy,
                correction_ratio,
                native_features,
                fused_native_inputs,
            ) = self._compute_block_input(
                block_idx, prev_block, completed,
                collect_alpha=need_alpha,
                compute_entropy=compute_entropy_flag,
                compute_correction_ratio=self._return_correction_ratios_flag,
                compute_native_features=native_feature_request,
                return_fused_native_inputs=(
                    fused_joint_native_request
                    or fused_native_device_request
                ),
                minimal_native_features=minimal_native_features,
            )
            if correction_ratio is not None:
                correction_ratios.append(correction_ratio)
            if self._return_native_features_flag:
                native_features_by_block.append(native_features)
            if collect_trace:
                native_vectors_by_block.append(
                    None
                    if routed is None
                    else {
                        "route_delta": (
                            routed[:, -1, :] - prev_block[:, -1, :]
                        ).detach().clone(),
                        "attnres_update": (
                            block_input[:, -1, :] - prev_block[:, -1, :]
                        ).detach().clone(),
                    }
                )
            if entropy is not None:
                entropy_accum = entropy if entropy_accum is None else entropy_accum + entropy
            if alpha is not None and self._return_alpha_flag:
                alpha_list.append(alpha)
            if sticky_collecting and alpha is not None and alpha.shape[-1] >= 2:
                sticky_observations.append((block_idx, alpha[..., -1].float().mean()))

            action_risk_prediction = None
            if local_action_risk_feature_request:
                if alpha is None or native_features is None or self.risk_surrogate is None:
                    raise RuntimeError(
                        f"missing AttnRes features for action-risk block {block_idx}"
                    )
                score_only_inference = (
                    inference_action_risk_request
                    and not training_action_risk_request
                    and bool((action_risk_cfg or {}).get("compile_selector"))
                )
                if score_only_inference:
                    action_risk_prediction = {
                        "risk_score": self.risk_surrogate.predict_block_risk_score(
                            block_idx,
                            alpha,
                            native_features,
                            cache_position,
                            block_input,
                        )
                    }
                else:
                    action_risk_prediction = self.risk_surrogate.predict_block(
                        block_idx,
                        alpha,
                        native_features,
                        cache_position,
                        block_input=block_input,
                        detach_features=True,
                    )
            early_gate_predictions = None
            if inference_early_fallback_request:
                if alpha is None or native_features is None:
                    raise RuntimeError(
                        "missing AttnRes features for early gate fallback"
                    )
                if bool(
                    early_fallback_cfg.get("triton_fused_selector", False)
                ):
                    if self._triton_early_joint_risk_selector is None:
                        raise RuntimeError(
                            "early fallback Triton selector is not configured"
                        )
                    early_gate_predictions = {
                        target_block: {"risk_score": risk_score}
                        for target_block, risk_score in (
                            self._triton_early_joint_risk_selector.predict(
                                alpha,
                                native_features,
                                cache_position,
                                block_input,
                            ).items()
                        )
                    }
                else:
                    early_gate_predictions = (
                        self.early_joint_risk_surrogate.predict_all_from_gate(
                            alpha,
                            native_features,
                            cache_position,
                            block_input,
                            detach_features=True,
                        )
                    )
                if self._skip_stats is not None:
                    self._skip_stats["early_gate_candidate_evaluations"] += 1
            if joint_action_risk_feature_request:
                if (
                    alpha is None
                    or (
                        native_features is None
                        and fused_native_inputs is None
                    )
                    or str(block_idx) not in self.joint_risk_surrogates
                ):
                    raise RuntimeError(
                        "missing AttnRes features for joint action-risk gate"
                    )
                triton_score_only = (
                    inference_joint_action_risk_request
                    and not training_joint_action_risk_request
                    and bool(
                        (action_risk_cfg or {}).get(
                            "triton_fused_selector", False
                        )
                    )
                )
                packed_gate_scores = None
                packed_gate_selector = None
                if triton_score_only:
                    selector = self._triton_joint_risk_selectors[
                        block_idx
                    ]
                    use_packed_scores = triton_packed_host_scores
                    if fused_joint_native_request:
                        (
                            source_values,
                            fused_routed,
                            fused_previous,
                        ) = fused_native_inputs
                        if use_packed_scores:
                            packed_gate_scores = (
                                selector.predict_fused_native_packed(
                                    alpha,
                                    cache_position,
                                    block_input,
                                    source_values,
                                    fused_routed,
                                    fused_previous,
                                )
                            )
                            selected_predictions = selector.unpack_scores(
                                packed_gate_scores,
                                (*alpha.shape[:-1], 1),
                            )
                        else:
                            selected_predictions = selector.predict_fused_native(
                                alpha,
                                cache_position,
                                block_input,
                                source_values,
                                fused_routed,
                                fused_previous,
                            )
                    else:
                        if use_packed_scores:
                            packed_gate_scores = selector.predict_packed(
                                alpha,
                                native_features,
                                cache_position,
                                block_input,
                            )
                            selected_predictions = selector.unpack_scores(
                                packed_gate_scores,
                                (*alpha.shape[:-1], 1),
                            )
                        else:
                            selected_predictions = selector.predict(
                                alpha,
                                native_features,
                                cache_position,
                                block_input,
                            )
                    if packed_gate_scores is not None:
                        packed_gate_selector = selector
                    gate_predictions = {
                        target_block: {"risk_score": risk_score}
                        for target_block, risk_score in (
                            selected_predictions.items()
                        )
                    }
                else:
                    gate_predictions = (
                        self.joint_risk_surrogates[str(block_idx)]
                        .predict_all_from_gate(
                            alpha,
                            native_features,
                            cache_position,
                            block_input,
                            detach_features=True,
                        )
                    )
                joint_action_risk_predictions.update(gate_predictions)
                if self._return_joint_action_risks_flag:
                    for target_block, prediction in (
                        gate_predictions.items()
                    ):
                        joint_action_risks_by_block[target_block] = prediction

                if (
                    inference_joint_action_risk_request
                    and device_conditional_request
                ):
                    if packed_gate_scores is None:
                        raise RuntimeError(
                            "device conditional runtime requires packed scores"
                        )
                    self._device_conditional_runtime.update_gate(
                        block_idx, packed_gate_scores, cache_position
                    )

                if (
                    inference_joint_action_risk_request
                    and not device_conditional_request
                ):
                    score_tensors = []
                    score_specs = []
                    for target_block in sorted(action_risk_thresholds):
                        if (
                            self.joint_risk_gate_by_block.get(target_block)
                            != block_idx
                        ):
                            continue
                        if (
                            action_risk_eligible is not None
                            and target_block not in action_risk_eligible
                        ):
                            continue
                        prediction = gate_predictions.get(
                            target_block
                        )
                        if prediction is None:
                            raise RuntimeError(
                                f"joint gate omitted target block {target_block}"
                            )
                        actions = self.risk_surrogate_actions[target_block]
                        scores = prediction["risk_score"].reshape(
                            -1, len(actions)
                        )[0]
                        for action_position, action in enumerate(actions):
                            threshold = action_risk_thresholds[
                                target_block
                            ].get(action)
                            if threshold is None:
                                continue
                            score_tensors.append(scores[action_position])
                            score_specs.append(
                                (target_block, action, float(threshold))
                            )
                    early_score_tensors = []
                    if early_gate_predictions is not None:
                        for target_block in early_fallback_targets:
                            prediction = early_gate_predictions.get(
                                target_block
                            )
                            if prediction is None:
                                raise RuntimeError(
                                    "early fallback omitted a target block"
                                )
                            early_score_tensors.append(
                                prediction["risk_score"].reshape(-1)[0]
                            )
                    combined_score_tensors = (
                        score_tensors + early_score_tensors
                    )
                    packed_blocks = tuple(
                        spec[0] for spec in score_specs
                    )
                    direct_packed_transfer = (
                        packed_gate_scores is not None
                        and packed_gate_selector is not None
                        and not early_score_tensors
                        and packed_blocks
                        == tuple(packed_gate_selector.target_blocks)
                    )
                    if direct_packed_transfer:
                        combined_score_values = (
                            packed_gate_selector.copy_scores_to_host(
                                packed_gate_scores
                            )
                        )
                    else:
                        combined_score_values = (
                            torch.stack(combined_score_tensors)
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                            if combined_score_tensors
                            else []
                        )
                    if combined_score_tensors and self._skip_stats is not None:
                        self._skip_stats["joint_gate_score_transfers"] += 1
                    score_values = combined_score_values[: len(score_tensors)]
                    action_ids, layer_offsets = (
                        self._select_joint_action_risk_actions(
                            score_values,
                            score_specs,
                            max_layer_skips=action_risk_max_layers,
                            initial_layer_count=action_risk_layer_count,
                        )
                    )
                    joint_action_ids_by_block.update(action_ids)
                    joint_layer_offsets_by_block.update(layer_offsets)
                    if (
                        ratio_fallback_cfg is not None
                        and block_idx == ratio_fallback_gate
                    ):
                        ratio_by_block = {
                            target_block: float(score)
                            / max(float(threshold), 1.0e-12)
                            for score, (
                                target_block,
                                _,
                                threshold,
                            ) in zip(score_values, score_specs)
                            if target_block in ratio_fallback_sources
                        }
                        if set(ratio_by_block) != set(
                            ratio_fallback_sources
                        ):
                            raise RuntimeError(
                                "ratio fallback is missing a gate score"
                            )
                        max_ratio = max(ratio_by_block.values())
                        predict_all_skip = max_ratio <= ratio_fallback_cfg[
                            "all_skip_upper"
                        ]
                        predict_all_compute = max_ratio >= (
                            ratio_fallback_cfg["all_compute_lower"]
                        )
                        ratio_fallback_decided = True
                        ratio_fallback_bypass = (
                            predict_all_skip or predict_all_compute
                        )
                        predicted_action_count = 0
                        if predict_all_skip:
                            future_regular_layers = sum(
                                len(offsets)
                                for offsets in layer_offsets.values()
                            )
                            predicted_specs = []
                            for target_block in ratio_fallback_targets:
                                action = self.risk_surrogate_actions[
                                    target_block
                                ][0]
                                predicted_specs.append(
                                    (
                                        target_block,
                                        action,
                                        action_risk_thresholds[
                                            target_block
                                        ][action],
                                    )
                                )
                            predicted_action_ids, predicted_layer_offsets = (
                                self._select_joint_action_risk_actions(
                                    [0.0] * len(predicted_specs),
                                    predicted_specs,
                                    max_layer_skips=action_risk_max_layers,
                                    initial_layer_count=(
                                        action_risk_layer_count
                                        + future_regular_layers
                                    ),
                                )
                            )
                            joint_action_ids_by_block.update(
                                predicted_action_ids
                            )
                            joint_layer_offsets_by_block.update(
                                predicted_layer_offsets
                            )
                            predicted_action_count = sum(
                                len(actions)
                                for actions in predicted_action_ids.values()
                            )
                        if self._skip_stats is not None:
                            self._skip_stats[
                                "ratio_gate_candidate_evaluations"
                            ] += 1
                            if ratio_fallback_bypass:
                                self._skip_stats[
                                    "ratio_gate_bypass_forwards"
                                ] += 1
                                self._skip_stats[
                                    "ratio_gate_predicted_actions"
                                ] += predicted_action_count
                            else:
                                self._skip_stats[
                                    "ratio_gate_fallback_forwards"
                                ] += 1
                    if early_score_tensors:
                        early_values = combined_score_values[
                            len(score_tensors) :
                        ]
                        decisions: list[bool | None] = []
                        for target_block, score in zip(
                            early_fallback_targets, early_values
                        ):
                            if score <= early_fallback_cfg[
                                "skip_thresholds"
                            ][target_block]:
                                decisions.append(True)
                            elif score >= early_fallback_cfg[
                                "no_skip_thresholds"
                            ][target_block]:
                                decisions.append(False)
                            else:
                                decisions.append(None)
                        early_fallback_decided = True
                        early_fallback_bypass = all(
                            decision is not None for decision in decisions
                        )
                        if early_fallback_bypass:
                            future_regular_layers = sum(
                                len(offsets)
                                for offsets in layer_offsets.values()
                            )
                            early_selected_values = []
                            early_selected_specs = []
                            for target_block, score, decision in zip(
                                early_fallback_targets,
                                early_values,
                                decisions,
                            ):
                                if not decision:
                                    continue
                                action = self.risk_surrogate_actions[
                                    target_block
                                ][0]
                                early_selected_values.append(score)
                                early_selected_specs.append(
                                    (
                                        target_block,
                                        action,
                                        early_fallback_cfg[
                                            "skip_thresholds"
                                        ][target_block],
                                    )
                                )
                            early_action_ids, early_layer_offsets = (
                                self._select_joint_action_risk_actions(
                                    early_selected_values,
                                    early_selected_specs,
                                    max_layer_skips=action_risk_max_layers,
                                    initial_layer_count=(
                                        action_risk_layer_count
                                        + future_regular_layers
                                    ),
                                )
                            )
                            joint_action_ids_by_block.update(
                                early_action_ids
                            )
                            joint_layer_offsets_by_block.update(
                                early_layer_offsets
                            )
                            if self._skip_stats is not None:
                                self._skip_stats[
                                    "early_gate_bypass_forwards"
                                ] += 1
                                self._skip_stats[
                                    "early_gate_predicted_actions"
                                ] += len(early_selected_specs)
                        elif self._skip_stats is not None:
                            self._skip_stats[
                                "early_gate_fallback_forwards"
                            ] += 1
            if action_risk_selector_mode == "joint_gate":
                action_risk_prediction = joint_action_risk_predictions.get(
                    block_idx
                )
            if self._return_action_risks_flag:
                action_risks_by_block.append(action_risk_prediction)

            skip_requested = block_idx in self._active_skip_blocks
            # Dyn-skip decision: per-block τ, eligible mask, max_skips cap.
            # Only sync w_recent to CPU when the Python-side gating passes,
            # so retrofit-full (no dyn cfg) and non-eligible blocks pay zero
            # per-block sync.
            dynamic_skip_requested = False
            if (
                dynamic_cfg is not None
                and alpha is not None
                and alpha.shape[-1] >= 2
            ):
                thr = dyn_thr_map.get(block_idx)
                if (
                    thr is not None
                    and (dyn_eligible is None or block_idx in dyn_eligible)
                    and (dyn_max_skips is None or dyn_skipped_count < dyn_max_skips)
                ):
                    if bool((alpha[..., -1].mean().float() > float(thr)).item()):
                        dynamic_skip_requested = True
            native_token_skip_requested = False
            if inference_native_feature_request:
                if (
                    native_device_conditional_request
                    and fused_native_device_request
                ):
                    if alpha is None or fused_native_inputs is None:
                        raise RuntimeError(
                            "fused native device selector lacks router inputs"
                        )
                    (
                        source_values,
                        fused_routed,
                        fused_previous,
                    ) = fused_native_inputs
                    self._native_device_conditional_runtime.update_block_fused(
                        block_idx,
                        alpha,
                        source_values,
                        fused_routed,
                        fused_previous,
                        cache_position,
                    )
                elif native_device_conditional_request:
                    if native_features is None:
                        raise RuntimeError(
                            "native device selector lacks native features"
                        )
                    self._native_device_conditional_runtime.update_block(
                        block_idx, native_features, cache_position
                    )
                else:
                    if native_features is None:
                        raise RuntimeError(
                            "host native selector lacks native features"
                        )
                    upper_bounds = native_bounds_map[block_idx]
                    lower_bounds = native_lower_bounds_map[block_idx]
                    native_decision = torch.stack(
                        [
                            (
                                native_features[name].reshape(-1)[0]
                                < float(upper_bounds[name])
                            )
                            & (
                                native_features[name].reshape(-1)[0]
                                >= float(lower_bounds[name])
                            )
                            for name in NATIVE_TOKEN_SKIP_FEATURES
                        ]
                    ).all()
                    native_token_skip_requested = bool(native_decision.item())
            action_risk_action_ids: tuple[int, ...] = ()
            action_risk_layer_offsets: tuple[int, ...] = ()
            if action_risk_selector_mode == "joint_gate":
                action_risk_action_ids = joint_action_ids_by_block.get(
                    block_idx, ()
                )
                action_risk_layer_offsets = (
                    joint_layer_offsets_by_block.get(block_idx, ())
                )
            elif inference_action_risk_request:
                if action_risk_prediction is None or self.risk_surrogate is None:
                    raise RuntimeError("action-risk inference produced no prediction")
                offsets = self.risk_surrogate_actions[block_idx]
                single_action_spec = action_risk_single_action_specs.get(block_idx)
                if single_action_spec is not None:
                    (
                        action_position,
                        action,
                        threshold,
                        group_offsets,
                    ) = single_action_spec
                    score_value = float(
                        action_risk_prediction["risk_score"]
                        .reshape(-1, len(offsets))[0, action_position]
                        .detach()
                        .float()
                        .item()
                    )
                    within_budget = (
                        action_risk_max_layers <= 0
                        or action_risk_layer_count + len(group_offsets)
                        <= action_risk_max_layers
                    )
                    if score_value <= threshold and within_budget:
                        action_risk_action_ids = (action,)
                        action_risk_layer_offsets = group_offsets
                else:
                    score_values = (
                        action_risk_prediction["risk_score"]
                        .reshape(-1, len(offsets))[0]
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    )
                    thresholds = action_risk_thresholds[block_idx]
                    candidates = sorted(
                        (
                            (float(score), int(action))
                            for score, action in zip(score_values, offsets)
                            if action in thresholds
                            and float(score) <= float(thresholds[action])
                        ),
                        key=lambda item: (item[0], item[1]),
                    )
                    selected_actions = []
                    selected_offsets: set[int] = set()
                    for _, action in candidates:
                        group_offsets = set(
                            self.risk_surrogate_groups[block_idx][action]
                        )
                        new_offsets = group_offsets.difference(selected_offsets)
                        if (
                            action_risk_max_layers > 0
                            and action_risk_layer_count
                            + len(selected_offsets)
                            + len(new_offsets)
                            > action_risk_max_layers
                        ):
                            continue
                        selected_actions.append(action)
                        selected_offsets.update(group_offsets)
                    action_risk_action_ids = tuple(sorted(selected_actions))
                    action_risk_layer_offsets = tuple(sorted(selected_offsets))
            profile_score = None
            if (
                profile_cfg is not None
                and block_idx == profile_gate
                and alpha is not None
                and (not profile_decode_only or incoming_cache_len > 0)
                and int(inputs_embeds.shape[1]) == 1
                and not profile_warmup_active
            ):
                if int(inputs_embeds.shape[0]) != 1:
                    raise ValueError(
                        "risk-profile token ReSkip currently requires batch size 1"
                    )
                selected_profile = ()
                if profile_cfg["selector_mode"] == "linear_native":
                    feature_vector = native_alpha_feature_vector(alpha)
                    level_scores = []
                    for level in profile_levels:
                        linear_model = level["linear_model"]
                        cache_key = (
                            str(feature_vector.device),
                            linear_model["mean"],
                            linear_model["scale"],
                            linear_model["coefficient"],
                            linear_model["intercept"],
                        )
                        tensors = self._linear_selector_tensor_cache.get(cache_key)
                        if tensors is None:
                            tensors = (
                                torch.tensor(
                                    linear_model["mean"],
                                    device=feature_vector.device,
                                    dtype=torch.float32,
                                ),
                                torch.tensor(
                                    linear_model["scale"],
                                    device=feature_vector.device,
                                    dtype=torch.float32,
                                ),
                                torch.tensor(
                                    linear_model["coefficient"],
                                    device=feature_vector.device,
                                    dtype=torch.float32,
                                ),
                                torch.tensor(
                                    linear_model["intercept"],
                                    device=feature_vector.device,
                                    dtype=torch.float32,
                                ),
                            )
                            self._linear_selector_tensor_cache[cache_key] = tensors
                        mean, scale, coefficient, intercept = tensors
                        if feature_vector.numel() != coefficient.numel():
                            raise ValueError(
                                "linear native selector feature dimension mismatch"
                            )
                        logit = (
                            ((feature_vector - mean) / scale) * coefficient
                        ).sum() + intercept
                        level_scores.append(torch.sigmoid(logit))
                    # One synchronization transfers every action score.
                    profile_score_values = (
                        torch.stack(level_scores).detach().float().cpu().tolist()
                    )
                    for level, score_value in zip(
                        profile_levels, profile_score_values
                    ):
                        if score_value > level["score_threshold"]:
                            selected_profile = level["profile_blocks"]
                    profile_score_value = float(profile_score_values[0])
                else:
                    if profile_cfg["score_mode"] == "recent":
                        profile_score_tensor = alpha[..., -1].float().reshape(-1)[0]
                    else:
                        profile_score_tensor = routing_confidence_score(
                            alpha,
                            margin_weight=profile_cfg["margin_weight"],
                            entropy_weight=profile_cfg["entropy_weight"],
                        ).reshape(-1)[0]
                    # One device-to-host synchronization controls every action.
                    profile_score_value = float(profile_score_tensor.item())
                    for level in profile_levels:
                        if profile_score_value > level["score_threshold"]:
                            selected_profile = level["profile_blocks"]
                        else:
                            break
                if selected_profile:
                    profile_active_blocks.update(selected_profile)
                    if self._skip_stats is not None:
                        self._skip_stats["risk_profile_selections"] += 1
                if collect_trace:
                    profile_score = profile_score_value
            risk_profile_skip_requested = block_idx in profile_active_blocks
            random_skip_requested = False
            if (
                random_cfg is not None
                and (rnd_eligible is None or block_idx in rnd_eligible)
                and (rnd_max_skips is None or random_skipped_count < rnd_max_skips)
                and rnd_rng is not None
                and rnd_rng.random() < rnd_p
            ):
                random_skip_requested = True
            sticky_skip_requested = block_idx in sticky_active_blocks

            deepstack_sensitive = self._block_contains_deepstack_layers(block_idx, deepstack_visual_embeds)
            native_action = (
                native_action_by_block.get(block_idx, "block")
                if native_token_skip_requested
                else None
            )
            non_native_full_skip_requested = (
                skip_requested
                or dynamic_skip_requested
                or risk_profile_skip_requested
                or random_skip_requested
                or sticky_skip_requested
            )
            should_skip = (
                (
                    non_native_full_skip_requested
                    or (
                        native_token_skip_requested
                        and native_action == "block"
                    )
                )
                and block_idx in self.skippable_block_set
                and not deepstack_sensitive
            )
            should_skip_mlp = (
                native_token_skip_requested
                and native_action == "mlp"
                and not non_native_full_skip_requested
                and block_idx in self.skippable_block_set
                and not deepstack_sensitive
            )
            should_skip_layers = (
                (
                    (
                        native_token_skip_requested
                        and native_action == "layer"
                    )
                    or bool(action_risk_layer_offsets)
                )
                and not non_native_full_skip_requested
                and block_idx in self.skippable_block_set
                and not deepstack_sensitive
            )
            mlp_skip_offsets = (
                native_mlp_offsets_by_block.get(
                    block_idx, tuple(range(self.layers_per_block))
                )
                if should_skip_mlp
                else ()
            )
            layer_skip_offsets = (
                (
                    action_risk_layer_offsets
                    if action_risk_layer_offsets
                    else native_layer_offsets_by_block.get(
                        block_idx, tuple(range(self.layers_per_block))
                    )
                )
                if should_skip_layers
                else ()
            )
            native_action_executed = (
                native_token_skip_requested
                and (should_skip or should_skip_mlp or should_skip_layers)
            )
            if dynamic_skip_requested and should_skip:
                dyn_skipped_count += 1
            if native_action_executed:
                native_skipped_count += 1
                native_profile_active_blocks.add(block_idx)
            action_risk_action_executed = bool(
                action_risk_layer_offsets and should_skip_layers
            )
            if action_risk_action_executed:
                action_risk_layer_count += len(layer_skip_offsets)
            if random_skip_requested and should_skip:
                random_skipped_count += 1
            if self._skip_stats is not None:
                if skip_requested:
                    self._skip_stats["static_skip_requests"] += 1
                    self._skip_stats["per_block_static_requests"][block_idx] += 1
                if dynamic_skip_requested:
                    self._skip_stats["dynamic_skip_requests"] += 1
                    self._skip_stats["per_block_dynamic_requests"][block_idx] += 1
                if native_token_skip_requested:
                    self._skip_stats["native_token_skip_requests"] += 1
                    self._skip_stats["per_block_native_token_requests"][block_idx] += 1
                if action_risk_action_executed:
                    self._skip_stats["action_risk_skip_requests"] += 1
                    self._skip_stats["action_risk_layer_requests"] += len(
                        layer_skip_offsets
                    )
                    self._skip_stats["per_block_action_risk_requests"][block_idx] += 1
                    self._skip_stats["per_block_action_risk_layer_requests"][
                        block_idx
                    ] += len(layer_skip_offsets)
                if risk_profile_skip_requested:
                    self._skip_stats["risk_profile_skip_requests"] += 1
                    self._skip_stats["per_block_risk_profile_requests"][block_idx] += 1
                if random_skip_requested:
                    self._skip_stats["random_skip_requests"] += 1
                    self._skip_stats["per_block_random_requests"][block_idx] += 1
                if sticky_skip_requested:
                    self._skip_stats["sticky_skip_requests"] += 1
                    self._skip_stats["per_block_sticky_requests"][block_idx] += 1
                if should_skip:
                    self._skip_stats["skip_events"] += 1
                    self._skip_stats["per_block_skips"][block_idx] += 1
                if should_skip_mlp:
                    self._skip_stats["partial_mlp_skip_events"] += 1
                    self._skip_stats["per_block_partial_mlp_skips"][block_idx] += 1
                    self._skip_stats["partial_mlp_layer_skip_events"] += len(
                        mlp_skip_offsets
                    )
                    self._skip_stats["per_block_partial_mlp_layer_skips"][
                        block_idx
                    ] += len(mlp_skip_offsets)
                if should_skip_layers:
                    self._skip_stats["partial_layer_skip_events"] += 1
                    self._skip_stats["per_block_partial_layer_skips"][block_idx] += 1
                    self._skip_stats["partial_decoder_layer_skip_events"] += len(
                        layer_skip_offsets
                    )
                    self._skip_stats["per_block_partial_decoder_layer_skips"][
                        block_idx
                    ] += len(layer_skip_offsets)

            if block_input is None:
                block_input = prev_block
                if collect_trace:
                    surrogate_outputs.append(None)
            elif collect_trace:
                surrogate_outputs.append(block_input)

            device_conditional_block = bool(
                device_conditional_request
                and action_risk_eligible is not None
                and block_idx in action_risk_eligible
                and not deepstack_sensitive
            )
            native_device_conditional_block = bool(
                native_device_conditional_request
                and not deepstack_sensitive
            )
            if native_device_conditional_block:
                h = block_input
                conditional_offsets = native_layer_offsets_by_block.get(
                    block_idx, ()
                )
                for layer_offset in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    if layer_offset in conditional_offsets:
                        h = self._native_device_conditional_runtime.run_layer(
                            block_idx,
                            layer_offset,
                            layer,
                            h,
                            past_key_values,
                            attention_mask,
                            text_position_ids,
                            position_embeddings,
                            cache_position,
                            kwargs,
                            self._update_layer_kv_only,
                        )
                    else:
                        h = self._native_device_conditional_runtime.run_full_layer(
                            layer,
                            h,
                            past_key_values,
                            attention_mask,
                            text_position_ids,
                            position_embeddings,
                            cache_position,
                            kwargs,
                            self._update_layer_kv_only,
                        )
                    layer_counter += 1
            elif device_conditional_block:
                h = block_input
                for layer_offset in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    h = self._device_conditional_runtime.run_layer(
                        block_idx,
                        layer_offset,
                        layer,
                        h,
                        past_key_values,
                        attention_mask,
                        text_position_ids,
                        position_embeddings,
                        cache_position,
                        kwargs,
                        self._update_layer_kv_only,
                    )
                    layer_counter += 1
            elif should_skip:
                h = block_input
                # Maintain KV-cache consistency for the skipped layers so
                # subsequent decode steps see a full-length per-layer cache.
                # We run only the layer's K/V path (input_layernorm →
                # k_proj/v_proj → k_norm → RoPE → cache.update) on the same
                # block_input; Q/attention/output-proj/MLP are fully skipped.
                # Correctness intuition: a skipped block is treated as an
                # identity transformation, so every layer inside it sees the
                # same block_input as its hidden state input.
                if past_key_values is not None and not sticky_skip_requested:
                    for l_off in range(self.layers_per_block):
                        layer = text_model.layers[layer_counter + l_off]
                        self._update_layer_kv_only(
                            layer,
                            h,
                            past_key_values,
                            position_embeddings,
                            cache_position,
                            key_only_rope=kv_only_key_rope,
                            triton_fused_k_norm_rope=(
                                triton_fused_k_norm_rope
                            ),
                        )
                elif past_key_values is not None and self._skip_stats is not None:
                    self._skip_stats["sticky_kv_projections_avoided"] += self.layers_per_block
                layer_counter += self.layers_per_block
            elif should_skip_layers:
                h = block_input
                for layer_offset in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    if layer_offset in layer_skip_offsets:
                        if past_key_values is not None:
                            self._update_layer_kv_only(
                                layer,
                                h,
                                past_key_values,
                                position_embeddings,
                                cache_position,
                                key_only_rope=kv_only_key_rope,
                                triton_fused_k_norm_rope=(
                                    triton_fused_k_norm_rope
                                ),
                            )
                    else:
                        h = layer(
                            h,
                            attention_mask=attention_mask,
                            position_ids=text_position_ids,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                        if isinstance(h, tuple):
                            h = h[0]
                    if (
                        deepstack_visual_embeds is not None
                        and layer_counter < len(deepstack_visual_embeds)
                    ):
                        h = text_model._deepstack_process(
                            h, visual_pos_masks, deepstack_visual_embeds[layer_counter]
                        )
                    layer_counter += 1
            elif should_skip_mlp:
                # Keep exact causal self-attention and KV-cache updates, but
                # bypass the MLP residual branch in every layer of this block.
                # AttnRes still supplies the block input and the decision is
                # recomputed independently for each cached-decode token.
                h = block_input
                for layer_offset in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    residual = h
                    normed = layer.input_layernorm(h)
                    attention_output, _ = layer.self_attn(
                        hidden_states=normed,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    h = residual + attention_output
                    if layer_offset not in mlp_skip_offsets:
                        residual = h
                        h = residual + layer.mlp(
                            layer.post_attention_layernorm(h)
                        )
                    if (
                        deepstack_visual_embeds is not None
                        and layer_counter < len(deepstack_visual_embeds)
                    ):
                        h = text_model._deepstack_process(
                            h, visual_pos_masks, deepstack_visual_embeds[layer_counter]
                        )
                    layer_counter += 1
            else:
                h = block_input
                dense_token_skip_mask = (self._token_skip_masks or {}).get(block_idx)
                dense_layer_skip_masks = (
                    self._token_layer_skip_masks or {}
                ).get(block_idx, {})
                for layer_offset in range(self.layers_per_block):
                    layer = text_model.layers[layer_counter]
                    layer_input = h
                    layer_output = layer(
                        h,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )
                    if isinstance(layer_output, tuple):
                        layer_output = layer_output[0]
                    if block_layer_vectors is not None:
                        block_layer_vectors.append(
                            {
                                "input": layer_input[:, -1, :].detach().clone(),
                                "residual": (
                                    layer_output[:, -1, :]
                                    - layer_input[:, -1, :]
                                )
                                .detach()
                                .clone(),
                            }
                        )
                    dense_layer_skip_mask = dense_layer_skip_masks.get(layer_offset)
                    h = (
                        torch.where(
                            dense_layer_skip_mask[..., None],
                            layer_input,
                            layer_output,
                        )
                        if dense_layer_skip_mask is not None
                        else layer_output
                    )
                    if (
                        deepstack_visual_embeds is not None
                        and layer_counter < len(deepstack_visual_embeds)
                    ):
                        h = text_model._deepstack_process(
                            h, visual_pos_masks, deepstack_visual_embeds[layer_counter]
                        )
                    if dense_token_skip_mask is not None:
                        h = torch.where(
                            dense_token_skip_mask[..., None], block_input, h
                        )
                    layer_counter += 1

            if collect_trace:
                layer_vectors_by_block.append(block_layer_vectors)

            if (
                device_conditional_block
                and self.recovery_mode == "block"
                and block_idx in self.recovery_blocks
            ):
                h = self._device_conditional_runtime.run_recovery(
                    block_idx,
                    self.recovery_adapters[
                        self._recovery_block_key(block_idx)
                    ],
                    h,
                )

            if (
                self.recovery_mode == "surrogate"
                and block_idx in self.recovery_blocks
            ):
                dense_surrogate_mask = (self._token_skip_masks or {}).get(block_idx)
                if should_skip or dense_surrogate_mask is not None:
                    surrogate_routed, _ = self.recovery_router.route(
                        block_idx + 1, completed
                    )
                    surrogate_delta = surrogate_routed - prev_block
                    surrogate = prev_block + self.recovery_adapters[
                        self._recovery_block_key(block_idx)
                    ](surrogate_delta)
                    if should_skip:
                        h = surrogate
                    else:
                        h = torch.where(
                            dense_surrogate_mask[..., None], surrogate, h
                        )
            elif self.recovery_mode == "block" and block_idx in self.recovery_blocks:
                dense_block_recovery_mask = (self._token_skip_masks or {}).get(
                    block_idx
                )
                dense_layer_recovery_mask = None
                for layer_mask in (
                    (self._token_layer_skip_masks or {})
                    .get(block_idx, {})
                    .values()
                ):
                    dense_layer_recovery_mask = (
                        layer_mask
                        if dense_layer_recovery_mask is None
                        else dense_layer_recovery_mask | layer_mask
                    )
                dense_recovery_mask = dense_block_recovery_mask
                if dense_layer_recovery_mask is not None:
                    dense_recovery_mask = (
                        dense_layer_recovery_mask
                        if dense_recovery_mask is None
                        else dense_recovery_mask | dense_layer_recovery_mask
                    )
                whole_token_recovery = (
                    should_skip or should_skip_mlp or should_skip_layers
                )
                if whole_token_recovery or dense_recovery_mask is not None:
                    recovery_adapter = self.recovery_adapters[
                        self._recovery_block_key(block_idx)
                    ]
                    recovered = h + recovery_adapter(_rms_norm(h))
                    if whole_token_recovery:
                        h = recovered
                    else:
                        h = torch.where(
                            dense_recovery_mask[..., None], recovered, h
                        )
            elif self.recovery_mode == "profile":
                for recovery_profile in self.recovery_profiles:
                    if block_idx != recovery_profile[-1]:
                        continue
                    dense_recovery_mask = (self._token_recovery_masks or {}).get(
                        recovery_profile
                    )
                    whole_token_recovery = (
                        static_recovery_profile == recovery_profile
                        or tuple(sorted(profile_active_blocks)) == recovery_profile
                        or tuple(sorted(native_profile_active_blocks))
                        == recovery_profile
                    )
                    if not whole_token_recovery and dense_recovery_mask is None:
                        continue
                    recovery_adapter = self.recovery_adapters[
                        self._recovery_profile_key(recovery_profile)
                    ]
                    recovered = h + recovery_adapter(_rms_norm(h))
                    if whole_token_recovery:
                        h = recovered
                    else:
                        h = torch.where(
                            dense_recovery_mask[..., None], recovered, h
                        )

            prev_block = h
            completed.append(h)
            if self._collect_block_states:
                block_inputs.append(block_input)
                block_outputs.append(h)
            if collect_trace:
                # All per-block sync / bookkeeping gated here so plain
                # inference does not pay 3 CUDA syncs + 2 kernel launches
                # per block. w_recent / top_source / gamma are only of
                # interest to calibration + analysis callers.
                if gamma_cpu is None:
                    gamma_cpu = self.gamma.detach().float().cpu().tolist()  # one sync
                if alpha is not None and alpha.shape[-1] >= 2:
                    w_recent_val = float(alpha[..., -1].float().mean().item())
                    top_source = int(alpha.float().mean(dim=(0, 1)).argmax().item())
                else:
                    w_recent_val = None
                    top_source = None
                native_feature_values = None
                if native_features is not None:
                    feature_names = NATIVE_TOKEN_SKIP_FEATURES
                    feature_values = torch.stack(
                        [
                            native_features[name].reshape(-1)[0]
                            for name in feature_names
                        ]
                    ).detach().float().cpu().tolist()
                    native_feature_values = dict(zip(feature_names, feature_values))
                skip_trace.append(
                    {
                        "block_idx": block_idx,
                        "used_attnres": block_idx in self.skippable_block_set,
                        "skipped": should_skip,
                        "partial_mlp_skipped": should_skip_mlp,
                        "partial_mlp_layer_offsets": list(mlp_skip_offsets),
                        "partial_layer_skipped": should_skip_layers,
                        "partial_decoder_layer_offsets": list(layer_skip_offsets),
                        "native_action": native_action,
                        "skip_requested": skip_requested,
                        "dynamic_skip_requested": dynamic_skip_requested,
                        "native_token_skip_requested": native_token_skip_requested,
                        "action_risk_skip_requested": action_risk_action_executed,
                        "action_risk_action_ids": list(action_risk_action_ids),
                        "action_risk_layer_offsets": list(
                            action_risk_layer_offsets
                        ),
                        "action_risk_scores": (
                            action_risk_prediction["risk_score"]
                            .reshape(-1)
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                            if action_risk_prediction is not None
                            else None
                        ),
                        "risk_profile_skip_requested": risk_profile_skip_requested,
                        "risk_profile_score": profile_score,
                        "random_skip_requested": random_skip_requested,
                        "sticky_skip_requested": sticky_skip_requested,
                        "w_recent": w_recent_val,
                        "deepstack_sensitive": deepstack_sensitive,
                        "gamma": gamma_cpu[block_idx],
                        "top_source": top_source,
                        "native_features": native_feature_values,
                    }
                )

        self._finish_sticky_observation(sticky_observations)
        if self._risk_profile_skip_config is not None and is_single_decode:
            self._risk_profile_decode_steps += 1
        hidden_states = text_model.norm(completed[-1])
        self._fwd_alpha_list = alpha_list if self._return_alpha_flag else None
        self._fwd_skip_trace = skip_trace if collect_trace else None
        if self._routing_dump_path and skip_trace:
            if self._routing_dump_limit <= 0 or self._routing_dump_count < self._routing_dump_limit:
                record = {
                    "idx": self._routing_dump_count,
                    "seq_len": int(hidden_states.shape[1]) if hidden_states.ndim >= 2 else None,
                    "w_recents": {
                        str(item["block_idx"]): item.get("w_recent")
                        for item in skip_trace
                        if item.get("w_recent") is not None
                    },
                    "native_features": {
                        str(item["block_idx"]): item["native_features"]
                        for item in skip_trace
                        if item.get("native_features") is not None
                    },
                    "skipped": [item["block_idx"] for item in skip_trace if item.get("skipped")],
                    "dynamic_skip_requested": [
                        item["block_idx"] for item in skip_trace if item.get("dynamic_skip_requested")
                    ],
                    "random_skip_requested": [
                        item["block_idx"] for item in skip_trace if item.get("random_skip_requested")
                    ],
                }
                with open(self._routing_dump_path, "a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                self._routing_dump_count += 1
        self._fwd_block_inputs = block_inputs if self._collect_block_states else None
        self._fwd_block_outputs = block_outputs if self._collect_block_states else None
        self._fwd_surrogate_outputs = surrogate_outputs if self._collect_block_states else None
        self._fwd_entropy = entropy_accum
        self._fwd_correction_ratios = (
            correction_ratios if self._return_correction_ratios_flag else None
        )
        self._fwd_native_features_by_block = (
            native_features_by_block
            if self._return_native_features_flag
            else None
        )
        self._fwd_native_vectors_by_block = (
            native_vectors_by_block
            if collect_trace
            else None
        )
        self._fwd_layer_vectors_by_block = (
            layer_vectors_by_block
            if collect_trace
            else None
        )
        self._fwd_action_risks_by_block = (
            action_risks_by_block
            if self._return_action_risks_flag
            else None
        )
        self._fwd_joint_action_risks_by_block = (
            joint_action_risks_by_block
            if self._return_joint_action_risks_flag
            else None
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        return_alpha: bool = False,
        return_block_states: bool = False,
        return_correction_ratios: bool = False,
        return_native_features: bool = False,
        return_action_risks: bool = False,
        return_joint_action_risks: bool = False,
        skip_block_indices: Iterable[int] | None = None,
        dynamic_skip_config: dict | None = None,
        native_token_skip_config: dict | None = None,
        action_risk_skip_config: dict | None = None,
        risk_profile_skip_config: dict | None = None,
        token_skip_masks: dict[int, torch.Tensor] | None = None,
        token_layer_skip_masks: (
            dict[int, dict[int, torch.Tensor]] | None
        ) = None,
        token_recovery_masks: (
            dict[tuple[int, ...], torch.Tensor] | None
        ) = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Qwen3VLAttnResRetrofitOutput:
        """dynamic_skip_config: dict with keys:
            thresholds: {block_idx: float τ_n}  — from phase-1 w_recent calibration
            eligible_blocks: set of int | None  — if None, all skippable allowed
            max_skips: int | None  — cap on blocks skipped per input

        native_token_skip_config: training-free AttnRes consistency selector:
            feature_upper_bounds: {block_idx: {source_dispersion,
                route_novelty, correction_ratio}}
            eligible_blocks: set of int | None
            max_skips: int | None — cap per token, never held across tokens
            decode_only: bool — true by default; prefill remains full depth

        risk_profile_skip_config: one current-token routing decision controls a
            multi-block profile. This amortises the mandatory device/host
            decision over multiple skipped blocks while remaining token-level.

        action_risk_skip_config: current-token per-layer thresholds for the
            jointly trained AttnRes risk surrogate. Decisions are made when
            each block is reached and cannot inspect later-block features.

        token_skip_masks: dense training-time simulation of native token-level
            skips. Maps block index to a bool [B,T] mask. A skipped token is
            reset to the block input after every inner layer while active
            tokens still attend to its K/V state. If recovery profiles are
            configured, an exact per-token profile match activates only that
            profile's post-skip recovery adapter.

        token_layer_skip_masks: exact teacher-forcing analogue of the native
            decoder-layer micro-action. Maps block index and inner-layer
            offset to bool [B,T] masks. The skipped token keeps the layer input
            while the frozen layer is still evaluated during training so its
            causal K/V contribution to later tokens remains exact.

        token_recovery_masks: explicit bool [B,T] activation masks for joint
            recovery profiles. This decouples the recovery action from a
            full-block training mask when the student uses micro-layer skips.

        return_native_features: expose source dispersion, route novelty, and
            correction ratio as differentiable [B,T] tensors for every
            AttnRes block. This is used by the single-stage adaptation trainer
            to construct native token masks without an auxiliary selector.

        use_cache: passes through to base_model. Skipped blocks still update
            each inner layer's KV cache by running only the cheap K/V
            projection path on block_input (treating the skipped block as an
            identity for its output), so subsequent decode steps can attend
            over a consistent-length cache.
        """
        self._fwd_alpha_list = None
        self._fwd_skip_trace = None
        self._fwd_block_inputs = None
        self._fwd_block_outputs = None
        self._fwd_surrogate_outputs = None
        self._fwd_entropy = None
        self._fwd_correction_ratios = None
        self._fwd_native_features_by_block = None
        self._fwd_native_vectors_by_block = None
        self._fwd_layer_vectors_by_block = None
        self._fwd_action_risks_by_block = None
        self._fwd_joint_action_risks_by_block = None
        self._token_recovery_masks = None
        self._active_skip_blocks = set(int(x) for x in (skip_block_indices or []))
        self._dynamic_skip_config = dynamic_skip_config
        if native_token_skip_config is not None and action_risk_skip_config is not None:
            raise ValueError(
                "native-token and action-risk selectors cannot be active together"
            )
        self.configure_native_token_skip(native_token_skip_config)
        self.configure_action_risk_skip(action_risk_skip_config)
        self.configure_risk_profile_skip(risk_profile_skip_config)
        normalized_masks: dict[int, torch.Tensor] = {}
        for raw_block, raw_mask in (token_skip_masks or {}).items():
            block_idx = int(raw_block)
            if block_idx not in self.skippable_block_set:
                raise ValueError(f"token skip mask targets non-skippable block {block_idx}")
            if input_ids is None:
                raise ValueError("token skip masks currently require input_ids")
            mask = raw_mask.to(device=input_ids.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    f"token skip mask for block {block_idx} has shape {tuple(mask.shape)}; "
                    f"expected {tuple(input_ids.shape)}"
                )
            normalized_masks[block_idx] = mask
        self._token_skip_masks = normalized_masks
        normalized_layer_masks: dict[int, dict[int, torch.Tensor]] = {}
        for raw_block, raw_offsets in (token_layer_skip_masks or {}).items():
            block_idx = int(raw_block)
            if block_idx not in self.skippable_block_set:
                raise ValueError(
                    f"token layer skip mask targets non-skippable block {block_idx}"
                )
            if block_idx in normalized_masks:
                raise ValueError(
                    f"block {block_idx} cannot have both full-block and layer masks"
                )
            if input_ids is None:
                raise ValueError("token layer skip masks currently require input_ids")
            block_masks: dict[int, torch.Tensor] = {}
            for raw_offset, raw_mask in raw_offsets.items():
                layer_offset = int(raw_offset)
                if layer_offset < 0 or layer_offset >= self.layers_per_block:
                    raise ValueError(
                        f"token layer offset {layer_offset} outside [0, "
                        f"{self.layers_per_block - 1}] for block {block_idx}"
                    )
                mask = raw_mask.to(device=input_ids.device, dtype=torch.bool)
                if tuple(mask.shape) != tuple(input_ids.shape):
                    raise ValueError(
                        f"token layer skip mask for block {block_idx}, offset "
                        f"{layer_offset} has shape {tuple(mask.shape)}; expected "
                        f"{tuple(input_ids.shape)}"
                    )
                block_masks[layer_offset] = mask
            if not block_masks:
                raise ValueError(
                    f"token layer skip masks for block {block_idx} are empty"
                )
            normalized_layer_masks[block_idx] = block_masks
        self._token_layer_skip_masks = normalized_layer_masks
        if (
            self.recovery_mode == "profile"
            and self.recovery_profiles
            and normalized_masks
        ):
            recovery_union = set().union(
                *(set(profile) for profile in self.recovery_profiles)
            )
            recovery_masks: dict[tuple[int, ...], torch.Tensor] = {}
            for profile in self.recovery_profiles:
                exact_action_mask = torch.ones_like(input_ids, dtype=torch.bool)
                profile_set = set(profile)
                for block_idx in recovery_union:
                    block_mask = normalized_masks.get(block_idx)
                    if block_idx in profile_set:
                        if block_mask is None:
                            exact_action_mask = torch.zeros_like(
                                exact_action_mask, dtype=torch.bool
                            )
                            break
                        exact_action_mask = exact_action_mask & block_mask
                    elif block_mask is not None:
                        exact_action_mask = exact_action_mask & ~block_mask
                recovery_masks[profile] = exact_action_mask
            self._token_recovery_masks = recovery_masks
        if token_recovery_masks:
            if self.recovery_mode != "profile":
                raise ValueError(
                    "explicit token recovery masks require recovery_mode='profile'"
                )
            explicit_recovery_masks: dict[tuple[int, ...], torch.Tensor] = {}
            known_profiles = set(self.recovery_profiles)
            for raw_profile, raw_mask in token_recovery_masks.items():
                profile = tuple(sorted(set(int(block) for block in raw_profile)))
                if profile not in known_profiles:
                    raise ValueError(f"unknown recovery profile {profile}")
                if input_ids is None:
                    raise ValueError("token recovery masks currently require input_ids")
                mask = raw_mask.to(device=input_ids.device, dtype=torch.bool)
                if tuple(mask.shape) != tuple(input_ids.shape):
                    raise ValueError(
                        f"token recovery mask for profile {profile} has shape "
                        f"{tuple(mask.shape)}; expected {tuple(input_ids.shape)}"
                    )
                explicit_recovery_masks[profile] = mask
            self._token_recovery_masks = explicit_recovery_masks
        self._random_skip_config = None
        self._collect_block_states = return_block_states
        self._return_alpha_flag = return_alpha
        self._return_correction_ratios_flag = return_correction_ratios
        self._return_native_features_flag = return_native_features
        self._return_action_risks_flag = return_action_risks
        self._return_joint_action_risks_flag = return_joint_action_risks

        out = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            use_cache=use_cache,
            **kwargs,
        )
        logits = out.logits

        loss = None
        if labels is not None:
            shifted = torch.cat(
                [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
            )
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), shifted.view(-1), ignore_index=-100
            )

        return Qwen3VLAttnResRetrofitOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=getattr(out, "hidden_states", None),
            alpha_list=self._fwd_alpha_list if return_alpha else None,
            skip_trace=self._fwd_skip_trace,
            block_inputs=self._fwd_block_inputs,
            block_outputs=self._fwd_block_outputs,
            surrogate_outputs=self._fwd_surrogate_outputs,
            entropy_penalty=self._fwd_entropy,
            correction_ratios=(
                self._fwd_correction_ratios if return_correction_ratios else None
            ),
            native_features_by_block=(
                self._fwd_native_features_by_block
                if return_native_features
                else None
            ),
            action_risks_by_block=(
                self._fwd_action_risks_by_block
                if return_action_risks
                else None
            ),
            joint_action_risks_by_block=(
                self._fwd_joint_action_risks_by_block
                if return_joint_action_risks
                else None
            ),
        )

    def freeze_vision(self):
        for p in self.base_model.model.visual.parameters():
            p.requires_grad = False

    def move_retrofit_modules(self, device=None, dtype=None):
        """Move only newly introduced parameters, leaving the base untouched.

        The pretrained model can intentionally retain selected FP32 buffers
        (notably rotary-position state) even when its weights are loaded in
        BF16. Calling ``wrapper.to(dtype=...)`` recursively would recast those
        buffers and break identity at gamma=0.  All callers should use this
        method after the already-placed base model is wrapped.
        """
        self.router.to(device=device, dtype=dtype)
        self.adapters.to(device=device, dtype=dtype)
        self.recovery_adapters.to(device=device, dtype=dtype)
        if self.risk_surrogate is not None:
            # Risk features and calibration thresholds are FP32. Keeping this
            # tiny module in FP32 avoids dtype casts inside every decode block.
            self.risk_surrogate.to(device=device, dtype=torch.float32)
        self.joint_risk_surrogates.to(
            device=device, dtype=torch.float32
        )
        if self.recovery_router is not None:
            self.recovery_router.to(device=device, dtype=dtype)
        with torch.no_grad():
            self.gamma.data = self.gamma.data.to(device=device, dtype=dtype)
        return self

    def freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

    def retrofit_parameters(self):
        return (
            list(self.router.parameters())
            + list(self.adapters.parameters())
            + list(self.recovery_adapters.parameters())
            + (
                list(self.recovery_router.parameters())
                if self.recovery_router is not None
                else []
            )
            + (
                list(self.risk_surrogate.parameters())
                if self.risk_surrogate is not None
                else []
            )
            + (
                list(self.joint_risk_surrogates.parameters())
            )
            + [self.gamma]
        )

    def late_block_parameters(self):
        params = []
        for block_idx in self.skippable_blocks:
            start = block_idx * self.layers_per_block
            end = start + self.layers_per_block
            for layer_idx in range(start, end):
                params.extend(list(self.text_layers[layer_idx].parameters()))
        return params

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
