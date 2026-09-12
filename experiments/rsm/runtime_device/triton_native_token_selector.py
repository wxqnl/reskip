from __future__ import annotations

import torch
import triton
import triton.language as tl


NATIVE_FEATURE_ORDER = (
    "source_dispersion",
    "route_novelty",
    "correction_ratio",
    "delta_path_ratio",
    "delta_cancellation_ratio",
    "recent_delta_ratio",
    "delta_trend_ratio",
    "recent_delta_cosine",
    "w_recent",
    "residual_strength_factor",
)


@triton.jit
def _native_action_mask_kernel(
    source_dispersion_ptr,
    route_novelty_ptr,
    correction_ratio_ptr,
    delta_path_ratio_ptr,
    delta_cancellation_ratio_ptr,
    recent_delta_ratio_ptr,
    delta_trend_ratio_ptr,
    recent_delta_cosine_ptr,
    w_recent_ptr,
    residual_strength_factor_ptr,
    lower_bounds_ptr,
    upper_bounds_ptr,
    layer_masks_ptr,
    block_masks_ptr,
    cache_position_ptr,
    history_layers_ptr,
    history_blocks_ptr,
    history_visited_ptr,
    block_idx: tl.constexpr,
    layer_bits: tl.constexpr,
    max_history: tl.constexpr,
):
    bound_base = block_idx * 10
    decision = (
        (tl.load(source_dispersion_ptr).to(tl.float32)
         >= tl.load(lower_bounds_ptr + bound_base + 0))
        & (tl.load(source_dispersion_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 0))
        & (tl.load(route_novelty_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 1))
        & (tl.load(route_novelty_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 1))
        & (tl.load(correction_ratio_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 2))
        & (tl.load(correction_ratio_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 2))
        & (tl.load(delta_path_ratio_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 3))
        & (tl.load(delta_path_ratio_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 3))
        & (tl.load(delta_cancellation_ratio_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 4))
        & (tl.load(delta_cancellation_ratio_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 4))
        & (tl.load(recent_delta_ratio_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 5))
        & (tl.load(recent_delta_ratio_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 5))
        & (tl.load(delta_trend_ratio_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 6))
        & (tl.load(delta_trend_ratio_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 6))
        & (tl.load(recent_delta_cosine_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 7))
        & (tl.load(recent_delta_cosine_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 7))
        & (tl.load(w_recent_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 8))
        & (tl.load(w_recent_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 8))
        & (tl.load(residual_strength_factor_ptr).to(tl.float32)
           >= tl.load(lower_bounds_ptr + bound_base + 9))
        & (tl.load(residual_strength_factor_ptr).to(tl.float32)
           < tl.load(upper_bounds_ptr + bound_base + 9))
    )
    value = decision.to(tl.int32)
    tl.store(block_masks_ptr + block_idx, value)

    position = tl.load(cache_position_ptr).to(tl.int64)
    valid_position = (position >= 0) & (position < max_history)
    for layer_offset in tl.static_range(0, 4):
        selected = (layer_bits >> layer_offset) & 1
        layer_value = value * selected
        layer_index = block_idx * 4 + layer_offset
        tl.store(layer_masks_ptr + layer_index, layer_value)
        tl.store(
            history_layers_ptr + position * 28 + layer_index,
            layer_value,
            mask=valid_position,
        )
    tl.store(
        history_blocks_ptr + position * 7 + block_idx,
        value,
        mask=valid_position,
    )
    tl.store(history_visited_ptr + position, 1, mask=valid_position)


@triton.jit
def _fused_native_action_mask_kernel(
    values_ptr,
    alpha_ptr,
    routed_ptr,
    previous_ptr,
    lower_bounds_ptr,
    upper_bounds_ptr,
    layer_masks_ptr,
    block_masks_ptr,
    cache_position_ptr,
    history_layers_ptr,
    history_blocks_ptr,
    history_visited_ptr,
    block_idx: tl.constexpr,
    layer_offset: tl.constexpr,
    source_count: tl.constexpr,
    hidden_size: tl.constexpr,
    need_route_novelty: tl.constexpr,
    need_delta_cancellation: tl.constexpr,
    need_recent_delta_ratio: tl.constexpr,
    need_delta_trend_ratio: tl.constexpr,
    block_size: tl.constexpr,
    max_history: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    hidden_mask = offsets < hidden_size
    previous_bf16 = tl.load(previous_ptr + offsets, mask=hidden_mask, other=0.0)
    routed_bf16 = tl.load(routed_ptr + offsets, mask=hidden_mask, other=0.0)
    previous = previous_bf16.to(tl.float32)
    routed = routed_bf16.to(tl.float32)
    previous_rms = tl.sqrt(
        tl.sum(previous * previous, axis=0) / hidden_size
    )
    previous_rms = tl.maximum(previous_rms, 1.0e-8)

    route_novelty = 0.0
    if need_route_novelty:
        # The reference path forms delta in BF16 before its FP32 RMS.
        routed_delta = (routed_bf16 - previous_bf16).to(tl.bfloat16).to(
            tl.float32
        )
        route_novelty = tl.sqrt(
            tl.sum(routed_delta * routed_delta, axis=0) / hidden_size
        ) / previous_rms

    delta_cancellation_ratio = 0.0
    recent_delta_ratio = 0.0
    delta_trend_ratio = 0.0
    if need_delta_cancellation or need_delta_trend_ratio:
        prefix_mass = 0.0
        path_rms = 0.0
        recent_delta_rms = 0.0
        older_delta_rms_sum = 0.0
        for source_offset in tl.static_range(0, source_count - 1):
            prefix_mass += tl.load(alpha_ptr + source_offset).to(tl.float32)
            earlier = tl.load(
                values_ptr + source_offset * hidden_size + offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            later = tl.load(
                values_ptr + (source_offset + 1) * hidden_size + offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            source_delta = later - earlier
            source_delta_rms = tl.sqrt(
                tl.sum(source_delta * source_delta, axis=0) / hidden_size
            )
            path_rms += prefix_mass * source_delta_rms
            if source_offset == source_count - 2:
                recent_delta_rms = source_delta_rms
            else:
                older_delta_rms_sum += source_delta_rms
        if need_delta_cancellation:
            route_delta = routed - previous
            route_delta_rms = tl.sqrt(
                tl.sum(route_delta * route_delta, axis=0) / hidden_size
            )
            delta_cancellation_ratio = route_delta_rms / tl.maximum(
                path_rms, 1.0e-8
            )
        if need_delta_trend_ratio:
            if source_count > 2:
                older_delta_rms = older_delta_rms_sum / (source_count - 2)
            else:
                older_delta_rms = recent_delta_rms
            delta_trend_ratio = recent_delta_rms / tl.maximum(
                older_delta_rms, 1.0e-8
            )
    elif need_recent_delta_ratio:
        earlier = tl.load(
            values_ptr + (source_count - 2) * hidden_size + offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        later = tl.load(
            values_ptr + (source_count - 1) * hidden_size + offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        recent_delta = later - earlier
        recent_delta_rms = tl.sqrt(
            tl.sum(recent_delta * recent_delta, axis=0) / hidden_size
        )
        recent_delta_ratio = recent_delta_rms / previous_rms

    bound_base = block_idx * 10
    decision = True
    if need_route_novelty:
        decision = decision & (
            route_novelty >= tl.load(lower_bounds_ptr + bound_base + 1)
        ) & (route_novelty < tl.load(upper_bounds_ptr + bound_base + 1))
    if need_delta_cancellation:
        decision = decision & (
            delta_cancellation_ratio
            >= tl.load(lower_bounds_ptr + bound_base + 4)
        ) & (
            delta_cancellation_ratio
            < tl.load(upper_bounds_ptr + bound_base + 4)
        )
    if need_recent_delta_ratio:
        decision = decision & (
            recent_delta_ratio >= tl.load(lower_bounds_ptr + bound_base + 5)
        ) & (
            recent_delta_ratio < tl.load(upper_bounds_ptr + bound_base + 5)
        )
    if need_delta_trend_ratio:
        decision = decision & (
            delta_trend_ratio >= tl.load(lower_bounds_ptr + bound_base + 6)
        ) & (
            delta_trend_ratio < tl.load(upper_bounds_ptr + bound_base + 6)
        )

    value = decision.to(tl.int32)
    layer_index = block_idx * 4 + layer_offset
    tl.store(layer_masks_ptr + layer_index, value)
    tl.store(block_masks_ptr + block_idx, value)
    position = tl.load(cache_position_ptr).to(tl.int64)
    valid_position = (position >= 0) & (position < max_history)
    tl.store(
        history_layers_ptr + position * 28 + layer_index,
        value,
        mask=valid_position,
    )
    tl.store(
        history_blocks_ptr + position * 7 + block_idx,
        value,
        mask=valid_position,
    )
    tl.store(history_visited_ptr + position, 1, mask=valid_position)


class DeviceNativeTokenMaskSelector:
    """GPU-only implementation of frozen AttnRes native interval rules."""

    def __init__(self, config: dict, device: torch.device):
        self.device = torch.device(device)
        self.max_history = int(config.get("device_conditional_max_positions", 8192))
        if self.max_history <= 0:
            raise ValueError("device_conditional_max_positions must be positive")
        self.eligible_blocks = tuple(sorted(int(v) for v in config["eligible_blocks"]))
        actions = config["action_by_block"]
        offsets = config["layer_offsets_by_block"]
        if any(actions[block] != "layer" for block in self.eligible_blocks):
            raise ValueError("native device selector supports layer actions only")
        self.layer_offsets = {}
        for block in self.eligible_blocks:
            selected = tuple(int(v) for v in offsets[block])
            if not selected:
                raise ValueError("native device selector requires a layer action")
            self.layer_offsets[block] = selected
        self.fused_feature_names = {
            1: ("delta_cancellation_ratio",),
            2: ("route_novelty", "delta_trend_ratio"),
            3: ("route_novelty",),
            4: ("delta_cancellation_ratio",),
            5: ("recent_delta_ratio",),
            6: ("recent_delta_ratio",),
        }
        if bool(config.get("device_fused_native_features", False)):
            if set(self.eligible_blocks) != set(self.fused_feature_names):
                raise ValueError(
                    "fused native selector requires the frozen tail.05 blocks"
                )
            fitted = (
                config.get("calibration", {}).get("fitted_rules", {}) or {}
            )
            fitted_features = {}
            for block in self.eligible_blocks:
                rule = fitted.get(block, fitted.get(str(block), {}))
                names = [str(rule.get("feature", ""))]
                if rule.get("guard_feature") is not None:
                    names.append(str(rule["guard_feature"]))
                fitted_features[block] = tuple(sorted(names))
            expected_features = {
                block: tuple(sorted(names))
                for block, names in self.fused_feature_names.items()
            }
            if fitted_features != expected_features:
                raise ValueError(
                    "fused native selector does not match frozen calibration rules"
                )

        lower = torch.full((7, 10), -float("inf"), dtype=torch.float32)
        upper = torch.full((7, 10), float("inf"), dtype=torch.float32)
        for block in self.eligible_blocks:
            for feature_index, feature_name in enumerate(NATIVE_FEATURE_ORDER):
                lower[block, feature_index] = float(
                    config["feature_lower_bounds"][block][feature_name]
                )
                upper[block, feature_index] = float(
                    config["feature_upper_bounds"][block][feature_name]
                )
        self.lower_bounds = lower.to(self.device)
        self.upper_bounds = upper.to(self.device)
        self.layer_masks = torch.zeros((7, 4), device=self.device, dtype=torch.int32)
        self.block_masks = torch.zeros(7, device=self.device, dtype=torch.int32)
        self.history_layer_masks = torch.zeros(
            (self.max_history, 7, 4), device=self.device, dtype=torch.int8
        )
        self.history_block_masks = torch.zeros(
            (self.max_history, 7), device=self.device, dtype=torch.int8
        )
        self.history_visited = torch.zeros(
            self.max_history, device=self.device, dtype=torch.int8
        )

    def reset_history(self):
        self.layer_masks.zero_()
        self.block_masks.zero_()
        self.history_layer_masks.zero_()
        self.history_block_masks.zero_()
        self.history_visited.zero_()

    def update_block(
        self,
        block_idx: int,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
    ):
        block_idx = int(block_idx)
        if block_idx not in self.layer_offsets:
            raise ValueError(f"block {block_idx} is not device-native eligible")
        if cache_position is None or cache_position.numel() != 1:
            raise ValueError("native device selector requires one cache position")
        values = [native_features[name] for name in NATIVE_FEATURE_ORDER]
        if any(value.numel() != 1 or not value.is_cuda for value in values):
            raise ValueError("native device selector requires scalar CUDA features")
        with torch.cuda.device(self.device):
            layer_bits = sum(
                1 << layer_offset for layer_offset in self.layer_offsets[block_idx]
            )
            _native_action_mask_kernel[(1,)](
                *values,
                self.lower_bounds,
                self.upper_bounds,
                self.layer_masks,
                self.block_masks,
                cache_position,
                self.history_layer_masks,
                self.history_block_masks,
                self.history_visited,
                block_idx=block_idx,
                layer_bits=layer_bits,
                max_history=self.max_history,
                num_warps=1,
            )

    def update_block_fused(
        self,
        block_idx: int,
        alpha: torch.Tensor,
        source_values: torch.Tensor,
        routed: torch.Tensor,
        previous: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        block_idx = int(block_idx)
        if block_idx not in self.fused_feature_names:
            raise ValueError(f"block {block_idx} lacks a fused native rule")
        if cache_position is None or cache_position.numel() != 1:
            raise ValueError("fused native selector requires one cache position")
        source_count = block_idx + 1
        hidden_size = int(previous.shape[-1])
        if (
            source_values.shape[0] != source_count
            or source_values.numel() != source_count * hidden_size
            or alpha.numel() != source_count
            or routed.numel() != hidden_size
            or previous.numel() != hidden_size
        ):
            raise ValueError("fused native selector received an unsupported shape")
        if not all(
            value.is_cuda
            for value in (alpha, source_values, routed, previous, cache_position)
        ):
            raise ValueError("fused native selector requires CUDA tensors")
        if not source_values.is_contiguous():
            raise ValueError("fused native source values must be contiguous")
        features = self.fused_feature_names[block_idx]
        if len(self.layer_offsets[block_idx]) != 1:
            raise ValueError("fused native selector supports one layer per block")
        block_size = triton.next_power_of_2(hidden_size)
        with torch.cuda.device(self.device):
            _fused_native_action_mask_kernel[(1,)](
                source_values,
                alpha,
                routed,
                previous,
                self.lower_bounds,
                self.upper_bounds,
                self.layer_masks,
                self.block_masks,
                cache_position,
                self.history_layer_masks,
                self.history_block_masks,
                self.history_visited,
                block_idx=block_idx,
                layer_offset=self.layer_offsets[block_idx][0],
                source_count=source_count,
                hidden_size=hidden_size,
                need_route_novelty="route_novelty" in features,
                need_delta_cancellation=(
                    "delta_cancellation_ratio" in features
                ),
                need_recent_delta_ratio="recent_delta_ratio" in features,
                need_delta_trend_ratio="delta_trend_ratio" in features,
                block_size=block_size,
                max_history=self.max_history,
                num_warps=8,
            )

    def layer_mask(self, block_idx: int, layer_offset: int) -> torch.Tensor:
        return self.layer_masks[block_idx, layer_offset : layer_offset + 1]

    def summarize(self) -> dict:
        block_counts = self.history_block_masks.to(torch.int64).sum(dim=0)
        layer_counts = self.history_layer_masks.to(torch.int64).sum(dim=(0, 2))
        packed = torch.cat(
            [
                block_counts,
                layer_counts,
                torch.count_nonzero(self.history_visited).reshape(1),
            ]
        ).cpu().tolist()
        return {
            "per_block_actions": [int(value) for value in packed[:7]],
            "per_block_layers": [int(value) for value in packed[7:14]],
            "visited_tokens": int(packed[14]),
        }
