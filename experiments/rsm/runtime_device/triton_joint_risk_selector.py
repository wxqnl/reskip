from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _joint_risk_score_kernel(
    alpha_ptr,
    source_dispersion_ptr,
    route_novelty_ptr,
    correction_ratio_ptr,
    cache_position_ptr,
    block_input_ptr,
    source_values_ptr,
    routed_ptr,
    previous_ptr,
    semantic_weight_ptr,
    first_weight_ptr,
    first_bias_ptr,
    second_weight_ptr,
    second_bias_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    SEMANTIC_DIM: tl.constexpr,
    FEATURE_DIM: tl.constexpr,
    FEATURE_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TARGET_COUNT: tl.constexpr,
    FLIP_WEIGHT: tl.constexpr,
    LOG_SOURCE_COUNT: tl.constexpr,
    LOG_POSITION_DENOMINATOR: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    FUSED_NATIVE: tl.constexpr,
):
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    block_input = tl.load(
        block_input_ptr + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    rms = tl.sqrt(
        tl.sum(block_input * block_input, axis=0) / HIDDEN_SIZE
    )
    inverse_rms = 1.0 / tl.maximum(rms, 1.0e-6)

    alpha_offsets = tl.arange(0, 8)
    alpha_mask = alpha_offsets < SOURCE_COUNT
    raw_probability = tl.load(
        alpha_ptr + alpha_offsets,
        mask=alpha_mask,
        other=0.0,
    ).to(tl.float32)
    probability = tl.where(
        alpha_mask, tl.maximum(raw_probability, 1.0e-8), 0.0
    )
    recent = tl.sum(
        tl.where(alpha_offsets == SOURCE_COUNT - 1, probability, 0.0),
        axis=0,
    )
    competitor = tl.max(
        tl.where(alpha_offsets < SOURCE_COUNT - 1, probability, -1.0e20),
        axis=0,
    )
    log_margin = tl.log(recent) - tl.log(competitor)
    normalized_entropy = -tl.sum(
        tl.where(alpha_mask, probability * tl.log(probability), 0.0),
        axis=0,
    ) / LOG_SOURCE_COUNT
    age_scale = SOURCE_COUNT - 1
    normalized_age = (
        (SOURCE_COUNT - 1 - alpha_offsets).to(tl.float32)
        / age_scale
    )
    expected_age = tl.sum(probability * normalized_age, axis=0)
    age_delta = normalized_age - expected_age
    age_variance = tl.sum(
        probability * age_delta * age_delta,
        axis=0,
    )
    if FUSED_NATIVE:
        routed = tl.load(
            routed_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        previous = tl.load(
            previous_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        previous_rms = tl.sqrt(
            tl.sum(previous * previous, axis=0) / HIDDEN_SIZE
        )
        previous_rms = tl.maximum(previous_rms, 1.0e-8)
        route_delta = routed - previous
        route_novelty = tl.sqrt(
            tl.sum(route_delta * route_delta, axis=0) / HIDDEN_SIZE
        ) / previous_rms
        correction_delta = block_input - previous
        correction_ratio = tl.sqrt(
            tl.sum(
                correction_delta * correction_delta,
                axis=0,
            )
            / HIDDEN_SIZE
        ) / previous_rms
        dispersion_by_hidden = tl.zeros(
            (BLOCK_HIDDEN,), dtype=tl.float32
        )
        for source_idx in tl.static_range(0, SOURCE_COUNT):
            source_value = tl.load(
                source_values_ptr
                + source_idx * HIDDEN_SIZE
                + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            source_weight = tl.sum(
                tl.where(
                    alpha_offsets == source_idx,
                    raw_probability,
                    0.0,
                ),
                axis=0,
            )
            source_delta = source_value - routed
            dispersion_by_hidden += (
                source_weight * source_delta * source_delta
            )
        dispersion = tl.sqrt(
            tl.sum(dispersion_by_hidden, axis=0) / HIDDEN_SIZE
        )
        routed_rms = tl.sqrt(
            tl.sum(routed * routed, axis=0) / HIDDEN_SIZE
        )
        source_dispersion = dispersion / tl.maximum(
            routed_rms, 1.0e-8
        )
    else:
        source_dispersion = tl.load(source_dispersion_ptr).to(tl.float32)
        route_novelty = tl.load(route_novelty_ptr).to(tl.float32)
        correction_ratio = tl.load(correction_ratio_ptr).to(tl.float32)
    position = tl.load(cache_position_ptr).to(tl.float32)
    log_position = tl.log(1.0 + position) / LOG_POSITION_DENOMINATOR

    feature_offsets = tl.arange(0, FEATURE_PAD)
    features = tl.zeros((FEATURE_PAD,), dtype=tl.float32)
    features = tl.where(feature_offsets == 0, recent, features)
    features = tl.where(feature_offsets == 1, competitor, features)
    features = tl.where(feature_offsets == 2, log_margin, features)
    features = tl.where(
        feature_offsets == 3, 1.0 - normalized_entropy, features
    )
    features = tl.where(feature_offsets == 4, expected_age, features)
    features = tl.where(feature_offsets == 5, age_variance, features)
    features = tl.where(feature_offsets == 6, source_dispersion, features)
    features = tl.where(feature_offsets == 7, route_novelty, features)
    features = tl.where(feature_offsets == 8, correction_ratio, features)
    features = tl.where(feature_offsets == 9, log_position, features)

    for semantic_idx in tl.static_range(0, SEMANTIC_DIM):
        semantic_weight = tl.load(
            semantic_weight_ptr
            + semantic_idx * HIDDEN_SIZE
            + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        semantic_value = (
            tl.sum(block_input * semantic_weight, axis=0) * inverse_rms
        )
        semantic_value = semantic_value * tl.sigmoid(semantic_value)
        features = tl.where(
            feature_offsets == 10 + semantic_idx,
            semantic_value,
            features,
        )

    head_offsets = tl.arange(0, HEAD_DIM)
    feature_matrix_offsets = (
        head_offsets[None, :] * FEATURE_DIM
        + feature_offsets[:, None]
    )
    feature_mask = feature_offsets[:, None] < FEATURE_DIM
    for target_idx in tl.static_range(0, TARGET_COUNT):
        first_weight = tl.load(
            first_weight_ptr
            + target_idx * HEAD_DIM * FEATURE_DIM
            + feature_matrix_offsets,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        hidden = tl.sum(features[:, None] * first_weight, axis=0)
        hidden += tl.load(
            first_bias_ptr + target_idx * HEAD_DIM + head_offsets
        ).to(tl.float32)
        hidden = hidden * tl.sigmoid(hidden)
        second_base = target_idx * 2 * HEAD_DIM
        raw_log_kl = tl.sum(
            hidden
            * tl.load(
                second_weight_ptr + second_base + head_offsets
            ).to(tl.float32),
            axis=0,
        ) + tl.load(second_bias_ptr + target_idx * 2).to(tl.float32)
        raw_flip = tl.sum(
            hidden
            * tl.load(
                second_weight_ptr
                + second_base
                + HEAD_DIM
                + head_offsets
            ).to(tl.float32),
            axis=0,
        ) + tl.load(second_bias_ptr + target_idx * 2 + 1).to(tl.float32)
        risk_score = tl.exp(raw_log_kl) + (
            FLIP_WEIGHT * tl.sigmoid(raw_flip)
        )
        tl.store(output_ptr + target_idx, risk_score)


class TritonJointRiskSelector:
    """One-kernel O13 score path for batch-1, one-token CUDA decode."""

    def __init__(self, surrogate):
        target_blocks = tuple(sorted(surrogate.actions_by_block))
        if surrogate.semantic_projection is None:
            raise ValueError("Triton selector requires semantic projection")
        if surrogate.semantic_dim != 16 or surrogate.hidden_dim != 64:
            raise ValueError(
                "Triton selector currently supports semantic-16 hidden-64 heads"
            )
        if any(
            len(surrogate.actions_by_block[block]) != 1
            for block in target_blocks
        ):
            raise ValueError(
                "Triton selector currently supports one action per target block"
            )
        heads = tuple(surrogate.heads[str(block)] for block in target_blocks)
        self.target_blocks = target_blocks
        self.hidden_size = int(surrogate.hidden_size)
        self.semantic_dim = int(surrogate.semantic_dim)
        self.feature_dim = int(surrogate.feature_dim + surrogate.semantic_dim)
        self.head_dim = int(surrogate.hidden_dim)
        self.flip_weight = float(surrogate.flip_weight)
        self.semantic_weight = (
            surrogate.semantic_projection.weight.detach().contiguous()
        )
        self.first_weight = torch.stack(
            [head[0].weight.detach() for head in heads], dim=0
        ).contiguous()
        self.first_bias = torch.stack(
            [head[0].bias.detach() for head in heads], dim=0
        ).contiguous()
        self.second_weight = torch.stack(
            [head[2].weight.detach() for head in heads], dim=0
        ).contiguous()
        self.second_bias = torch.stack(
            [head[2].bias.detach() for head in heads], dim=0
        ).contiguous()
        self._host_score_buffers: dict[
            tuple[str, int | None, int], torch.Tensor
        ] = {}

    def _predict_impl(
        self,
        alpha: torch.Tensor,
        source_dispersion: torch.Tensor,
        route_novelty: torch.Tensor,
        correction_ratio: torch.Tensor,
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
        source_values: torch.Tensor,
        routed: torch.Tensor,
        previous: torch.Tensor,
        *,
        fused_native: bool,
    ) -> torch.Tensor:
        if (
            not alpha.is_cuda
            or alpha.shape[0] != 1
            or alpha.shape[-2] != 1
            or block_input.shape[0] != 1
            or block_input.shape[-2] != 1
        ):
            raise ValueError(
                "Triton joint selector requires batch-1, one-token CUDA decode"
            )
        source_count = int(alpha.shape[-1])
        if source_count < 2 or source_count > 8:
            raise ValueError("unsupported AttnRes source count")
        output = torch.empty(
            len(self.target_blocks),
            device=alpha.device,
            dtype=torch.float32,
        )
        block_hidden = triton.next_power_of_2(self.hidden_size)
        _joint_risk_score_kernel[(1,)](
            alpha,
            source_dispersion,
            route_novelty,
            correction_ratio,
            cache_position,
            block_input,
            source_values,
            routed,
            previous,
            self.semantic_weight,
            self.first_weight,
            self.first_bias,
            self.second_weight,
            self.second_bias,
            output,
            HIDDEN_SIZE=self.hidden_size,
            SOURCE_COUNT=source_count,
            SEMANTIC_DIM=self.semantic_dim,
            FEATURE_DIM=self.feature_dim,
            FEATURE_PAD=32,
            HEAD_DIM=self.head_dim,
            TARGET_COUNT=len(self.target_blocks),
            FLIP_WEIGHT=self.flip_weight,
            LOG_SOURCE_COUNT=math.log(source_count),
            LOG_POSITION_DENOMINATOR=math.log1p(32768.0),
            BLOCK_HIDDEN=block_hidden,
            FUSED_NATIVE=fused_native,
            num_warps=8,
        )
        return output

    def unpack_scores(
        self,
        output: torch.Tensor,
        output_shape: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        if output.ndim != 1 or output.numel() != len(self.target_blocks):
            raise ValueError("packed selector output has the wrong shape")
        return {
            block: output[position].reshape(output_shape)
            for position, block in enumerate(self.target_blocks)
        }

    def copy_scores_to_host(self, output: torch.Tensor) -> list[float]:
        """Copy one packed gate result through a reusable pinned buffer."""
        if (
            not output.is_cuda
            or output.dtype != torch.float32
            or output.ndim != 1
            or output.numel() != len(self.target_blocks)
        ):
            raise ValueError("packed selector scores must be a CUDA float vector")
        key = (
            output.device.type,
            output.device.index,
            int(output.numel()),
        )
        host_output = self._host_score_buffers.get(key)
        if host_output is None:
            host_output = torch.empty(
                output.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            self._host_score_buffers[key] = host_output
        host_output.copy_(output, non_blocking=True)
        # The next decoder block depends on this token-local decision.  The
        # synchronization cannot be removed without moving block execution
        # into a device-side conditional runtime, but the pageable transfer
        # and score-stack launch can be removed.
        torch.cuda.current_stream(output.device).synchronize()
        return host_output.tolist()

    def predict(
        self,
        alpha: torch.Tensor,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        output = self._predict_impl(
            alpha,
            native_features["source_dispersion"],
            native_features["route_novelty"],
            native_features["correction_ratio"],
            cache_position,
            block_input,
            block_input,
            block_input,
            block_input,
            fused_native=False,
        )
        return self.unpack_scores(output, (*alpha.shape[:-1], 1))

    def predict_packed(
        self,
        alpha: torch.Tensor,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
    ) -> torch.Tensor:
        return self._predict_impl(
            alpha,
            native_features["source_dispersion"],
            native_features["route_novelty"],
            native_features["correction_ratio"],
            cache_position,
            block_input,
            block_input,
            block_input,
            block_input,
            fused_native=False,
        )

    def predict_fused_native(
        self,
        alpha: torch.Tensor,
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
        source_values: torch.Tensor,
        routed: torch.Tensor,
        previous: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        source_count = int(alpha.shape[-1])
        expected_values_shape = (
            source_count,
            int(alpha.shape[0]),
            int(alpha.shape[-2]),
            self.hidden_size,
        )
        if tuple(source_values.shape) != expected_values_shape:
            raise ValueError(
                "fused selector source values have shape "
                f"{tuple(source_values.shape)}, expected "
                f"{expected_values_shape}"
            )
        if (
            not source_values.is_cuda
            or not routed.is_cuda
            or not previous.is_cuda
            or routed.shape != block_input.shape
            or previous.shape != block_input.shape
        ):
            raise ValueError(
                "fused selector inputs must be matching CUDA tensors"
            )
        output = self._predict_impl(
            alpha,
            block_input,
            block_input,
            block_input,
            cache_position,
            block_input,
            source_values,
            routed,
            previous,
            fused_native=True,
        )
        return self.unpack_scores(output, (*alpha.shape[:-1], 1))

    def predict_fused_native_packed(
        self,
        alpha: torch.Tensor,
        cache_position: torch.Tensor,
        block_input: torch.Tensor,
        source_values: torch.Tensor,
        routed: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        source_count = int(alpha.shape[-1])
        expected_values_shape = (
            source_count,
            int(alpha.shape[0]),
            int(alpha.shape[-2]),
            self.hidden_size,
        )
        if tuple(source_values.shape) != expected_values_shape:
            raise ValueError(
                "fused selector source values have shape "
                f"{tuple(source_values.shape)}, expected "
                f"{expected_values_shape}"
            )
        if (
            not source_values.is_cuda
            or not routed.is_cuda
            or not previous.is_cuda
            or routed.shape != block_input.shape
            or previous.shape != block_input.shape
        ):
            raise ValueError(
                "fused selector inputs must be matching CUDA tensors"
            )
        return self._predict_impl(
            alpha,
            block_input,
            block_input,
            block_input,
            cache_position,
            block_input,
            source_values,
            routed,
            previous,
            fused_native=True,
        )
