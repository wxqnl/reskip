from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate1_action_mask_kernel(
    scores_ptr,
    thresholds_ptr,
    layer_masks_ptr,
    block_masks_ptr,
    status_ptr,
    all_skip_upper_ptr,
    all_compute_lower_ptr,
):
    offsets = tl.arange(0, 32)
    score_1 = tl.load(scores_ptr + 0).to(tl.float64)
    score_2 = tl.load(scores_ptr + 1).to(tl.float64)
    threshold_1 = tl.load(thresholds_ptr + 0)
    threshold_2 = tl.load(thresholds_ptr + 1)
    skip_1 = score_1 <= threshold_1
    skip_2 = score_2 <= threshold_2
    max_ratio = tl.maximum(
        score_1 / tl.maximum(threshold_1, 1.0e-12),
        score_2 / tl.maximum(threshold_2, 1.0e-12),
    )
    all_skip = max_ratio <= tl.load(all_skip_upper_ptr)
    all_compute = max_ratio >= tl.load(all_compute_lower_ptr)
    status = tl.where(all_skip, 1, tl.where(all_compute, 2, 0))

    layer_values = tl.zeros((32,), dtype=tl.int32)
    # Block 1 action skips layer offsets 1, 2, 3.
    layer_values = tl.where(
        (offsets == 5) | (offsets == 6) | (offsets == 7),
        skip_1,
        layer_values,
    )
    # Block 2 action skips layer offset 1.
    layer_values = tl.where(offsets == 9, skip_2, layer_values)
    # The ratio all-skip bypass predicts the single action at blocks 3, 4, 5.
    layer_values = tl.where(
        (offsets == 12)
        | (offsets == 13)
        | (offsets == 17)
        | (offsets == 20)
        | (offsets == 23),
        all_skip,
        layer_values,
    )
    tl.store(layer_masks_ptr + offsets, layer_values, mask=offsets < 28)

    block_offsets = tl.arange(0, 8)
    block_values = tl.zeros((8,), dtype=tl.int32)
    block_values = tl.where(block_offsets == 1, skip_1, block_values)
    block_values = tl.where(block_offsets == 2, skip_2, block_values)
    block_values = tl.where(
        (block_offsets == 3) | (block_offsets == 4) | (block_offsets == 5),
        all_skip,
        block_values,
    )
    tl.store(block_masks_ptr + block_offsets, block_values, mask=block_offsets < 7)
    tl.store(status_ptr, status)


@triton.jit
def _gate2_action_mask_kernel(
    scores_ptr,
    thresholds_ptr,
    layer_masks_ptr,
    block_masks_ptr,
    status_ptr,
    cache_position_ptr,
    history_layers_ptr,
    history_blocks_ptr,
    history_status_ptr,
    max_history: tl.constexpr,
):
    status = tl.load(status_ptr)
    use_gate2 = status == 0
    skip_3 = (
        tl.load(scores_ptr + 0).to(tl.float64) <= tl.load(thresholds_ptr + 0)
    )
    skip_4 = (
        tl.load(scores_ptr + 1).to(tl.float64) <= tl.load(thresholds_ptr + 1)
    )
    skip_5 = (
        tl.load(scores_ptr + 2).to(tl.float64) <= tl.load(thresholds_ptr + 2)
    )

    offsets = tl.arange(0, 32)
    current_layers = tl.load(
        layer_masks_ptr + offsets, mask=offsets < 28, other=0
    )
    current_layers = tl.where(
        use_gate2 & ((offsets == 12) | (offsets == 13)),
        skip_3,
        current_layers,
    )
    current_layers = tl.where(
        use_gate2 & (offsets == 17), skip_4, current_layers
    )
    current_layers = tl.where(
        use_gate2 & ((offsets == 20) | (offsets == 23)),
        skip_5,
        current_layers,
    )
    tl.store(layer_masks_ptr + offsets, current_layers, mask=offsets < 28)

    block_offsets = tl.arange(0, 8)
    current_blocks = tl.load(
        block_masks_ptr + block_offsets, mask=block_offsets < 7, other=0
    )
    current_blocks = tl.where(
        use_gate2 & (block_offsets == 3), skip_3, current_blocks
    )
    current_blocks = tl.where(
        use_gate2 & (block_offsets == 4), skip_4, current_blocks
    )
    current_blocks = tl.where(
        use_gate2 & (block_offsets == 5), skip_5, current_blocks
    )
    tl.store(block_masks_ptr + block_offsets, current_blocks, mask=block_offsets < 7)

    position = tl.load(cache_position_ptr).to(tl.int64)
    valid_position = (position >= 0) & (position < max_history)
    history_layer_base = position * 28
    history_block_base = position * 7
    tl.store(
        history_layers_ptr + history_layer_base + offsets,
        current_layers,
        mask=(offsets < 28) & valid_position,
    )
    tl.store(
        history_blocks_ptr + history_block_base + block_offsets,
        current_blocks,
        mask=(block_offsets < 7) & valid_position,
    )
    tl.store(
        history_status_ptr + position,
        status + 1,
        mask=valid_position,
    )


class DeviceActionMaskSelector:
    """Exact GPU implementation of the frozen one-action policy."""

    _EXPECTED_GROUPS = {
        1: (1, 2, 3),
        2: (1,),
        3: (0, 1),
        4: (1,),
        5: (0, 3),
    }

    def __init__(self, config: dict, risk_groups: dict, device: torch.device):
        thresholds = config["thresholds_by_block"]
        groups = {
            int(block): tuple(int(value) for value in risk_groups[int(block)][0])
            for block in self._EXPECTED_GROUPS
        }
        if groups != self._EXPECTED_GROUPS:
            raise ValueError(
                "O27 device selector requires the frozen O26 action groups"
            )
        if any(tuple(values) != (0,) for values in config.get("actions_by_block", {}).values()):
            raise ValueError("O27 device selector supports one frozen action per block")
        if int(config.get("max_layer_skips", 0)) not in (0, 9):
            raise ValueError("O27 device selector requires the frozen nine-layer budget")
        fallback = config.get("gate1_ratio_fallback")
        if fallback is not None and (
            tuple(fallback["source_blocks"]) != (1, 2)
            or tuple(fallback["target_blocks"]) != (3, 4, 5)
        ):
            raise ValueError("O27 device selector requires the frozen fallback map")
        self.ratio_fallback_enabled = fallback is not None

        self.device = torch.device(device)
        self.max_history = int(config.get("device_conditional_max_positions", 8192))
        if self.max_history <= 0:
            raise ValueError("device_conditional_max_positions must be positive")
        threshold_values = [float(thresholds[block][0]) for block in range(1, 6)]
        self.gate1_thresholds = torch.tensor(
            threshold_values[:2], device=self.device, dtype=torch.float64
        )
        self.gate2_thresholds = torch.tensor(
            threshold_values[2:], device=self.device, dtype=torch.float64
        )
        # Without a certified ratio envelope, force the exact second gate on
        # every decode token.  +/-inf makes both bypass predicates false in the
        # existing Triton kernel, preserving one persistent device path without
        # inventing late-block actions from gate-1 scores.
        self.all_skip_upper = torch.tensor(
            float(fallback["all_skip_upper"]) if fallback is not None else -float("inf"),
            device=self.device,
            dtype=torch.float64,
        )
        self.all_compute_lower = torch.tensor(
            float(fallback["all_compute_lower"]) if fallback is not None else float("inf"),
            device=self.device,
            dtype=torch.float64,
        )
        self.layer_masks = torch.zeros(
            (7, 4), device=self.device, dtype=torch.int32
        )
        self.block_masks = torch.zeros(7, device=self.device, dtype=torch.int32)
        self.gate_status = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.history_layer_masks = torch.zeros(
            (self.max_history, 7, 4), device=self.device, dtype=torch.int8
        )
        self.history_block_masks = torch.zeros(
            (self.max_history, 7), device=self.device, dtype=torch.int8
        )
        # Zero means unvisited; stored statuses are runtime status + 1.
        self.history_gate_status = torch.zeros(
            self.max_history, device=self.device, dtype=torch.int8
        )

    def reset_history(self):
        self.history_layer_masks.zero_()
        self.history_block_masks.zero_()
        self.history_gate_status.zero_()

    def update_gate1(self, packed_scores: torch.Tensor):
        if packed_scores.numel() != 2 or not packed_scores.is_cuda:
            raise ValueError("gate 1 must provide two CUDA scores")
        _gate1_action_mask_kernel[(1,)](
            packed_scores,
            self.gate1_thresholds,
            self.layer_masks,
            self.block_masks,
            self.gate_status,
            self.all_skip_upper,
            self.all_compute_lower,
            num_warps=1,
        )

    def update_gate2(
        self, packed_scores: torch.Tensor, cache_position: torch.Tensor
    ):
        if packed_scores.numel() != 3 or not packed_scores.is_cuda:
            raise ValueError("gate 2 must provide three CUDA scores")
        if cache_position is None or cache_position.numel() != 1:
            raise ValueError("device selector requires one cache position")
        _gate2_action_mask_kernel[(1,)](
            packed_scores,
            self.gate2_thresholds,
            self.layer_masks,
            self.block_masks,
            self.gate_status,
            cache_position,
            self.history_layer_masks,
            self.history_block_masks,
            self.history_gate_status,
            max_history=self.max_history,
            num_warps=1,
        )

    def layer_mask(self, block_idx: int, layer_offset: int) -> torch.Tensor:
        return self.layer_masks[block_idx, layer_offset : layer_offset + 1]

    def block_mask(self, block_idx: int) -> torch.Tensor:
        return self.block_masks[block_idx : block_idx + 1]

    def summarize(self) -> dict:
        block_counts = self.history_block_masks.to(torch.int64).sum(dim=0)
        layer_counts = self.history_layer_masks.to(torch.int64).sum(dim=(0, 2))
        statuses = self.history_gate_status
        visited = statuses > 0
        if self.ratio_fallback_enabled:
            fallback_count = torch.count_nonzero(statuses == 1)
            all_skip_count = torch.count_nonzero(statuses == 2)
            all_compute_count = torch.count_nonzero(statuses == 3)
        else:
            fallback_count = torch.zeros((), device=self.device, dtype=torch.int64)
            all_skip_count = torch.zeros((), device=self.device, dtype=torch.int64)
            all_compute_count = torch.zeros((), device=self.device, dtype=torch.int64)
        packed = torch.cat(
            [
                block_counts,
                layer_counts,
                torch.stack(
                    [
                        torch.count_nonzero(visited),
                        fallback_count,
                        all_skip_count,
                        all_compute_count,
                    ]
                ),
            ]
        ).cpu().tolist()
        return {
            "per_block_actions": [int(value) for value in packed[:7]],
            "per_block_layers": [int(value) for value in packed[7:14]],
            "visited_tokens": int(packed[14]),
            "ratio_fallback_tokens": int(packed[15]),
            "ratio_all_skip_tokens": int(packed[16]),
            "ratio_all_compute_tokens": int(packed[17]),
        }
