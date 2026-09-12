from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _key_norm_rope_kernel(
    key_ptr,
    norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    EPSILON: tl.constexpr,
):
    head_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HEAD)
    mask = offsets < HEAD_DIM
    base = head_idx * HEAD_DIM

    raw = tl.load(key_ptr + base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    inverse_rms = 1.0 / tl.sqrt(
        tl.sum(raw * raw, axis=0) / HEAD_DIM + EPSILON
    )
    normalized = (raw * inverse_rms).to(tl.bfloat16)
    norm_weight = tl.load(
        norm_weight_ptr + offsets, mask=mask, other=0.0
    )
    weighted = (
        normalized.to(tl.float32) * norm_weight.to(tl.float32)
    ).to(tl.bfloat16)

    half_dim = HEAD_DIM // 2
    partner_offsets = tl.where(
        offsets < half_dim,
        offsets + half_dim,
        offsets - half_dim,
    )
    partner_raw = tl.load(
        key_ptr + base + partner_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    partner_normalized = (partner_raw * inverse_rms).to(tl.bfloat16)
    partner_weight = tl.load(
        norm_weight_ptr + partner_offsets,
        mask=mask,
        other=0.0,
    )
    partner_weighted = (
        partner_normalized.to(tl.float32)
        * partner_weight.to(tl.float32)
    ).to(tl.bfloat16)
    rotated = tl.where(
        offsets < half_dim,
        -partner_weighted,
        partner_weighted,
    )

    cosine = tl.load(cos_ptr + offsets, mask=mask, other=0.0)
    sine = tl.load(sin_ptr + offsets, mask=mask, other=0.0)
    direct_product = (
        weighted.to(tl.float32) * cosine.to(tl.float32)
    ).to(tl.bfloat16)
    rotated_product = (
        rotated.to(tl.float32) * sine.to(tl.float32)
    ).to(tl.bfloat16)
    output = (
        direct_product.to(tl.float32)
        + rotated_product.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(output_ptr + base + offsets, output, mask=mask)


def fused_key_norm_rope(
    key_projection: torch.Tensor,
    norm_weight: torch.Tensor,
    variance_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Fuse the skipped-layer K RMSNorm and key-only rotary path.

    This bounded inference kernel supports the actual Qwen3-VL batch-1,
    one-token BF16 decode shape. Other shapes stay on the native PyTorch path.
    """
    if (
        not key_projection.is_cuda
        or key_projection.dtype != torch.bfloat16
        or key_projection.ndim != 3
        or key_projection.shape[0] != 1
        or key_projection.shape[1] != 1
        or key_projection.shape[-1] % head_dim != 0
        or norm_weight.dtype != torch.bfloat16
        or not norm_weight.is_cuda
        or norm_weight.numel() != head_dim
        or cos.shape != sin.shape
        or cos.numel() != head_dim
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError(
            "fused K norm/RoPE requires contiguous batch-1 one-token BF16 CUDA inputs"
        )
    num_heads = int(key_projection.shape[-1] // head_dim)
    output = torch.empty(
        (1, num_heads, 1, head_dim),
        device=key_projection.device,
        dtype=key_projection.dtype,
    )
    _key_norm_rope_kernel[(num_heads,)](
        key_projection,
        norm_weight,
        cos,
        sin,
        output,
        HEAD_DIM=head_dim,
        BLOCK_HEAD=triton.next_power_of_2(head_dim),
        EPSILON=float(variance_epsilon),
        num_warps=4,
    )
    return output
