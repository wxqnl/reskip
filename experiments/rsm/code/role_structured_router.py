"""Role-structured, multi-head routing for AttnRes retrofit experiments.

The module changes only the native AttnRes source softmax.  It never observes
skip actions, counterfactual skip losses, coverage targets, or task labels.

The multi-head path follows MHAR: the existing D-dimensional pseudo-query is
reshaped into H disjoint D/H subspace queries, giving each subspace its own
softmax over depth without adding query parameters.  The proposed extension
adds a zero-initialized, low-rank relative-depth prior to those same logits.
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch
import torch.nn as nn

from qwen3vl_attnres_retrofit import _rms_norm


SUPPORTED_VARIANTS = {
    "attnres",
    "mhar",
    "unstructured",
    "role_factor",
    "role_structured",
}


@dataclass(frozen=True)
class RouterInstallReport:
    variant: str
    num_heads: int
    num_basis: int
    additional_parameters: int
    basis_type: str | None


def _triangular_depth_basis(
    position: int,
    num_basis: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fixed local-to-global basis over relative source age.

    Source ``position - 1`` is the most recent source (relative age 0), while
    source 0 is the oldest available source (relative age 1).  Rows form a
    partition of unity, making the same basis useful for interpretable routing
    profiles as well as the structural logit term.
    """
    if position <= 0 or num_basis <= 0:
        raise ValueError("position and num_basis must be positive")
    if position == 1:
        ages = torch.zeros(1, device=device, dtype=dtype)
    else:
        ages = torch.arange(
            position - 1,
            -1,
            -1,
            device=device,
            dtype=dtype,
        ) / float(position - 1)
    if num_basis == 1:
        return torch.ones(position, 1, device=device, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, num_basis, device=device, dtype=dtype)
    width = 1.0 / float(num_basis - 1)
    basis = torch.relu(1.0 - (ages[:, None] - centers[None, :]).abs() / width)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _build_router_basis(
    num_sources: int,
    num_basis: int,
    basis_type: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    result = torch.zeros(
        num_sources,
        num_sources,
        num_basis,
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for position in range(1, num_sources):
        structured = _triangular_depth_basis(
            position,
            num_basis,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if basis_type == "relative_depth":
            current = structured
        elif basis_type == "fixed_random":
            current = torch.randn(
                position,
                num_basis,
                generator=generator,
                dtype=torch.float32,
            )
            current = current / current.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            target_norm = structured.norm(dim=-1).mean()
            current = current * target_norm
        else:
            raise ValueError(f"unknown router basis type: {basis_type}")
        result[position, :position] = current
    return result.to(device=device, dtype=dtype)


def install_role_router(
    model,
    *,
    variant: str,
    num_heads: int = 4,
    num_basis: int = 4,
    role_scale: float = 1.0,
    random_basis_seed: int = 314159,
) -> RouterInstallReport:
    """Install MHAR and, when requested, a native depth-role factorization.

    ``attnres`` leaves the original single-head implementation untouched.
    ``mhar`` only reshapes the existing query and therefore adds no parameters.
    The remaining variants add one zero-initialized coefficient vector per
    AttnRes position and routing head.  ``unstructured`` uses a fixed random
    source basis as a parameter-matched control; the role variants use the
    relative-depth basis.
    """
    variant = str(variant).lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported router variant: {variant}")
    router = model.router
    if hasattr(router, "role_variant"):
        raise RuntimeError("role router is already configured")
    if variant == "attnres":
        router.role_variant = variant
        router.role_num_heads = 1
        router.role_num_basis = 0
        router.role_head_alpha_by_position = {}
        router.role_head_alpha_raw_by_position = {}
        router.role_keys_by_position = {}
        return RouterInstallReport(variant, 1, 0, 0, None)
    if num_heads <= 0 or router.hidden_size % num_heads != 0:
        raise ValueError("hidden size must be divisible by num_heads")
    if num_basis <= 0 or not math.isfinite(role_scale) or role_scale < 0.0:
        raise ValueError("invalid role basis size or scale")

    basis_type = None
    if variant == "unstructured":
        basis_type = "fixed_random"
    elif variant in {"role_factor", "role_structured"}:
        basis_type = "relative_depth"

    router.role_variant = variant
    router.role_num_heads = int(num_heads)
    router.role_num_basis = int(num_basis)
    router.role_scale = float(role_scale)
    router.role_head_alpha_by_position = {}
    router.role_head_alpha_raw_by_position = {}
    # Training-only cache for Full-path functional query identification.  It
    # exposes exactly the normalized native AttnRes keys used by the source
    # softmax and does not add parameters or alter the routed output.
    router.role_keys_by_position = {}

    additional_parameters = 0
    if basis_type is not None:
        router.register_parameter(
            "role_coefficients",
            nn.Parameter(
                torch.zeros(
                    router.num_sources,
                    num_heads,
                    num_basis,
                    device=router.w_query.device,
                    dtype=router.w_query.dtype,
                )
            ),
        )
        router.register_buffer(
            "role_router_basis",
            _build_router_basis(
                router.num_sources,
                num_basis,
                basis_type,
                device=router.w_query.device,
                dtype=router.w_query.dtype,
                seed=random_basis_seed,
            ),
            persistent=True,
        )
        additional_parameters = int(router.role_coefficients.numel())

    def _role_route_impl(
        self,
        position: int,
        completed_outputs: list[torch.Tensor],
        compute_source_dispersion: bool,
    ):
        values = torch.stack(completed_outputs, dim=0)
        keys = _rms_norm(values)
        if self.key_pos_bias is not None:
            positional = self.key_pos_bias[:position].to(keys.dtype)
            keys = keys + positional[:, None, None, :]

        source_count, batch, tokens, hidden = values.shape
        heads = self.role_num_heads
        head_dim = hidden // heads
        query = self.w_query[position].to(keys.dtype).view(heads, head_dim)
        key_heads = keys.view(source_count, batch, tokens, heads, head_dim)
        scale = math.sqrt(head_dim) * self.base_temperature
        scores = torch.einsum("hd,nbthd->nbth", query, key_heads) / scale

        if hasattr(self, "role_coefficients"):
            basis = self.role_router_basis[position, :position].to(scores.dtype)
            coefficients = self.role_coefficients[position].to(scores.dtype)
            role_logits = torch.einsum("nr,hr->nh", basis, coefficients)
            scores = scores + self.role_scale * role_logits[:, None, None, :]

        head_alpha = torch.softmax(scores.float(), dim=0).to(values.dtype)
        value_heads = values.view(source_count, batch, tokens, heads, head_dim)
        routed_heads = torch.einsum(
            "nbth,nbthd->bthd",
            head_alpha,
            value_heads,
        )
        routed = routed_heads.reshape(batch, tokens, hidden)
        self.role_head_alpha_raw_by_position[position] = head_alpha
        self.role_head_alpha_by_position[position] = head_alpha.permute(1, 2, 3, 0)
        self.role_keys_by_position[position] = keys

        # Keep the existing wrapper API intact.  The mean route is used only
        # for legacy diagnostics; training and role analysis consume the full
        # [B,T,H,N] tensor stored above.
        mean_alpha = head_alpha.mean(dim=-1)
        native_delta_features = None
        if compute_source_dispersion:
            values_fp = values.float()
            value_heads_fp = value_heads.float()
            routed_fp = routed.float()
            routed_heads_fp = routed_heads.float()
            head_alpha_fp = head_alpha.float()
            dispersion = (
                head_alpha_fp[..., None]
                * (value_heads_fp - routed_heads_fp[None, ...]).pow(2)
            ).sum(dim=0).mean(dim=(-1, -2)).sqrt()
            routed_scale = routed_fp.pow(2).mean(dim=-1).sqrt().clamp_min(1.0e-8)
            source_dispersion = dispersion / routed_scale
            source_deltas = values_fp[1:] - values_fp[:-1]
            latest_scale = values_fp[-1].pow(2).mean(dim=-1).sqrt().clamp_min(1.0e-8)
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
                prefix_mass = mean_alpha.float().cumsum(dim=0)[:-1]
                path_rms = (prefix_mass * source_delta_rms).sum(dim=0)
                recent_delta = source_deltas[-1]
                recent_delta_rms = source_delta_rms[-1]
                older_delta_rms = (
                    source_delta_rms[:-1].mean(dim=0)
                    if source_delta_rms.shape[0] > 1
                    else recent_delta_rms
                )
                delta_trend_ratio = recent_delta_rms / older_delta_rms.clamp_min(1.0e-8)
                recent_delta_cosine = (
                    (route_delta * recent_delta).sum(dim=-1)
                    / (
                        route_delta.norm(dim=-1).clamp_min(1.0e-8)
                        * recent_delta.norm(dim=-1).clamp_min(1.0e-8)
                    )
                )
            native_delta_features = {
                "source_dispersion": source_dispersion,
                "delta_path_ratio": path_rms / latest_scale,
                "delta_cancellation_ratio": route_delta_rms / path_rms.clamp_min(1.0e-8),
                "recent_delta_ratio": recent_delta_rms / latest_scale,
                "delta_trend_ratio": delta_trend_ratio,
                "recent_delta_cosine": recent_delta_cosine,
            }
        return (
            routed,
            mean_alpha.permute(1, 2, 0),
            native_delta_features,
            values,
        )

    router._route_impl = types.MethodType(_role_route_impl, router)
    return RouterInstallReport(
        variant,
        int(num_heads),
        int(num_basis),
        additional_parameters,
        basis_type,
    )


def capture_head_alpha_list(
    model,
    alpha_list: list[torch.Tensor | None] | None,
) -> list[torch.Tensor | None]:
    """Return one [B,T,H,N] native route tensor for every retrofit block."""
    result: list[torch.Tensor | None] = []
    stored = getattr(model.router, "role_head_alpha_by_position", {})
    for block in range(model.num_blocks):
        position = block + 1
        if position in stored:
            result.append(stored[position])
            continue
        alpha = None if alpha_list is None else alpha_list[block]
        result.append(None if alpha is None else alpha.unsqueeze(-2))
    return result


def routing_role_statistics(
    head_alpha_list: list[torch.Tensor | None],
    *,
    num_bins: int = 4,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Differentiable route entropy and block-role separation statistics."""
    profiles: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for alpha in head_alpha_list:
        if alpha is None:
            continue
        probabilities = alpha.float().clamp_min(1.0e-8)
        entropies.append(
            -(probabilities * probabilities.log()).sum(dim=-1).mean()
        )
        source_count = probabilities.shape[-1]
        if source_count < 2:
            continue
        basis = _triangular_depth_basis(
            source_count,
            num_bins,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        token_profiles = torch.einsum("bthn,nk->bthk", probabilities, basis)
        profile = token_profiles.mean(dim=(0, 1, 2))
        profiles.append(profile / profile.sum().clamp_min(1.0e-8))

    if not entropies:
        raise RuntimeError("no AttnRes routing tensors were captured")
    route_entropy = torch.stack(entropies).mean()
    if len(profiles) < 2:
        zero = route_entropy * 0.0
        return {
            "route_entropy": route_entropy,
            "role_js": zero,
            "pairwise_js": zero,
            "effective_rank": zero + 1.0,
            "profiles": profiles,
        }

    matrix = torch.stack(profiles)
    mean_profile = matrix.mean(dim=0)
    mean_entropy = -(mean_profile * mean_profile.clamp_min(1.0e-8).log()).sum()
    block_entropy = -(
        matrix * matrix.clamp_min(1.0e-8).log()
    ).sum(dim=-1).mean()
    role_js = mean_entropy - block_entropy

    pairwise: list[torch.Tensor] = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            midpoint = 0.5 * (matrix[left] + matrix[right])
            pairwise.append(
                0.5
                * (
                    (
                        matrix[left]
                        * (
                            matrix[left].clamp_min(1.0e-8).log()
                            - midpoint.clamp_min(1.0e-8).log()
                        )
                    ).sum()
                    + (
                        matrix[right]
                        * (
                            matrix[right].clamp_min(1.0e-8).log()
                            - midpoint.clamp_min(1.0e-8).log()
                        )
                    ).sum()
                )
            )
    pairwise_js = torch.stack(pairwise).mean()

    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    normalized = singular / singular.sum().clamp_min(1.0e-8)
    effective_rank = torch.exp(
        -(normalized * normalized.clamp_min(1.0e-8).log()).sum()
    )
    return {
        "route_entropy": route_entropy,
        "role_js": role_js,
        "pairwise_js": pairwise_js,
        "effective_rank": effective_rank,
        "profiles": profiles,
    }
