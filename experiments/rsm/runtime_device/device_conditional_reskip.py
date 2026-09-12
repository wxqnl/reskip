from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

try:
    from .triton_device_action_selector import DeviceActionMaskSelector
    from .triton_native_token_selector import DeviceNativeTokenMaskSelector
except ImportError:
    from triton_device_action_selector import DeviceActionMaskSelector
    from triton_native_token_selector import DeviceNativeTokenMaskSelector


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    module_name = "o27_device_conditional_graph_ext_v2"
    try:
        _EXTENSION = importlib.import_module(module_name)
        return _EXTENSION
    except ImportError:
        pass
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / "csrc" / "device_conditional_graph_ext.cu"
    _EXTENSION = load(
        name=module_name,
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        verbose=False,
    )
    return _EXTENSION


def _rms_norm(value: torch.Tensor, epsilon: float = 1.0e-6) -> torch.Tensor:
    fp32 = value.float()
    inverse = torch.rsqrt(fp32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (fp32 * inverse).to(value.dtype)


def _static_tensor(value: torch.Tensor | None):
    return torch.empty_like(value) if value is not None else None


class _ConditionalLayerGraph:
    def __init__(
        self,
        mask: torch.Tensor,
        layer,
        hidden_states: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
        kwargs: dict,
        kv_only_function,
    ):
        extension = _load_extension()
        self.mask = mask
        self.layer = layer
        self.past_key_values = past_key_values
        self.static_input = torch.empty_like(hidden_states)
        self.static_output = torch.empty_like(hidden_states)
        self.static_attention_mask = _static_tensor(attention_mask)
        self.static_position_ids = _static_tensor(position_ids)
        self.static_cos = torch.empty_like(position_embeddings[0])
        self.static_sin = torch.empty_like(position_embeddings[1])
        self.static_cache_position = torch.empty_like(cache_position)
        self.static_tensor_kwargs = {
            key: torch.empty_like(value)
            for key, value in kwargs.items()
            if torch.is_tensor(value)
        }
        self.scalar_kwargs = {
            key: value for key, value in kwargs.items() if not torch.is_tensor(value)
        }
        self.conditional = extension.DeviceConditionalGraph(mask)
        self._copy_inputs(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            cache_position,
            kwargs,
        )

        # Initialize cuBLAS/attention/Triton paths before stream capture. Static
        # cache writes target the same slot and are overwritten by the replay.
        kv_only_function(
            layer,
            self.static_input,
            past_key_values,
            (self.static_cos, self.static_sin),
            self.static_cache_position,
            key_only_rope=True,
            triton_fused_k_norm_rope=True,
        )
        _ = self._full_layer()
        torch.cuda.synchronize(hidden_states.device)

        capture_stream = torch.cuda.Stream(device=hidden_states.device)
        capture_stream.wait_stream(torch.cuda.current_stream(hidden_states.device))
        with torch.cuda.stream(capture_stream):
            # mask == 1: exact KV-only identity branch.
            self.conditional.capture_begin(0)
            kv_only_function(
                layer,
                self.static_input,
                past_key_values,
                (self.static_cos, self.static_sin),
                self.static_cache_position,
                key_only_rope=True,
                triton_fused_k_norm_rope=True,
            )
            self.static_output.copy_(self.static_input)
            self.conditional.capture_end()

            # mask == 0: complete decoder layer.
            self.conditional.capture_begin(1)
            full_output = self._full_layer()
            self.static_output.copy_(full_output)
            self.conditional.capture_end()
        torch.cuda.current_stream(hidden_states.device).wait_stream(capture_stream)
        self.conditional.instantiate()

    def _full_layer(self):
        kwargs = dict(self.scalar_kwargs)
        kwargs.update(self.static_tensor_kwargs)
        output = self.layer(
            self.static_input,
            attention_mask=self.static_attention_mask,
            position_ids=self.static_position_ids,
            past_key_values=self.past_key_values,
            cache_position=self.static_cache_position,
            position_embeddings=(self.static_cos, self.static_sin),
            **kwargs,
        )
        return output[0] if isinstance(output, tuple) else output

    def _copy_inputs(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
        kwargs,
    ):
        if hidden_states.shape != self.static_input.shape:
            raise ValueError("device conditional hidden shape changed")
        self.static_input.copy_(hidden_states)
        self.static_cos.copy_(position_embeddings[0])
        self.static_sin.copy_(position_embeddings[1])
        self.static_cache_position.copy_(cache_position)
        if self.static_attention_mask is not None:
            if attention_mask is None or attention_mask.shape != self.static_attention_mask.shape:
                raise ValueError("device conditional attention-mask shape changed")
            self.static_attention_mask.copy_(attention_mask)
        elif attention_mask is not None:
            raise ValueError("device conditional attention mask changed from None")
        if self.static_position_ids is not None:
            if position_ids is None or position_ids.shape != self.static_position_ids.shape:
                raise ValueError("device conditional position-id shape changed")
            self.static_position_ids.copy_(position_ids)
        elif position_ids is not None:
            raise ValueError("device conditional position ids changed from None")
        for key, target in self.static_tensor_kwargs.items():
            value = kwargs.get(key)
            if value is None or value.shape != target.shape:
                raise ValueError(f"device conditional kwarg {key} changed shape")
            target.copy_(value)

    def run(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
        kwargs,
    ) -> torch.Tensor:
        self._copy_inputs(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            cache_position,
            kwargs,
        )
        self.conditional.replay()
        return self.static_output


class _ConditionalRecoveryGraph:
    def __init__(
        self,
        mask: torch.Tensor,
        adapter,
        hidden_states: torch.Tensor,
    ):
        extension = _load_extension()
        self.adapter = adapter
        self.static_input = torch.empty_like(hidden_states)
        self.static_output = torch.empty_like(hidden_states)
        self.conditional = extension.DeviceConditionalGraph(mask)
        self.static_input.copy_(hidden_states)
        _ = self._recover()
        torch.cuda.synchronize(hidden_states.device)
        capture_stream = torch.cuda.Stream(device=hidden_states.device)
        capture_stream.wait_stream(torch.cuda.current_stream(hidden_states.device))
        with torch.cuda.stream(capture_stream):
            self.conditional.capture_begin(0)
            self.static_output.copy_(self._recover())
            self.conditional.capture_end()
            self.conditional.capture_begin(1)
            self.static_output.copy_(self.static_input)
            self.conditional.capture_end()
        torch.cuda.current_stream(hidden_states.device).wait_stream(capture_stream)
        self.conditional.instantiate()

    def _recover(self):
        return self.static_input + self.adapter(_rms_norm(self.static_input))

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape != self.static_input.shape:
            raise ValueError("device conditional recovery shape changed")
        self.static_input.copy_(hidden_states)
        self.conditional.replay()
        return self.static_output


class DeviceConditionalReskipRuntime:
    """Long-lived batch-1 cached-decode runtime for the frozen O26 policy."""

    def __init__(self, config: dict, risk_groups: dict, device: torch.device):
        self.selector = DeviceActionMaskSelector(config, risk_groups, device)
        self.layer_graphs: dict[int, _ConditionalLayerGraph] = {}
        self.recovery_graphs: dict[int, _ConditionalRecoveryGraph] = {}
        self.cache_identity: int | None = None

    def prepare_cache(self, past_key_values):
        identity = id(past_key_values)
        if self.cache_identity is None:
            self.cache_identity = identity
        elif self.cache_identity != identity:
            torch.cuda.synchronize(self.selector.device)
            self.layer_graphs.clear()
            self.recovery_graphs.clear()
            self.cache_identity = identity

    def reset_history(self):
        self.selector.reset_history()

    def update_gate(
        self,
        gate_block: int,
        packed_scores: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        if gate_block == 1:
            self.selector.update_gate1(packed_scores)
        elif gate_block == 2:
            self.selector.update_gate2(packed_scores, cache_position)
        else:
            raise ValueError(f"unsupported O27 gate block {gate_block}")

    def run_layer(
        self,
        block_idx: int,
        layer_offset: int,
        layer,
        hidden_states: torch.Tensor,
        past_key_values,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
        kwargs,
        kv_only_function,
    ) -> torch.Tensor:
        self.prepare_cache(past_key_values)
        layer_index = int(layer.self_attn.layer_idx)
        graph = self.layer_graphs.get(layer_index)
        if graph is None:
            graph = _ConditionalLayerGraph(
                self.selector.layer_mask(block_idx, layer_offset),
                layer,
                hidden_states,
                past_key_values,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
                kwargs,
                kv_only_function,
            )
            self.layer_graphs[layer_index] = graph
        return graph.run(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            cache_position,
            kwargs,
        )

    def run_recovery(self, block_idx: int, adapter, hidden_states):
        graph = self.recovery_graphs.get(block_idx)
        if graph is None:
            graph = _ConditionalRecoveryGraph(
                self.selector.block_mask(block_idx), adapter, hidden_states
            )
            self.recovery_graphs[block_idx] = graph
        return graph.run(hidden_states)

    def summarize(self) -> dict:
        return self.selector.summarize()


class DeviceConditionalNativeReskipRuntime:
    """Persistent decode runtime for frozen, parameter-free native rules."""

    def __init__(self, config: dict, device: torch.device):
        self.selector = DeviceNativeTokenMaskSelector(config, device)
        self.full_compute_mask = torch.zeros(
            1, device=self.selector.device, dtype=torch.int32
        )
        self.layer_graphs: dict[int, _ConditionalLayerGraph] = {}
        self.cache_identity: int | None = None

    def prepare_cache(self, past_key_values):
        identity = id(past_key_values)
        if self.cache_identity is None:
            self.cache_identity = identity
        elif self.cache_identity != identity:
            torch.cuda.synchronize(self.selector.device)
            self.layer_graphs.clear()
            self.cache_identity = identity

    def reset_history(self):
        self.selector.reset_history()

    def update_block(
        self,
        block_idx: int,
        native_features: dict[str, torch.Tensor],
        cache_position: torch.Tensor,
    ):
        self.selector.update_block(block_idx, native_features, cache_position)

    def update_block_fused(
        self,
        block_idx: int,
        alpha: torch.Tensor,
        source_values: torch.Tensor,
        routed: torch.Tensor,
        previous: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        self.selector.update_block_fused(
            block_idx,
            alpha,
            source_values,
            routed,
            previous,
            cache_position,
        )

    def run_layer(
        self,
        block_idx: int,
        layer_offset: int,
        layer,
        hidden_states: torch.Tensor,
        past_key_values,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
        kwargs,
        kv_only_function,
    ) -> torch.Tensor:
        self.prepare_cache(past_key_values)
        layer_index = int(layer.self_attn.layer_idx)
        graph = self.layer_graphs.get(layer_index)
        if graph is None:
            graph = _ConditionalLayerGraph(
                self.selector.layer_mask(block_idx, layer_offset),
                layer,
                hidden_states,
                past_key_values,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
                kwargs,
                kv_only_function,
            )
            self.layer_graphs[layer_index] = graph
        return graph.run(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            cache_position,
            kwargs,
        )

    def run_full_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        past_key_values,
        attention_mask,
        position_ids,
        position_embeddings,
        cache_position,
        kwargs,
        kv_only_function,
    ) -> torch.Tensor:
        """Replay a persistent full-compute graph for a non-candidate layer."""
        self.prepare_cache(past_key_values)
        layer_index = int(layer.self_attn.layer_idx)
        graph = self.layer_graphs.get(layer_index)
        if graph is None:
            graph = _ConditionalLayerGraph(
                self.full_compute_mask,
                layer,
                hidden_states,
                past_key_values,
                attention_mask,
                position_ids,
                position_embeddings,
                cache_position,
                kwargs,
                kv_only_function,
            )
            self.layer_graphs[layer_index] = graph
        return graph.run(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            cache_position,
            kwargs,
        )

    def summarize(self) -> dict:
        return self.selector.summarize()
