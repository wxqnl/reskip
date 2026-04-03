from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .config_reskip_transformer import ReskipTransformerConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bs,d->bsd", position_ids.to(self.inv_freq.dtype), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, :, :].to(dtype=dtype)
        sin = emb.sin()[:, None, :, :].to(dtype=dtype)
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(dtype=x.dtype)


class BlockAttnRes(nn.Module):
    def __init__(self, d_model: int, num_blocks: int, block_idx: int, temperature: float = 1.0):
        super().__init__()
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.temperature = temperature
        self.block_idx = block_idx
        self.num_blocks = num_blocks

    def forward(self, block_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(block_outputs) == 1:
            return block_outputs[0]
        sources = torch.stack(block_outputs, dim=2)
        _, _, _, hidden_size = sources.shape
        keys = self.key_proj(sources)
        values = self.value_proj(sources)
        query = self.w_query.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = (query * keys).sum(dim=-1) / (math.sqrt(hidden_size) * self.temperature)
        weights = F.softmax(scores, dim=-1)
        combined = (weights.unsqueeze(-1) * values).sum(dim=2)
        return self.norm(combined)


class ReskipAttention(nn.Module):
    def __init__(self, config: ReskipTransformerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta) if config.use_rope else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            cos, sin = self.rotary(position_ids, q.dtype)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        attn_mask = None
        is_causal = attention_mask is None
        if attention_mask is not None:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
            )[None, None, :, :]
            key_padding_mask = attention_mask[:, None, None, :].to(torch.bool)
            attn_mask = causal_mask & key_padding_mask

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return (self.o_proj(attn_output),)


class ReskipMLP(nn.Module):
    def __init__(self, config: ReskipTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ReskipDecoderLayer(nn.Module):
    def __init__(self, config: ReskipTransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ReskipAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = ReskipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )[0]
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states), **kwargs)
        return (hidden_states,)


class ReskipTransformerPreTrainedModel(PreTrainedModel):
    config_class = ReskipTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReskipDecoderLayer"]
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class ReskipTransformerModel(ReskipTransformerPreTrainedModel):
    def __init__(self, config: ReskipTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ReskipDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False

        self.use_attn_res = config.use_attn_res
        self.layers_per_block = max(1, config.num_hidden_layers // max(config.attn_res_num_blocks, 1))
        self.num_blocks = math.ceil(config.num_hidden_layers / self.layers_per_block)
        if self.use_attn_res:
            self.block_attn_res = nn.ModuleList(
                [
                    BlockAttnRes(
                        config.hidden_size,
                        self.num_blocks,
                        block_idx,
                        temperature=config.attn_res_temperature,
                    )
                    for block_idx in range(self.num_blocks)
                ]
            )
            self.block_output_norm = (
                nn.ModuleList([RMSNorm(config.hidden_size, eps=config.norm_eps) for _ in range(self.num_blocks)])
                if config.attn_res_output_norm
                else None
            )
        else:
            self.block_attn_res = None
            self.block_output_norm = None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del past_key_values, cu_seqlens
        output_attentions = False if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        use_cache = False if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        if output_attentions:
            output_attentions = False
        if use_cache:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if inputs_embeds is None:
            hidden_states = self.embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)

        all_hidden_states = () if output_hidden_states else None
        block_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )[0]

            if self.use_attn_res:
                is_block_end = ((layer_idx + 1) % self.layers_per_block == 0) or (layer_idx == len(self.layers) - 1)
                if is_block_end:
                    block_idx = len(block_outputs)
                    if block_outputs:
                        routed = self.block_attn_res[block_idx](block_outputs)
                        hidden_states = hidden_states + routed
                        if self.block_output_norm is not None:
                            hidden_states = self.block_output_norm[block_idx](hidden_states)
                    block_outputs.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class ReskipTransformerForCausalLM(ReskipTransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ReskipTransformerConfig):
        super().__init__(config)
        self.model = ReskipTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        del past_key_values, use_cache
        if inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update({"attention_mask": attention_mask})
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_linear_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
