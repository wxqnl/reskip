from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class ReskipQwen3Config(PretrainedConfig):
    model_type = "reskip_qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = 4096,
        max_window_layers: int = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        fuse_norm: bool = False,
        fuse_cross_entropy: bool = False,
        fuse_linear_cross_entropy: bool = False,
        use_attn_res: bool = False,
        attn_res_num_blocks: int = 8,
        attn_res_temperature: float = 1.0,
        attn_res_output_norm: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 1000000.0,
        }
        self.attention_bias = attention_bias
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types or [
            "sliding_attention" if self.sliding_window is not None and i >= self.max_window_layers else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
        self.attention_dropout = attention_dropout

        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy

        self.use_attn_res = use_attn_res
        self.attn_res_num_blocks = attn_res_num_blocks
        self.attn_res_temperature = attn_res_temperature
        self.attn_res_output_norm = attn_res_output_norm

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
