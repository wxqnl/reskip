from transformers.configuration_utils import PretrainedConfig


class ReskipTransformerConfig(PretrainedConfig):
    model_type = "reskip_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        fuse_norm: bool = False,
        fuse_cross_entropy: bool = False,
        fuse_linear_cross_entropy: bool = False,
        use_attn_res: bool = False,
        attn_res_num_blocks: int = 8,
        attn_res_temperature: float = 1.0,
        attn_res_output_norm: bool = True,
        enable_skipping: bool = False,
        skip_threshold: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.attention_bias = attention_bias
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy

        self.use_attn_res = use_attn_res
        self.attn_res_num_blocks = attn_res_num_blocks
        self.attn_res_temperature = attn_res_temperature
        self.attn_res_output_norm = attn_res_output_norm
        self.enable_skipping = enable_skipping
        self.skip_threshold = skip_threshold

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
