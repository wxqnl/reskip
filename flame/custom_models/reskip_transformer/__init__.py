from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_reskip_transformer import ReskipTransformerConfig
from .modeling_reskip_transformer import (
    ReskipTransformerForCausalLM,
    ReskipTransformerModel,
)

__all__ = [
    "ReskipTransformerConfig",
    "ReskipTransformerModel",
    "ReskipTransformerForCausalLM",
]

AutoConfig.register("reskip_transformer", ReskipTransformerConfig)
AutoModel.register(ReskipTransformerConfig, ReskipTransformerModel)
AutoModelForCausalLM.register(ReskipTransformerConfig, ReskipTransformerForCausalLM)
