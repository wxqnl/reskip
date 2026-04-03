from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_reskip_qwen3 import ReskipQwen3Config
from .modeling_reskip_qwen3 import (
    ReskipQwen3ForCausalLM,
    ReskipQwen3Model,
)

__all__ = [
    "ReskipQwen3Config",
    "ReskipQwen3Model",
    "ReskipQwen3ForCausalLM",
]

AutoConfig.register("reskip_qwen3", ReskipQwen3Config)
AutoModel.register(ReskipQwen3Config, ReskipQwen3Model)
AutoModelForCausalLM.register(ReskipQwen3Config, ReskipQwen3ForCausalLM)
