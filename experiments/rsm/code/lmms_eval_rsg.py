"""lmms-eval entry point for residual-strength-gated AttnRes."""
from __future__ import annotations

import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_SNAPSHOT_ROOT = os.path.join(CODE_ROOT, "base_snapshot")
ACTIVE_EVAL_ROOT = (
    "/data/Minko/starVLA_workspace/"
    "reskip_ar_v2_skip_consistent_20260820/retrofit/eval"
)
ACTIVE_RETROFIT_ROOT = os.path.dirname(ACTIVE_EVAL_ROOT)

# The plugin imports this module by name. Preloading the frozen snapshot keeps
# the benchmark on exactly the same wrapper code as training and held-out eval.
sys.path.insert(0, BASE_SNAPSHOT_ROOT)
sys.path.insert(0, CODE_ROOT)
import qwen3vl_attnres_retrofit  # noqa: F401

sys.path.insert(0, ACTIVE_EVAL_ROOT)
sys.path.insert(0, ACTIVE_RETROFIT_ROOT)
import lmms_eval_retrofit as plugin

from checkpoint_utils import load_role_retrofit


def _load_rsg_checkpoint(self, state_path, num_blocks, adapter_rank):
    del num_blocks, adapter_rank
    base_model = self._model
    dtype = next(base_model.parameters()).dtype
    device = next(base_model.parameters()).device
    wrapper, checkpoint = load_role_retrofit(
        base_model,
        state_path,
        device=device,
        dtype=dtype,
    )
    config = checkpoint.get("config", {})
    print(
        f"[rsg-retrofit] state loaded: variant={config.get('router_variant')} "
        f"heads={config.get('routing_heads')} "
        f"residual_strength_gate={config.get('residual_strength_gate', False)} "
        f"additional_params={config.get('additional_trainable_parameters', 0)} "
        f"gamma_mean={float(wrapper.gamma.detach().float().mean()):+.3f}",
        flush=True,
    )
    return wrapper


plugin.Qwen3_VL_Retrofit._load_retrofit = _load_rsg_checkpoint


if __name__ == "__main__":
    import lmms_eval.__main__ as main

    plugin._install_qwen3vl_task_prompt_alias(main)
    main.cli_evaluate()
