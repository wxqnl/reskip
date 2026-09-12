# ReSkip / Residual-Strength Modulation

This branch contains the current RSM extension of ReSkip, the frozen token-level skip policies, the paper source, and the compact evidence package used by the manuscript.

Residual-Strength Modulation (RSM) is trained inside the full AttnRes path. For block \(b\), it rescales the existing AttnRes correction \(u_b\):

```text
g_b = 1 + 0.5 tanh((v_b^T RMSNorm(u_b) + c_b) / sqrt(d))
x_b = h_(b-1) + g_b u_b
```

The projection starts at zero, so `g_b = 1` at initialization. Training uses the full path and adds no skip labels, compute target, selector head, or skip-specific loss.

At inference time, ReSkip reads the AttnRes routing weight on the most recent completed block,

```text
w_recent(b,t) = alpha_(b-1 -> b,t),
```

and compares it with a frozen, block-specific threshold. The selected action skips only calibrated decoder-layer offsets and still updates the K/V cache. Decisions are made independently for each decode token.

## Active directories

| Path | Contents |
|---|---|
| [`paper/`](paper/) | Current manuscript, canonical figures, result tables, analysis, policies, protocols, and runtime receipts |
| [`experiments/rsm/`](experiments/rsm/) | Exact RSM training, evaluation, calibration, device-runtime, LM, and VLA source snapshots |
| [`retrofit/`](retrofit/) | Earlier AR-Retrofit baseline retained for context; the frozen run dependency is under `experiments/rsm/code/base_snapshot/` |
| [`starVLA/`](starVLA/) | VLA framework used for the LIBERO transfer experiments |
| [`flash-linear-attention/`](flash-linear-attention/) and [`flame/`](flame/) | Original from-scratch ReSkip model and training stack |

Start with [`paper/README.md`](paper/README.md). The authoritative paper numbers and their limits are in [`paper/PAPER_NUMBERS.md`](paper/PAPER_NUMBERS.md). Implementation entry points are documented in [`experiments/rsm/README.md`](experiments/rsm/README.md).

## Evidence status

- Qwen3-VL-2B/4B full-path RSM results use three training seeds.
- Qwen3-VL-2B/4B ReSkip results use frozen, matched-seed operating points.
- Fixed-64 H100 timing covers six tasks, 12 requests per task, and three repeats.
- SmolVLM2 supports the cross-architecture analysis; InternVL did not pass the positive-transfer criterion.
- LIBERO results are mixed across scale and training path. The incomplete 2B matched rerun is retained for continuation and is excluded from formal paper numbers.
- Seven zero-shot language-model benchmarks were completed on 2026-09-11 and remain supplemental evidence.

Large checkpoints, datasets, generated caches, logs, and compiled CUDA binaries stay on the experiment server. [`paper/reproducibility/SOURCE_PATHS.md`](paper/reproducibility/SOURCE_PATHS.md) records their canonical locations.

## Environment

The recorded experiments ran from `/data/Minko` on the `New-H100-2` host with H100 GPUs. Model and dataset paths are supplied through each entry point's command-line arguments or the preserved VLA configs. The device runtime compiles its CUDA extension locally on first use.

## License

MIT
