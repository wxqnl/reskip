# VLA landscape research notes (2026-04-29)

Background research on LIBERO 4-suite numbers from recent VLA papers, to expand the
paper's `tab:vla_landscape` table.

## Candidate rows (cited / re-run, no new experiments)

| Model | Params | Spatial | Object | Goal | Long-10 | Avg | Citation | Confidence |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 7B | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | kim2025openvlaoft (arXiv:2502.19645) | self-reported |
| CogACT | 7B | 97.2 | 98.0 | 90.2 | 88.8 | 93.55 | li2024cogact (arXiv:2411.19650) | re-run by MemoryVLA harness (within 1pp of CogACT self-report) |
| π0-FAST | 3B | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | pertsch2025fast (arXiv:2501.09747) | re-run by InternVLA-M1 / MemoryVLA |
| π0.5-KI | ~3B | 98.0 | 97.8 | 95.6 | 85.8 | 94.3 | physicalintelligence2025pi05 | re-run by InternVLA-M1 |
| NORA-Long | 3B | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 | hung2025nora (arXiv:2504.19854) | self-reported |
| CoT-VLA | 7B | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 | zhao2025cotvla (arXiv:2503.22020) | re-run by InternVLA-M1 / MemoryVLA |
| GR00T N1 | ~2.7B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 | nvidia2025groot (arXiv:2503.14734) | re-run by InternVLA-M1 |
| MemoryVLA | 7B+0.3B | 98.4 | 98.4 | 96.4 | 93.4 | 96.65 | shi2025memoryvla (arXiv:2508.19236) | self-reported |
| InternVLA-M1 | 4.1B | 98.0 | 99.0 | 93.8 | 92.6 | 95.85 | shanghaiailab2025internvlam1 (arXiv:2510.13778) | self-reported |
| VLA-Adapter | 0.5B+97M | 99.6 | 99.6 | 98.2 | 96.4 | 98.45 | wu2025vlaadapter (arXiv:2509.xxxxx) | self-reported (small-backbone outlier) |
| Diffusion Policy | <100M | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | chi2023diffusionpolicy | re-run by OpenVLA-OFT |
| MDT (DiT) | <100M | 78.5 / 84.2 | 87.5 / 96.3 | 73.5 / 85.4 | 64.8 / 63.8 | 76.1 / 82.4 | reuss2024mdt; hou2024dit | re-run by OpenVLA-OFT |

## Excluded (and why)

- **Magma (8B)**: only 10-trajectory few-shot LIBERO bar; not full-finetune comparable.
- **DexVLA, TinyVLA, RDT-1B, GR-1, GR-MG, RoboFlamingo, MoLe-VLA**: do not report LIBERO 4-suite (real-world only / Meta-World / bimanual / RLBench).
- **MiniVLA**: LIBERO-90 only, not the 4 suites.

## Typical landscape-table size in 2025-2026 VLA papers

- SpatialVLA (Qu 2025): 5 rows
- NORA (Hung 2025): 6 rows
- π0-FAST (Pertsch 2025): 5 rows (Fig.6)
- OpenVLA-OFT (Kim 2025): 7 rows
- CogACT-derived re-runs (MemoryVLA 2025): 14 rows
- InternVLA-M1 (2025): 9 rows
- VLA-Adapter (2025): 8 rows

**Median ~7 rows; range 5-14**.

## Recommended additions for our `tab:vla_landscape` (5 strong rows)

Existing 5 (Octo / OpenVLA / TraceVLA / SpatialVLA / π0) → keep. Add:

1. **OpenVLA-OFT (7B)** — direct head ancestor of our method.
2. **CogACT (7B)** — most-cited 2024 VLA, comparable param tier.
3. **π0-FAST (3B)** — closest matched-budget to our 4B retrofit.
4. **NORA-Long (3B)** — small-VLM peer at 2B/3B param tier.
5. **GR00T N1 (~2.7B)** — NVIDIA reference point.

Optionally add MemoryVLA / VLA-Adapter as "frontier ceiling" rows showing our 96-97% mean is consistent with the SOTA cluster.

Final table size: 10 rows + 2 ours = 12 rows. Comfortably within the InternVLA-M1 / MemoryVLA range.

## Sources

- [OpenVLA-OFT](https://arxiv.org/html/2502.19645v2)
- [CogACT](https://arxiv.org/abs/2411.19650)
- [SpatialVLA](https://arxiv.org/html/2501.15830v2)
- [NORA](https://arxiv.org/html/2504.19854v1)
- [InternVLA-M1](https://arxiv.org/html/2510.13778v1)
- [MemoryVLA](https://arxiv.org/html/2508.19236v1)
- [VLA-Adapter](https://vla-adapter.github.io/)
- [π0-FAST](https://arxiv.org/html/2501.09747v1)
