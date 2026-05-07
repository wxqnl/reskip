# Paper drafts

This directory is an archived Codex draft area. The final paper data source is `../Reskip_5_7_3.pdf`, and the canonical project-side data table is `../paper/Reskip_5_7_3_FINAL_DATA.md`.

| File | Title | Core framing | Status |
|---|---|---|---|
| `main.tex` / `main.pdf` | "AR-Retrofit: Retrofitting Pretrained Decoders with Attention Residuals" | Archived source snapshot; use `../paper/main.tex` for the active source | archived |
| `main_v1.tex` / `main_v1.pdf` | "Attention Residuals as Adaptive Computation Routers: Input-Dependent Depth for LLMs and Vision-Language-Action Models" | Three parallel contributions: ReSkip, ReLoop, VLA modality-aware skip | v1, kept for reference |

## Build

```bash
bash compile.sh            # builds main.pdf (v2 retrofit)
pdflatex main_v1.tex && bibtex main_v1 && pdflatex main_v1.tex && pdflatex main_v1.tex  # builds main_v1.pdf (v1)
```

## When to update which

- **`main.tex`**: archived. Do not use as paper-facing truth unless it has been manually re-synced with `../paper/Reskip_5_7_3_FINAL_DATA.md`.
- **`main_v1.tex`**: only update if we need to revive the parallel-contributions framing (e.g., if retrofit fails and we fall back to from-scratch-only story). Currently frozen.

## Figures

All figures under `figures/` are shared between versions. See `figures/generate_reskip_figures.py` for the reproducible generation script (uses real 340M FineWeb-Edu results from `../outputs/`).
