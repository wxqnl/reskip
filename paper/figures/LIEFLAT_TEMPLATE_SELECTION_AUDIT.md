# Lieflat chart-selection audit

Global system: **Porcelain**. It matches an academic/technical paper and uses one ordered blue-lightness system throughout. No chart mixes palettes.

| Figure | Candidate 1 | Candidate 2 | Candidate 3 | Selected | Reason |
|---|---|---|---|---|---|
| S1 gate scale | F2 Hairline Line | F8 Plumb Scatter | L11 Trend Lineage | F2 | Four ordered scalar settings form a sparse sequence; a hairline preserves the response shape without implying dense sampling. |
| S2 block × token | L4 Arc Matrix | L16 Matrix Heat | F10 Dot Heat | L16 + F8 inset | The 6/8 × 3 values are continuous specialization strengths, so lightness is honest; F10 area would imply counts. The causal inset is a genuine two-variable scatter. |
| S3 Pareto | F2 Hairline Line | F8 Plumb Scatter | L20 Parallel Coordinates | F8 | Both axes are observed continuous quantities and there are fewer than 20 points. Plumb lines make actual compute readable. |
| S4 runtime | F6 Paired Rungs | F12 Dumbbell Queue | F2 Hairline Line | F6 | Each coverage cell compares two commensurate speedup ratios; one rung is explicitly 0.01×. F12 beads would falsely imply time saved as countable units. |

Template source: the real `basics-gallery.html` F2/F6/F8 skeletons and `lupi-gallery.html` L16 matrix skeleton. Color tokens come from the real `color-presets.js` Porcelain preset.
