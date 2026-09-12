# Figure 1 · ImageGen prompt

Date: 2026-09-12

Generator: built-in ImageGen. New raster method figure, followed by one targeted bypass-arrow correction. The selected PNG is `fig1_method_imagegen_v1.png`. Existing figures and manuscript references were not replaced.

## Generation prompt

```text
Use case: scientific-educational.
Create one NEW, high-resolution, publication-quality Figure 1 for a machine-learning research paper about AR-Retrofit, Residual-Strength Modulation (RSM), and ReSkip. This is a clean conceptual METHOD figure for a two-column conference paper, not a technical report, implementation flowchart, poster, or slide. English labels only. Wide landscape canvas, approximately 2.5:1 aspect ratio, white background. Extremely crisp vector-like flat shapes, restrained blue/teal and muted amber accents, dark charcoal lines and typography, generous whitespace, consistent thin arrows, readable publication typography. No shadows, gradients, 3D, decorative icons, brain/robot illustrations, logos, watermark, benchmark numbers, or large overall title.

COMPOSITION: Three adjacent panels with clear headings at the top. The center is the largest. Panel widths roughly 20%, 45%, 35%. All actual hidden-state computation flows top to bottom. Panel labels exactly:
"(a) Standard residual"
"(b) AR-Retrofit with RSM"
"(c) ReSkip at inference"

PANEL A:
A minimal conventional residual schematic: input h at top splits into a vertical path through a gray rectangle labeled "Decoder transformation" and an identity bypass around it. These join at a small circled plus; below is output h'. Show the standard residual idea only. Keep this small and visually quiet.

PANEL B:
The main design. At the top, show a compact row of three small history-state tiles, with labels "h₀", "⋯", "h" (h is the most recent completed state); bracket caption "Completed states". All feed a blue rectangle "Cross-layer aggregation". Tiny unequal attention weights/connection thicknesses suggest content-dependent source selection, not sequence attention.
Its output r feeds a blue rectangle "Residual correction" with a small, legible sublabel "u = γ A(r − h)".
Its output u then passes vertically through a muted-amber rectangle "RSM" with the sublabel "Token-wise strength" and output label "g · u". RSM is INSIDE the correction branch, not a separate skip controller.
A distinct dark identity path starts from the most recent state h, runs around the left side of this branch, and joins g · u at a circled PLUS below RSM. Label the combined state "x = h + g · u".
Then a gray rectangle "Frozen decoder block" receives x; below it is output h'.
Place a subtle blue grouping boundary around Cross-layer aggregation + Residual correction + RSM, labeled "Joint Full-path adaptation". The gray frozen decoder block is outside this blue grouping boundary.
A short note below this panel reads "Identity-preserving initialization".
Crucial topology: h is added exactly ONCE, after the correction has been scaled. RSM scales u, NOT the whole hidden state or decoder output. γ belongs to residual injection; do not mark γ as a learned gate. Do not show a skip training branch.

PANEL C:
Explain information reuse, not a new trained predictor. Near the top, one compact light neutral box labeled "Frozen selection rule", with smaller sublabel "Calibrated offline".
Dashed blue/amber information arrows originate from panel B's Cross-layer aggregation and RSM, travel cleanly in the whitespace between the panels, and enter this rule box. Label these arrows collectively "Reuse native signals"; near the amber signal write "g (optional)" and near the blue signal write "Routing weights". Ensure these are information arrows, not hidden-state computation arrows.
Below the rule, illustrate TWO token-dependent execution paths side by side. Heading of left path "Token A"; heading of right path "Token B". Each path consists of four aligned small gray decoder-layer rectangles in a vertical stack, labeled "L1", "L2", "L3", "L4".
For Token A, a solid teal path passes through all four layers; label below "Full execution".
For Token B, solid teal path passes through L1 and L3; L2 and L4 remain visible as pale dashed-outline rectangles, with clearly drawn dashed bypass arrows around those two specific layer computations. Label below "Selective execution".
Both paths keep the same input and output representation dimensions. Do NOT draw an entire decoder block disappearing, early exit, or different output heads. These are illustrative per-token paths, not exact threshold specifications.
A small note below reads "No additional parameter training". No safety guarantee or claimed accuracy equivalence.

LEGEND:
A tiny shared bottom legend, using simple swatches/lines only: blue fill "Trainable adaptation"; gray fill "Frozen backbone"; dashed path "Bypassed computation". RSM's amber is a highlight inside trainable adaptation, not a third contribution.
Keep all text short and legible. Math labels must be exactly as given. Avoid overlapping arrows, awkward line breaks, duplicated labels, long captions, and visual clutter. The figure must communicate these two ideas immediately: learn cross-layer residual correction with internal strength modulation; then reuse its native signals for selective inference.
```

## Final targeted edit

```text
Use case: precise-object-edit. Image 1 is the edit target: the paper Figure 1 just generated.
Make ONE targeted correction: fix the Token B bypass connections in panel (c). Preserve the canvas, panel sizes, all headings, all existing text, typography, colors, and the entirety of panels (a) and (b), including every formula. Preserve the Token A path unchanged.

The Token B execution topology must be: input -> L1 -> bypass L2 -> L3 -> bypass L4 -> output.
Keep L2 and L4 as pale dashed-outline boxes, but disconnect them completely from the execution path.
Delete the current solid teal arrows leading into and out of L2 and L4. Delete the two curved dashed arrows whose arrowheads currently point INTO L2 and L4; those arrows are semantically wrong.
Instead, draw:
1. A single neat dashed dark-teal orthogonal arrow from the bottom edge of L1, detouring to the RIGHT around the entire L2 box, then turning back LEFT into the TOP edge of L3. Its arrowhead must point into L3, not L2.
2. A single neat dashed dark-teal orthogonal arrow from the bottom edge of L3, detouring to the RIGHT around the entire L4 box, then turning back LEFT to the original output line BELOW L4. Its arrowhead must point to the output below L4, not into L4.
The dashed paths must visibly bypass those full boxes with clear white space. No path should cross or enter L2 or L4. Keep the solid teal arrow entering L1. Retain the "Selective execution" label below, and do not add any new text.
Do not change any other element. Preserve high-resolution crisp publication quality.
```
