# Figure 1 · ImageGen redesign v2

Date: 2026-09-12

Generator: built-in ImageGen. Status: layout/structure draft for author review, not publication-final. This version replaces long text boxes with a residual ladder, explicit cross-depth fan-in and side-branch gain, and two execution paths over one shared layer stack. No existing figures or manuscript references were replaced.

Known visual issue: two redundant arrowheads near h2 in panel (b) survived the final cleanup attempt; the intended directions are h2 -> weighted aggregation and h2 -> subtraction, not feedback into h2. This draft needs that cleanup before publication use.

Selected PNG: `fig1_method_imagegen_v2.png`.

## Initial redesign

```text
Use case: scientific-educational.
Asset: a NEW, substantially redesigned Figure 1 for a machine-learning research paper about AR-Retrofit, Residual Strength Modulation (RSM), and ReSkip. Generate a crisp, publication-style raster diagram on pure white, landscape approximately 2400 × 1000. This is an ARCHITECTURE drawing, not a workflow infographic. Communicate primarily with actual neural-network connections, residual loops, fan-in edges, and shared execution paths. Keep text exceptionally sparse.

Three balanced panels, left 25%, middle 43%, right 32%. Short headings at the bottom, exactly "(a) Standard residual", "(b) AR-Retrofit + RSM", "(c) ReSkip". Align the architecture drawings above these captions. Fine unobtrusive pale vertical dividers. Flat gray backbone modules, blue cross-depth adaptation, amber strength modulation, charcoal ordinary paths, teal selective token path. Thin consistent strokes, generous whitespace, modest sans-serif labels and italic math. No shadows, gradients, huge headings, prose inside boxes, numbered workflow stages, decorative icons, or implementation notes.

(a) Draw a genuine residual decoder STACK, flowing BOTTOM TO TOP, not a single box. Three compact gray decoder units vertically aligned. Each unit visibly contains two short modules, "Attn" then "MLP" in upward order, and the two residual bypass loops rejoining at small circled plus operators after the respective sublayers. A narrow normalization stripe can be shown before each sublayer without any text. Small dark hidden-state nodes between complete units, labeled h₀ at input, h₁, h₂, h₃ at output. Neat recurring geometry makes ordinary adjacent-depth residual connections obvious. The three units have small side labels D₁, D₂, D₃. Do not add history fan-in links here.

(b) Draw the SAME three-unit frozen decoder stack on the RIGHT half of this panel, with D₁, D₂, D₃ and dark state nodes h₀, h₁, h₂, h₃. Simplified gray units may use only their D labels because (a) already exposes their internals. Concentrate the colored adaptation drawing in the open space to the LEFT of this backbone. Show one representative retrofit site BETWEEN h₂ and D₃; lower sites need not be expanded.

Essential connectivity:
- Three blue curved edges branch from the completed-state nodes h₀, h₁, and h₂ and FAN INTO one small blue weighted-sum circle labelled "Σα". Use unequal line widths to encode different weights; a tiny group of three blue weight bars immediately beside the sum makes learned depth selection visible. This must look like cross-depth connectivity attached to the backbone, NOT boxes in a serial pipeline.
- The weighted-sum output is r. In this SIDE BRANCH, r and a thin tap from current state h₂ meet at a compact subtract operator representing r − h₂; mark the incoming signs unambiguously. Feed this into a small blue adapter module labelled "γA". Its output is u.
- The side-branch correction u passes through a small multiplication circle. An AMBER mini-controller labelled "RSM" sits beside this circle and supplies g on a short arrow. Use a small amber gain dial/short variable-width bar so strength is visually represented, not explained in a paragraph. RSM controls ONLY the correction branch. It must not replace or gate the main identity stream.
- The main BLACK line from h₂ runs directly upward into a circled + located immediately below D₃. The blue/amber scaled correction joins this SAME plus from the side. Above this plus, x enters D₃ and D₃ outputs h₃.
- Put only the compact equation "x = h₂ + g·u" next to that junction, outside any box. No other long formulas or explanatory sentences.
- A light blue grouping contour can loosely enclose the side-branch adaptation circuit, excluding the frozen backbone. Small outside labels "AR" near the fan-in and "RSM" only on its mini-controller; never a big paragraph block.
This panel must visibly distinguish WHERE depth information comes from (blue fan-in) from HOW STRONGLY the correction is injected (amber gain).

(c) Draw ONE SHARED decoder stack, not two separate columns of network boxes. Four wide pale-gray horizontal units, with small labels D₁, D₂, D₃, D₄ on their left edges, ordered BOTTOM TO TOP. Overlay two thin separated vertical execution lanes across these SAME four shared units. Place small token circles below the stack, "tₐ" on the left lane and "tᵦ" on the right lane; matching output circles at the top.
- Charcoal tₐ path runs straight upward through all four units.
- Teal tᵦ path goes through D₁, then detours completely to the RIGHT around D₂ and returns into D₃, then detours to the RIGHT around D₄ and rejoins at its output above D₄.
- The bypass sections are DASHED teal and visibly stay outside the full rectangular units. No teal segment enters D₂ or D₄. Arrowheads on the first bypass lead into D₃, never D₂; arrowheads on the second bypass lead to the output above D₄, never D₄. The charcoal path still traverses D₂ and D₄.
- At the upper-left of this panel, a tiny two-row native-signal glyph: blue weight bars labelled "α", amber gain bar labelled "g". A single short arrow labelled "fixed rule" leads from this glyph toward the route-selection junctions. Thin dashed taps from the blue weighting and amber gain in (b) can connect to this glyph across the panel boundary, routed in clear whitespace. This represents reuse of existing signals, NOT an additional learned network.
Only the two small captions "full" and "selective" beneath the corresponding token circles. A tiny dashed sample labelled "bypass" at the lower-right is enough of a legend.

Final priorities: scientific topology correct; sparse labels; genuinely visual structure rather than a list of text boxes; similar visual density across the three panels; restrained beautiful conference-paper design; no extra explanatory copy, no speed numbers, no training-loss, CUDA, cache or calibration machinery.
```

## Topology correction

```text
Use case: precise-object-edit.
Image 1 is the edit target: the newly redesigned three-panel research diagram. Preserve the overall layout, bottom captions, colors, font style, whitespace and all module locations. Correct NETWORK CONNECTIONS, not the visual style. There are three precise corrections.

1. PANEL (c): DELETE the erroneous straight teal path through D₂ and D₄. Keep the existing dashed teal bypass curves to their RIGHT.
For the lower bypass, teal must leave the TOP of D₁, travel right and upward AROUND D₂, and reenter the BOTTOM of D₃. Remove the entire straight teal vertical segment between D₁ and D₃, including the teal segment and teal arrowheads INSIDE D₂.
For the upper bypass, teal must leave the TOP of D₃, travel right and upward AROUND D₄, then reach the teal output dot ABOVE D₄. Remove the entire straight teal segment from D₃ up to the output-line merge, including all teal lines and teal arrowheads INSIDE D₄.
After editing, D₂ and D₄ contain ONLY the straight BLACK execution lane. D₁ and D₃ contain both black and teal execution lanes. Preserve the black path unchanged. This is the most important correction.

2. PANEL (a): correct the repeated residual units. Each of D₁, D₂ and D₃ must contain exactly one lower module "Attn" and one upper module "MLP". D₁ currently wrongly has two Attn labels; change the upper one to "MLP".
Inside EACH unit, redraw a correct pre-norm residual transformer:
input -> thin Norm stripe -> Attn -> first small circled plus -> thin Norm stripe -> MLP -> second small circled plus -> output, flowing upward.
Draw one identity loop from the INPUT of that unit into its FIRST plus (after Attn).
Draw a separate identity loop from the FIRST plus into its SECOND plus (after MLP).
The two sums are distinct. Every module and every plus must have an upward data-flow arrow. There must be NO downward feedback arrows. Keep each decoder outer gray boundary and the inter-unit h₀,h₁,h₂,h₃ nodes.

3. PANEL (b): make the cross-depth fan-in physically connect to the backbone.
REMOVE the three duplicated h₀,h₁,h₂ dots currently in a row below the blue adaptation region, and remove their dangling fan-in stems.
Instead, draw THREE curved blue weighted connections branching from the actual black state nodes ON THE BACKBONE: h₀ below D₁, h₁ between D₁ and D₂, and h₂ between D₂ and the injection plus. Add the small black h₂ junction dot on that existing black line if necessary. All three curved blue links converge directly into the EXISTING blue "Σα" circle to the left. Distinguish weights with thin/medium/thick blue lines. This is a lateral cross-depth fan-in graph attached to the backbone, not a disconnected memory row.
The negative black input to the subtraction circle must now come from that SAME actual h₂ junction on the backbone, not from any deleted copy. Use a thin tidy black line.
Preserve the rest of (b): r -> subtraction -> γA -> multiplication by amber g -> main plus -> D₃. Preserve the direct BLACK h₂ identity path to that main plus. Keep "x = h₂ + g·u", the RSM controller and native-signal taps. Do not move the frozen decoder stack or other adaptation operators.

Do not add any extra words, prose, legends, panels, cards, decorations or formulas. The final picture must remain as sparse as the edit target but with these precise scientific connections corrected.
```

## Residual ladder and backbone fan-in

```text
Edit target: Image 1. Preserve panel (c) EXACTLY: its selective bypasses are now correct. Preserve all three bottom captions, the white background, typography, palette, overall composition, and the α/g signal glyph. Make these architecture-drawing corrections only.

PANEL (a): Replace the ENTIRE internal drawing above its caption with a clean repeated residual-stream schematic. Do NOT show Attn/MLP/Norm internals.
A black vertical identity stream flows UPWARD on the RIGHT of panel (a), with three circled + junctions at evenly spaced heights. The input is h₀ at the bottom; label the states after the three sums h₁, h₂, h₃. For each stage, branch from the stream BEFORE its sum, detour LEFT through one pale gray rectangle F₁, F₂, or F₃, then feed RIGHT into that stage's sum. The main black stream bypasses each F box directly. All flow is upward, except the lateral branch arrows. This creates THREE distinct residual loops in a repeated ladder:
h₁ = h₀ + F₁(h₀), then h₂ = h₁ + F₂(h₁), then h₃ = h₂ + F₃(h₂).
These equations describe the drawing but MUST NOT be printed. Print only F₁,F₂,F₃,h₀,h₁,h₂,h₃ and the plus symbols. Keep the ladder visually substantial, using the full panel height. The diagram itself must be exact, clean, and rich in connections without tiny internals.

PANEL (b): Keep every operator and frozen D box at its current location. Remove the THREE duplicated state dots and their labels in the bottom-left row (h₀,h₁,h₂), including every wire touching that removed row. Remove the black extra dot beside the LEFT edge of D₂ and its connection to the side of D₂; that is not a valid output node.
Keep the existing real h₀ black node BELOW D₁, and the existing real h₁ node BETWEEN D₁ and D₂. Put the real h₂ black node on the VERTICAL BLACK LINE BETWEEN THE TOP OF D₂ AND THE MAIN INJECTION PLUS.
Now draw exactly three curved BLUE fan-in arrows: real h₀ -> Σα; real h₁ -> Σα; real h₂ -> Σα. All arrowheads are AT Σα, and every tail is at its named REAL node on the right backbone. There are NO state copies and NO nodes underneath the blue region.
Draw a thin BLACK tap from the real h₂ node to the negative input of the subtract circle. It branches only after D₂ has completed.
Retain the rest of the blue and amber correction branch exactly as it is, including γA, u, RSM, multiplication, and its arrow entering the main plus. Preserve the direct black identity route from real h₂ to the main plus and then into D₃.

No other edits. No extra prose or decorative elements.
```

## Fan-in source correction

```text
Precise edit of Image 1. Preserve ALL of panels (a) and (c). In panel (b), preserve every symbol, box, text label, position and existing operator connection, except for the four input wires described here.

The blue circle Σα needs THREE inputs from THREE DIFFERENT black backbone dots, not repeated inputs from h₀.
1. Keep ONE existing blue arrow from the black h₀ dot below D₁ into Σα. DELETE the second redundant bowed blue arrow from h₀.
2. ADD a curved blue arrow from the black h₁ dot BETWEEN D₁ and D₂ into the RIGHT EDGE of Σα. Its tail must touch the black h₁ dot; its arrowhead must touch Σα.
3. ADD a curved blue arrow from the black h₂ dot BETWEEN D₂ and the injection + into the UPPER-RIGHT EDGE of Σα. Its tail must touch that black h₂ dot; its arrowhead must touch Σα. Route in the white space left of D₂.
4. The existing horizontal black wire into the minus circle wrongly starts at the LEFT EDGE of the gray D₂ rectangle. Disconnect it from that rectangle. Connect its source to the black h₂ DOT ABOVE D₂ instead: start at that dot, go LEFT outside the rectangle, turn DOWN, then LEFT into the existing negative port of the minus circle. The black line must NOT touch the left edge of the D₂ box.

After editing, there must be one blue input each from h₀, h₁, and h₂ into Σα, plus one black input from h₂ into the minus circle. No copies of h nodes, no extra dots, no other edits.
```

## Attempted arrowhead cleanup

```text
Image 1 is the edit target. Make only one microscopic cleanup in panel (b): remove TWO redundant arrowheads next to the black h₂ dot above D₂.
- The curved blue line connects h₂ to Σα. Remove its arrowhead at the h₂ END, and keep its arrowhead at the Σα END. Keep the blue curve attached to h₂ as a plain source tail.
- The thin black orthogonal line connects h₂ to the negative port of the minus circle. Remove its arrowhead at the h₂ END, and keep its arrowhead at the minus-circle END. Keep the black line attached to h₂ as a plain source tail.
Thus both lines flow OUT of h₂ toward the side branch; neither line points back into h₂. Do not remove the separate vertical black backbone arrow from D₂ toward h₂.
Preserve every other pixel and all text, modules, paths, colors and layout. Do not redraw the diagram. No other changes.
```
