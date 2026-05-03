result: RESKIP’s safety margin under threshold/distribution mismatch is high (no degradation when the rule rarely fires),
but deployment requires per-distribution threshold calibration, as already noted for the cross-modality VLA case (§F.4).
D.6
RESKIP full Pareto and latency curves at 340M
6.6
6.8
7.0
7.2
7.4
7.6
7.8
8.0
Average blocks executed per forward (of 8)
1.0
1.2
1.4
1.6
1.8
2.0
PPL ratio (skip / full-depth)
5% PPL tolerance
Dynamic skip Pareto frontier (340M, FineWeb-Edu proxy set)
single ({5}, lowest I)
late positions
all interior
other subsets
combined {3,5}, max_skips=2, q=0.85 (ours)
full-depth baseline
(a) Accuracy–compute Pareto. P={3, 5} (orange star) sits below the 5%
PPL tolerance.
0
10
20
30
40
50
60
ms / batch (seq\_len=8192, batch\_size=1)
full-depth
attn, [5] skip=1 q=0.95
attn, [5] skip=1 q=0.85
prev, [2, 3, 5] skip=2 q=0.8
attn, [3, 5] skip=2 q=0.9
attn, [3, 5] skip=2 q=0.85
1.00×
1.13×
1.14×
1.15×
1.17×
1.19×
Wall-clock latency comparison (340M, single H100)
(b) Wall-clock at seq 8192.
{3, 5}/M=2/q=0.85
reaches 1.19×.
Figure 3: RESKIP on 340M FineWeb-Edu: position-calibrated dynamic skip with P={3, 5} strictly dominates the
single-position baseline at zero benchmark drop.
18

## E
## Retrofit Pareto and breakdowns
E.1
LoRA baselines
We compare against parameter-matched LoRA baselines on the same 50/50 UltraChat + LLaVA mix at 2× the
canonical retrofit’s step count. LoRA r=32 on q, v: LAMBADA acc 0.540 / 0.516 (two seeds), HellaSwag 0.510
/ 0.524. LoRA r=16 on q, k, v, o: LAMBADA 0.534, HellaSwag 0.492. LoRA r=8 on MLP: LAMBADA 0.514,
HellaSwag 0.510. Mean LAMBADA across the four runs is 0.526, −0.6pp below base (0.532) and −5.0pp below our
γ→1 retrofit (0.576). Mean HellaSwag 0.509 is +0.3pp over base versus retrofit’s +1.6pp. Attribution: the retrofit’s
gain comes from the ATTNRES routing structure, not from a few extra trainable parameters on SFT data.
E.2
Static-pruning baseline at the VLM scale (Gromov drop-k)
The 340M decision-rule ablation (Tab. 10) shows that single-position static skip and random skip both collapse on
language tasks. The complementary question at the VLM scale is whether full-layer static pruning—the standard
“remove the least useful blocks” baseline used by Gromov et al. [16]—can match the accuracy floor that the retrofit
operates above. We rank the 28 Qwen3-VL-2B layers by the Gromov angular-distance criterion (computed on a
FineWeb-Edu calibration set with the LM-head only; no fine-tuning), remove the k lowest-distance layers, and
re-evaluate on lmms-eval. Tab. 12 reports drop-4 (layers {23, 24, 25, 26}) and drop-8 (additionally {12, 13, 14, 22}).
Table 12:
Static layer pruning on Qwen3-VL-2B (Gromov-style angular-distance ranking, no retraining),
lmms-eval full splits. Companion to the 340M in-block decision-rule ablation in Tab. 10: spans the full-layer
vs. single-position-within-block skip dimension. Pruning 4 of 28 layers (14%) costs −16pp AI2D, −6pp MMMU,
−5pp MMStar, −65pp OCRBench; pruning 8 (29%) is catastrophic on every cell. The retrofit (Tab. 2) operates at
0 removed layers, 0pp loss, and additionally lifts the base; RESKIP on the retrofit at q=0.85 holds within 1pp on
LAMBADA (Tab. 13).
Configuration
AI2D
MMMU
MMStar
OCRBench
RWQA
Qwen3-VL-2B (base, Lrm=0)
0.736
0.414
0.536
0.772
0.648
Gromov drop-4 (layers {23, 24, 25, 26})
0.5732
0.3556
0.4906
0.1240
0.3791
Gromov drop-8 (drop-4 ∪{12, 13, 14, 22})
0.0683
0.2389
0.0190
0.0030
0.1451
+ AR-RETROFIT v3 (L=4, 10k; Tab. 2)
0.758
0.432
0.536
0.814
0.661
The pattern matches the in-block 340M observation: any unconditional block removal (single-position static at
340M, full-layer Gromov at 2B) destroys benchmark accuracy at the VLM scale even at modest pruning fractions, while
input-dependent skip on top of an ATTNRES routing structure preserves it. We do not attempt to retrain the pruned
model; the intent is a free static-baseline floor for the cost-quality trade-off, not a tuned competitor.*
E.3
RESKIP on the canonical retrofit: text Pareto and cross-modality consistency
On the canonical L=4 v3 10k retrofit, applied to LAMBADA-500 as the language-modality cross-check of the LIBERO
Pareto (Tab. 5), RESKIP produces the same shape as on action: lossless at the conservative end, collapsing once q
moves below the action distribution’s mean. Same P={1, 4}, Mmax=2 as the 2B VLA cell.
The legacy v1 (5k, P={4, 6, 11}) Pareto on the H_r256_5k retrofit lifted LAMBADA above its full-path value at
q=0.95, M=1 (0.593 vs. 0.576), but that retrofit is no longer canonical for any reported result; we list the v1 Pareto in
Tab. 14 for completeness.
E.4
Wall-clock latency: full table including legacy L=2 and VLA in-backbone
Tab. 3 in the main body reports the canonical L=4 result. Tab. 15 below shows the full set: the legacy L=2 partition
(14 blocks at 2B) carries a 1.39× structural cost over stock Qwen3-VL-2B because each token pays 14 router calls, the
*Pruned cells additionally exhibit POPE generation drift below the strict yes/no surface (drop-4 pope-acc 0.021, drop-8 0.500 via accidental
all-“yes” decoding), so we drop POPE from the table.
19

Table 13: Cross-modality RESKIP consistency. LAMBADA-500 on the canonical L=4 v3 retrofit, P={1, 4}, M=2,
τ calibrated on 32 held-out LAMBADA prefixes at quantile q. Same operating-point structure as the LIBERO 4-suite
Pareto: near-lossless above the calibration distribution mean (q≥0.85); rapid collapse below, where the threshold
instructs the router to skip blocks it is normally executing.
q
LAMBADA acc
LAMBADA ppl
∆acc vs no-skip
avg. skips / forward
no-skip (M=0)
0.5700
4.526
—
0.00 / ≤7
0.85
0.5600
5.258
−1.0
0.19 / 2
0.50
0.4120
12.550
−15.8
1.06 / 2
0.30
0.3900
14.005
−18.0
1.17 / 2
Table 14: (Legacy) Dynamic skip Pareto on the v1 H_r256_5k retrofit, P={4, 6, 11}. Retained for reproducibility of
the earlier draft.
q
Mmax
LAMBADA acc
LAMBADA ppl
avg. skips / 3 eligible
0.50
1
0.5500
5.45
0.82
0.70
1
0.5600
5.24
0.64
0.85
1
0.5733
4.90
0.37
0.95
1
0.5933
4.77
0.18
0.50
2
0.4733
6.42
1.41
0.70
2
0.5300
5.62
0.97
0.85
2
0.5600
5.03
0.55
0.95
2
0.5800
4.84
0.26
VLA in-backbone forward (the same retrofit run inside the OFT trainer’s backbone forward, §4) tracks the VLM retrofit
within 1% at both partitions, and torch.compile closes the L=2 residual to 1.16× but is more dramatic on L=4
where the router count is halved. This table is the basis for the canonical-partition switch from L=2 to L=4.
The structural per-block router stack (stack→RMSNorm→softmax→einsum over N completed blocks) does
not go away under cached decode: at T=1 it fires N times per decoded token. Adapter rank ablation (Appendix C.2)
confirms the adapter is not the bottleneck (≤2%); the router itself, when uncompiled, is the floor. torch.compile
(rows 5–8) collapses the small router gemms into a captured graph and tunes the matmul kernels for retrofit’s small
shapes, removing ∼92% of the L=4 structural gap. The remaining 2.9% is the small overhead of 7 surviving router
calls per token; closing it would require a Triton-fused router that issues a single kernel for the entire stack–→–einsum
sequence (open future work).
Skip on top of compile.
Eager-mode RESKIP contributes a further 5–9% when used on its own at q=0.99, but
cannot currently be composed with torch.compile in a single forward: the dyn-skip rule requires an .item()
on a per-token threshold comparison, which forces a CPU sync and breaks the captured CUDA graph. Production
deployments pick one mode per latency budget—compile for high-throughput batch decode, eager-with-skip for
variable-batch interactive workloads. A static-graph variant of the skip rule (compile-safe top-k with a fixed routing
decision per shape bucket) is a known direction.
E.5
Compile vs. eager: accuracy parity
The compiled-overhead result of §3 requires that torch.compile preserve the retrofit’s accuracy. We verified two
parity tests on the canonical L=4 v3 10k retrofit.
LAMBADA-500 accuracy parity.
Running LAMBADA-500 with the retrofit wrapped in torch.compile using
default mode and dynamic shapes: acc 0.5720 compiled vs. 0.5700 eager (∆=+0.20pp), ppl 4.534 vs. 4.526. Per-target
20

Table 15: Forward-pass latency, full sweep on 1×H100, bf16, batch 1, median of 20 runs after 5 warmup. Eager rows:
stock-HF base vs. retrofit at the indicated L. Compile rows wrap both with torch.compile, dynamic=False;
reduce-overhead captures CUDA graphs, max-autotune additionally tunes per-kernel matmuls. The VLA
in-backbone row is the canonical L=4 retrofit loaded into the OFT trainer’s backbone forward (a separate code path) —
within 1% of the VLM retrofit cell, confirming the per-block router stack, not the VLA harness, is the structural cost.
Bold: the canonical operating point of the paper.
Configuration
seq
mode
base (ms)
retrofit (ms)
retrofit / base
(1) L=2 retrofit (legacy)
1024
eager
15.42
21.25
1.378×
(2) L=2 retrofit (legacy)
2048
eager
26.03
36.14
1.388×
(3) L=4 retrofit (canonical)
1024
eager
14.91
16.66
1.117×
(4) L=4 retrofit (canonical)
2048
eager
25.49
28.53
1.119×
(5) L=2 retrofit
1024
reduce-overhead
9.31
10.78
1.158×
(6) L=2 retrofit
2048
reduce-overhead
21.25
24.45
1.151×
(7) L=4 retrofit
2048
reduce-overhead
21.31
22.50
1.056×
(8) L=4 retrofit
2048
max-autotune
21.81
22.43
1.029×
(9) VLA in-backbone L=4 retrofit
2048
eager
25.49
28.79
1.130×
argmax agreement against eager: 98.60% — the residual 1.4% are tokens where eager and compiled both fall within
bf16 SDPA jitter and the rank-1/rank-2 logits flip.
Per-token logit parity.
On real prompt tokens with mode="reduce-overhead" (the inference-time mode used
for the speed table): per-token argmax agreement 98.51%, max |∆logits| = 0.50, RMSE 0.060 — same magnitude as
the cache-on/cache-off SDPA jitter on the stock base (Appendix B.6). Compile preserves the retrofit’s forward to within
bf16 evaluation noise; the 1.029× base_compiled number is at matched accuracy.
E.6
Block partition sweep
Table 16: Block partition sweep (v3 recipe, r=256, γ-curriculum). L is layers-per-block, N=Ltotal/L. Per-layer
L=1 underperforms by 2–9pp; the plateau L ∈{2, 4, 7} on 2B reproduces Kimi Team et al. [24]’s S ∈{2, 4, 8}. We
recommend L=4 on both scales because it preserves quality while reducing router calls.
Scale
L
N
LAMBADA
∆L
HellaSwag MMBench MMMU MMStar AI2D OCRBench RWQA
2B base
— —
0.532
—
0.506
75.77
0.414
0.536
0.736
0.772
0.648
2B retrofit
1
28
0.5645
+3.3
0.490
76.20
0.388
0.499
0.743
0.809
0.652
2B retrofit
2
14
0.5755
+4.4
0.494
77.23
0.426
0.532
0.748
0.803
0.663
2B (rec.)
4
7
0.5650
+3.3
0.500
78.87
0.432
0.536
0.758
0.814
0.661
2B retrofit
7
4
0.5155
−1.7
0.492
77.49
0.427
0.530
0.756
0.808
0.656
4B base
— —
0.576
—
0.562
83.33
0.490
0.624
0.819
0.819
0.715
4B retrofit
1
36
0.5575
−1.9
0.523
83.33
0.497
0.538
0.783
0.768
0.694
4B retrofit
2
18
–
–
–
84.28
0.523
0.587
0.816
0.813
0.708
4B (rec.)
4
9
0.6625
+8.7
0.552
85.22
0.521
0.632
0.825
0.824
0.718
4B retrofit
6
6
0.6540
+7.8
0.554
84.79
0.531
0.623
0.817
0.824
0.715
21

Table 17: MMStar subcategory breakdown for the v3 retrofit. The math, logical, and sci&tech triple is the “reasoning
over images” cell where the retrofit shows the largest gain (+7.9pp math on 2B; +3.9pp on 4B). Perception cells
(coarse / fine / instance) are at parity or a small regression, consistent with a router that lifts deliberate-reasoning paths
rather than altering early perception.
Configuration
math
logical
sci&tech
coarse
fine
instance
Qwen3-VL-2B (base)
0.413
0.429
0.408
0.714
0.520
0.710
+ AR-RETROFIT v3 (2B, L=4)
0.492
0.432
0.353
0.734
0.505
0.683
∆vs. 2B base
+7.9
+0.3
−5.5
+2.0
−1.5
−2.7
Qwen3-VL-4B (base)
0.549
0.626
0.465
0.788
0.611
0.705
+ AR-RETROFIT v3 (4B, L=4)
0.588
0.602
0.467
0.812
0.606
0.714
∆vs. 4B base
+3.9
−2.4
+0.2
+2.4
−0.5
+0.9
E.7
MMStar subcategory breakdown
## F
## VLA appendix
F.1
VLA seed variance on 2B
We ran two seeds for the two 2B suites where Path 0 / Path B are within evaluation noise on a single seed;
libero_spatial and libero_object are single runs. Aggregating both repeated rollouts gives Path 0 96.05
Table 18: Per-seed Qwen3-VL-2B numbers. Seed 1 is the initial run; seed 2 is a fresh environment-reseeded rollout of
the same trained checkpoint. The last column shows the values used in Table 4.
Suite
Policy
Seed 1
Seed 2
Used in Tab. 4
libero_goal
Path 0 (30k)
97.6
97.4
97.4
libero_goal
Path B (30k)
97.4
98.6
98.6
libero_10
Path 0 (30k)
92.8
91.4
91.4
libero_10
Path B (30k)
92.6
90.6
92.6
and Path B 96.75 (∆=+0.70pp), while the Table 4 operating-point values give ∆=+1.30pp. Both summaries preserve
the ranking Path B > Path 0.
Binomial standard error.
At p≈0.96 the per-suite binomial SE is σsuite=
p
p(1−p)/500≈0.88pp and the 4-suite
aggregate SE is σagg=
p
p(1−p)/2,000≈0.44pp. The repeated-rollout 2B gain ∆= + 0.70pp is therefore ≈1.6 σagg,
while the Table 4 operating point is ≈3.0 σagg. We do not interpret the 4-suite gain as a statistically tight win in the
strict frequentist sense; the load-bearing pattern is the per-suite ranking on the suites where a stable backbone matters
most: Spatial +3.0pp, Goal +1.2pp in Table 4, and Long-10 showing sensitivity across repeated rollouts. The 4B
clean-base +0.65pp single-run cell sits at ≈1.5 σagg and is reported with the same caveat.
F.2
VLA pipeline pitfall: unfrozen action-token embeddings
Two Qwen3-VL-4B base VLMs we tried produced qualitatively different Path B behaviour. Our first attempt used an
“Instruct-Action” variant that adds 2,048 <robot_action_*> embeddings (initialised at the mean of the existing
vocab, kept unfrozen). The OFT head we train does not predict those tokens, but the rows for them sit in the tied
embed_tokens / lm_head matrix and receive gradient from the retrofit’s distillation loss and from label smoothing.
Path B under this dirty base reaches 94.55% 4-suite avg (Long-10 84.8%); Path B under the clean base—no extra
tokens added—reaches 96.70% (Long-10 94.2%). A +9.4pp Long-10 swing is entirely explained by whether 2,048
unused rows in the shared matrix receive stray gradient. Use the clean base for retrofit-+-OFT; only add bespoke action
vocabulary when the head actually predicts those tokens.
22

F.3
VLA landscape: open VLAs on standard benchmarks
Table 19: Open VLA landscape on standard benchmarks. LIBERO entries are success rates (%) per task suite. Following
common practice, π0 and Octo-Base LIBERO numbers are taken from OpenVLA-OFT [23], which re-ran all three
policies under a uniform LIBERO protocol. Our rows use the 30k Path B numbers from Table 4.
Model
#Params
Spatial
Object
Goal
Long-10
ref.
Octo-Base
93M
78.9
85.7
84.6
51.1
[33, 23]
OpenVLA
7B
84.7
88.4
79.2
53.7
[22, 23]
TraceVLA
7B
84.6
85.2
75.1
54.1
[51]
SpatialVLA
4B
88.2
89.9
78.6
55.5
[37]
π0
3B
96.8
98.8
95.8
85.2
[3, 23]
Qwen3-VL-2B + AR-Retrofit + OFT (ours)
2B
97.8
99.6
98.6
92.6
this work
Qwen3-VL-4B + AR-Retrofit + OFT (ours)
4B
94.6
99.8
98.2
94.2
this work
F.4
VLA RESKIP setup: per-block drift and eligible-set selection
The RESKIP eligible set P for each scale is selected from a per-block ablation on the trained Path B policy, not on the
VLM retrofit alone, because the VLA action stream activates a different routing distribution than the language stream.
We measure per-block action-drift MSE on a 24-sample sim trajectory: the policy is run with no skip and with each
block individually skipped; the per-step action MSE between the skipped and no-skip rollouts is the per-block “cost of
removing this block on the action distribution”. The two safest blocks per scale form P.
2B Path B v2 30k per-block action-drift MSE (sorted ascending, in units ×10−3): block 1: 0.4; block 4: 34.0;
block 0: 58.1; block 5: 87.0; remaining blocks >100. Selecting the two lowest: P2B = {1, 4}.
4B Path B 30k clean per-block action-drift MSE: block 1: 0.6; block 2: 11.0; block 0: 34.0; block 3: 63.0;
remaining >120. Selecting: P4B = {1, 2}.
Cross-scale transfer of P fails.
Naively transferring 2B’s P={1, 4} onto 4B catastrophically drops LIBERO-Spatial
to 12.4% at q=0.85 — block-4’s drift on 4B is roughly 10× that of block-2, so the same eligible-set rule produces
wildly different operating points across scales. P must be calibrated on each model’s own sim distribution; this is the
per-scale half of the modality-aware skip protocol.
Threshold calibration.
Per-block thresholds τn are the q-th empirical quantile of wrecent,n over 31,286 (2B) / 31,257
(4B) sim-rollout records (one record per decoded action token across many trajectories). The Tab. 5 numbers use this
protocol. A separate “Method A” that calibrates on retrofit pretokenized data (as if the VLA were just a long-context
LM) accidentally lands at a conservative operating point that mimics q=0.99 on the action distribution; we use it as a
sensitivity comparator in our internal ablations and recommend the action-distribution-calibrated thresholds for any
deployment.
Cross-modality threshold transfer is not safe.
The LAMBADA-calibrated thresholds at q=0.85 from Tab. 13,
applied unchanged to the same Path B policy at LIBERO inference, drop LIBERO-Spatial from 99.5% (a L=4 v3
warm-start, partial eval) to 64%. The action distribution has more low-wrecent mass than the language distribution at the
same blocks; the language q=0.85 corresponds to roughly the action q=0.50, which over-skips. This is the empirical
hard motivation for per-modality, per-token-class calibration in deployments that mix modalities within a single rollout.
F.5
VLA planned follow-ups
Per-modality, per-token-class RESKIP. The Tab. 5 thresholds are calibrated on the action distribution as a whole;
an obvious refinement is per-modality (P(m), τ (m)
n
, M (m)
max) within a single rollout (vision tokens, language tokens,
action tokens). Working hypothesis: vision α concentrates on early blocks (allows aggressive late-block skip on
23

perception steps); action α spreads to late blocks (restricts late-block skip on action prediction steps). The calibration
harness (retrofit/eval/calibrate_vla_thresholds.py) is in place; the failed cross-modality transfer
of Appendix F.4 is the empirical motivation. Compile + skip composition. A static-graph variant of the skip rule
(compile-safe top-k with a fixed routing decision per shape bucket) would let the compiled full-depth mode and the
eager skip mode coexist in a single forward and stack their 1.029× and 5–9% savings. Edge hardware. Wall-clock is
H100 / bf16; RTX 4090 / Jetson Orin direct measurement is planned.
## G
## NeurIPS Paper Checklist
1. Claims. Yes. The abstract, introduction, and conclusions state the main claims and separate the retrofit
contribution from calibrated skipping, VLA transfer, and systems characterization.
2. Limitations. Yes. Appendix A lists architecture coverage, compile / dynamic-skip composition, hardware scope,
and per-modality calibration limits.
3. Theory assumptions. Not applicable. The paper is empirical and algorithmic; all equations define implemented
objectives or routing rules.
4. Experimental reproducibility. Yes. Sections 2.2–4.1 and the appendix specify model scales, block partitions,
training steps, datasets, calibration rules, and evaluation protocols.
5. Open access to code and data. Partially. Experiments use public datasets and benchmarks where available;
release details for retrofit / VLA training scripts will follow the anonymous-submission policy.
6. Compute. Yes. Main text and appendices report GPU type, training steps, rough GPU-minutes / GPU-hours, and
latency measurement settings.
7. Dataset and benchmark provenance. Yes. All public datasets and benchmarks are cited; LIBERO and
lmms-eval protocols are described with rollout counts or split usage.
8. Human subjects. Not applicable. No new human-subject data are collected.
9. Privacy. Not applicable beyond the privacy considerations of the cited public datasets.
10. Licenses. Partially. Public resources are cited; final camera-ready release will include the exact code / model /
data license table.
11. Broader impacts. Yes. The work targets lower inference cost and adaptive computation in pretrained models; no
safety-critical deployment is claimed.
12. Safeguards. Yes. VLA thresholds are calibrated per model and action distribution; cross-modality transfer
failures are explicitly documented as a deployment risk.
24