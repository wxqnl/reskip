## ReSkip: Retrofitting Intrinsic Depth Routing into Pretrained
## Transformers
## via Attention Residuals
Anonymous Author(s)
Affiliation withheld
anonymous@submission.invalid
May 1, 2026
Abstract
Pretrained transformers execute nearly the same depth for every input. We study how to retrofit a pretrained
transformer with an intrinsic pre-execution routing signal that can drive dynamic depth decisions without
training a separate router or retraining the model from scratch. We introduce AR-RETROFIT, an identity-
preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with
well under 1% new parameters. The resulting routing weights are computed before each block executes,
enabling RESKIP, a calibrated dynamic block-skipping rule that selects eligible blocks using offline safety
checks and applies input-dependent skipping at inference. On Qwen3-VL-2B and 4B, AR-RETROFIT
preserves or improves most VLM benchmarks and substantially improves LAMBADA, while RESKIP
provides calibrated input-dependent depth allocation on the retrofitted model, with conservative operating
points preserving quality within 1pp on LAMBADA-500. A 340M controlled study shows that the dynamic
routing rule outperforms static or random schedules at matched skip rates, and LIBERO experiments
provide supporting evidence that the same routed-depth signal remains usable for calibrated action-stream
skipping. These results show that pretrained transformers can acquire usable depth-axis routing through a
lightweight retrofit rather than from-scratch residual pretraining.
## 1
## Introduction
Different inputs should not always spend the same computation. This is a design motivation rather than a claim of
cognitive equivalence. Kahneman’s dual-process framework [19] distinguishes fast and deliberate processing, and
biological vision offers a concrete computational analogue: easy stimuli can be recognised through shallow feedforward
sweeps, while harder stimuli recruit recurrent and deeper circuits [26, 20, 10]. Recurrent vision models capture human
representational dynamics [22], and confidence-thresholded recurrence spends more steps on harder images to trade
speed for accuracy [42]. Since finite recurrence can be unrolled into feedforward depth [44], these observations motivate
a transformer-side question: can a pretrained model acquire an intrinsic signal for allocating effective depth per input?
Large language models already expose one test-time compute axis: tokens. Test-time compute scaling [34, 8, 41],
chain-of-thought prompting [45], and self-thought methods [48] spend more compute on hard problems by emitting
more reasoning tokens before answering. Yet every token in such a trace still traverses every layer of the underlying
transformer, regardless of whether that token is a routine connector or a load-bearing inferential step. The missing
complementary axis is therefore depth: deciding which blocks to invoke for this input, at this point in the sequence.
Existing depth-axis methods do not quite match this motivation because they externalise the decision. Early-exit
classifiers [40, 12] add auxiliary heads per layer; Mixture-of-Depths [39] learns a token router with a capacity-balancing

loss; layer-pruning [16, 32] is post-hoc and static; Universal Transformers [9] share weights with ACT-style halting [15].
In each case a separate component decides whether the rest of the network should run. The biological analogue
motivating this paper is different: control is embedded in the processing substrate rather than attached as a separate
decision head.
ATTNRES [25] gives a natural handle. It replaces the fixed scalar-1 residual with a learned softmax-attention over
previous block outputs; the routing weights αi→n are input-dependent, competitively normalised, and computed before
block n runs, as part of forming its input. They therefore emit a per-block depth-allocation signal as a side-effect of the
network’s own forward, not as a separate head. The practical obstacle is that existing ATTNRES models are trained
from scratch with the modified residual; reproducing this at the 2–7B scale that real applications need costs tens of
thousands of GPU-hours (Appendix D.1). The central problem in this paper is how to install such a signal into standard
pretrained transformers through a short fine-tune, while preserving their original capabilities and making the signal
usable for calibrated depth skipping.
The paper has one main contribution and two supporting claims.
• AR-RETROFIT installs intrinsic depth routing into a pretrained model (§2.2). A γ-gated residual injection
adds an ATTNRES-style routing path to a frozen transformer through a short fine-tune, using ∼0.7% new
parameters and an identity-preserving initialization.
• RESKIP uses the installed signal for pre-execution dynamic depth (§2.3, §3.3). The skip decision is made
before the current block runs, and eligible blocks are selected by ablation or action-drift because routing weight
is a useful signal but not a safety certificate.
• VLM and VLA experiments validate the retrofit pathway (§3, §4). The retrofit preserves or improves
pretrained VLM capability, dynamic RESKIP gives calibrated adaptive-depth behavior, and LIBERO provides
supporting transfer evidence.
## 2
## Method
2.1
Attention Residuals as a pre-execution routing signal
A standard transformer uses fixed scalar-1 combination, xl = xl−1 + fl(xl−1), emitting no routing signal. AT-
TNRES [25] replaces this with a learned attention over depth. Each block l maintains a pseudo-query wl ∈Rd; keys are
projections ki = WKhi of earlier block outputs hi, and
αi→l =
exp(w⊤
l ki/
√
d)
Pl−1
j=0 exp(w⊤
l kj/
√
d)
,
xl =
l−1
X
i=0
αi→l hi.
(1)
The key property is two-phase execution. Computing the input to block l requires only {hi}i<l and wl—both available
before fl runs. We call the pre-execution attention computation phase 1 and fl itself phase 2. Phase 1 is the routing
signal we exploit. Block-ATTNRES groups layers into N blocks and applies the softmax at the block level, which is the
granularity used throughout. Algorithm 1 gives the full two-phase forward with RESKIP’s skip check folded in.
2.2
AR-RETROFIT: installing the routing path
The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-
residual regime and produce an ATTNRES-capable model that (P1) preserves benchmark quality, (P2) emits a routing α
informative enough to drive RESKIP, and (P3) trains in a single short fine-tune over off-the-shelf SFT data, with the
base frozen. To satisfy the first constraint, the modification must be exactly identity-preserving at step 0; to satisfy the
second, the routing must feed the real forward path, not merely observe it.
We organise the decoder into N blocks of adjacent layers (Qwen3-VL-2B uses 7 blocks of 4 layers in the main
experiments). For every block n ≥1 we add an ATTNRES router wn, a tiny adapter An (down-up bottleneck of rank r,

Figure 1: Overview of RESKIP and AR-RETROFIT. Left: a standard transformer residual gives no intrinsic routing
signal. Middle: AR-RETROFIT injects an identity-preserving γ-gated ATTNRES path into the frozen pretrained
backbone. Right: the phase-1 ATTNRES weights are computed before block execution and are calibrated into RESKIP
decisions.
SiLU), and a scalar gate γn:
rn =
n−1
X
i=0
αi→n hi,
xn = hn−1 + γn An(rn −hn−1),
hn = Blockn(xn).
(2)
Blockn is the pretrained block group, verbatim; ATTNRES enters between blocks, as a correction. The retrofit parameters
{wn, γn, An} amount to ∼15M (∼0.7% of base) at r=256 on 2B; the entire base—embeddings, vision tower, decoder
layers, LM head—is frozen.
With γn=0 we have xn = hn−1 exactly, so the forward at step 0 is bit-identical to the pretrained model. The
adapter up-projection is small-random (N(0, 0.022)) so that ∂L/∂γn ̸= 0 at init, avoiding gradient deadlock. We then
ramp γn via a linear curriculum 0 →1 over the first 30–50% of training. At γn=1, the ATTNRES-routed correction is
fully active, but the model remains a residual correction on the pretrained stream as defined in Eq. 2. Appendix C.1
compares observer-only, interpolation, and informed-initialisation controls.
The training objective follows from the three constraints above. We minimise
L = LCE(full path) + λkl LKL(skip path ∥teacher) + λent Lent(α),
(3)
where the full-path CE is computed on assistant tokens, while the skip-branch KL samples one block at random per
step, runs the forward with it skipped, and pulls the resulting logits toward a frozen pretrained teacher. This makes
the surrogate xn useful both when the block runs and when it is skipped. Lent is implemented with a negative sign in
the loss, i.e., a light entropy-maximising prior against premature collapse of α. Default λkl=1, entropy weight 0.02,
lr 10−3 on retrofit params, cosine schedule with 100-step warmup. Hyperparameters and data mixtures are listed in
Appendix B.3.

This is not ordinary SFT. The CE term keeps the full path useful for the downstream task, but by itself it does
not train the skipped surrogate to replace a real block. The skip-branch KL supplies that constraint by tying one
skipped-block forward per step to the frozen teacher. Conversely, teacher imitation alone would merely reproduce the
base model. The weak entropy prior prevents early one-source collapse while still allowing the router to sharpen later.
This three-term structure is what makes the retrofit identity-preserving at step 0, task-improving after training, and
skip-ready at inference.
2.3
RESKIP: dynamic block skipping from the installed signal
RESKIP uses the installed routing path rather than a separate controller. For block n, phase 1 computes α·→n before the
block body runs. We then read the weight wrecent(n) that the router assigns to the immediate predecessor and decide
whether to execute phase 2:
skip block n ⇐⇒wrecent(n) > τn and n ∈P and P
m<n ⊮[skippedm] < Mmax.
(4)
The decision is available before the current block executes, so a trigger can actually save that block. When the rule
fires, we short-circuit the block as hn ←xn, where xn is the routed-correction input in Eq. 2. To keep autoregressive
decoding cache-consistent, a skipped decoder layer still runs the K/V-producing slice (LayerNorm→k/v_proj→
RoPE→cache.update) and skips q_proj, attention, o_proj, and MLP. The cache-enabled skip path matches
the cacheless path in last-position argmax (Appendix B.6).
2.4
Calibration and safety selection
Routing weight is a usable depth signal, not a certificate that a block can be removed. Each eligible set P is therefore
selected offline from two complementary diagnostics. The ATTNRES importance score
I(n) = max
l>n E[αn→l]
measures how often future blocks refer to block n, while an ablation or drift score measures how costly it is to remove
that block on the relevant distribution. For language experiments we use A(n) = PPL(n removed)/PPL(full); for
VLA we use action-drift MSE on simulated rollouts. Thresholds τn are the q-th empirical quantile of wrecent(n) on
held-out calibration data. This separation is central: α decides when a calibrated block looks locally redundant, while
ablation or drift decides which blocks are eligible to skip.
## 3
## Experiments: AR-RETROFIT on Qwen3-VL-2B and 4B
3.1
Q1: Does AR-RETROFIT preserve or improve pretrained capability?
We retrofit Qwen3-VL-2B [2] (L=28, d=2048) at N=7 blocks of L=4 and Qwen3-VL-4B (L=36, d=2560) at N=9
blocks of L=4. All base parameters are frozen; ∼15M (2B) / ∼23.7M (4B) retrofit parameters at r=256 train via
Eq. 3. The final SFT mixture contains 60% LLaVA-OneVision-Data [28], 20% UltraChat [11], 10% NuminaMath-
CoT [1], and 10% OpenThoughts-114k [17] for 10k steps; Appendix C.6 gives the concrete mixture. Evaluation uses
LAMBADA [35]/HellaSwag [49] for text and six lmms-eval benchmarks for VLM (MMBench [30], MMMU [47],
MMStar [7], AI2D [21], OCRBench [31], RealWorldQA [46]). Identity-at-init is verified before training (γ=0
reproduces base; max |∆logits| = 0.375 bf16, 100% argmax agreement; Appendix B.5).
The retrofit answers Q1 positively (Tab. 1). It improves base on five of six VLM benchmarks at both scales, with the
largest gain on the MMStar math subcategory (+7.9pp at 2B, +3.9pp at 4B; full 6-cell decomposition in Appendix E.7).
The win concentrates on deliberate reasoning over images rather than perception, consistent with a router that lifts
deeper reasoning paths. Text quality also lifts: +3.3pp LAMBADA / −16% ppl on 2B and +8.7pp / −32% on 4B;
HellaSwag is within ±1.1pp at both scales. Auxiliary parameter-matched LoRA controls on a related SFT mix do not
recover the LAMBADA gain (Appendix E.1). These controls do not prove that all quality gains come uniquely from
routing, but they rule out a simple parameter-count-only explanation. We therefore treat the routed path as load-bearing
for skip-ready behavior, while interpreting quality gains as consistent with, but not uniquely attributable to, the routed
correction.

Table 1: AR-RETROFIT on Qwen3-VL-2B/4B at the canonical L=4 recipe (lmms-eval full splits for VLM,
n=2000 for LAMBADA/HellaSwag). The retrofit improves base on 5/6 VLM benchmarks at both scales and lifts text
on LAMBADA. MMStar overall holds; the math sub-category (Appendix E.7) jumps +7.9pp on 2B / +3.9pp on 4B,
consistent with a router that lifts deliberate-reasoning paths. Auxiliary parameter-matched LoRA controls do not show
the same text-gain pattern (Appendix E.1).
Text
VLM (lmms-eval full split)
Configuration
LAMBADA
HellaSwag
MMBench
MMMU
MMStar
AI2D
OCRBench
RWQA
Qwen3-VL-2B (base)
0.532
0.506
75.77
0.414
0.536
0.736
0.772
0.648
+ AR-RETROFIT (L=4, 10k)
0.5650
0.500
78.87
0.432
0.536
0.758
0.814
0.661
∆vs. base
+3.3
−0.6
+3.10
+1.8
0.0
+2.2
+4.2
+1.3
Qwen3-VL-4B (base)
0.576
0.562
83.33
0.490
0.624
0.819
0.819
0.715
+ AR-RETROFIT (L=4, 10k)
0.6625
0.5515
85.22
0.521
0.632
0.825
0.824
0.718
∆vs. base
+8.7
−1.1
+1.9
+3.1
+0.8
+0.6
+0.5
+0.3
Downstream block l
Source block n
0.29
0.13
0.08
0.10
0.12
0.07
0.17
0.43
0.10
0.06
0.06
0.06
0.07
0.52
0.13
0.07
0.07
0.04
0.56
0.17
0.13
0.10
0.46
0.18
0.16
0.40
0.23
0.50
(a) AttnRes routing weights αn →l
Block index n
0.0
0.2
0.4
0.6
0.8
1.0
AttnRes importance I(n)
Blocks 3, 5:
our dynamic skip
positions
({3}, {5}-axis pair)
(b) Importance ≠ irreplaceability
AttnRes importance I(n)
Static removal PPL ratio
0.0
0.1
0.2
0.3
0.4
0.5
0.6
PPL ratio after removal
Block 2:
moderate I but
highest removal
PPL (8.1×)
Figure 2: ATTNRES routing weights do not predict ablation impact (340M). (a) Learned block-level αn→l. (b) I(n)
(blue, left axis) and A(n) (orange, right axis). RESKIP uses routing for input-dependent decisions, but eligible blocks
still need ablation-based safety selection.
3.2
Q2: Is the routing signal necessary for RESKIP?
We first test the mechanism in the setting where ATTNRES is trained in from scratch. On a 340M block-ATTNRES
transformer (d=1024, L=24, N=8 blocks; FineWeb-Edu 100BT [36]; lm-eval-harness [14]), RESKIP at
P={3, 5}, Mmax=2, q=0.85 matches the full-depth ATTNRES scores across the expanded language benchmark set
and gives 1.19× wall-clock at sequence length 8192 (Tab. 2; Pareto curves in Appendix D.6). The observed skip count
averages 0.88 blocks and varies 0–2 per forward, confirming input-dependent allocation rather than a threshold that
never fires.
The routing signal matters beyond the choice of skip locations. At a matched skip budget on the same 340M model,
an input-dependent rule based on phase-1 routing preserves substantially more quality than static or random schedules
(Tab. 3). The dynamic rule is not always lossless at this more aggressive calibration point, but it is clearly better than
skipping the same positions unconditionally.

Table 2: RESKIP on 340M from-scratch ATTNRES. The selected operating point preserves the full-depth AT-
TNRES scores on the expanded task set while actually firing: ReSkip averages 0.88 skipped blocks over 48 sequence-
8192 batches and gives 1.19× wall-clock at that length. Headline: acc_norm for PIQA/HellaSwag/ARC-E/ARC-
C/OpenBookQA, plain acc for MMLU/LAMBADA.
Configuration
Avg. skip
Speed
LAMBADA acc LAMBADA ppl HellaSwag
PIQA
ARC-E ARC-C MMLU OpenBookQA
Vanilla (no ATTNRES)
–
–
0.3790
24.71
0.4436
0.6779
0.5602
0.3046
0.2594
0.3320
ATTNRES full-depth
0.00
1.00×
0.4054
20.20
0.4607
0.6893
0.5438
0.3012
0.2555
0.3580
+ RESKIP ({3, 5}, M=2, q=0.85)
0.88
1.19×
0.4054
20.20
0.4607
0.6893
0.5438
0.3012
0.2555
0.3580
Table 3: Dynamic, static, and random skip rules at matched rate on 340M. All rows use the same trained ATTNRES
weights, P={3, 5}, and Mmax=1. Dynamic RESKIP uses a calibrated phase-1 statistic; static and random controls
remove the same positions without input dependence. Full decision-rule ablation and observed trigger rates are in
Appendix D.5.
Rule
LAMBADA
HellaSwag
PIQA
ARC-E
OpenBookQA
Full-depth ATTNRES
0.4054
0.4607
0.6893
0.5438
0.3580
Dynamic RESKIP
0.3445
0.4471
0.6774
0.5391
0.3540
Static skip P=3
0.2624
0.3968
0.6610
0.5206
0.3300
Static skip P=5
0.2189
0.4223
0.6638
0.5253
0.3260
Random {P=3 or P=5}
0.2416
0.4028
0.6420
0.5101
0.3300
3.3
Q3: Does the installed signal support calibrated dynamic depth after retrofit?
After AR-RETROFIT installs the routing path, RESKIP can be applied without adding a separate router. On the canonical
2B retrofit, LAMBADA-500 at the conservative language operating point stays within 1.0pp of the no-skip retrofit
while skipping input-dependent blocks (Tab. 4). The aggressive thresholds show the expected calibration boundary:
once the threshold instructs the model to skip blocks it normally executes, quality drops sharply. Together with the
runtime characterization in Tab. 5, this shows the target-scale behavior we want: the installed routing path preserves or
improves model capability at near-base full-depth runtime, and RESKIP turns that path into calibrated input-dependent
depth allocation.
The full-depth retrofit path remains close to the base when both models run under the same fixed-shape inference
setting (Tab. 5). This is the practical runtime point: adding the routing path does not erase the quality gains with a large
inference penalty, and the dynamic skip rule operates on top of that installed path.
The block partition and mixture choices are reported in the appendix rather than treated as the main story. L=4 sits
on the accuracy plateau alongside nearby block sizes and reduces router calls; the final mixture is the one that preserves
VLM capability while training the routed correction. Full block-partition, data-mixture, and failed-design ablations are
in Appendices E.6, C.6, C.3.
## 4
## VLA Transfer Evidence
A VLA test asks whether the retrofitted routed-depth signal remains usable beyond VLM evaluation. Vision tokens
(perceptual, early-layer [38]), language tokens (task spec.), and action tokens (motor planning) have no principled reason
to share effective depth; ATTNRES routing gives a per-position handle on this without token-level gating machinery.
We attach the OpenVLA-OFT [24] action head (L1 regression, no chunking) to Qwen3-VL-2B/4B backbones (ViT
frozen) and train on the pooled libero_all mix from LIBERO [29]. The main comparison is deliberately narrow:
Path 0 uses the stock backbone with OFT, while Path B warm-starts from our canonical L=4 VLM retrofit and keeps
γn=1 throughout. Both paths share the same action head, data, and 30k-step OFT schedule on 4×H100 ZeRO-2 with
effective batch 32. Each policy is evaluated for 500 rollouts per suite (2,000 per policy). VLA-only retrofit controls,
longer schedules, and pipeline diagnostics are reported in the appendix.

Table 4: Quality–skip behavior after retrofit. LAMBADA-500 on the canonical L=4 2B retrofit, P={1, 4}, Mmax=2,
with thresholds calibrated on 32 held-out LAMBADA prefixes. The conservative point preserves quality while producing
input-dependent block skipping; lower thresholds show the expected calibration boundary.
Configuration
LAMBADA acc
LAMBADA ppl
∆acc vs no-skip
Avg. skipped blocks
No-skip retrofit
0.5700
4.526
—
0.00
AR-RETROFIT + RESKIP (q=0.85)
0.5600
5.258
−1.0pp
0.19
AR-RETROFIT + RESKIP (q=0.50)
0.4120
12.550
−15.8pp
1.06
AR-RETROFIT + RESKIP (q=0.30)
0.3900
14.005
−18.0pp
1.17
Table 5: Full-depth runtime of the installed routing path. Forward-pass latency on 1×H100, bf16, batch 1, median
of 20 runs after 5 warmup. The canonical compiled setting runs the retrofitted model within 2.9% of the compiled base,
while improving the quality metrics in Tab. 1.
Mode
seq
base (ms)
retrofit (L=4, ms)
retrofit / base
vs uncompiled base
Uncompiled
14.91
16.66
1.117×
1.117×
Uncompiled
25.49
28.53
1.119×
1.119×
Compiled (reduce-overhead)
21.31
22.50
1.056×
0.855×
Compiled (max-autotune)
21.81
22.43
1.029×
0.864×
4.1
LIBERO results
Path B reaches 97.15% on 2B and 96.70% on 4B. The gain concentrates on Spatial (+3.0pp at 2B) and Long-10
(+2.0pp at 4B), where we observe the clearest gains in this evaluation. We use these results as downstream transfer
evidence for AR-RETROFIT, not as a broad robotics claim. Appendix F.1 reports repeated-rollout sensitivity.
4.2
RESKIP on the action stream: calibrated conservative skipping
We now apply the same RESKIP rule at inference on the trained Path B policies, with thresholds calibrated on a sim-
rollout distribution rather than language tokens (the rationale is in the next paragraph). Eligible blocks are picked per
scale from a per-block action-drift sweep (Appendix F.3: P={1, 4} on 2B, P={1, 2} on 4B) and Mmax=2 throughout.
Tab. 7 shows the 4-suite Pareto across q.
Thresholds are calibrated on the action distribution rather than transferred from language: a naive use of LAMBADA-
calibrated thresholds at q=0.85 on the same policy reduces LIBERO-Spatial to 64% (Appendix F.3). The reason is that
q is not itself a skip rate; it indexes a sharp empirical distribution of wrecent. The recommended VLA point is therefore
conservative (q=0.99): it skips rarely, but only on blocks whose action-drift sweep says they are locally safe.
## 5
## Related Work
For test-time compute, o1 [34], DeepSeek-R1 [8], chain-of-thought [45], and optimal test-time compute analyses [41]
allocate compute by emitting more reasoning tokens; per-token depth remains fixed. Depth-axis methods include ACT
/ Universal Transformers [15, 9], CALM [40], Mixture-of-Depths [39], LayerSkip [13], and static pruning [16, 32].
These add a halting head, exit classifier, router, speculative decoder, or static removal rule. We instead use ATTNRES
as an intrinsic routing signal produced by the forward pass itself, then retrofit that signal into an existing pretrained
backbone. Appendix D.5 compares dynamic, static, and random skip rules at 340M, and Appendix E.2 gives a full-layer
static-pruning baseline at the VLM scale.
Table 8 summarises the closest method-level distinction. The relevant distinction is the combination of frozen-
backbone retrofitting and a pre-execution signal that is also part of the forward path.
The neural motivation is recurrence as flexible effective depth. The dual-process distinction [19] maps naturally
onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent
circuits [26, 20, 10]. Recurrent networks capture human representational dynamics [22]; Spoerer et al. [42] are the

Table 6: LIBERO 4-suite success rates (%) for Qwen3-VL-2B/4B. Entries are success rates over 500 rollouts per
suite (2,000 per policy). Path 0 is the stock-backbone OFT baseline; Path B warm-starts from our L=4 VLM retrofit.
The table is supporting transfer evidence for the retrofit pathway rather than a broad robotics claim.
Scale
Path (steps)
Spatial
Object
Goal
Long-10
Avg
∆vs. Path 0
2B
Path 0 (30k)
94.8
99.8
97.4
91.4
95.85
—
2B
Path B (30k, ours)
97.8
99.6
98.6
92.6
97.15
+1.30
4B
Path 0 (30k)
95.0
99.2
97.8
92.2
96.05
—
4B
Path B (30k, ours)
94.6
99.8
98.2
94.2
96.70
+0.65
Table 7: RESKIP 4-suite Pareto on the trained Path B policies. Eligible blocks P and threshold τ are calibrated per
scale on the action distribution; q is the empirical quantile that produces τ. The conservative end (q=0.99) preserves
success rates in this evaluation.
q (sim-calibrated)
Spatial
Object
Goal
Long-10
Avg
Qwen3-VL-2B Path B, P={1, 4}, M=2, no-skip ref 96.25
0.30
4.7
—
—
—
(sharp drop)
0.85
80.0
98.0
87.3∗
67.2
83.1
0.95
95.0
99.4
97.6
86.8
94.7
0.99
97.6
99.2
99.0
93.6
97.35
no-skip (ref)
97.4
98.6
98.0
91.0
96.25
Qwen3-VL-4B Path B, P={1, 2}, M=2, no-skip ref 96.25
0.30
89.6
95.4
96.4
86.0
91.85
0.50
91.2
98.0
98.2
89.6
94.25
0.85
93.6
99.2
97.6
93.2
95.90
0.95
95.6
98.0
98.0
93.0
96.15
0.99
96.4
98.2
98.4
92.8
96.45
no-skip (ref)
97.4
98.2
98.0
91.4
96.25
∗Partial eval (332/500 trials) — driver schedule cut short; reported as the rate over completed trials.
closest conceptual antecedent, showing that recurrent CNNs can spend more steps on harder images to explain speed–
accuracy behaviour. Recurrence also has a depth interpretation: finite recurrence can be unrolled into feedforward
depth, but the recurrent form reuses a fixed substrate for flexible effective depth [44]. We take this as a design target,
not a cognitive-faithfulness claim: cortical microcircuits and dendritic association [3, 6, 27] motivate intrinsic routing,
while our implementation is a transformer retrofit.
In residual architectures, DenseNet [18] and Highway networks [43] modify residual flow; ATTNRES [25] gener-
alises residual combination with depth attention. To our knowledge, this is the first work to use ATTNRES weights as a
pre-execution skip signal and to retrofit an ATTNRES-style routing path into pretrained transformers. In VLA, RT-2 [5],
Octo [33], OpenVLA [23], and Pi0 [4] established the paradigm, while efficiency work has focused mainly on action
chunking [50] and post-hoc compression. We contribute modality-aware adaptive depth for the VLM backbone.
## 6
## Discussion
AR-RETROFIT shows that a frozen pretrained transformer can acquire a pre-execution depth-routing path through a
short identity-preserving fine-tune. RESKIP shows that this installed signal can be calibrated into input-dependent block
execution, with ablation or drift checks determining which blocks are safe to skip. The current evidence is strongest
for the retrofit mechanism and calibrated dynamic-depth behavior; broader backbone coverage, stronger end-to-end
acceleration, and per-modality token-level calibration remain open.

Table 8: Closest method comparison. The relevant distinction is the combination of frozen-backbone retrofitting and a
pre-execution signal that is also part of the forward path.
Method
Pretrained backbone Extra decision head/router Decision before current block Applies to frozen pretrained backbone Same signal forms input and skip
Early exit / CALM
yes
yes
no
no
no
Mixture-of-Depths
train-time design
yes
yes
no
no
Static pruning
yes
no
yes
yes
no
LayerSkip
continued training
no / separate exit policy
partly / speculative
no
no
ATTNRES pretraining
no
no
yes
no
yes
AR-RETROFIT + RESKIP
yes
no
yes
yes
yes
## References
[1] AI-MO. NuminaMath-CoT: A dataset of competition-level mathematics problems with chain-of-thought solutions.
https://huggingface.co/datasets/AI-MO/NuminaMath-CoT, 2024.
[2] Shuai Bai et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.
[3] Andre M. Bastos, W. Martin Usrey, Rick A. Adams, George R. Mangun, Pascal Fries, and Karl J. Friston.
Canonical microcircuits for predictive coding. Neuron, 76(4):695–711, 2012. doi: 10.1016/j.neuron.2012.10.038.
[4] Kevin Black, Noah Brown, Danny Driess, et al. π0: A vision-language-action flow model for general robot control.
arXiv:2410.24164, 2024.
[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, et al. RT-2:
Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818,
2023.
[6] Timothy J. Buschman and Earl K. Miller. Top-down versus bottom-up control of attention in the prefrontal and
posterior parietal cortices. Science, 315(5820):1860–1862, 2007. doi: 10.1126/science.1138071.
[7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao,
Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Neural
Information Processing Systems (NeurIPS), 2024. MMStar benchmark; arXiv:2403.20330.
[8] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025.
[9] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers.
In ICLR, 2019.
[10] James J. DiCarlo, Davide Zoccolan, and Nicole C. Rust. How does the brain solve visual object recognition?
Neuron, 73(3):415–434, 2012. doi: 10.1016/j.neuron.2012.01.010.
[11] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen
Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint
arXiv:2305.14233, 2023.
[12] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer. In ICLR, 2020.
[13] Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas
Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu.
LayerSkip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12622–12642. Association
for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.681. URL https://aclanthology.
org/2024.acl-long.681/.
[14] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, et al. A framework for few-shot language model
evaluation. https://github.com/EleutherAI/lm-evaluation-harness, 2024.

[15] Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.
[16] Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts. The unreasonable
ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887, 2024.
[17] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina,
Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric
Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas
Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-
Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett,
Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan,
Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy,
Alexandros G. Dimakis, and Ludwig Schmidt. OpenThoughts: Data recipes for reasoning models, 2025. URL
https://arxiv.org/abs/2506.04178.
[18] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely connected convolutional
networks. In CVPR, 2017. doi: 10.1109/CVPR.2017.243.
[19] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011.
[20] Kohitij Kar, Jonas Kubilius, Kailyn Schmidt, Elias B. Issa, and James J. DiCarlo. Evidence that recurrent circuits
are critical to the ventral stream’s execution of core object recognition behavior. Nature Neuroscience, 22(6):
974–983, 2019. doi: 10.1038/s41593-019-0392-5.
[21] Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A
diagram is worth a dozen images. In European Conference on Computer Vision (ECCV), 2016.
[22] Tim C. Kietzmann, Courtney J. Spoerer, Lynn K. A. Sörensen, Radoslaw M. Cichy, Olaf Hauk, and Nikolaus
Kriegeskorte. Recurrence is required to capture the representational dynamics of the human visual system.
Proceedings of the National Academy of Sciences, 116(43):21854–21863, 2019. doi: 10.1073/pnas.1905544116.
[23] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov,
Ethan Foster, Grace Lam, Maja Vosshall, et al. Openvla: An open-source vision-language-action model. arXiv
preprint arXiv:2406.09246, 2024.
[24] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and
success. arXiv preprint arXiv:2502.19645, 2025.
[25] Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo
Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie,
Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui,
Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu,
Zhilin Yang, and Xinyu Zhou. Attention residuals, 2026. URL https://arxiv.org/abs/2603.15031.
Technical report.
[26] Victor A.F. Lamme and Pieter R. Roelfsema. The distinct modes of vision offered by feedforward and recurrent
processing. Trends in Neurosciences, 23(11):571–579, 2000. doi: 10.1016/S0166-2236(00)01657-X.
[27] Matthew Larkum. A cellular mechanism for cortical associations: an organizing principle for the cerebral cortex.
Trends in Neurosciences, 36(3):141–151, 2013. doi: 10.1016/j.tins.2012.11.006.
[28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei
Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. arXiv preprint arXiv:2408.03326,
2024.

[29] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking
knowledge transfer for lifelong robot learning. In NeurIPS Datasets and Benchmarks, 2023.
[30] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang,
Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?
In European Conference on Computer Vision (ECCV), 2024. arXiv:2307.06281.
[31] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu,
Lianwen Jin, and Xiang Bai. OCRBench: On the hidden mystery of OCR in large multimodal models. Science
China Information Sciences, 67:220102, 2024. doi: 10.1007/s11432-024-4235-6. arXiv:2305.07895.
[32] Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen.
Shortgpt: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853,
2024.
[33] Octo Model Team. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.
[34] OpenAI.
Learning
to
reason
with
LLMs.
https://openai.com/index/
learning-to-reason-with-llms/, 2024.
[35] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle,
Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a
broad discourse context. In ACL, 2016.
[36] Guilherme Penedo, Hynek Kydlíˇcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro
Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. NeurIPS
Datasets and Benchmarks, 2024.
[37] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, Jiayuan Gu, Bin Zhao,
Dong Wang, and Xuelong Li. SpatialVLA: Exploring spatial representations for visual-language-action model.
arXiv preprint arXiv:2501.15830, 2025.
[38] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision
transformers see like convolutional neural networks? In NeurIPS, 2021.
[39] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro.
Mixture-of-depths: Dynamically allocating compute in transformer-based language models. arXiv preprint
arXiv:2404.02258, 2024.
[40] Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler.
Confident adaptive language modeling. In NeurIPS, 2022.
[41] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more
effective than scaling model parameters. In International Conference on Learning Representations (ICLR), 2025.
[42] Courtney J. Spoerer, Tim C. Kietzmann, Johannes Mehrer, Ian Charest, and Nikolaus Kriegeskorte. Recurrent
neural networks can explain flexible trading of speed and accuracy in biological vision. PLOS Computational
Biology, 16(10):e1008215, 2020. doi: 10.1371/journal.pcbi.1008215.
[43] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway networks. In ICML Deep Learning
Workshop, 2015.
[44] Ruben S. van Bergen and Nikolaus Kriegeskorte. Going in circles is the way forward: the role of recurrence in
visual inference. Current Opinion in Neurobiology, 65:176–193, 2020. doi: 10.1016/j.conb.2020.11.009.

[45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and
Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Neural Information
Processing Systems (NeurIPS), 2022.
[46] xAI. RealWorldQA: A benchmark for real-world spatial understanding. https://huggingface.co/
datasets/xai-org/RealworldQA, 2024.
[47] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang,
Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning
benchmark for expert AGI. In CVPR, 2024.
[48] Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah D. Goodman. Quiet-STaR:
Language models can teach themselves to think before speaking. In Conference on Language Modeling (COLM),
2024.
[49] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish
your sentence? In ACL, 2019.
[50] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation
with low-cost hardware. In RSS, 2023.
[51] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé, Andrey Kolobov, Furong Huang, and
Jianwei Yang. TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic
policies. arXiv preprint arXiv:2412.10345, 2024.
## A
## Limitations and future work
Limitations.
(i) Two retrofit targets from one VLM family; extension to InternVL / Gemma-VL / text-only LMs is
planned. (ii) Runtime measurements are H100 / bf16 / batch 1 and use standard PyTorch inference paths; lower-end
GPU / edge measurement is not yet done. (iii) The current RESKIP rule is block-level, inherited from ATTNRES;
per-token routing and fused production kernels are future work. (iv) The VLA Pareto thresholds are sim-calibrated per
scale, not per modality within a single rollout; per-modality, per-token-class calibration is the next sweep.
Future work: weight-shared AttnRes (RELOOP).
A natural extension shares block weights across depth and
differentiates each application via position-indexed pseudo-queries; the ATTNRES routing statistic can then serve both
RESKIP’s high-immediate-predecessor skip rule and a RELOOP halt rule based on low marginal change across recurrent
applications. A 74M validation produced a smooth depth–quality Pareto (25% compute saved at < 1% ppl change);
scaling and combining with the retrofit framework is open.
## B
## Implementation details
B.1
Online softmax merge with skip
When a block is skipped, the online softmax merge simply does not process its output. The running state (s, m, e) is
unchanged. The output is mathematically equivalent to an ATTNRES model trained without that block contributing—
modulo the calibration of downstream wl.
B.2
Pseudo-query initialisation (Section 2.3 from-scratch)
For from-scratch training we use wl ∼N(0, 0.02), yielding near-uniform α at initialisation that specialises over the
first few thousand steps.

B.3
Retrofit hyperparameters
AdamW with β=(0.9, 0.95), weight decay 0, gradient clip 1.0, bfloat16. Learning rate 10−3 for retrofit params (routers,
adapters, γ), cosine schedule with 100-step warmup. Sequence length 1536 (training), 2048 (evaluation). Loss weights
λkl=1, entropy weight 0.02, KD temperature 1. The entropy term is subtracted from the loss, so it maximises routing
entropy early in training rather than minimising it. Skip-branch sampling: one eligible block chosen uniformly at
random per step. Adapter bottleneck rank r=256 (canonical); position-bias on router keys enabled; router temperature
fixed at 1. γ-curriculum: 0 →1 linearly over the first 30% of total steps (2B 5k/10k, 4B 5k), with ramp-fraction 0.5 on
4B ×10k to stabilise a late-stage γ=1 transition divergence (Appendix B.7). Total training wall-clock on a single H100:
∼22 min for 2B 5k, ∼44 min for 2B 10k, ∼54 min for 4B 10k.
B.4
Block granularity
For the final recipe we use L=4 layers per block on both scales: Qwen3-VL-2B has N=7 blocks and Qwen3-VL-4B
has N=9 blocks. Earlier L=2 runs remain in the appendix as ablation controls, but L=4 is the canonical setting
because it preserves the accuracy plateau while halving router calls relative to L=2. Each block receives one router,
one residual adapter (r=256 default), and a scalar γ gate.
B.5
Identity-at-init details
With γn=0 and the adapter up-projection initialised at N(0, 0.022), the retrofit forward is arithmetically identical to
the pretrained model (both produce xn = hn−1, which is passed into the unmodified block). We verified end-to-end:
at n=500 on LAMBADA the retrofit and the base agree to within bfloat16 evaluation noise (acc 0.530 vs. 0.532; ppl
5.572 vs. 5.547); on MMMU and MMStar they are bit-equal at the answer-choice level. The smoke test at the token
level shows max |∆logits| = 0.375 (bfloat16 SDPA noise) and 100% argmax agreement on our calibration prompts.
B.6
Skip under use_cache=True: correctness verification
We verify that the K/V-only skip path described in §2.2 produces an argmax-consistent output with the cache-
less path. Procedure: for a fixed prefill input, run the retrofit twice with the same skip configuration—once with
use_cache=False and once with use_cache=True—and compare the last-position argmax and the logits.
We swept skip sets {[4], [4, 10], [4, 10, 12], [2, 4, 10]} on H_r256_5k on Qwen3-VL-2B. In every configuration the
last-position argmax matches between the two cache regimes, and the maximum logit delta is 0.19–0.37, identical to
the bfloat16 SDPA jitter observed on the stock base with no retrofit and no skip. Multi-step autoregressive divergence
starting around position 27–40 is an inherent property of HF + bf16 + SDPA that the stock base exhibits on the same
prompts (measured on “Once upon a time” and “def fibonacci”); it is not a property of our skip path.
B.7
γ=1 transition stability
The γ curriculum is used to keep the frozen residual stream close to its pretrained operating point while the routed
correction becomes active. Across the reported 2B and 4B canonical runs, we use a ramp that reaches γ=1 after at least
5,000 optimisation steps. This schedule was stable across the scales and block partitions reported in the paper and does
not confound the quality comparisons in Appendix E.6.
## C
## Retrofit ablations and inference machinery
C.1
Design controls
We compared the γ-gated residual injection against three simpler designs: observer-only routing, block-input interpo-
lation, and direct ATTNRES replacement with informed pseudo-query initialisation. These controls either made the
routing weights non-load-bearing, moved the frozen backbone too abruptly away from its pretrained residual stream, or
required substantially more tuning than the short retrofit budget. The γ-gated injection of §2.2 is the stable recipe used
in all main experiments.

C.2
Adapter-rank ablation
Wall-clock latency is essentially unchanged across ranks (±2% at all measured sequence lengths): the adapter’s
2048→r→2048 bottleneck is a small contributor compared to the router stack/softmax/einsum cost. Rank therefore
affects quality but not speed. The canonical r=256 was selected from the γ→1 ablation of Appendix C.3; at equal step
budgets lower ranks leave more of the MMBench drop on the table without compensating latency savings.
C.3
γ→1 retrofit ablation
We swept variants of the canonical γ→1 recipe along four axes: adapter rank r ∈{32, 64, 128, 256}, training steps
∈{5k, 10k, 20k}, LLaVA-Instruct fraction of the mix ∈{50%, 80%}, and γ ramp-fraction ∈{0.3, 0.7}. All other
hyperparameters (loss, optimizer, schedule family, seed) are held constant.
Table 9: Retrofit ablation on Qwen3-VL-2B. All rows use the γ-curriculum 0→1 over the first 30% of steps unless
noted. “n=300 MMBench noise floor ±1 question” corresponds to ±0.33pp.
Name
γ sched
rank
steps
VLM%
ramp
LAMBADA
HellaSwag
MMBench
Base Qwen3-VL-2B
—
—
—
—
—
0.532
0.506
0.727
Canonical
0→1 fast
5k
0.3
0.576
0.522
0.710
+ low rank
0→1 fast
10k
0.3
0.570
0.530
0.697
+ very low rank
0→1 fast
10k
0.3
0.568
0.526
0.683
+ low rank, VLM-heavy
0→1 fast
10k
0.3
0.560
0.524
0.660
+ mid rank, VLM-heavy
0→1 fast
10k
0.3
0.564
0.520
0.653
+ VLM-heavy
0→1 fast
10k
0.3
0.560
0.520
0.617
+ longer, 10k
0→1 fast
10k
0.3
0.586
—
0.587
+ slow ramp
0→1 slow
10k
0.7
0.568
0.520
0.517
+ longer, 20k
0→1 fast
20k
0.3
0.568
0.520
0.513
Over-training signature at γ=1.
Holding (r=256, 50% VLM, fast ramp) fixed and varying only the total step count,
MMBench is strictly monotonic in steps: 5k →0.710 (preserved), 10k →0.587 (−14pp), 20k →0.513 (−21pp).
LAMBADA peaks at 10k (0.586) and regresses at 20k (0.568), so longer training buys no text-domain gain either. The
γ=1 regime has a narrow compute sweet spot; past it the router+adapter over-specialise to the training distribution and
the frozen backbone, unable to adapt further, loses its multimodal calibration.
Rank and data are subordinate.
Halving rank at 10k recovers some MMBench (r=32: 0.683, r=64: 0.697), but
never reaches the 5k-r=256 level (0.710). Pushing the mix to 80% LLaVA worsens MMBench at every rank, indicating
the regression is router over-adaptation to any narrow training distribution, not insufficient visual exposure.
Interpretation.
These ablations are used only to set the canonical recipe. They are not the main evidence for RESKIP;
the load-bearing evidence is that the installed routing path participates in the forward computation and supports the
calibrated skip rule in §3.3.
C.4
Per-block skip importance
Running single-block removals on the canonical 2B retrofit gives the eligibility set used in the main text, P={1, 4}
(Tab. 4). The same qualitative ranking pattern appears across the 340M from-scratch and 2B retrofit settings: some
blocks near the embedding or late residual-refinement stages are costly to remove, while selected mid-depth positions
are safer candidates after per-model calibration.

C.5
Calibration set (dynamic skip)
Calibration uses 32 held-out LAMBADA prefixes (truncated to 512 tokens each). Per-block wrecent(n) samples are
collected with a single retrofit forward and the empirical quantile at q is used as τn. We verified that varying the
calibration draw shifts τn by < 0.01 in absolute terms at the chosen q=0.95. For VLA deployment, the identical calibra-
tion pipeline applied to the VLA in-backbone forward (retrofit/eval/calibrate_vla_thresholds.py)
produces thresholds byte-identical to the LAMBADA-calibrated set on matched inputs, confirming that the VLA and
VLM forwards instantiate the same router.
C.6
Canonical data mixture
Table 10 gives the final SFT mixture used for the canonical retrofit. The mix is intentionally VLM-anchored, with a
smaller math/reasoning component to strengthen deliberate reasoning without moving the frozen VLM backbone away
from visual instruction following.
Table 10: Canonical retrofit SFT mixture. The same source proportions are used for the 2B and 4B canonical L=4
recipes in Tab. 1.
Source
Proportion
Role
LLaVA-OneVision-Data [28]
60%
visual instruction anchor
UltraChat [11]
20%
general instruction following
NuminaMath-CoT [1]
10%
mathematical reasoning traces
OpenThoughts-114k [17]
10%
open-ended reasoning traces
This mixture is the only data recipe used for the canonical results in the main paper.
C.7
Two-phase forward with dynamic skip (algorithm)
Algorithm 1 Two-phase ATTNRES forward with dynamic RESKIP.
1: Input: embeddings h0; block functions f1, . . . , fN; pseudo-queries {wn}; eligible P; thresholds {τn}; max skips
Mmax.
2: Initialise online-softmax state (s, m, e) ←INIT(h0); skip counter k ←0.
3: for n = 1 to N do
4:
Compute α·→n over completed block outputs (phase 1).
5:
Form routed-correction input xn as in Eq. 2.
6:
wrecent(n) ←Etoken[αn−1→n].
7:
if n ∈P and k < Mmax and wrecent(n) > τn then
8:
Skip: set hn ←xn; under use_cache=True, run K/V-only slice on each constituent layer; k ←k + 1.
9:
else
10:
Execute block n on the merged input; update (s, m, e) with hn.
11:
end if
12: end for
13: Output: final hidden state s/e.
## D
## Phase 1: 340M from-scratch validation
D.1
Motivation: ATTNRES cost from-scratch is prohibitive
We measured forward-pass latency on 1×H100 / bf16 / batch 1 for two 340M models trained under identical FineWeb-
Edu 100BT recipes: a vanilla standard-residual transformer and an ATTNRES transformer with N=8 blocks. Block-
ATTNRES adds +78% (8192 seq) to +128% (1024 seq) wall-clock vs. vanilla. At the 2B VL scale a stock Qwen3-

VL-2B forward already takes ∼25.6 ms at seq 256 on 1×H100, so a 35–45% block-level ATTNRES overhead would
push past common 50 Hz / 20 Hz robotic control budgets. Pretraining a 2B ATTNRES-VLM costs at least what the
matched standard-residual base costs (≥10k–25k H100-h reading published Qwen3-VL / InternVL3 / OpenVLA /
LLaVA-OneVision tech reports), so the only practical access path is a retrofit on the already-trained standard model.
D.2
Phase 1: cross-scale validation
The 340M from-scratch ATTNRES (§2.3, Table 2) is the load-bearing existence proof. A 110M cross-scale check on the
same FineWeb-Edu recipe with an 8-block partition reproduces the zero-degradation pattern at a safer operating point
(q=0.97, Mmax=1 vs. 340M’s q=0.85, Mmax=2); skip trigger rate is lower at 110M, consistent with larger models
carrying more redundant computation. 1.3B / 2B from-scratch are not load-bearing because the retrofit (§3) directly
targets a pretrained 2.13B model, answering the 2B-scale question without paying the pretrain bill.
D.3
RESKIP method comparison and full Pareto / latency at 340M
Table 11: RESKIP versus representative adaptive-computation methods.
Method
Auxiliary network
Routing is
Train-time changes
Architectural changes
CALM [40]
exit classifiers
per-token
yes
small
Mixture-of-Depths [39]
router
per-token
yes, capacity loss
yes
Static pruning [16]
none
static
no
block removed
LayerSkip [13]
spec. decoder
per-token
yes, cont. pretrain
decoding path
RESKIP (ours)
none
per-batch
none
none
D.4
RESKIP position-set ablation at 340M
Table 12: Dynamic skip as a function of P on 340M from-scratch ATTNRES (Mmax=2, q=0.85). Picking by I alone
({5}) or A alone ({3}) leaves speed or quality on the table; combining the two ({3, 5}) is strictly best.
P
PPL ratio
Speedup
{5} (I-only)
0.993
1.14×
{3} (A-only)
1.009
1.02×
{2, 3}
1.009
1.17×
{4, 5}
1.008
0.97×
{3, 4, 5}
0.993
1.02×
{2, 3, 4, 5, 6}
1.40
1.16×
{3, 5} (ours)
0.991
1.19×
D.5
Decision rule: dynamic vs. static-rate-matched vs. random
The position-set ablation above answers which blocks to skip; this subsection answers whether the input-dependent
decision matters at all. We hold the 340M from-scratch ATTNRES weights fixed, fix P={3, 5} and Mmax=1 (block-
level skip rate ≤12.5%), and only vary the runtime decision rule:
• B0. No-skip upper bound (full ATTNRES forward).
• B1.b/B2.a. Dynamic, fire when wrecent,n > τn (our RESKIP signal at L=1 block granularity).
• B1.c, B1.d. Static, every-token skip at P=3 or P=5 (rate-matched, 12.5%).
• B1.e/B2.d. Random, per-call uniform draw from {keep-P=3, keep-P=5} (rate-matched, 12.5%).

• B2.b. Dynamic, fire when block-1 entropy Hn < τn (router-confidence rule).
• B2.c. Dynamic, fire when wrecent,n −wembed,n > τn (relative-recent rule).
All three thresholds are calibrated as the q=0.5 per-position quantile on 32×8192 FineWeb-Edu tokens (§C.5). Results
in Tab. 13; observed skip rates on the eval distribution in Tab. 14.
Table 13: Decision-rule and threshold-rule ablation on 340M from-scratch ATTNRES, lm-evaluation-harness with
P={3, 5}, Mmax=1. Dynamic rules whose threshold actually fires on the eval distribution (B2.c) preserve LAMBADA
at 0.345 vs. static / random at the same ∼12% rate (0.22–0.26). Dynamic rules whose calibrated thresholds rarely cross
on lm-eval data (B1.b at 3.1% observed; B2.b at 0.0%, see Tab. 14) are nearly equivalent to no-skip and are not the
rate-matched comparison.
Cell
LAMBADA
HellaSwag
PIQA
ARC-e
OpenBookQA
B0. no-skip (upper bound)
0.4054
0.4607
0.6893
0.5438
0.3580
Dynamic, calibrated q=0.5 on FineWeb-Edu
B1.b. recent_weight_gt
0.4011
0.4534
0.6839
0.5412
0.3580
B2.b. entropy_lt
0.4036
0.4608
0.6839
0.5417
0.3580
B2.c. recent_minus_embed_gt
0.3445
0.4471
0.6774
0.5391
0.3540
Static / random, rate-matched at 12.5%
B1.c. static skip P=3 every
0.2624
0.3968
0.6610
0.5206
0.3300
B1.d. static skip P=5 every
0.2189
0.4223
0.6638
0.5253
0.3260
B1.e/B2.d. random {P=3 OR P=5}
0.2416
0.4028
0.6420
0.5101
0.3300
Table 14: Observed skip rate on 8 × 512 LAMBADA tokens. The calibration set (FineWeb-Edu) and the eval set
(LAMBADA) have different routing-statistic distributions; B2.b’s entropy_lt threshold never fires on LAMBADA
and B1.b fires only 3.1%, so their headline-table accuracy is misleadingly close to no-skip. B2.c is the strategy whose
calibration target (∼12.5%) actually transfers, and is therefore the load-bearing rate-matched comparison against
B1.c/d/e in Tab. 13.
Cell
Calibration target
Observed (LAMBADA)
Notes
B0
0%
0.00%
sanity check, τn=∞
B1.b
≤12.5%
3.12%
τ3=0.4645, τ5=0.4010
B2.b
≤12.5%
0.00%
τ3=0.8631, τ5=0.8764
B2.c
≤12.5%
10.94%
τ3=0.1655, τ5=0.2472
B1.c
12.5%
12.50%
forced, every-token P=3
B1.d
12.5%
12.50%
forced, every-token P=5
B1.e
12.5%
per-batch toggle
uniform {P=3, P=5}
Conclusion.
At a fair rate-matched comparison (∼12% fired skips), input-dependent dynamic skip (B2.c) preserves
LAMBADA at 0.345 while every static or random alternative degrades to 0.22–0.26 (−8 to −13pp). The same ordering
holds on HellaSwag, PIQA, ARC-easy and OpenBookQA. The MoD-style “you might be getting a free lunch from
any same-rate schedule” critique is rejected: at 340M from-scratch, schedule choice matters and the ATTNRES routing
weights carry the load.
Threshold-transfer caveat.
Two of the three dynamic rules (B1.b recent_weight_gt and B2.b entropy_lt)
fire far below the 12.5% FineWeb-Edu calibration target on the LAMBADA distribution (Tab. 14). Their near-no-skip
accuracy is therefore not a positive datapoint for those specific rules; it is consistent with “the threshold protected the
output by accidentally not firing.” The load-bearing rate-matched comparison is B2.c recent_minus_embed_gt

(which does fire close to target) versus the static/random alternatives. We treat the two non-firing rows as a sensitivity
result: RESKIP’s safety margin under threshold/distribution mismatch is high (no degradation when the rule rarely fires),
but deployment requires per-distribution threshold calibration, as already noted for the cross-modality VLA case (§F.3).
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
## E
## Retrofit Pareto and breakdowns
E.1
LoRA baselines
These auxiliary controls ask whether adding a similar number of trainable parameters through a standard adaptation
path is enough to reproduce the text gains. They are pilot controls on an earlier 50/50 UltraChat + LLaVA mix, not a
final-mixture baseline for Tab. 1. LoRA r=32 on q, v: LAMBADA acc 0.540 / 0.516 (two seeds), HellaSwag 0.510
/ 0.524. LoRA r=16 on q, k, v, o: LAMBADA 0.534, HellaSwag 0.492. LoRA r=8 on MLP: LAMBADA 0.514,
HellaSwag 0.510. Mean LAMBADA across the four runs is 0.526, −0.6pp below base (0.532) and −5.0pp below the
matched γ→1 retrofit on that mix (0.576). Mean HellaSwag 0.509 is +0.3pp over base versus retrofit’s +1.6pp. These
controls rule out a simple parameter-count-only explanation under a related SFT setting, but they do not prove that all
quality gains come uniquely from routing.
E.2
Static-pruning baseline at the VLM scale (Gromov drop-k)
The 340M decision-rule ablation (Tab. 13) shows that single-position static skip and random skip both degrade on
language tasks. The complementary question at the VLM scale is whether full-layer static pruning—the standard
“remove the least useful blocks” baseline used by Gromov et al. [16]—can match the accuracy floor that the retrofit
operates above. We rank the 28 Qwen3-VL-2B layers by the Gromov angular-distance criterion (computed on a
FineWeb-Edu calibration set with the LM-head only; no fine-tuning), remove the k lowest-distance layers, and
re-evaluate on lmms-eval. Tab. 15 reports drop-4 (layers {23, 24, 25, 26}) and drop-8 (additionally {12, 13, 14, 22}).
The pattern matches the in-block 340M observation: any unconditional block removal (single-position static at
340M, full-layer Gromov at 2B) destroys benchmark accuracy at the VLM scale even at modest pruning fractions, while
input-dependent skip on top of an ATTNRES routing structure preserves it. We do not attempt to retrain the pruned
model; the intent is a free static-baseline floor for the cost-quality trade-off, not a tuned competitor.*
*Pruned cells additionally exhibit POPE generation drift below the strict yes/no surface (drop-4 pope-acc 0.021, drop-8 0.500 via accidental

Table 15:
Static layer pruning on Qwen3-VL-2B (Gromov-style angular-distance ranking, no retraining),
lmms-eval full splits. Companion to the 340M in-block decision-rule ablation in Tab. 13: spans the full-layer
vs. single-position-within-block skip dimension. Pruning 4 of 28 layers (14%) costs −16pp AI2D, −6pp MMMU,
−5pp MMStar, −65pp OCRBench; pruning 8 (29%) sharply degrades every cell. The retrofit (Tab. 1) operates at
0 removed layers, 0pp loss, and additionally lifts the base; RESKIP on the retrofit at q=0.85 holds within 1pp on
LAMBADA (Tab. 16).
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
+ AR-RETROFIT v3 (L=4, 10k; Tab. 1)
0.758
0.432
0.536
0.814
0.661
E.3
RESKIP on the canonical retrofit: text Pareto and cross-modality consistency
On the canonical L=4 v3 10k retrofit, applied to LAMBADA-500 as the language-modality cross-check of the LIBERO
Pareto (Tab. 7), RESKIP produces the same shape as on action: lossless at the conservative end, collapsing once q
moves below the action distribution’s mean. Same P={1, 4}, Mmax=2 as the 2B VLA cell.
Table 16: Cross-modality RESKIP consistency. LAMBADA-500 on the canonical L=4 v3 retrofit, P={1, 4}, M=2,
τ calibrated on 32 held-out LAMBADA prefixes at quantile q. Same operating-point structure as the LIBERO 4-suite
Pareto: near-lossless above the calibration distribution mean (q≥0.85); sharp degradation below, where the threshold
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
E.4
Wall-clock latency: full table including earlier L=2 and VLA in-backbone
Tab. 5 in the main body reports the canonical L=4 result. Tab. 17 below shows the full set: the earlier L=2 partition
(14 blocks at 2B) carries a 1.39× structural cost over stock Qwen3-VL-2B because each token pays 14 router calls, the
VLA in-backbone forward (the same retrofit run inside the OFT trainer’s backbone forward, §4) tracks the VLM retrofit
within 1% at both partitions, and torch.compile closes the L=2 residual to 1.16× but is more dramatic on L=4
where the router count is halved. This table is the basis for the canonical-partition switch from L=2 to L=4.
The structural per-block router stack (stack→RMSNorm→softmax→einsum over N completed blocks) is
small but visible in eager inference. Adapter rank ablation (Appendix C.2) confirms the adapter is not the bottleneck
(≤2%). torch.compile (rows 5–8) collapses the small router gemms into a captured graph and tunes the matmul
kernels for retrofit’s small shapes, removing most of the L=4 structural gap. The remaining 2.9% is the cost of the
installed routing path at the canonical operating point; fusing the router and skip decision into a single production kernel
is an implementation direction rather than a change to the method.
E.5
Compile vs. eager: accuracy parity
The compiled-overhead result of §3 requires that torch.compile preserve the retrofit’s accuracy. We verified two
parity tests on the canonical L=4 v3 10k retrofit.
all-“yes” decoding), so we drop POPE from the table.

Table 17: Forward-pass latency, full sweep on 1×H100, bf16, batch 1, median of 20 runs after 5 warmup. Eager rows:
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
(1) L=2 retrofit (earlier)
eager
15.42
21.25
1.378×
(2) L=2 retrofit (earlier)
eager
26.03
36.14
1.388×
(3) L=4 retrofit (canonical)
eager
14.91
16.66
1.117×
(4) L=4 retrofit (canonical)
eager
25.49
28.53
1.119×
(5) L=2 retrofit
reduce-overhead
9.31
10.78
1.158×
(6) L=2 retrofit
reduce-overhead
21.25
24.45
1.151×
(7) L=4 retrofit
reduce-overhead
21.31
22.50
1.056×
(8) L=4 retrofit
max-autotune
21.81
22.43
1.029×
(9) VLA in-backbone L=4 retrofit
eager
25.49
28.79
1.130×
LAMBADA-500 accuracy parity.
Running LAMBADA-500 with the retrofit wrapped in torch.compile using
default mode and dynamic shapes: acc 0.5720 compiled vs. 0.5700 eager (∆=+0.20pp), ppl 4.534 vs. 4.526. Per-target
argmax agreement against eager: 98.60% — the residual 1.4% are tokens where eager and compiled both fall within
bf16 SDPA jitter and the rank-1/rank-2 logits flip.
Per-token logit parity.
On real prompt tokens with mode="reduce-overhead" (the inference-time mode used
for the speed table): per-token argmax agreement 98.51%, max |∆logits| = 0.50, RMSE 0.060 — same magnitude as
the cache-on/cache-off SDPA jitter on the stock base (Appendix B.6). Compile preserves the retrofit’s forward to within
bf16 evaluation noise; the 1.029× base_compiled number is at matched accuracy.
E.6
Block partition sweep
Table 18: Block partition sweep (v3 recipe, r=256, γ-curriculum). L is layers-per-block, N=Ltotal/L. Per-layer
L=1 underperforms by 2–9pp; the plateau L ∈{2, 4, 7} on 2B reproduces Kimi Team et al. [25]’s S ∈{2, 4, 8}. We
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
0.6540
+7.8
0.554
84.79
0.531
0.623
0.817
0.824
0.715

Table 19: MMStar subcategory breakdown for the v3 retrofit. The math, logical, and sci&tech triple is the “reasoning
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
Table 20: Per-seed Qwen3-VL-2B numbers. Seed 1 is the initial run; seed 2 is a fresh environment-reseeded rollout of
the same trained checkpoint. The last column shows the values used in Table 6.
Suite
Policy
Seed 1
Seed 2
Used in Tab. 6
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
and Path B 96.75 (∆=+0.70pp), while the Table 6 operating-point values give ∆=+1.30pp. Both summaries preserve
the ranking Path B > Path 0. The main text therefore treats LIBERO as supporting transfer evidence rather than as the
primary contribution.
F.2
VLA landscape: open VLAs on standard benchmarks
Table 21: Open VLA landscape on standard benchmarks. LIBERO entries are success rates (%) per task suite. Following
common practice, π0 and Octo-Base LIBERO numbers are taken from OpenVLA-OFT [24], which re-ran all three
policies under a uniform LIBERO protocol. Our rows use the 30k Path B numbers from Table 6.
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
[33, 24]
OpenVLA
7B
84.7
88.4
79.2
53.7
[23, 24]
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
[4, 24]
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

F.3
VLA RESKIP setup: per-block drift and eligible-set selection
The RESKIP eligible set P for each scale is selected from a per-block ablation on the trained Path B policy, not on the
VLM retrofit alone, because the VLA action stream activates a different routing distribution than the language stream.
We measure per-block action-drift MSE on a 24-sample sim trajectory: the policy is run with no skip and with each
block individually skipped; the per-step action MSE between the skipped and no-skip rollouts is the per-block “cost of
removing this block on the action distribution”. The two safest blocks per scale form P.
2B Path B 30k per-block action-drift MSE (sorted ascending, in units ×10−3): block 1: 0.4; block 4: 34.0; block 0:
58.1; block 5: 87.0; remaining blocks >100. Selecting the two lowest: P2B = {1, 4}.
4B Path B 30k per-block action-drift MSE: block 1: 0.6; block 2: 11.0; block 0: 34.0; block 3: 63.0; remaining
>120. Selecting: P4B = {1, 2}.
Per-scale eligibility.
The eligible set is selected separately for each model scale because the action-stream routing
distribution and per-block drift are scale dependent. This is the per-scale half of the modality-aware skip protocol: α
provides the online signal, while action-drift calibration selects the safe candidate blocks for the policy being deployed.
Threshold calibration.
Per-block thresholds τn are the q-th empirical quantile of wrecent,n over 31,286 (2B) / 31,257
(4B) sim-rollout records (one record per decoded action token across many trajectories). The Tab. 7 numbers use this
protocol. A separate “Method A” that calibrates on retrofit pretokenized data (as if the VLA were just a long-context
LM) lands at a conservative operating point that mimics q=0.99 on the action distribution; we use it as a sensitivity
comparator in our ablations and recommend the action-distribution-calibrated thresholds for any deployment.
Cross-modality threshold calibration.
Language-calibrated thresholds and action-calibrated thresholds correspond
to different effective skip rates because the action stream has a different wrecent distribution. We therefore calibrate τn
on the action distribution for LIBERO rather than transferring the LAMBADA thresholds unchanged. This is the same
calibration principle used in the main method: routing decides when to skip, while the eligible set and thresholds are
selected on the deployment distribution.
F.4
VLA planned follow-ups
Per-modality, per-token-class RESKIP. The Tab. 7 thresholds are calibrated on the action distribution as a whole;
an obvious refinement is per-modality (P(m), τ (m)
n
, M (m)
max) within a single rollout (vision tokens, language tokens,
action tokens). Working hypothesis: vision α concentrates on early blocks, while action α spreads to late blocks and
benefits from more conservative late-block skipping. Fused routing kernels. A static-graph variant of the skip rule
could further reduce the installed-path runtime. Edge hardware. Wall-clock is H100 / bf16; RTX 4090 / Jetson Orin
direct measurement is planned.
## G
## NeurIPS Paper Checklist
1. Claims. Yes. The abstract, introduction, and conclusions state the main claims and separate the retrofit
contribution from calibrated skipping, VLA transfer, and systems characterization.
2. Limitations. Yes. Appendix A lists architecture coverage, hardware scope, block-level routing granularity, and
per-modality calibration limits.
3. Theory assumptions. Not applicable. The paper is empirical and algorithmic; all equations define implemented
objectives or routing rules.
4. Experimental reproducibility. Yes. Sections 2.3–4.1 and the appendix specify model scales, block partitions,
training steps, datasets, calibration rules, and evaluation protocols.
5. Open access to code and data. Yes. Experiments use public datasets and benchmarks; implementation details
and release plan are described under the anonymous submission policy.

6. Compute. Yes. Main text and appendices report GPU type, training steps, rough GPU-minutes / GPU-hours, and
latency measurement settings.
7. Dataset and benchmark provenance. Yes. All public datasets and benchmarks are cited; LIBERO and
lmms-eval protocols are described with rollout counts or split usage.
8. Human subjects. Not applicable. No new human-subject data are collected.
9. Privacy. Not applicable beyond the privacy considerations of the cited public datasets.
10. Licenses. Yes. Public datasets/models are cited; release artifacts will include license metadata.
11. Broader impacts. Yes. The work targets lower inference cost and adaptive computation in pretrained models; no
safety-critical deployment is claimed.
12. Safeguards. Yes. VLA thresholds are calibrated per model and action distribution; cross-modality transfer
failures are explicitly documented as a deployment risk.