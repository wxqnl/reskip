## Implementation details
B.1
Online softmax merge with skip
When a block is skipped, the online softmax merge simply does not process its output. The running state (s, m, e) is
unchanged. The output is mathematically equivalent to an ATTNRES model trained without that block contributing—
modulo the calibration of downstream wl.
B.2
Pseudo-query initialisation (Section 2.2 from-scratch)
For from-scratch training we use wl ∼N(0, 0.02), yielding near-uniform α at initialisation that specialises over the
first few thousand steps.
13

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
has N=9 blocks. Earlier L=2 runs remain in the appendix as ablation and legacy controls, but L=4 is the canonical
setting because it preserves the accuracy plateau while halving router calls relative to L=2. Each block receives one
router, one residual adapter (r=256 default), and a scalar γ gate.
B.5
Identity-at-init details
With γn=0 and the adapter up-projection initialised at N(0, 0.022), the retrofit forward is arithmetically identical to
the pretrained model (both produce xn = hn−1, which is passed into the unmodified block). We verified end-to-end:
at n=500 on LAMBADA the retrofit and the base agree to within bfloat16 evaluation noise (acc 0.530 vs. 0.532; ppl
5.572 vs. 5.547); on MMMU and MMStar they are bit-equal at the answer-choice level. The smoke test at the token
level shows max |∆logits| = 0.375 (bfloat16 SDPA noise) and 100% argmax agreement on our calibration prompts.
B.6
Skip under use_cache=True: correctness verification
We verify that the K/V-only skip path described in §2.3 produces an argmax-consistent output with the cache-
less path. Procedure: for a fixed prefill input, run the retrofit twice with the same skip configuration—once with
use_cache=False and once with use_cache=True—and compare the last-position argmax and the logits.
We swept skip sets {[4], [4, 10], [4, 10, 12], [2, 4, 10]} on H_r256_5k on Qwen3-VL-2B. In every configuration the
last-position argmax matches between the two cache regimes, and the maximum logit delta is 0.19–0.37, identical to
the bfloat16 SDPA jitter observed on the stock base with no retrofit and no skip. Multi-step autoregressive divergence
starting around position 27–40 is an inherent property of HF + bf16 + SDPA that the stock base exhibits on the same
prompts (measured on “Once upon a time” and “def fibonacci”); it is not a property of our skip path.
B.7
γ=1 transition stability
Under the v3 mix with ramp-fraction 0.3 on 10k steps, Qwen3-VL-4B diverged at step ∼3,125 (training CE climbed
from 0.9 to 6.5+). The same configuration at 5k steps, and at 10k steps with the v3-VL-only mix (same ramp-fraction
but reduced reasoning-gradient density), converged stably. Extending the ramp to fraction 0.5 (ramp ends at step 5,000
rather than step 3,000) recovered convergence on 4B ×10k with no measurable effect on final quality: 4B ×10k at
ramp 0.5 is within 1pp of 4B ×5k at ramp 0.3 on every VLM benchmark, so the ramp-fraction difference does not
confound the between-cell comparison in Appendix E.6. Working rule: the γ=1 transition is stable whenever the
schedule reaches γ=1 at step ≥5,000, across every scale and partition we tested.
## C
## Retrofit ablations and inference machinery
C.1
Earlier design pilots (rejected)
Earlier pilots tried (i) observer-only (no forward change; α trained by distillation), (ii) interpolation at block-input
with β=σ(logit) driven toward 1, and (iii) pure AttnRes with informed init wn ←c ¯kn−1 + temperature annealing.
(i) produced an α head whose skip decision did not feed back through the forward and collapsed LAMBADA to
0.12 accuracy; (ii) pushed the frozen backbone off-distribution once β grew and required pretraining data to recover
14

(defeating the “light fine-tune” premise); (iii) was highly sensitive to the consecutive-layer key-similarity of the
pretrained backbone, and did not converge within our training budget. The γ-gated residual injection of §2.3 is the
converged recipe.
C.2
Adapter-rank ablation
Wall-clock latency is essentially unchanged across ranks (±2% at all measured sequence lengths): the adapter’s
2048→r→2048 bottleneck is a small contributor compared to the router stack/softmax/einsum cost. Rank therefore
affects quality but not speed. The canonical r=256 was selected from the γ→1 ablation of Appendix C.3; at equal step
budgets lower ranks leave more of the MMBench drop on the table without compensating latency savings.
C.3
γ→1 retrofit ablation
We swept 8 variants of the canonical γ→1 recipe along four axes: adapter rank r ∈{32, 64, 128, 256}, training steps
∈{5k, 10k, 20k}, LLaVA-Instruct fraction of the mix ∈{50%, 80%}, and γ ramp-fraction ∈{0.3, 0.7}. All other
hyperparameters (loss, optimizer, schedule family, seed) are held constant. A separate γ-free control is run for the same
number of steps with γn ≡0 trainable around its zero-init (Appendix Table 7).
Table 7: Retrofit ablation on Qwen3-VL-2B. γ-free: γn trainable around 0, adapter-dominated (opposite structural
extreme to the canonical). All others: γ-curriculum 0→1 over the first 30% of steps. “n=300 MMBench noise floor ±1
question” corresponds to ±0.33pp.
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
γ-free (A)
γ≈0 trainable
128
10k
50
—
0.576
0.512
0.720
Canonical
0→1 fast
256
5k
50
0.3
0.576
0.522
0.710
+ low rank
0→1 fast
64
10k
50
0.3
0.570
0.530
0.697
+ very low rank
0→1 fast
32
10k
50
0.3
0.568
0.526
0.683
+ low rank, VLM-heavy
0→1 fast
64
10k
80
0.3
0.560
0.524
0.660
+ mid rank, VLM-heavy
0→1 fast
128
10k
80
0.3
0.564
0.520
0.653
+ VLM-heavy
0→1 fast
256
10k
80
0.3
0.560
0.520
0.617
+ longer, 10k
0→1 fast
256
10k
50
0.3
0.586
—
0.587
+ slow ramp
0→1 slow
256
10k
50
0.7
0.568
0.520
0.517
+ longer, 20k
0→1 fast
256
20k
50
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
γ-free as the opposite extreme.
The γ-free control (row 2) achieves the same LAMBADA gain (+4.4pp) as the
canonical with slightly weaker HellaSwag (+0.6pp vs. +1.6pp) and slightly stronger MMBench (0.720 vs. 0.710). It
reaches these numbers by treating the adapter as a learnable correction on top of the original residual path—the model
still computes xn ≈hn−1 + adapter(·), never structurally ATTNRES. Because the paper’s claim is that pretrained
transformers can be rewired into the ATTNRES form, we take the γ→1 canonical as our headline and report this row as
a non-pure-ATTNRES control showing the same gain is achievable with the standard residual path left in place.
15

C.4
Per-block skip importance
Running the 5k Qwen3-VL-2B retrofit (H_r256_5k) with a single block statically removed (hn ←xn) and measuring
LAMBADA at n=300: block 1 is catastrophic (−55% acc), block 10 severe (−46%), blocks 4, 6, 11 are safest (−11
to −14%). We use this sweep to pick P={4, 6, 11} for the dynamic-skip eligibility set. The same ranking structurally
reproduces across the 340M from-scratch (§2.2) and the 2B retrofit: the block nearest the embedding and the one
late-layer block that concentrates residual-refinement traffic are the worst to remove, while the three mid-depth positions
with collapsed-onto-predecessor α are the safest.
C.5
Calibration set (dynamic skip)
Calibration uses 32 held-out LAMBADA prefixes (truncated to 512 tokens each). Per-block wrecent(n) samples are
collected with a single retrofit forward and the empirical quantile at q is used as τn. We verified that varying the
calibration draw shifts τn by < 0.01 in absolute terms at the chosen q=0.95. For VLA deployment, the identical calibra-
tion pipeline applied to the VLA in-backbone forward (retrofit/eval/calibrate_vla_thresholds.py)
produces thresholds byte-identical to the LAMBADA-calibrated set on matched inputs, confirming that the VLA and
VLM forwards instantiate the same router.
C.6
Data-mix ablation (v1 →v2 →v3)
Table 8 collects the data-mix ablation that motivates the v3 canonical. Three intermediate cells show how the
mix landscape partitions: v1 (narrow VL: UltraChat + LLaVA-Instruct-VSFT) is the initial canonical and pre-
serves base within noise at 5k steps but collapses when trained longer; v2 (aggressive math-CoT: 40% Numina-
Math/OpenThoughts/OpenMath2 +30% VL) recovers MMStar reasoning subtasks on 2B but crashes AI2D by 45pp
(on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive
on VL at both scales.
Table 8: Data-mix ablation. Same γ→1 retrofit recipe (r=256, L=2, ramp-fraction per-scale) varying only the data
mix and step count. v1 at 5k is preserved, but at 10k it collapses on 4B (AI2D −23pp). v2 crashes diagram-reasoning
on 2B despite raising MMStar math subtasks. v3 at 10k is net-positive on every VL benchmark at 2B and matches base
at 4B (with the L=4 partition of Appendix E.6, 4B × v3 is strictly positive). All numbers from lmms-eval full splits.
Scale
Mix (steps)
AI2D
MMBench
MMMU
MMStar
OCRBench
RealWorldQA
2B
base
0.736
75.77
0.414
0.536
0.772
0.648
2B
v1 (5k)
0.684
72.85
0.406
0.386
0.795
0.447
2B
v1 (10k)
0.677
73.71
0.404
0.471
0.801
0.642
2B
v2 (5k)
0.283
73.80
0.421
0.422
0.806
0.512
2B
v3 (10k)
0.765
79.30
0.439
0.534
0.809
0.668
4B
base
0.819
83.33
0.490
0.624
0.819
0.715
4B
v1 (5k)
0.810
83.76
0.510
0.579
0.812
0.707
4B
v1 (10k)
0.580
81.79
0.432
0.437
0.812
0.689
4B
v2 (5k)
0.603
81.87
0.477
0.333
0.808
0.686
4B
v3 (10k, L=2)
0.816
84.28
0.523
0.587
0.813
0.708
The table crystallises three working rules for the retrofit’s data mix. (i) Anchor VL at ≥60%. Dropping VL below
30% (v2) catastrophically breaks diagram reasoning. Holding VL at 50% (v1) works at 5k but not at 10k. Holding VL
at 60% with a rich OneVision anchor (v3) works at 10k at both scales. (ii) Cap math-CoT text at ≤20%. v2’s 40%
math-CoT share on 2B dropped AI2D to 0.283—the retrofit’s router learnt to route away from vision-heavy paths when
the text side of the mix presented a strong symbolic-reasoning gradient. v3 keeps math-CoT at 20% and recovers AI2D
to 0.765. (iii) More steps does not rescue a narrow mix. v1 at 10k crashes AI2D by 23pp on 4B while barely moving
2B, disproving the hypothesis that the v1→v2 transition was just under-training. The mix quality is binding.
C.7
Two-phase forward with dynamic skip (algorithm)
16

Algorithm 1 Two-phase ATTNRES forward with dynamic RESKIP.
1: Input: embeddings h0; block functions f1, . . . , fN; pseudo-queries {wn}; eligible P; thresholds {τn}; max skips
Mmax.
2: Initialise online-softmax state (s, m, e) ←INIT(h0); skip counter k ←0.
3: for n = 1 to N do
4:
Compute α·→n over completed block outputs (phase 1).
5:
wrecent(n) ←Etoken[αn−1→n].
6:
if n ∈P and k < Mmax and wrecent(n) > τn then
7:
Skip: set hn ←hn−1; under use_cache=True, run K/V-only slice on each constituent layer; k ←k + 1.
8:
else
9:
Execute block n on the merged input; update (s, m, e) with hn.
10:
end if
11: end for
12: Output: final hidden state s/e.
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
The 340M from-scratch ATTNRES (§2.2, Table 1) is the load-bearing existence proof. A 110M cross-scale check on the
same FineWeb-Edu recipe with an 8-block partition reproduces the zero-degradation pattern at a safer operating point
(q=0.97, Mmax=1 vs. 340M’s q=0.85, Mmax=2); skip trigger rate is lower at 110M, consistent with larger models
carrying more redundant computation. 1.3B / 2B from-scratch are not load-bearing because the retrofit (§3) directly
targets a pretrained 2.13B model, answering the 2B-scale question without paying the pretrain bill.
D.3
RESKIP method comparison and full Pareto / latency at 340M
Table 9: RESKIP versus representative adaptive-computation methods.