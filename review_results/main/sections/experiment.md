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
12

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
13

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
14

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
15

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