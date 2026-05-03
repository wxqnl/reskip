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
1

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