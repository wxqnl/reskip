## Introduction
Different inputs should not always spend the same computation. The dual-process distinction [18] contrasts System 1
(fast, automatic) with System 2 (slow, deliberate); the same dichotomy has direct correlates in cortical processing,
where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper
circuits [25, 19, 21, 42]. Both perspectives suggest the same computational target: the agent decides its own depth, per
input. We use this analogy only as a design motivation, not a cognitive-faithfulness claim. The technical question is
whether depth allocation can be exposed as an intrinsic signal of the transformer forward pass, rather than delegated to
a separate router or exit head.
Large language models already expose one System-2-like axis: tokens. Test-time compute scaling [34, 8, 41],
chain-of-thought prompting [45], and self-thought methods [48] spend more compute on hard problems by emitting
more reasoning tokens before answering. Yet every token in such a trace still traverses every layer of the underlying
transformer, regardless of whether that token is a routine connector or a load-bearing inferential step. The missing
complementary axis is therefore depth: deciding which blocks to invoke for this input, at this point in the sequence.
Existing depth-axis methods do not quite match this motivation because they externalise the decision. Early-exit
classifiers [40, 12] add auxiliary heads per layer; Mixture-of-Depths [39] learns a token router with a capacity-balancing
1

loss; layer-pruning [16, 32] is post-hoc and static; Universal Transformers [9] share weights with ACT-style halting [15].
In each case a separate component decides whether the rest of the network should run. Cognitive and neural System-2
control, in contrast, is not a separate module; it is a property of the substrate.
ATTNRES [24] gives that intrinsic handle. It replaces the fixed scalar-1 residual with a learned softmax-attention
over previous block outputs; the routing weights αi→n are input-dependent, competitively normalised, and computed
before block n runs, as part of forming its input. They therefore emit a per-block depth-allocation signal as a side-effect
of the network’s own forward, not as a separate head. The practical obstacle is that every existing ATTNRES model
is pretrained from scratch with the modified residual; reproducing this at the 2–7B scale that real applications need
costs tens of thousands of GPU-hours (Appendix D.1). The community has already invested in strong standard-residual
transformers—Qwen3-VL, LLaVA-OneVision, InternVL—that no group will retrain just to acquire a depth-axis routing
signal. The central problem in this paper is how to install such a signal into a standard pretrained transformer through a
short fine-tune, while preserving its original capabilities and making the signal usable for skipping.
The paper makes one main contribution and two supporting ones.
• AR-RETROFIT (§2.3, main). A γ-gated residual injection installs ATTNRES into a frozen pretrained transformer
through a single short fine-tune (∼0.7% new parameters, identity-at-init, tens of GPU-minutes per billion), with
every block converging to γn=1. On Qwen3-VL-2B/4B the retrofit improves the base on 5/6 VLM benchmarks
at both scales, at iso-cost compiled inference (1.029× base_compiled, §3).
• Routing weights as a usable depth signal (§2.2). Once installed, the pre-execution ATTNRES weights drive
a per-input layer-skip rule (RESKIP) with no auxiliary head. The signal is necessary but not sufficient—we
additionally require an offline ablation safety check—and at a rate-matched comparison the input-dependent rule
strictly beats static and random skip schedules at the same skip rate (Appendix D.5).
• Transfer to embodied control (§4). A LIBERO VLA warm-started from the retrofit improves a matched
pure-OFT baseline at both 2B and 4B; RESKIP on the action stream, with thresholds re-calibrated on the action
distribution, further improves the trained policy at the conservative operating point.
## 2