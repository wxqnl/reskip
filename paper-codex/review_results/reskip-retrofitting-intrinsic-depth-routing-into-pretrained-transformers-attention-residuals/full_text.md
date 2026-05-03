article
neurips_2026
inputenc
fontenc
hyperref
url
booktabs
amsfonts
amsmath
amssymb
nicefrac
microtype
graphicx
xcolor
algorithm
algorithmic
subcaption
multirow
wrapfig
\@undefined
figures/./outputs/standard/analysis/./outputs/vla/
AttnRes
ReSkip
ReLoop
AR-Retrofit
Route-A
Route-B
Route-C
E
R
softmax
[1]red[TODO: #1]
ReSkip: Retrofitting Intrinsic Depth Routing into Pretrained Transformers\ Attention Residuals
 Anonymous Author(s) \\
 Affiliation withheld \\
 anonymous@submission.invalid
document
abstract
Pretrained transformers normally execute all layers for every token, even though different inputs may require different effective depth. We study whether a standard pretrained transformer can be equipped with an intrinsic depth-routing pathway without retraining from scratch. We build on Attention Residuals, whose routing weights over previous block outputs are computed before the next block executes, and introduce , a -gated residual-injection fine-tune that installs this pathway while freezing the base model and adding well under new parameters. The retrofit is identity-preserving at initialization, trains in a short SFT-style run, and converges to a model whose block inputs are routed by rather than by a fixed residual path. On Qwen3-VL-2B and 4B, improves the pretrained backbone on most lmms-eval benchmarks and produces gains that parameter-matched LoRA controls do not reproduce. We then use the learned routing weights as a calibrated dynamic-depth signal: validates the signal on a 340M from-scratch model, supports conservative skipping on the retrofitted VLM, and transfers to LIBERO policies through a VLA warm-start. The resulting system should be read as a practical retrofit mechanism for intrinsic depth routing, with adaptive skipping and VLA transfer as supporting evidence rather than as universal claims about optimal inference.
abstract
## Introduction
sec:intro
Different inputs should not always spend the same computation. This is a design motivation rather than a claim of cognitive equivalence. Kahneman's dual-process framework~kahneman2011thinking distinguishes fast and deliberate processing, and biological vision offers a concrete computational analogue: easy stimuli can be recognised through shallow feedforward sweeps, while harder stimuli recruit recurrent and deeper circuits~lamme2000distinct, kar2019recurrent, dicarlo2012does. Recurrent vision models capture human representational dynamics~kietzmann2019recurrence, and confidence-thresholded recurrence spends more steps on harder images to trade speed for accuracy~spoerer2020recurrent. Since finite recurrence can be unrolled into feedforward depth~vanbergen2020going, these observations motivate a transformer-side question: can a pretrained model acquire an intrinsic signal for allocating effective depth per input?
Large language models already expose one test-time compute axis: tokens. Test-time compute scaling~openai2024o1, deepseek2025r1, snell2024scaling, chain-of-thought prompting~wei2022chain, and self-thought methods~zelikman2024quietstar spend more compute on hard problems by emitting more reasoning tokens before answering. Yet every token in such a trace still traverses every layer of the underlying transformer, regardless of whether that token is a routine connector or a load-bearing inferential step. The missing complementary axis is therefore depth: deciding which blocks to invoke for this input, at this point in the sequence.
Existing depth-axis methods do not quite match this motivation because they externalise the decision. Early-exit classifiers~schuster2022confident, elbayad2020depth add auxiliary heads per layer; Mixture-of-Depths~raposo2024mixture learns a token router with a capacity-balancing loss; layer-pruning~gromov2024unreasonable, men2024shortgpt is post-hoc and static; Universal Transformers~dehghani2018universal share weights with ACT-style halting~graves2016adaptive. In each case a separate component decides whether the rest of the network should run. The biological analogue motivating this paper is different: control is embedded in the processing substrate rather than attached as a separate decision head.
~chen2026attnres gives a natural handle. It replaces the fixed scalar- residual with a learned softmax-attention over previous block outputs; the routing weights are input-dependent, competitively normalised, and computed before block runs, as part of forming its input. They therefore emit a per-block depth-allocation signal as a side-effect of the network's own forward, not as a separate head. The practical obstacle is that existing models are trained from scratch with the modified residual; reproducing this at the 2--7B scale that real applications need costs tens of thousands of GPU-hours (Appendix~app:motivation). The central problem in this paper is how to install such a signal into standard pretrained transformers through a short fine-tune, while preserving their original capabilities and making the signal usable for calibrated depth skipping.
The paper has one main contribution and two supporting claims.
itemize
 is the main method (:retrofit). A -gated residual injection installs into a frozen pretrained transformer through a short fine-tune, using new parameters and an identity-preserving initialization.
 The installed routing signal is usable (:reskip, :exp_retrofit). treats weights as a calibrated depth signal. The key method insight is that routing weight alone is not a safety certificate; eligible positions must be selected by ablation or drift.
 The retrofit transfers beyond VLM evaluation (:vla). A LIBERO policy warm-started from the retrofit improves the matched OFT baseline in our evaluated operating point, and the same routing signal supports conservative action-stream skipping after calibration.
itemize
## Method
sec:method
### Background: Attention Residuals
sec:background
A standard transformer uses fixed scalar- combination, , emitting no routing signal. ~chen2026attnres replaces this with a learned attention over depth. Each block maintains a pseudo-query ; keys are projections of earlier block outputs , and
The key property is two-phase execution. Computing the input to block requires only and ---both available before runs. We call the pre-execution attention computation phase~1 and itself phase~2. Phase~1 is the routing signal we exploit. Block- groups layers into blocks and applies the softmax at the block level (the granularity used throughout). Algorithm~alg:two_phase (Appendix~app:algorithm) gives the full two-phase forward with 's skip check folded in.
### 
: AttnRes-guided layer skipping
sec:reskip
 turns the phase-1 signal into a per-input skip rule. Each block~ has two offline scores: an importance (how often block is referenced; forward passes only) and an ablation impact n (is block irreplaceable; ablated forwards). At inference, for each we read the mean weight that position 's router places on the immediate predecessor and apply
with calibrated per position as the -th empirical quantile of on held-out long-context sequences ( default). The skip decision is a scalar comparison on a quantity already computed in phase~1; no auxiliary network, no train-time changes, no architectural modification (cf.\ CALM / MoD / pruning / LayerSkip in Appendix~app:method_comparison).
 and are both needed because depth-attention is a routing signal, not a complete safety certificate. On our 340M from-scratch (Figure~fig:reskip_routing), block~ has middle-of-the-pack importance () yet is catastrophic to remove ( PPL), while block~ has the lowest ablation impact yet higher than block~. should therefore use both signals---low so skip triggers often, low so it is safe. Picking by either alone is strictly worse: alone (low-) gives ; combined gives at PPL slightly below full depth. Full position sweep in Appendix~app:reskip_position_ablation.
This rule first asks whether contains a usable intrinsic depth signal at all. On a 340M block- transformer (, , blocks; FineWeb-Edu 100BT~penedo2024fineweb; lm-eval-harness~eval-harness), at reaches wall-clock at zero benchmark drop on LAMBADA~paperno2016lambada, HellaSwag~zellers2019hellaswag, and ARC-E/C~clark2018arc (Table~tab:reskip_benchmark; Pareto and latency curves in Appendix~app:reskip_results). Skip count varies -- per forward (mean over batches), confirming genuine input-dependence rather than a fixed-block removal. A 110M cross-scale replication on the same FineWeb-Edu recipe reproduces the zero-degradation pattern at (Appendix~app:phase1). This validates the mechanism when is trained in. The harder question---how to acquire it in an already-trained model---is :retrofit.
### 
: installing AttnRes into a pretrained transformer
sec:retrofit
The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an -capable model that (P1) preserves benchmark quality, (P2) emits a routing informative enough to drive , and (P3) trains in a single short fine-tune over off-the-shelf SFT data, with the base frozen. To satisfy the first constraint, the modification must be exactly identity-preserving at step ; to satisfy the second, the routing must feed the real forward path, not merely observe it.
We organise the decoder layers into blocks of layers each (Qwen3-VL-2B: ). For every block we add an router , a tiny adapter (down-up bottleneck of rank , SiLU), and a scalar gate :
 is the pretrained block, verbatim; enters between blocks, as a correction. The retrofit parameters amount to M ( of base) at on 2B; the entire base---embeddings, vision tower, decoder layers, LM head---is frozen.
With we have exactly, so the forward at step is bit-identical to the pretrained model. The adapter up-projection is small-random () so that at init, avoiding gradient deadlock. We then ramp via a linear curriculum over the first -- of training; every block converges to , leaving the retrofit structurally pure (every block consumes the routed sum, not the standard residual). Three cleaner-looking alternatives we tried first---an observer-only head, an interpolation at the block input, and an informed-init pseudo-query with temperature anneal---all fail (Appendix~app:route_ablations).
 uses the same path rather than a new mechanism. Its phase-1 decision reads the already-computed and, when the rule triggers, short-circuits the block: . To make this consistent with use\_cache=True, on a skipped block we still run the per-layer K/V-producing slice (), which costs -- of a full-layer forward, and skip q\_proj / attention / o\_proj / MLP. The combined skip\,+\,use\_cache=True path matches the cacheless path in last-position argmax (Appendix~app:kv_equiv).
The training objective follows from the three constraints above. We minimise
where the full-path CE is computed on assistant tokens, while the skip-branch KL samples one block at random per step, runs the forward with it skipped, and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate is useful both when the block runs and when it is skipped). is implemented with a negative sign in the loss, i.e., a light entropy-maximising prior against premature collapse of . Default , entropy weight , lr on retrofit params, cosine schedule with -step warmup. Hyperparameters and the v1 / v3 data mixes used throughout are listed in Appendix~app:retrofit_hparams.
This is not ordinary SFT. The CE term keeps the full path useful for the downstream task, but by itself it does not train the skipped surrogate to replace a real block. The skip-branch KL supplies that constraint by tying one skipped-block forward per step to the frozen teacher. Conversely, teacher imitation alone would merely reproduce the base model. The weak entropy prior prevents early one-source collapse while still allowing the router to sharpen later. This three-term structure is what makes the retrofit identity-preserving at step , task-improving after training, and skip-ready at inference.
## Experiments: 
 on Qwen3-VL-2B and 4B
sec:experiments
sec:exp_retrofit
The first empirical question is whether the retrofit solves the installation problem without merely trading away the pretrained model's capabilities. We retrofit Qwen3-VL-2B~bai2025qwen3vl () at blocks of and Qwen3-VL-4B () at blocks of (block-partition sweep in Appendix~app:block_partition). is the canonical partition for both scales: it sits on the accuracy plateau alongside and halves the per-token router count, which is the dominant factor in the inference-cost result below. All base parameters are frozen; M (B) / M (B) retrofit parameters at train via Eq.~eq:retrofit_loss. We compare two data mixes: v1 (/ UltraChat-k~ding2023ultrachat + LLaVA-Instruct-VSFT~liu2023llava, k steps; GPU-min on B), used as a LoRA-comparable narrow-mix control; and v3 ( LLaVA-OneVision-Data~li2024llavaonevision + UltraChat + NuminaMath + OpenThoughts, k steps), the canonical mix used for every result we report. Evaluation uses LAMBADA~paperno2016lambada/HellaSwag~zellers2019hellaswag for text and six lmms-eval benchmarks for VLM (MMBench~liu2023mmbench, MMMU~yue2024mmmu, MMStar~chen2024mmstar, AI2D~kembhavi2016ai2d, OCRBench~liu2024ocrbench, RealWorldQA~grok2024realworldqa); text-side eligible sets are selected from per-block static removal, while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~app:block_removal,\,app:vla_pareto). Identity-at-init is verified before training ( reproduces base; max bf16, argmax agreement; Appendix~app:identity).
The canonical retrofit answers this first question positively (Tab.~tab:retrofit_main). It is strictly above base on five of six VLM benchmarks at B and at B, with the largest gain on the MMStar math subcategory (pp at B, pp at B; full -cell decomposition in Appendix~app:mmstar_subcat). The win concentrates on deliberate reasoning over images rather than perception, consistent with a router that lifts deeper reasoning paths. Text quality also lifts: pp LAMBADA / ppl on B and pp / on B; HellaSwag is within pp at both scales. Every block converges to , so the retrofit is structurally pure .
The second question is attribution: are these gains caused by the pathway or simply by adding a small trainable module? Parameter-matched LoRA baselines on the same data produce pp LAMBADA over four runs (pp behind the retrofit, Appendix~app:lora_baselines). LoRA at the same parameter scale and twice the narrow-mix step count averages LAMBADA versus for the frozen base and for the matching v1 run. Observer-only routing learns an signal that never feeds back into the forward, and -free / interpolation variants destabilise the frozen backbone before the router stabilises. The converged recipe therefore depends on the identity-preserving bridge, the real forward-path signal, and the skip-aware surrogate loss together.
The third question is whether the installed signal is usable for depth allocation. With per-scale eligible sets (B: ; B: ) and per-block thresholds calibrated as the -th quantile of on a -prefix held-out set, dyn-skip preserves LAMBADA at the conservative end ( holds within pp of no-skip; full sweep in Appendix~app:retrofit_pareto). The same Pareto shape reappears on action prediction (:vla): at is the conservative operating point on the LIBERO action stream. We use this as a calibration test of the installed signal, not as evidence that alone certifies safe removal.
Finally, the installation problem is only useful if it does not make the forward path impractically expensive. Two architectural choices put the accuracy results on the same fast path. First, the canonical partition halves the per-token router count vs , closing of the eager-vs-eager gap on its own (, seq~, cache, H100/bf16; Tab.~tab:retrofit_latency). Second, wrapping both base and retrofit in torch.compile (max-autotune, dynamic=False) fuses the small router gemms. The retrofit measures base\_compiled on the same shape and compile setting, while compile preserves accuracy ( argmax agreement on LAMBADA-, logit RMSE ; Appendix~app:compile_accuracy). Eager-mode contributes a further -- standalone (Appendix~app:retrofit_pareto); the two paths cannot currently be composed because dyn-skip breaks the captured CUDA graph. The systems result is therefore a two-mode characterization: compiled full-depth retrofit for near-matched-cost accuracy, and eager calibrated skipping for additional dynamic-depth savings.
 and the v3 data mix are not arbitrary engineering choices; they are the settings that make the preceding claims simultaneously true. sits on the accuracy plateau alongside (within pp on every VLM benchmark) and halves the per-token router count, which is what makes the compiled-overhead result possible. Per-layer underperforms by --pp; the plateau reproduces chen2026attnres's pretrain observation. The LLaVA-OneVision-anchored v3 mix is canonical at both scales; v1 and v2 expose a binding ``mix-quality'' regime---v1 collapses VLM at k, v2 crashes diagram-reasoning at k. Full block-partition sweep, v1v2v3 trail, and -free / informed-init / observer-only baselines are in Appendices~app:block_partition,\,app:data_mix,\,app:retrofit_ablation. The retrofit is still not free in eager mode: its router stack adds -- at . The deployment claim is narrower: after both base and retrofit are compiled under the same setting, the fixed-shape overhead falls to . Dynamic remains a separate eager-mode latency lever because thresholding currently graph-breaks torch.compile.
## VLA Transfer Evidence
sec:vla
A VLA test isolates whether the retrofit transfers to a downstream control task and whether its warm-start is substitutable by simply running the -curriculum on VLA data. Vision tokens (perceptual, early-layer~raghu2021vision), language tokens (task spec.), and action tokens (motor planning) have no principled reason to share effective depth; routing gives a per-position handle on this without token-level gating machinery.
We attach the OpenVLA-OFT~kim2025openvlaoft action head ( regression, no chunking) to Qwen3-VL-2B/4B backbones (ViT frozen) and train on the pooled libero\_all mix from LIBERO~liu2023libero. The experiment is designed around two questions: whether the VLM retrofit is a useful warm-start for control, and whether the same architecture can catch up by learning only on VLA data. Three paths share the same head and data: Path 0 (stock backbone OFT, baseline), Path B (warm-start from our canonical v3 k VLM retrofit, throughout), and Path C (same architecture as Path B but no VLM retrofit; routers / adapters random-init, ramps on VLA data). H100 ZeRO-2, batch effective, k-step default with a k overtraining cell. Each policy is evaluated for rollouts per suite ( per policy); repeated-rollout sensitivity is reported in Appendix~app:vla_variance.
### LIBERO results
sec:vla_results
Path B answers the warm-start question positively. It reaches on B and on the clean-base B cell. The gain concentrates on Spatial (pp at B) and Long- (pp at B), which are the suites where action prediction benefits most from a stable high-level backbone. We use these results as downstream transfer evidence for , not as a broad robotics claim.
Path C answers the substitution question negatively. It plateaus at at k, pp behind Path B; at k it closes only some of the gap (, still pp behind Path B k). A k total budget (k VLM retrofit + k VLA) outperforms k of VLA-only training on the suites where a stable pretrained router most matters: Spatial task goes from (Path C k) and (Path C k) to (Path B k). Longer training is not uniformly helpful either: Path B k drops pp on B and pp on B even as Path 0 on B keeps improving, so we use Path B at k as the VLA operating point.
### 
 on the action stream: calibrated conservative skipping
sec:vla_reskip
We now apply the same rule at inference on the trained Path B policies, with thresholds calibrated on a sim-rollout distribution rather than language tokens (the rationale is in the next paragraph). Eligible blocks are picked per scale from a per-block action-drift sweep (Appendix~app:vla_pareto: on B, on B) and throughout. Tab.~tab:vla_reskip_pareto shows the -suite Pareto across .
Thresholds are calibrated on the action distribution rather than transferred from language: a naive use of LAMBADA-calibrated thresholds at on the same policy collapses LIBERO-Spatial to (Appendix~app:vla_pareto). The reason is that is not itself a skip rate; it indexes a sharp empirical distribution of . The recommended VLA point is therefore conservative (): it skips rarely, but only on blocks whose action-drift sweep says they are locally safe. The per-block action-drift sweep that picks per scale, the cross-modality transfer failure, the per-seed breakdown, the pipeline pitfall, and a public-VLA landscape table ( / OpenVLA / SpatialVLA on LIBERO) live in Appendices~app:vla_variance,\,app:vla_pitfall,\,app:vla_landscape,\,app:vla_pareto.
## Related Work
sec:related
For test-time compute, o1~openai2024o1, DeepSeek-R1~deepseek2025r1, chain-of-thought~wei2022chain, and optimal test-time compute analyses~snell2024scaling allocate compute by emitting more reasoning tokens; per-token depth remains fixed. Depth-axis methods include ACT / Universal Transformers~graves2016adaptive, dehghani2018universal, CALM~schuster2022confident, Mixture-of-Depths~raposo2024mixture, LayerSkip~elhoushi2024layerskip, and static pruning~gromov2024unreasonable, men2024shortgpt. These add a halting head, exit classifier, router, speculative decoder, or static removal rule. We instead use as an intrinsic routing signal produced by the forward pass itself, then retrofit that signal into an existing pretrained backbone. Appendix~app:b1b2_decision_rule compares dynamic, static, and random skip rules at 340M, and Appendix~app:gromov_pruning gives a full-layer static-pruning baseline at the VLM scale.
The neural motivation is recurrence as flexible effective depth. The dual-process distinction~kahneman2011thinking maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent circuits~lamme2000distinct, kar2019recurrent, dicarlo2012does. Recurrent networks capture human representational dynamics~kietzmann2019recurrence; spoerer2020recurrent are the closest conceptual antecedent, showing that recurrent CNNs can spend more steps on harder images to explain speed--accuracy behaviour. Recurrence also has a depth interpretation: finite recurrence can be unrolled into feedforward depth, but the recurrent form reuses a fixed substrate for flexible effective depth~vanbergen2020going. We take this as a design target, not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~bastos2012canonical, buschman2007topdown, larkum2013cellular motivate intrinsic routing, while our implementation is a transformer retrofit.
In residual architectures, DenseNet~huang2017densely and Highway networks~srivastava2015highway modify residual flow; ~chen2026attnres generalises residual combination with depth attention. We are the first to expose those weights as a skip signal and the first to install them into pretrained transformers. In VLA, RT-2~brohan2023rt, Octo~team2024octo, OpenVLA~kim2024openvla, and Pi0~black2024pi0 established the paradigm, while efficiency work has focused mainly on action chunking~zhao2023learning and post-hoc compression. We contribute modality-aware adaptive depth for the VLM backbone.
## Discussion
sec:discussion
 is the main claim: a standard pretrained transformer can be given an intrinsic depth-routing pathway through a short, identity-preserving fine-tune rather than from-scratch pretraining. At the canonical v3 recipe it improves / VLM benchmarks at both Qwen3-VL scales, lifts LAMBADA by /pp, and concentrates the largest gain on MMStar math. and LIBERO then test the usefulness of the installed signal: from-scratch reaches wall-clock at zero benchmark drop, the retrofitted VLM supports conservative text/action skipping after calibration, and the VLA warm-start improves the matched OFT operating point. The boundary is calibration. We do not claim that the current policy is universally optimal, nor that depth attention alone certifies safe removal; eligible sets must be selected by ablation or action-drift and thresholds must be calibrated on the decoded modality. We also separate compiled full-depth inference from eager dynamic skipping, since current CUDA graph capture does not support the dynamic branch. The central result is that a pretrained model already capable of token-axis test-time scaling can acquire a second, depth-axis allocation mechanism without retraining from scratch.
references
## Limitations and future work
app:limitations
Limitations.
(i)~Two retrofit targets from one VLM family; extension to InternVL / Gemma-VL / text-only LMs is planned. (ii)~The compile-on overhead result requires torch.compile; eager-mode retrofit alone is at , while eager contributes a further -- but cannot currently be composed in a single forward with compile (dyn-skip's .item() guard breaks the captured CUDA graph). Deployments pick one mode per latency budget; a Triton-fused router that lets the two co-exist is open future work. (iii)~Wall-clock is H100 / bf16; lower-end GPU / edge measurement is not yet done. (iv)~The VLA Pareto thresholds are sim-calibrated per scale, not per modality within a single rollout; per-modality, per-token-class calibration is the next sweep. (v)~Block-level granularity is inherited from ; per-token routing is future work.
Future work: weight-shared AttnRes ().
A natural extension shares block weights across depth and differentiates each application via position-indexed pseudo-queries; the routing weight then doubles as the halt signal, unifying (``low skip block'') and a halt rule (``low mean-recent halt iteration''). A 74M validation produced a smooth depth--quality Pareto ( compute saved at ppl change); scaling and combining with the retrofit framework is open.
## Implementation details
app:implementation
### Online softmax merge with skip
When a block is skipped, the online softmax merge simply does not process its output. The running state is unchanged. The output is mathematically equivalent to an model trained without that block contributing---modulo the calibration of downstream .
### Pseudo-query initialisation (Section~sec:reskip
 from-scratch)
For from-scratch training we use , yielding near-uniform at initialisation that specialises over the first few thousand steps.
### Retrofit hyperparameters
app:retrofit_hparams
AdamW with , weight decay~, gradient clip~, bfloat16. Learning rate for retrofit params (routers, adapters, ), cosine schedule with -step warmup. Sequence length~ (training), (evaluation). Loss weights , entropy weight , KD temperature~. The entropy term is subtracted from the loss, so it maximises routing entropy early in training rather than minimising it. Skip-branch sampling: one eligible block chosen uniformly at random per step. Adapter bottleneck rank (canonical); position-bias on router keys enabled; router temperature fixed at~. -curriculum: linearly over the first of total steps (2B k/k, 4B k), with ramp-fraction on 4B k to stabilise a late-stage transition divergence (Appendix~app:gamma_stability). Total training wall-clock on a single H100: min for 2B k, min for 2B k, min for 4B k.
### Block granularity
app:block_granularity
For the final recipe we use layers per block on both scales: Qwen3-VL-2B has blocks and Qwen3-VL-4B has blocks. Earlier runs remain in the appendix as ablation and legacy controls, but is the canonical setting because it preserves the accuracy plateau while halving router calls relative to . Each block receives one router, one residual adapter ( default), and a scalar gate.
### Identity-at-init details
app:identity
With and the adapter up-projection initialised at , the retrofit forward is arithmetically identical to the pretrained model (both produce , which is passed into the unmodified block). We verified end-to-end: at on LAMBADA the retrofit and the base agree to within bfloat16 evaluation noise (acc vs.\ ; ppl vs.\ ); on MMMU and MMStar they are bit-equal at the answer-choice level. The smoke test at the token level shows max (bfloat16 SDPA noise) and argmax agreement on our calibration prompts.
### Skip under use\_cache=True
: correctness verificationSkip under use cache: correctness verification
app:kv_equiv
We verify that the K/V-only skip path described in :retrofit produces an argmax-consistent output with the cacheless path. Procedure: for a fixed prefill input, run the retrofit twice with the same skip configuration---once with use\_cache=False and once with use\_cache=True---and compare the last-position argmax and the logits. We swept skip sets , , , on H\_r256\_5k on Qwen3-VL-2B. In every configuration the last-position argmax matches between the two cache regimes, and the maximum logit delta is --, identical to the bfloat16 SDPA jitter observed on the stock base with no retrofit and no skip. Multi-step autoregressive divergence starting around position -- is an inherent property of HF bf16 SDPA that the stock base exhibits on the same prompts (measured on ``Once upon a time'' and ``def fibonacci''); it is not a property of our skip path.
### transition stability
gamma=1 transition stability
app:gamma_stability
Under the v3 mix with ramp-fraction on k steps, Qwen3-VL-4B diverged at step (training CE climbed from to ). The same configuration at k steps, and at k steps with the v3-VL-only mix (same ramp-fraction but reduced reasoning-gradient density), converged stably. Extending the ramp to fraction (ramp ends at step rather than step ) recovered convergence on B k with no measurable effect on final quality: B k at ramp is within pp of B k at ramp on every VLM benchmark, so the ramp-fraction difference does not confound the between-cell comparison in Appendix~app:block_partition. Working rule: the transition is stable whenever the schedule reaches at step , across every scale and partition we tested.
## Retrofit ablations and inference machinery
app:ablations
### Earlier design pilots (rejected)
app:route_ablations
Earlier pilots tried (i)~observer-only (no forward change; trained by distillation), (ii)~interpolation at block-input with driven toward~, and (iii)~pure AttnRes with informed init + temperature annealing. (i) produced an head whose skip decision did not feed back through the forward and collapsed LAMBADA to accuracy; (ii) pushed the frozen backbone off-distribution once grew and required pretraining data to recover (defeating the ``light fine-tune'' premise); (iii) was highly sensitive to the consecutive-layer key-similarity of the pretrained backbone, and did not converge within our training budget. The -gated residual injection of :retrofit is the converged recipe.
### Adapter-rank ablation
app:rank_ablation
Wall-clock latency is essentially unchanged across ranks ( at all measured sequence lengths): the adapter's bottleneck is a small contributor compared to the router stack/softmax/einsum cost. Rank therefore affects quality but not speed. The canonical was selected from the ablation of Appendix~app:retrofit_ablation; at equal step budgets lower ranks leave more of the MMBench drop on the table without compensating latency savings.
### retrofit ablation
gamma-to-1 retrofit ablation
app:retrofit_ablation
We swept variants of the canonical recipe along four axes: adapter rank , training steps , LLaVA-Instruct fraction of the mix , and ramp-fraction . All other hyperparameters (loss, optimizer, schedule family, seed) are held constant. A separate -free control is run for the same number of steps with trainable around its zero-init (Appendix Table~tab:retrofit_ablation).
Over-training signature at . Holding (, VLM, fast ramp) fixed and varying only the total step count, MMBench is strictly monotonic in steps: k (preserved), k (14pp), k (21pp). LAMBADA peaks at k () and regresses at k (), so longer training buys no text-domain gain either. The regime has a narrow compute sweet spot; past it the router+adapter over-specialise to the training distribution and the frozen backbone, unable to adapt further, loses its multimodal calibration.
Rank and data are subordinate. Halving rank at k recovers some MMBench (: , : ), but never reaches the k- level (). Pushing the mix to LLaVA worsens MMBench at every rank, indicating the regression is router over-adaptation to any narrow training distribution, not insufficient visual exposure.
-free as the opposite extreme. The -free control (row 2) achieves the same LAMBADA gain (pp) as the canonical with slightly weaker HellaSwag (pp vs.\ pp) and slightly stronger MMBench ( vs.\ ). It reaches these numbers by treating the adapter as a learnable correction on top of the original residual path---the model still computes , never structurally . Because the paper's claim is that pretrained transformers can be rewired into the form, we take the canonical as our headline and report this row as a non-pure- control showing the same gain is achievable with the standard residual path left in place.
### Per-block skip importance
app:block_removal
Running the k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed () and measuring LAMBADA at : block~ is catastrophic ( acc), block~ severe (), blocks~, , are safest ( to ). We use this sweep to pick for the dynamic-skip eligibility set. The same ranking structurally reproduces across the 340M from-scratch (:reskip) and the 2B retrofit: the block nearest the embedding and the one late-layer block that concentrates residual-refinement traffic are the worst to remove, while the three mid-depth positions with collapsed-onto-predecessor are the safest.
### Calibration set (dynamic skip)
app:calibration
Calibration uses 32 held-out LAMBADA prefixes (truncated to 512 tokens each). Per-block samples are collected with a single retrofit forward and the empirical quantile at is used as . We verified that varying the calibration draw shifts by in absolute terms at the chosen . For VLA deployment, the identical calibration pipeline applied to the VLA in-backbone forward (retrofit/eval/calibrate\_vla\_thresholds.py) produces thresholds byte-identical to the LAMBADA-calibrated set on matched inputs, confirming that the VLA and VLM forwards instantiate the same router.
### Data-mix ablation (v1 v2 v3)
Data-mix ablation (v1 to v2 to v3)
app:data_mix
Table~tab:data_mix_ablation collects the data-mix ablation that motivates the v3 canonical. Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at k steps but collapses when trained longer; v2 (aggressive math-CoT: NuminaMath/OpenThoughts/OpenMath2 VL) recovers MMStar reasoning subtasks on 2B but crashes AI2D by pp (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales.
The table crystallises three working rules for the retrofit's data mix.
(i)~Anchor VL at . Dropping VL below (v2) catastrophically breaks diagram reasoning. Holding VL at (v1) works at k but not at k. Holding VL at with a rich OneVision anchor (v3) works at k at both scales.
(ii)~Cap math-CoT text at . v2's math-CoT share on 2B dropped AI2D to ---the retrofit's router learnt to route away from vision-heavy paths when the text side of the mix presented a strong symbolic-reasoning gradient. v3 keeps math-CoT at and recovers AI2D to .
(iii)~More steps does not rescue a narrow mix. v1 at k crashes AI2D by pp on 4B while barely moving 2B, disproving the hypothesis that the v1v2 transition was just under-training. The mix quality is binding.
### Two-phase forward with dynamic skip (algorithm)
app:algorithm
## Phase 1: 340M from-scratch validation
app:phase1_section
### Motivation: 
 cost from-scratch is prohibitive
app:motivation
We measured forward-pass latency on H100 / bf16 / batch for two M models trained under identical FineWeb-Edu 100BT recipes: a vanilla standard-residual transformer and an transformer with blocks. Block- adds ( seq) to ( seq) wall-clock vs.\ vanilla. At the B VL scale a stock Qwen3-VL-2B forward already takes \,ms at seq on H100, so a -- block-level overhead would push past common \,Hz / \,Hz robotic control budgets. Pretraining a B -VLM costs at least what the matched standard-residual base costs (k--k H100-h reading published Qwen3-VL / InternVL3 / OpenVLA / LLaVA-OneVision tech reports), so the only practical access path is a retrofit on the already-trained standard model.
### Phase 1: cross-scale validation
app:phase1
The 340M from-scratch (:reskip, Table~tab:reskip_benchmark) is the load-bearing existence proof. A 110M cross-scale check on the same FineWeb-Edu recipe with an -block partition reproduces the zero-degradation pattern at a safer operating point ( vs.\ M's ); skip trigger rate is lower at 110M, consistent with larger models carrying more redundant computation. 1.3B / 2B from-scratch are not load-bearing because the retrofit (:exp_retrofit) directly targets a pretrained B model, answering the 2B-scale question without paying the pretrain bill.
### 
 method comparison and full Pareto / latency at 340M
app:method_comparison
### 
 position-set ablation at 340M
app:reskip_position_ablation
### Decision rule: dynamic vs.\ static-rate-matched vs.\ random
app:b1b2_decision_rule
The position-set ablation above answers which blocks to skip; this
subsection answers whether the input-dependent decision matters at
all. We hold the M from-scratch weights fixed, fix
 and (block-level skip rate
), and only vary the runtime decision rule:
itemize0pt
 B0. No-skip upper bound (full forward).
 B1.b/B2.a. Dynamic, fire when (our signal at block granularity).
 B1.c, B1.d. Static, every-token skip at or (rate-matched, ).
 B1.e/B2.d. Random, per-call uniform draw from keep-, keep- (rate-matched, ).
 B2.b. Dynamic, fire when block- entropy (router-confidence rule).
 B2.c. Dynamic, fire when (relative-recent rule).
itemize
All three thresholds are calibrated as the per-position quantile
on FineWeb-Edu tokens (:calibration). Results
in Tab.~tab:b1b2_decision_rule; observed skip rates on the eval
distribution in Tab.~tab:b1b2_observed_rate.
Conclusion. At a fair rate-matched comparison (
fired skips), input-dependent dynamic skip (Bc) preserves
LAMBADA at while every static or random alternative collapses
to -- ( to pp). The same ordering holds on
HellaSwag, PIQA, ARC-easy and OpenBookQA. The MoD-style ``you might be
getting a free lunch from any same-rate schedule'' critique is
rejected: at M from-scratch, schedule choice matters and the
 routing weights carry the load.
Threshold-transfer caveat. Two of the three dynamic rules
(Bb recent\_weight\_gt and Bb entropy\_lt)
fire far below the FineWeb-Edu calibration target on the
LAMBADA distribution (Tab.~tab:b1b2_observed_rate). Their
near-no-skip accuracy is therefore not a positive datapoint for those
specific rules; it is consistent with ``the threshold protected the
output by accidentally not firing.'' The load-bearing rate-matched
comparison is Bc recent\_minus\_embed\_gt (which does
fire close to target) versus the static/random alternatives. We treat
the two non-firing rows as a sensitivity result: 's safety
margin under threshold/distribution mismatch is high (no degradation
when the rule rarely fires), but deployment requires per-distribution
threshold calibration, as already noted for the cross-modality VLA
case (:vla_pareto).
### 
 full Pareto and latency curves at 340M
app:reskip_results
## Retrofit Pareto and breakdowns
app:retrofit_extras
### LoRA baselines
app:lora_baselines
We compare against parameter-matched LoRA baselines on the same UltraChat LLaVA mix at the canonical retrofit's step count. LoRA on : LAMBADA acc / (two seeds), HellaSwag / . LoRA on : LAMBADA , HellaSwag . LoRA on MLP: LAMBADA , HellaSwag . Mean LAMBADA across the four runs is , pp below base () and pp below our retrofit (). Mean HellaSwag is pp over base versus retrofit's pp. Attribution: the retrofit's gain comes from the routing structure, not from a few extra trainable parameters on SFT data.
### Static-pruning baseline at the VLM scale (Gromov drop-)
Static-pruning baseline at the VLM scale (Gromov drop-k)
app:gromov_pruning
The M decision-rule ablation (Tab.~tab:b1b2_decision_rule)
shows that single-position static skip and random skip both
collapse on language tasks. The complementary question at the VLM
scale is whether full-layer static pruning---the standard
``remove the least useful blocks'' baseline used by
gromov2024unreasonable---can match the accuracy floor that the
retrofit operates above. We rank the Qwen-VL-B layers by
the Gromov angular-distance criterion (computed on a
FineWeb-Edu calibration set with the LM-head only; no fine-tuning),
remove the lowest-distance layers, and re-evaluate on
lmms-eval. Tab.~tab:gromov_pruning_2b reports drop-
(layers ) and drop- (additionally
).
The pattern matches the in-block M observation: any
unconditional block removal (single-position static at
M, full-layer Gromov at B) destroys benchmark accuracy at
the VLM scale even at modest pruning fractions, while
input-dependent skip on top of an routing structure
preserves it. We do not attempt to retrain the pruned model; the
intent is a free static-baseline floor for the cost-quality
trade-off, not a tuned competitor.Pruned cells additionally
exhibit POPE generation drift below the strict yes/no surface
(drop- pope-acc , drop- via accidental
all-``yes'' decoding), so we drop POPE from the table.
### 
 on the canonical retrofit: text Pareto and cross-modality consistency
app:retrofit_pareto
On the canonical v3 k retrofit, applied to LAMBADA- as the language-modality cross-check of the LIBERO Pareto (Tab.~tab:vla_reskip_pareto), produces the same shape as on action: lossless at the conservative end, collapsing once moves below the action distribution's mean. Same , as the B VLA cell.
The legacy v1 (k, ) Pareto on the H\_r256\_5k retrofit lifted LAMBADA above its full-path value at ( vs.\ ), but that retrofit is no longer canonical for any reported result; we list the v1 Pareto in Tab.~tab:retrofit_pareto_v1_legacy for completeness.
### Wall-clock latency: full table including legacy and VLA in-backbone
Wall-clock latency: full table including legacy L=2 and VLA in-backbone
app:latency
Tab.~tab:retrofit_latency in the main body reports the canonical result. Tab.~tab:retrofit_latency_full below shows the full set: the legacy partition ( blocks at B) carries a structural cost over stock Qwen3-VL-2B because each token pays router calls, the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward, :vla) tracks the VLM retrofit within at both partitions, and torch.compile closes the residual to but is more dramatic on where the router count is halved. This table is the basis for the canonical-partition switch from to .
The structural per-block router stack (stackRMSNormsoftmaxeinsum over completed blocks) does not go away under cached decode: at it fires times per decoded token. Adapter rank ablation (Appendix~app:rank_ablation) confirms the adapter is not the bottleneck (); the router itself, when uncompiled, is the floor. torch.compile (rows 5--8) collapses the small router gemms into a captured graph and tunes the matmul kernels for retrofit's small shapes, removing of the structural gap. The remaining is the small overhead of surviving router calls per token; closing it would require a Triton-fused router that issues a single kernel for the entire stack----einsum sequence (open future work).
Skip on top of compile. Eager-mode contributes a further -- when used on its own at , but cannot currently be composed with torch.compile in a single forward: the dyn-skip rule requires an .item() on a per-token threshold comparison, which forces a CPU sync and breaks the captured CUDA graph. Production deployments pick one mode per latency budget---compile for high-throughput batch decode, eager-with-skip for variable-batch interactive workloads. A static-graph variant of the skip rule (compile-safe top- with a fixed routing decision per shape bucket) is a known direction.
### Compile vs.\ eager: accuracy parity
app:compile_accuracy
The compiled-overhead result of :exp_retrofit requires that torch.compile preserve the retrofit's accuracy. We verified two parity tests on the canonical v3 k retrofit.
LAMBADA- accuracy parity. Running LAMBADA- with the retrofit wrapped in torch.compile using default mode and dynamic shapes: acc compiled vs.\ eager (pp), ppl vs.\ . Per-target argmax agreement against eager: — the residual are tokens where eager and compiled both fall within bf16 SDPA jitter and the rank-/rank- logits flip.
Per-token logit parity. On real prompt tokens with mode="reduce-overhead" (the inference-time mode used for the speed table): per-token argmax agreement , max , RMSE — same magnitude as the cache-on/cache-off SDPA jitter on the stock base (Appendix~app:kv_equiv). Compile preserves the retrofit's forward to within bf16 evaluation noise; the base\_compiled number is at matched accuracy.
### Block partition sweep
app:block_partition
### MMStar subcategory breakdown
app:mmstar_subcat
## VLA appendix
app:vla_section
### VLA seed variance on 2B
app:vla_variance
We ran two seeds for the two B suites where Path 0 / Path B are within evaluation noise on a single seed; libero\_spatial and libero\_object are single runs.
Aggregating both repeated rollouts gives Path 0 and Path B (pp), while the Table~tab:vla_libero operating-point values give pp. Both summaries preserve the ranking Path B Path 0.
Binomial standard error. At the per-suite binomial SE is pp and the -suite aggregate SE is pp. The repeated-rollout B gain pp is therefore , while the Table~tab:vla_libero operating point is . We do not interpret the -suite gain as a statistically tight win in the strict frequentist sense; the load-bearing pattern is the per-suite ranking on the suites where a stable backbone matters most: Spatial pp, Goal pp in Table~tab:vla_libero, and Long- showing sensitivity across repeated rollouts. The B clean-base pp single-run cell sits at and is reported with the same caveat.
### VLA pipeline pitfall: unfrozen action-token embeddings
app:vla_pitfall
Two Qwen3-VL-4B base VLMs we tried produced qualitatively different Path B behaviour. Our first attempt used an ``Instruct-Action'' variant that adds <robot\_action\_*> embeddings (initialised at the mean of the existing vocab, kept unfrozen). The OFT head we train does not predict those tokens, but the rows for them sit in the tied embed\_tokens / lm\_head matrix and receive gradient from the retrofit's distillation loss and from label smoothing. Path B under this dirty base reaches 4-suite avg (Long- ); Path B under the clean base---no extra tokens added---reaches (Long- ). A pp Long- swing is entirely explained by whether unused rows in the shared matrix receive stray gradient. Use the clean base for retrofit--OFT; only add bespoke action vocabulary when the head actually predicts those tokens.
### VLA landscape: open VLAs on standard benchmarks
app:vla_landscape
### VLA 
 setup: per-block drift and eligible-set selection
app:vla_pareto
The eligible set for each scale is selected from a per-block ablation on the trained Path B policy, not on the VLM retrofit alone, because the VLA action stream activates a different routing distribution than the language stream. We measure per-block action-drift MSE on a -sample sim trajectory: the policy is run with no skip and with each block individually skipped; the per-step action MSE between the skipped and no-skip rollouts is the per-block ``cost of removing this block on the action distribution''. The two safest blocks per scale form .
2B Path B v2 30k per-block action-drift MSE (sorted ascending, in units ):
block~: ; block~: ; block~: ; block~: ; remaining blocks . Selecting the two lowest: .
4B Path B 30k clean per-block action-drift MSE: block~: ; block~: ; block~: ; block~: ; remaining . Selecting: .
Cross-scale transfer of fails. Naively transferring B's onto B catastrophically drops LIBERO-Spatial to at — block-'s drift on B is roughly that of block-, so the same eligible-set rule produces wildly different operating points across scales. must be calibrated on each model's own sim distribution; this is the per-scale half of the modality-aware skip protocol.
Threshold calibration. Per-block thresholds are the -th empirical quantile of over (B) / (B) sim-rollout records (one record per decoded action token across many trajectories). The Tab.~tab:vla_reskip_pareto numbers use this protocol. A separate ``Method A'' that calibrates on retrofit pretokenized data (as if the VLA were just a long-context LM) accidentally lands at a conservative operating point that mimics on the action distribution; we use it as a sensitivity comparator in our internal ablations and recommend the action-distribution-calibrated thresholds for any deployment.
Cross-modality threshold transfer is not safe. The LAMBADA-calibrated thresholds at from Tab.~tab:retrofit_pareto, applied unchanged to the same Path B policy at LIBERO inference, drop LIBERO-Spatial from (a warm-start, partial eval) to . The action distribution has more low- mass than the language distribution at the same blocks; the language corresponds to roughly the action , which over-skips. This is the empirical hard motivation for per-modality, per-token-class calibration in deployments that mix modalities within a single rollout.
### VLA planned follow-ups
app:vla_future
Per-modality, per-token-class . The Tab.~tab:vla_reskip_pareto thresholds are calibrated on the action distribution as a whole; an obvious refinement is per-modality within a single rollout (vision tokens, language tokens, action tokens). Working hypothesis: vision concentrates on early blocks (allows aggressive late-block skip on perception steps); action spreads to late blocks (restricts late-block skip on action prediction steps). The calibration harness (retrofit/eval/calibrate\_vla\_thresholds.py) is in place; the failed cross-modality transfer of Appendix~app:vla_pareto is the empirical motivation. Compile skip composition. A static-graph variant of the skip rule (compile-safe top- with a fixed routing decision per shape bucket) would let the compiled full-depth mode and the eager skip mode coexist in a single forward and stack their and -- savings. Edge hardware. Wall-clock is H100 / bf16; RTX 4090 / Jetson Orin direct measurement is planned.
## NeurIPS Paper Checklist
app:checklist
enumerate
 Claims. Yes. The abstract, introduction, and conclusions state the main claims and separate the retrofit contribution from calibrated skipping, VLA transfer, and systems characterization.
 Limitations. Yes. Appendix~app:limitations lists architecture coverage, compile / dynamic-skip composition, hardware scope, and per-modality calibration limits.
 Theory assumptions. Not applicable. The paper is empirical and algorithmic; all equations define implemented objectives or routing rules.
 Experimental reproducibility. Yes. Sections~sec:reskip--sec:vla_results and the appendix specify model scales, block partitions, training steps, datasets, calibration rules, and evaluation protocols.
 Open access to code and data. Partially. Experiments use public datasets and benchmarks where available; release details for retrofit / VLA training scripts will follow the anonymous-submission policy.
 Compute. Yes. Main text and appendices report GPU type, training steps, rough GPU-minutes / GPU-hours, and latency measurement settings.
 Dataset and benchmark provenance. Yes. All public datasets and benchmarks are cited; LIBERO and lmms-eval protocols are described with rollout counts or split usage.
 Human subjects. Not applicable. No new human-subject data are collected.
 Privacy. Not applicable beyond the privacy considerations of the cited public datasets.
 Licenses. Partially. Public resources are cited; final camera-ready release will include the exact code / model / data license table.
 Broader impacts. Yes. The work targets lower inference cost and adaptive computation in pretrained models; no safety-critical deployment is claimed.
 Safeguards. Yes. VLA thresholds are calibrated per model and action distribution; cross-modality transfer failures are explicitly documented as a deployment risk.
enumerate
document