## Discussion
sec:discussion
 installs an intrinsic depth-routing signal into a frozen pretrained transformer through a short fine-tune; the resulting routing weights drive as a natural application. At the canonical v3 recipe, the retrofit improves VLM benchmarks on Qwen3-VL-B/B at iso-cost compiled inference, and lifts LAMBADA by /pp with the largest gain concentrated on MMStar math (consistent with a router that lifts deliberate-reasoning paths). The installed signal is usable: at M from-scratch, reaches wall-clock at zero benchmark drop, and a rate-matched comparison at skip rate shows the input-dependent rule strictly beats static and random schedules. A LIBERO VLA warm-started from the retrofit improves the matched pure-OFT baseline at both 2B and 4B; on the action stream, with thresholds re-calibrated on the action distribution, further improves the trained policy at the conservative operating point.
Scope of claims. The result should be read as evidence that pretrained transformers can acquire a usable depth-routing pathway through lightweight retrofitting, not that the resulting policy is universally optimal or that low weight alone certifies safe layer removal. In all large-model and VLA experiments, skip eligibility is calibrated by per-block ablation or action-drift, and thresholds are calibrated on the decoded modality. We separate compiled full-depth inference from eager dynamic skipping, since current CUDA graph capture cannot accommodate the dynamic branch; the iso-cost result requires the former and the dynamic-skip savings require the latter, and combining them is open systems work. The retrofit is demonstrated on one VLM family at two scales; transfer to other backbones is open.
references
## Limitations and future work
app:limitations
Limitations.
(i)~Two retrofit targets from one VLM family; extension to InternVL / Gemma-VL / text-only LMs is planned. (ii)~The compile-on iso-cost result requires torch.compile; eager-mode retrofit alone is at , while eager contributes a further -- but cannot currently be composed in a single forward with compile (dyn-skip's .item() guard breaks the captured CUDA graph). Deployments pick one mode per latency budget; a Triton-fused router that lets the two co-exist is open future work. (iii)~Wall-clock is H100 / bf16; lower-end GPU / edge measurement is not yet done. (iv)~The VLA Pareto thresholds are sim-calibrated per scale, not per modality within a single rollout; per-modality, per-token-class calibration is the next sweep. (v)~Block-level granularity is inherited from ; per-token routing is future work.
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
The iso-cost result of :exp_retrofit requires that torch.compile preserve the retrofit's accuracy. We verified two parity tests on the canonical v3 k retrofit.
LAMBADA- accuracy parity. Running LAMBADA- with the retrofit wrapped in torch.compile using default mode and dynamic shapes: acc compiled vs.\ eager (pp), ppl vs.\ . Per-target argmax agreement against eager: — the residual are tokens where eager and compiled both fall within bf16 SDPA jitter and the rank-/rank- logits flip.
Per-token logit parity. On real prompt tokens with mode="reduce-overhead" (the inference-time mode used for the speed table): per-token argmax agreement , max , RMSE — same magnitude as the cache-on/cache-off SDPA jitter on the stock base (Appendix~app:kv_equiv). Compile preserves the retrofit's forward to within bf16 evaluation noise; the base\_compiled number is at iso-accuracy.
### Block partition sweep
app:block_partition
### MMStar subcategory breakdown
app:mmstar_subcat
## VLA appendix
app:vla_section
### VLA seed variance on 2B
app:vla_variance
For the two B suites where the k Path 0 and Path B success rates fall within typical evaluation noise on a single seed, we ran two independent rollout seeds. libero\_spatial and libero\_object are single rollouts at both scales.
Path B improves Path 0 on libero\_goal on at least one seed and ties on the other; on libero\_10 the two paths are within evaluation noise. The 2B aggregate gain is therefore carried mainly by libero\_spatial and libero\_goal; libero\_10 is a non-loss on this scale rather than a robust win. The binomial standard error at over trials is pp at the suite level and pp on the -suite aggregate, which sets the resolution at which we read these per-suite numbers.
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
Per-modality, per-token-class . The Tab.~tab:vla_reskip_pareto thresholds are calibrated on the action distribution as a whole; an obvious refinement is per-modality within a single rollout (vision tokens, language tokens, action tokens). Working hypothesis: vision concentrates on early blocks (allows aggressive late-block skip on perception steps); action spreads to late blocks (restricts late-block skip on action prediction steps). The calibration harness (retrofit/eval/calibrate\_vla\_thresholds.py) is in place; the failed cross-modality transfer of Appendix~app:vla_pareto is the empirical motivation. Compile skip composition. A static-graph variant of the skip rule (compile-safe top- with a fixed routing decision per shape bucket) would let the iso-cost compile mode and the eager skip mode coexist in a single forward and stack their and -- savings. Edge hardware. Wall-clock is H100 / bf16; RTX 4090 / Jetson Orin direct measurement is planned.
## NeurIPS Paper Checklist
app:checklist
enumerate
 Claims. Yes. The abstract, introduction, and conclusions state the main claims and report per-seed VLA numbers in the appendix where repeated rollouts exist.
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