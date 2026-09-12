# Paper-ready VLM sensitivity insert

## Recommended placement

- **Main text:** Figure S1 as the Full-path RSM ablation and Figure S2 as the mechanism analysis. These two figures support the central claim that RSM benefits AttnRes itself and learns structured residual usage; neither figure depends on ReSkip decisions.
- **Main text or first appendix page:** Figure S3, because it connects the accepted Full-path representation to inference-only token-level compute removal.
- **Appendix:** Figure S4 and the complete runtime table. Fixed-64 timing is valuable, but its hardware- and process-state boundary deserves the space of an appendix protocol.

## Suggested results text

### Full-path residual-strength sensitivity

We first vary only the residual-strength scale used during ordinary AttnRes adaptation, without activating a skip branch, providing skip labels, or imposing a compute target. On Qwen3-VL-2B, the original Full AttnRes control ($s=0$) obtains a six-task macro score of 67.357. Scales $s\in\{0.25,0.50,0.75\}$ obtain 67.615, 67.970, and 67.892, respectively. Thus, the accepted setting $s=0.50$ improves the Full-path macro by 0.613 points, while both neighboring nonzero scales remain above the original AttnRes control. This one-seed sweep supports local robustness of the Full-path benefit rather than a population-level significance claim.

### Block- and token-dependent residual specialization

We analyze the accepted 2B and 4B Full-path checkpoints on 128 held-out examples, separating visual-prefix, text-prompt, and assistant-response tokens. Normalized native route specialization varies from 0.032 to 0.719 across 2B blocks and from 0.025 to 0.620 across 4B blocks, revealing clear block-wise and token-role structure. To test whether these differences are functionally relevant, we replace one learned block at a time with the identity residual on 32 paired requests per block. Every tested block produces nonzero output divergence: learned-to-identity KL ranges from 0.00184 to 0.00534 for 2B and from 0.00142 to 0.00451 for 4B, with top-1 flip rates of approximately 1.2%–2.4%. Gate deviation is associated with ablation KL more strongly for 4B (Spearman $\rho=0.667$, paired-request bootstrap 95% CI [0.405, 0.738]) than for 2B ($\rho=0.200$, [0.029, 0.429]). We therefore use the correlation only as an association; functional relevance is supported by the explicit identity intervention.

### Inference-only quality–compute frontier

Keeping the Full-path checkpoint fixed, we calibrate four native token-level ReSkip policies and report observed removal rather than requested coverage. Under the pre-registered six-task macro tolerance of $-0.30$ points, Qwen3-VL-2B safely removes 0.959 block-equivalents per decode token with a 0.257-point macro decrease, while Qwen3-VL-4B safely removes 0.767 block-equivalents with a 0.251-point decrease. The next more aggressive cells fail the same gate: 2B removes 1.019 block-equivalents at $-0.309$ points, and 4B removes 1.089 at $-0.512$ points. This boundary is inference-only: policy calibration does not add a skip controller, skip supervision, or a second training stage.

### Measured runtime at the safe operating points

On paired batch-1 H100 runs using real inputs from all six benchmarks and a fixed 64-token decode, the safe 2B operating point is 1.0438× faster than Base (95% paired-request bootstrap CI [1.0357, 1.0532]) and 1.1752× faster than Full RSM. The safe 4B point is 1.0286× faster than Base ([1.0227, 1.0363]) and 1.0715× faster than Full RSM. Each interval measures within-cell request variation on one H100; it does not include independent-process, cross-GPU, or cross-host variance.

## Compact operating-point table

| Model | Safe target coverage | Observed block-eq/token | Δ macro vs Full RSM | Speedup vs Base | Speedup vs Full RSM |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | 0.45 | 0.959 | −0.257 pp | 1.0438× | 1.1752× |
| Qwen3-VL-4B | 0.25 | 0.767 | −0.251 pp | 1.0286× | 1.0715× |

## Figure captions

**Figure S1 — Full-path sensitivity to residual-strength scale.** Six-task macro accuracy for Qwen3-VL-2B after matched 3k-step AttnRes adaptation. The $s=0$ and $s=0.50$ cells are protocol-matched reused controls; $s=0.25$ and $s=0.75$ are freshly trained pre-registered cells. The right panel reports the learned residual-factor range and mean within-block standard deviation. No skip branch, skip labels, or compute target are used during training.

**Figure S2 — RSM learns structured token- and block-dependent residual usage.** Top: normalized native route specialization for visual-prefix, text-prompt, and assistant-response tokens over 128 held-out examples. Bottom: functional single-block identity interventions on 32 paired requests per block. Filled and hollow points denote the 2B and 4B checkpoints. The reported Spearman association is descriptive across blocks; causal language is reserved for the explicit identity ablation.

**Figure S3 — Quality–compute frontier of frozen native token-level policies.** Macro change relative to Full RSM versus observed removed block-equivalents per decode token. The shaded region marks the pre-registered $\Delta\geq-0.30$ point acceptance band. Requested coverage is used only to calibrate a frozen policy and is not the plotted compute quantity. Per-task changes appear in the companion table and tasks are not treated as independent replicates.

**Figure S4 — Fixed-64 end-to-end runtime sensitivity.** Paired batch-1 H100 speedups over real six-task images and prompts, with 64 generated tokens and three repetitions. Filled points compare ReSkip with Base and include paired-request 95% bootstrap intervals; hollow points compare with Full RSM. Amendment 015 preserves the original concurrently measured 2B-q10 estimate but uses its pre-specified quiescent rerun in the figure.

## Claim discipline

- Say **“RSM improves Full-path AttnRes and exposes native token-level depth adaptation as an inference-time by-product.”**
- Say **“associated with”** for the across-block gate/KL correlation; use **“functionally affects”** only for the explicit identity ablation.
- Do not describe the six tasks as iid replicates or turn their range into a confidence interval.
- Do not claim safe removal of 1.02 block-equivalents for 2B: that cell misses the registered macro gate by 0.009 points. The strict safe 2B value is 0.959.
- Do not reuse the original 2B-q10 1.1340× timing in the paper. It remains in the audit table because the run overlapped other project evaluations; the quiescent value is 1.0259× [1.0047, 1.0505].
