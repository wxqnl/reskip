# Deep Review Report

**Paper**: `/home/user01/Minko/reskip2/reskip/paper/main.tex` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-05-01 05:25 | **Venue**: NeurIPS2026
**Artifacts**: `/home/user01/Minko/reskip2/reskip/review_results/reskip-brain-inspired-adaptive-depth-for-pretrained-transformers-attention-residuals`

## Overall Assessment

Deep review found 1 major, 3 moderate, 0 minor issues. The highest-priority concerns are: Abstract and conclusion claims need explicit evidence traceability; Cross-section numeric consistency should be reconciled.

- **Major**: 1
- **Moderate**: 3
- **Minor**: 0

## Academic Pre-Review Committee

### Editor (Desk Reject Screen)

## Editor Pre-Screen (1-10)

Score: 4.0/10
Verdict: Desk Reject

### Desk-Reject Triggers (if any)
- Abstract and conclusion claims need explicit evidence traceability

### Top 3 Reasons (no hedging)
1. Abstract and conclusion claims need explicit evidence traceability

### Fast Fixes (within 1-2 days)
- Clarify abstract to address abstract and conclusion claims need explicit evidence traceability.
- Clarify introduction to address cross-section numeric consistency should be reconciled.
- Clarify related_work to address novelty claim should be grounded against the closest prior work.

### Reviewer 1 (Theory Contribution)

## Theory Contribution Review

### 3 Fatal Theory Holes
1. (abstract) Abstract and conclusion claims need explicit evidence traceability — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.
2. (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Concrete Moves
- Tighten the paper's theoretical positioning in abstract to resolve abstract and conclusion claims need explicit evidence traceability.
- Tighten the paper's theoretical positioning in related_work to resolve novelty claim should be grounded against the closest prior work.

### Reviewer 3 (Literature Dialogue)

## Literature Dialogue Review

### Closest Prior Work Risks
- (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Gap Claim Risks
- The claimed gap should be defended more explicitly: Novelty claim should be grounded against the closest prior work.

### Fast Fixes
- Name the closest prior comparator in related_work and explain the real novelty delta.

### Reviewer 2 (Methodology & Transparency)

## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (introduction) "The dual-process distinction~kahneman2011thinking contrasts System~1 (fast, automatic) with System~2 (slow, deliberate); the same dichotomy has direct correlates in cortical processing, where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~lamme2000distinct, kar2019recurrent, kietzmann2019recurrence, spoerer2020recurrent." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (method) "is the canonical partition for both scales: it sits on the accuracy plateau alongside and halves the per-token router count, which is the dominant factor in the inference-cost result below." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.

### Reviewer 4 (Logic Chain)

## Logic Chain Review

### Breakpoints
- (abstract) Abstract and conclusion claims need explicit evidence traceability — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

### Structural Fix Moves
- Add one explicit bridge sentence in abstract so the argument chain closes cleanly.

### Committee Consensus

## Committee Consensus

Overall Score: 4.0/10
Editor Verdict: Desk Reject

### Score Formula
- base 9.0
- minus 1.5 * major (1)
- minus 0.7 * moderate (3)
- minus 0.2 * minor (0)
- floor 1.0
- desk reject cap 4.0

### Top 3 Issues To Fix First
1. Abstract and conclusion claims need explicit evidence traceability
2. Cross-section numeric consistency should be reconciled
3. Comparison protocol should make fairness assumptions explicit

## Paper Summary

# Paper Summary: ReSkip: Brain-Inspired Adaptive Depth for Pretrained Transformers\ Attention Residuals

## Research Question
- Pretrained transformers execute every layer for every input

## Core Thesis
- The central problem in this paper is how to install such a signal into a standard pretrained transformer through a short fine-tune, while preserving its original capabilities and making the signal usable for skipping.

## Headline Claims
- The central problem in this paper is how to install such a signal into a standard pretrained transformer through a short fine-tune, while preserving its original capabilities and making the signal usable for skipping.

## Section Map
- abstract (57-61): 195 words
- introduction (62-90): 656 words
- method (91-330): 4077 words
- related (331-340): 262 words
- discussion (341-937): 7242 words

## Closure Targets
- No closure target was extracted automatically.

## Major Issues

### M1: Abstract and conclusion claims need explicit evidence traceability
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: low
- **Section**: abstract
- **Related Sections**: abstract, results, conclusion
- **Root Cause Key**: `abstract-and-conclusion-claims-need-explicit-evidence-traceability`
- **Quote Verified**: no
- **Quote**: —
- **Explanation**: At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

## Moderate Issues

### M1: Cross-section numeric consistency should be reconciled
- **Type**: presentation
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: medium
- **Section**: introduction
- **Related Sections**: introduction, method, related
- **Root Cause Key**: `cross-section-numeric-consistency-should-be-reconciled`
- **Quote Verified**: yes
- **Quote**: `The dual-process distinction~kahneman2011thinking contrasts System~1 (fast, automatic) with System~2 (slow, deliberate); the same dichotomy has direct correlates in cortical processing, where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~lamme2000distinct, kar2019recurrent, kietzmann2019recurrence, spoerer2020recurrent.`
- **Explanation**: Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.

### M2: Comparison protocol should make fairness assumptions explicit
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: method
- **Related Sections**: method
- **Root Cause Key**: `comparison-protocol-should-make-fairness-assumptions-explicit`
- **Quote Verified**: yes
- **Quote**: `is the canonical partition for both scales: it sits on the accuracy plateau alongside and halves the per-token router count, which is the dominant factor in the inference-cost result below.`
- **Explanation**: Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### M3: Novelty claim should be grounded against the closest prior work
- **Type**: claim_accuracy
- **Source**: [LLM] via `prior_art_and_novelty_grounding`
- **Confidence**: low
- **Section**: related_work
- **Related Sections**: related_work, results
- **Root Cause Key**: `novelty-claim-should-be-grounded-against-the-closest-prior-work`
- **Quote Verified**: no
- **Quote**: —
- **Explanation**: The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

## Phase 0 Automated Findings

### [Script] BIB

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Check: /home/user01/Minko/reskip2/reskip/paper/main.tex |
| --- | Minor | PASS |
| --- | Minor | entries: 0 |
| --- | Minor | entries: 0 |

### [Script] CITATIONS

| Line | Severity | Issue |
|------|----------|-------|
| 66 | Critical | Citation stacking: 5 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 68 | Critical | Citation stacking: 5 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 70 | Critical | Citation stacking: 7 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 335 | Critical | Citation stacking: 11 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 337 | Critical | Citation stacking: 10 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 339 | Critical | Citation stacking: 8 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |

### [Script] DEAI

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Use --analyze for full analysis |

### [Script] EXPERIMENT

| Line | Severity | Issue |
|------|----------|-------|
| 236 | Major | Performance claim lacks an explicit baseline or comparator. |
| 249 | Major | Performance claim lacks an explicit baseline or comparator. |
| 289 | Major | Performance claim lacks an explicit baseline or comparator. |
| 347 | Critical | Conclusion overreaches the reported evidence; avoid universal or guarantee-style claims. |
| 181 | Minor | No statistical significance, variance, or confidence information is mentioned. |
| 774 | Major | Performance claim lacks an explicit baseline or comparator. |
| 775 | Major | Performance claim lacks an explicit baseline or comparator. |
| 776 | Major | Performance claim lacks an explicit baseline or comparator. |
| 795 | Major | Performance claim lacks an explicit baseline or comparator. |
| 341 | Major | Discussion may lack depth: low ratio of explanatory/attribution language (8/451 lines). Add causal analysis explaining why results occur. |

### [Script] FIGURES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | figures in /home/user01/Minko/reskip2/reskip/paper/main.tex... |
| --- | Minor | 4 figures. |
| --- | Minor | Line 83: Image not found: reskip_main_figure.png |
| --- | Minor | Line 126: Image not found: reskip_routing_importance_vs_ablation.pdf |
| --- | Minor | Line 648: Image not found: reskip_pareto.pdf |
| --- | Minor | Line 652: Image not found: reskip_latency.pdf |
| --- | Minor | ⚠️ Found 4 potential issues. |

### [Script] FORMAT

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | ============================================================ |
| --- | Minor | Format Check Report |
| --- | Minor | ============================================================ |
| --- | Minor | /home/user01/Minko/reskip2/reskip/paper/main.tex |
| --- | Minor | UNAVAILABLE |
| --- | Minor | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| --- | Minor | MODE] chktex not available |
| --- | Minor | chktex for detailed format checking |

### [Script] GRAMMAR

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | No rule-based issues detected in selected scope. |

### [Script] LOGIC

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | /METHODOLOGY: No rule-based coherence issues detected. |

### [Script] REFERENCES

| Line | Severity | Issue |
|------|----------|-------|
| 86 | Minor | Unreferenced label: \label{fig:overview} is never cited in text |
| 103 | Minor | Unreferenced label: \label{eq:attnres} is never cited in text |
| 118 | Minor | Unreferenced label: \label{eq:skip_rule} is never cited in text |
| 122 | Minor | Reference before definition: \ref{fig:reskip_routing} at line 122 appears before label definition at line 128 |
| 131 | Minor | Reference before definition: \ref{tab:reskip_benchmark} at line 131 appears before label definition at line 137 |
| 136 | Minor | Reference before definition: \ref{tab:b1b2_observed_rate} at line 136 appears before label definition at line 602 |
| 163 | Minor | Unreferenced label: \label{eq:gamma_gate} is never cited in text |
| 213 | Minor | Reference before definition: \ref{tab:attribution} at line 213 appears before label definition at line 217 |
| 234 | Minor | Reference before definition: \ref{tab:retrofit_latency} at line 234 appears before label definition at line 240 |
| 296 | Minor | Reference before definition: \ref{tab:vla_reskip_pareto} at line 296 appears before label definition at line 300 |
| 408 | Minor | Reference before definition: \ref{tab:retrofit_ablation} at line 408 appears before label definition at line 412 |
| 451 | Minor | Reference before definition: \ref{tab:data_mix_ablation} at line 451 appears before label definition at line 455 |
| 538 | Minor | Unreferenced label: \label{tab:reskip_position_ablation} is never cited in text |
| 573 | Minor | Reference before definition: \ref{tab:b1b2_decision_rule} at line 573 appears before label definition at line 578 |
| 656 | Minor | Unreferenced label: \label{fig:reskip_pareto_latency} is never cited in text |
| 680 | Minor | Reference before definition: \ref{tab:gromov_pruning_2b} at line 680 appears before label definition at line 686 |
| 685 | Minor | Reference before definition: \ref{tab:retrofit_pareto} at line 685 appears before label definition at line 720 |
| 734 | Minor | Reference before definition: \ref{tab:retrofit_pareto_v1_legacy} at line 734 appears before label definition at line 738 |
| 758 | Minor | Reference before definition: \ref{tab:retrofit_latency_full} at line 758 appears before label definition at line 762 |
| 801 | Minor | Unreferenced label: \label{tab:block_partition} is never cited in text |
| 827 | Minor | Unreferenced label: \label{tab:mmstar_subcat} is never cited in text |
| 855 | Minor | Unreferenced label: \label{tab:vla_seed_variance} is never cited in text |
| 878 | Minor | Unreferenced label: \label{tab:vla_landscape} is never cited in text |

### [Script] SENTENCES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | SENTENCE (Line 58, 41 words, 4 clauses) |
| --- | Minor | Building on Attention Residuals (\attnres{})~\citep{chen2026attnres}, whose softmax routing over previous block outputs is computed \emph{before} each block runs, our main contribution is \retrofit{}, an identity-preserving  -gated residual-injection fine-tune that freezes the base and adds well under   new parameters. |
| --- | Minor | Building on Attention Residuals (\attnres{})~\citep{chen2026attnres}. whose softmax routing over previous block outputs is computed \emph{before} each block runs. our main contribution is \retrofit{}. an identity-preserving  -gated residual-injection fine-tune that freezes the base and adds well under   new parameters.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 58, 42 words, 4 clauses) |
| --- | Minor | We further show that the installed routing weights can be calibrated, with a complementary safety check, into a dynamic block-skip rule (\reskip{}) with no auxiliary head, and that the same signal supports adaptive depth on the action stream of the trained policy. |
| --- | Minor | We further show that the installed routing weights can be calibrated. with a complementary safety check. into a dynamic block-skip rule (\reskip{}) with no auxiliary head. and that the same signal supports adaptive depth on the action stream of the trained policy.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 66, 48 words, 6 clauses) |
| --- | Minor | The dual-process distinction~\citep{kahneman2011thinking} contrasts \emph{System~1} (fast, automatic) with \emph{System~2} (slow, deliberate); the same dichotomy has direct correlates in cortical processing, where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~\citep{lamme2000distinct, kar2019recurrent, kietzmann2019recurrence, spoerer2020recurrent}. |
| --- | Minor | The dual-process distinction~\citep{kahneman2011thinking} contrasts \emph{System~1} (fast. automatic) with \emph{System~2} (slow. deliberate); the same dichotomy has direct correlates in cortical processing. where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~\citep{lamme2000distinct. kar2019recurrent. kietzmann2019recurrence. spoerer2020recurrent}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 68, 29 words, 4 clauses) |
| --- | Minor | Test-time compute scaling~\citep{openai2024o1, deepseek2025r1, snell2024scaling}, chain-of-thought prompting~\citep{wei2022chain}, and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering. |
| --- | Minor | Test-time compute scaling~\citep{openai2024o1. deepseek2025r1. snell2024scaling}. chain-of-thought prompting~\citep{wei2022chain}. and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 131, 43 words, 5 clauses) |
| --- | Minor | On a 340M block-\attnres{} transformer ( ,  ,   blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}), \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}, HellaSwag~\citep{zellers2019hellaswag}, and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ). |
| --- | Minor | On a 340M block-\attnres{} transformer (. blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}). \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}. HellaSwag~\citep{zellers2019hellaswag}. and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 133, 51 words, 4 clauses) |
| --- | Minor | \paragraph{Dynamic vs.\ static at the same skip rate.} Holding the \attnres{} weights fixed and varying only the runtime decision rule, an input-dependent rule that fires close to its calibrated   rate preserves LAMBADA at  , while static-position and random-toggle skip schedules at the same rate collapse to  --  on the same lm-eval suite. |
| --- | Minor | \paragraph{Dynamic vs.\ static at the same skip rate.} Holding the \attnres{} weights fixed and varying only the runtime decision rule. an input-dependent rule that fires close to its calibrated   rate preserves LAMBADA at. while static-position and random-toggle skip schedules at the same rate collapse to  --  on the same lm-eval suite.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 156, 55 words, 4 clauses) |
| --- | Minor | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality, (P2) emits a routing   informative enough to drive \reskip{}, and (P3) trains in a single short fine-tune over off-the-shelf SFT data, with the base frozen. |
| --- | Minor | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality. (P2) emits a routing   informative enough to drive \reskip{}. and (P3) trains in a single short fine-tune over off-the-shelf SFT data. with the base frozen.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 169, 37 words, 4 clauses) |
| --- | Minor | To make this consistent with \texttt{use\_cache=True}, on a skipped block we still run the per-layer K/V-producing slice ( ), which costs  --  of a full-layer forward, and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP. |
| --- | Minor | To make this consistent with \texttt{use\_cache=True}. on a skipped block we still run the per-layer K/V-producing slice ( ). which costs  --  of a full-layer forward. and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 173, 32 words, 5 clauses) |
| --- | Minor | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\,\mathcal{L}_{\text{KL}}(\text{skip path}\,\|\,\text{teacher}) + \lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}}(\alpha) , |
| --- | Minor | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\. \mathcal{L}_{\text{KL}}(\text{skip path}\. \|\. \text{teacher}) + \lambda_{\text{ent}}\. \mathcal{L}_{\text{ent}}(\alpha). |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 176, 51 words, 5 clauses) |
| --- | Minor | where the full-path CE is computed on assistant tokens, while the skip-branch KL samples one block at random per step, runs the forward with it skipped, and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped). |
| --- | Minor | where the full-path CE is computed on assistant tokens. while the skip-branch KL samples one block at random per step. runs the forward with it skipped. and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 186, 45 words, 4 clauses) |
| --- | Minor | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava},  k steps;   GPU-min on  B), used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts,  k steps), the canonical mix used for every result we report. |
| --- | Minor | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava}. k steps;   GPU-min on  B). used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts. k steps). the canonical mix used for every result we report.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 186, 59 words, 9 clauses) |
| --- | Minor | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}, MMMU~\citep{yue2024mmmu}, MMStar~\citep{chen2024mmstar}, AI2D~\citep{kembhavi2016ai2d}, OCRBench~\citep{liu2024ocrbench}, RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal, while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~ ,\, ). |
| --- | Minor | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}. MMMU~\citep{yue2024mmmu}. MMStar~\citep{chen2024mmstar}. AI2D~\citep{kembhavi2016ai2d}. OCRBench~\citep{liu2024ocrbench}. RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal. while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~. \. ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 234, 23 words, 5 clauses) |
| --- | Minor | First, the canonical   partition halves the per-token router count vs  , closing   of the eager-vs-eager gap on its own ( , seq~ , cache, H100/bf16; Tab.~ ). |
| --- | Minor | First. the canonical   partition halves the per-token router count vs. closing   of the eager-vs-eager gap on its own (. seq~. cache. H100/bf16; Tab.~ ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 236, 64 words, 5 clauses) |
| --- | Minor | \paragraph{Two deployment modes.} We separate three speed claims that are easy to conflate: (i)~at the  M from-scratch scale, \reskip{} on \attnres{} reaches   wall-clock with zero benchmark drop (\S , mechanism validation); (ii)~at the Qwen3-VL retrofit scale, the compiled full-depth retrofit runs at   \texttt{base\_compiled} (this section, accuracy gain at iso-cost); (iii)~eager-mode \reskip{} on the retrofit contributes a further  --  standalone (Appendix~ ). |
| --- | Minor | \paragraph{Two deployment modes.} We separate three speed claims that are easy to conflate: (i)~at the  M from-scratch scale. \reskip{} on \attnres{} reaches   wall-clock with zero benchmark drop (\S. mechanism validation); (ii)~at the Qwen3-VL retrofit scale. the compiled full-depth retrofit runs at   \texttt{base\_compiled} (this section. accuracy gain at iso-cost); (iii)~eager-mode \reskip{} on the retrofit contributes a further  --  standalone (Appendix~ ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 255, 15 words, 6 clauses) |
| --- | Minor | Full block-partition sweep, v1 v2 v3 trail, and  -free / informed-init / observer-only baselines are in Appendices~ ,\, ,\, . |
| --- | Minor | Full block-partition sweep. v1 v2 v3 trail. and  -free / informed-init / observer-only baselines are in Appendices~. \. \. .. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 265, 48 words, 5 clauses) |
| --- | Minor | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT, baseline), \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit,   throughout), and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init,   ramps   on VLA data). |
| --- | Minor | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT. baseline). \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit. throughout). and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init. ramps   on VLA data).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 328, 30 words, 11 clauses) |
| --- | Minor | The per-block action-drift sweep that picks   per scale, the cross-modality transfer failure, the per-seed breakdown, the pipeline pitfall, and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~ ,\, ,\, ,\, . |
| --- | Minor | The per-block action-drift sweep that picks   per scale. the cross-modality transfer failure. the per-seed breakdown. the pipeline pitfall. and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~. \. \. \. .. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 335, 31 words, 4 clauses) |
| --- | Minor | For test-time compute, o1~\citep{openai2024o1}, DeepSeek-R1~\citep{deepseek2025r1}, chain-of-thought~\citep{wei2022chain}, and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed. |
| --- | Minor | For test-time compute. o1~\citep{openai2024o1}. DeepSeek-R1~\citep{deepseek2025r1}. chain-of-thought~\citep{wei2022chain}. and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 335, 24 words, 6 clauses) |
| --- | Minor | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive, dehghani2018universal}, CALM~\citep{schuster2022confident}, Mixture-of-Depths~\citep{raposo2024mixture}, LayerSkip~\citep{elhoushi2024layerskip}, and static pruning~\citep{gromov2024unreasonable, men2024shortgpt}. |
| --- | Minor | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive. dehghani2018universal}. CALM~\citep{schuster2022confident}. Mixture-of-Depths~\citep{raposo2024mixture}. LayerSkip~\citep{elhoushi2024layerskip}. and static pruning~\citep{gromov2024unreasonable. men2024shortgpt}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 335, 14 words, 4 clauses) |
| --- | Minor | These add a halting head, exit classifier, router, speculative decoder, or static removal rule. |
| --- | Minor | These add a halting head. exit classifier. router. speculative decoder. or static removal rule.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 337, 28 words, 4 clauses) |
| --- | Minor | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}. |
| --- | Minor | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps. while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct. kar2019recurrent. dicarlo2012does}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 337, 30 words, 5 clauses) |
| --- | Minor | We take this as a design target, not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical, buschman2007topdown, larkum2013cellular} motivate intrinsic routing, while our implementation is a transformer retrofit. |
| --- | Minor | We take this as a design target. not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical. buschman2007topdown. larkum2013cellular} motivate intrinsic routing. while our implementation is a transformer retrofit.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 339, 32 words, 6 clauses) |
| --- | Minor | In VLA, RT-2~\citep{brohan2023rt}, Octo~\citep{team2024octo}, OpenVLA~\citep{kim2024openvla}, and Pi0~\citep{black2024pi0} established the paradigm, while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression. |
| --- | Minor | In VLA. RT-2~\citep{brohan2023rt}. Octo~\citep{team2024octo}. OpenVLA~\citep{kim2024openvla}. and Pi0~\citep{black2024pi0} established the paradigm. while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 434, 28 words, 5 clauses) |
| --- | Minor | \paragraph{Over-training signature at  .} Holding ( ,   VLM, fast ramp) fixed and varying only the total step count, MMBench is strictly monotonic in steps:  k   (preserved),  k   ( 14pp),  k   ( 21pp). |
| --- | Minor | \paragraph{Over-training signature at  .} Holding (. VLM. fast ramp) fixed and varying only the total step count. MMBench is strictly monotonic in steps:  k   (preserved). k   ( 14pp). k   ( 21pp).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 442, 28 words, 4 clauses) |
| --- | Minor | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc), block~  severe ( ), blocks~ ,  ,   are safest (  to  ). |
| --- | Minor | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc). block~  severe ( ). blocks~. are safest (  to  ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 451, 74 words, 1 clauses) |
| --- | Minor | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| --- | Minor | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 510, 30 words, 4 clauses) |
| --- | Minor | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \,ms at seq   on  H100, so a  --  block-level \attnres{} overhead would push past common  \,Hz /  \,Hz robotic control budgets. |
| --- | Minor | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \. ms at seq   on  H100. so a  --  block-level \attnres{} overhead would push past common  \. Hz /  \. Hz robotic control budgets.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 758, 67 words, 4 clauses) |
| --- | Minor | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls, the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward, \S ) tracks the VLM retrofit within   at both partitions, and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved. |
| --- | Minor | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls. the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward. \S ) tracks the VLM retrofit within   at both partitions. and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 786, 53 words, 4 clauses) |
| --- | Minor | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at  , but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison, which forces a CPU sync and breaks the captured CUDA graph. |
| --- | Minor | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at. but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison. which forces a CPU sync and breaks the captured CUDA graph.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 925, 17 words, 5 clauses) |
| --- | Minor | Sections~ --  and the appendix specify model scales, block partitions, training steps, datasets, calibration rules, and evaluation protocols. |
| --- | Minor | Sections~ --  and the appendix specify model scales. block partitions. training steps. datasets. calibration rules. and evaluation protocols.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |

## Decision Signals

- **Committee Score**: 4.0/10
- **Editor Verdict**: Desk Reject
- **Reviewer Recommendation**: Major Revision
- **Issue Bundle**: 1 major / 3 moderate / 0 minor

## Revision Roadmap

### Priority 1

- [ ] Abstract and conclusion claims need explicit evidence traceability ([LLM]; abstract)

### Priority 2

- [ ] Cross-section numeric consistency should be reconciled ([LLM]; introduction)
- [ ] Comparison protocol should make fairness assumptions explicit ([LLM]; method)
- [ ] Novelty claim should be grounded against the closest prior work ([LLM]; related_work)
