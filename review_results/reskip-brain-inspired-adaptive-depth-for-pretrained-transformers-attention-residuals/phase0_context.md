# Phase 0: Automated Audit Results

**File**: `/home/user01/Minko/reskip2/reskip/paper/main.tex` | **Language**: en | **Mode**: quick-audit
**Venue**: NeurIPS2026
**Generated**: 2026-05-01T05:25:11.348545

## Issue Summary (181 total)
- Critical: 7
- Major: 8
- Minor: 166

## Issues by Module

### BIB

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Check: /home/user01/Minko/reskip2/reskip/paper/main.tex |
| 2 | — | Minor | P2 | PASS |
| 3 | — | Minor | P2 | entries: 0 |
| 4 | — | Minor | P2 | entries: 0 |

### CITATIONS

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 66 | Critical | P0 | Citation stacking: 5 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 2 | 68 | Critical | P0 | Citation stacking: 5 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 3 | 70 | Critical | P0 | Citation stacking: 7 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 4 | 335 | Critical | P0 | Citation stacking: 11 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 5 | 337 | Critical | P0 | Citation stacking: 10 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 6 | 339 | Critical | P0 | Citation stacking: 8 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |

### DEAI

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Use --analyze for full analysis |

### EXPERIMENT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 236 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 2 | 249 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 3 | 289 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 4 | 347 | Critical | P0 | Conclusion overreaches the reported evidence; avoid universal or guarantee-style claims. |
| 5 | 181 | Minor | P2 | No statistical significance, variance, or confidence information is mentioned. |
| 6 | 774 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 7 | 775 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 8 | 776 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 9 | 795 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 10 | 341 | Major | P1 | Discussion may lack depth: low ratio of explanatory/attribution language (8/451 lines). Add causal analysis explaining why results occur. |

### FIGURES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | figures in /home/user01/Minko/reskip2/reskip/paper/main.tex... |
| 2 | — | Minor | P2 | 4 figures. |
| 3 | — | Minor | P2 | Line 83: Image not found: reskip_main_figure.png |
| 4 | — | Minor | P2 | Line 126: Image not found: reskip_routing_importance_vs_ablation.pdf |
| 5 | — | Minor | P2 | Line 648: Image not found: reskip_pareto.pdf |
| 6 | — | Minor | P2 | Line 652: Image not found: reskip_latency.pdf |
| 7 | — | Minor | P2 | ⚠️ Found 4 potential issues. |

### FORMAT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | ============================================================ |
| 2 | — | Minor | P2 | Format Check Report |
| 3 | — | Minor | P2 | ============================================================ |
| 4 | — | Minor | P2 | /home/user01/Minko/reskip2/reskip/paper/main.tex |
| 5 | — | Minor | P2 | UNAVAILABLE |
| 6 | — | Minor | P2 | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| 7 | — | Minor | P2 | MODE] chktex not available |
| 8 | — | Minor | P2 | chktex for detailed format checking |

### GRAMMAR

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | No rule-based issues detected in selected scope. |

### LOGIC

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | /METHODOLOGY: No rule-based coherence issues detected. |

### REFERENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 86 | Minor | P2 | Unreferenced label: \label{fig:overview} is never cited in text |
| 2 | 103 | Minor | P2 | Unreferenced label: \label{eq:attnres} is never cited in text |
| 3 | 118 | Minor | P2 | Unreferenced label: \label{eq:skip_rule} is never cited in text |
| 4 | 122 | Minor | P2 | Reference before definition: \ref{fig:reskip_routing} at line 122 appears before label definition at line 128 |
| 5 | 131 | Minor | P2 | Reference before definition: \ref{tab:reskip_benchmark} at line 131 appears before label definition at line 137 |
| 6 | 136 | Minor | P2 | Reference before definition: \ref{tab:b1b2_observed_rate} at line 136 appears before label definition at line 602 |
| 7 | 163 | Minor | P2 | Unreferenced label: \label{eq:gamma_gate} is never cited in text |
| 8 | 213 | Minor | P2 | Reference before definition: \ref{tab:attribution} at line 213 appears before label definition at line 217 |
| 9 | 234 | Minor | P2 | Reference before definition: \ref{tab:retrofit_latency} at line 234 appears before label definition at line 240 |
| 10 | 296 | Minor | P2 | Reference before definition: \ref{tab:vla_reskip_pareto} at line 296 appears before label definition at line 300 |
| 11 | 408 | Minor | P2 | Reference before definition: \ref{tab:retrofit_ablation} at line 408 appears before label definition at line 412 |
| 12 | 451 | Minor | P2 | Reference before definition: \ref{tab:data_mix_ablation} at line 451 appears before label definition at line 455 |
| 13 | 538 | Minor | P2 | Unreferenced label: \label{tab:reskip_position_ablation} is never cited in text |
| 14 | 573 | Minor | P2 | Reference before definition: \ref{tab:b1b2_decision_rule} at line 573 appears before label definition at line 578 |
| 15 | 656 | Minor | P2 | Unreferenced label: \label{fig:reskip_pareto_latency} is never cited in text |
| 16 | 680 | Minor | P2 | Reference before definition: \ref{tab:gromov_pruning_2b} at line 680 appears before label definition at line 686 |
| 17 | 685 | Minor | P2 | Reference before definition: \ref{tab:retrofit_pareto} at line 685 appears before label definition at line 720 |
| 18 | 734 | Minor | P2 | Reference before definition: \ref{tab:retrofit_pareto_v1_legacy} at line 734 appears before label definition at line 738 |
| 19 | 758 | Minor | P2 | Reference before definition: \ref{tab:retrofit_latency_full} at line 758 appears before label definition at line 762 |
| 20 | 801 | Minor | P2 | Unreferenced label: \label{tab:block_partition} is never cited in text |
| 21 | 827 | Minor | P2 | Unreferenced label: \label{tab:mmstar_subcat} is never cited in text |
| 22 | 855 | Minor | P2 | Unreferenced label: \label{tab:vla_seed_variance} is never cited in text |
| 23 | 878 | Minor | P2 | Unreferenced label: \label{tab:vla_landscape} is never cited in text |

### SENTENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | SENTENCE (Line 58, 41 words, 4 clauses) |
| 2 | — | Minor | P2 | Building on Attention Residuals (\attnres{})~\citep{chen2026attnres}, whose softmax routing over previous block outputs is computed \emph{before} each block runs, our main contribution is \retrofit{}, an identity-preserving  -gated residual-injection fine-tune that freezes the base and adds well under   new parameters. |
| 3 | — | Minor | P2 | Building on Attention Residuals (\attnres{})~\citep{chen2026attnres}. whose softmax routing over previous block outputs is computed \emph{before} each block runs. our main contribution is \retrofit{}. an identity-preserving  -gated residual-injection fine-tune that freezes the base and adds well under   new parameters.. |
| 4 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 5 | — | Minor | P2 | SENTENCE (Line 58, 42 words, 4 clauses) |
| 6 | — | Minor | P2 | We further show that the installed routing weights can be calibrated, with a complementary safety check, into a dynamic block-skip rule (\reskip{}) with no auxiliary head, and that the same signal supports adaptive depth on the action stream of the trained policy. |
| 7 | — | Minor | P2 | We further show that the installed routing weights can be calibrated. with a complementary safety check. into a dynamic block-skip rule (\reskip{}) with no auxiliary head. and that the same signal supports adaptive depth on the action stream of the trained policy.. |
| 8 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 9 | — | Minor | P2 | SENTENCE (Line 66, 48 words, 6 clauses) |
| 10 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} contrasts \emph{System~1} (fast, automatic) with \emph{System~2} (slow, deliberate); the same dichotomy has direct correlates in cortical processing, where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~\citep{lamme2000distinct, kar2019recurrent, kietzmann2019recurrence, spoerer2020recurrent}. |
| 11 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} contrasts \emph{System~1} (fast. automatic) with \emph{System~2} (slow. deliberate); the same dichotomy has direct correlates in cortical processing. where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~\citep{lamme2000distinct. kar2019recurrent. kietzmann2019recurrence. spoerer2020recurrent}.. |
| 12 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 13 | — | Minor | P2 | SENTENCE (Line 68, 29 words, 4 clauses) |
| 14 | — | Minor | P2 | Test-time compute scaling~\citep{openai2024o1, deepseek2025r1, snell2024scaling}, chain-of-thought prompting~\citep{wei2022chain}, and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering. |
| 15 | — | Minor | P2 | Test-time compute scaling~\citep{openai2024o1. deepseek2025r1. snell2024scaling}. chain-of-thought prompting~\citep{wei2022chain}. and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering.. |
| 16 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 17 | — | Minor | P2 | SENTENCE (Line 131, 43 words, 5 clauses) |
| 18 | — | Minor | P2 | On a 340M block-\attnres{} transformer ( ,  ,   blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}), \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}, HellaSwag~\citep{zellers2019hellaswag}, and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ). |
| 19 | — | Minor | P2 | On a 340M block-\attnres{} transformer (. blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}). \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}. HellaSwag~\citep{zellers2019hellaswag}. and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ).. |
| 20 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 21 | — | Minor | P2 | SENTENCE (Line 133, 51 words, 4 clauses) |
| 22 | — | Minor | P2 | \paragraph{Dynamic vs.\ static at the same skip rate.} Holding the \attnres{} weights fixed and varying only the runtime decision rule, an input-dependent rule that fires close to its calibrated   rate preserves LAMBADA at  , while static-position and random-toggle skip schedules at the same rate collapse to  --  on the same lm-eval suite. |
| 23 | — | Minor | P2 | \paragraph{Dynamic vs.\ static at the same skip rate.} Holding the \attnres{} weights fixed and varying only the runtime decision rule. an input-dependent rule that fires close to its calibrated   rate preserves LAMBADA at. while static-position and random-toggle skip schedules at the same rate collapse to  --  on the same lm-eval suite.. |
| 24 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 25 | — | Minor | P2 | SENTENCE (Line 156, 55 words, 4 clauses) |
| 26 | — | Minor | P2 | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality, (P2) emits a routing   informative enough to drive \reskip{}, and (P3) trains in a single short fine-tune over off-the-shelf SFT data, with the base frozen. |
| 27 | — | Minor | P2 | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality. (P2) emits a routing   informative enough to drive \reskip{}. and (P3) trains in a single short fine-tune over off-the-shelf SFT data. with the base frozen.. |
| 28 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 29 | — | Minor | P2 | SENTENCE (Line 169, 37 words, 4 clauses) |
| 30 | — | Minor | P2 | To make this consistent with \texttt{use\_cache=True}, on a skipped block we still run the per-layer K/V-producing slice ( ), which costs  --  of a full-layer forward, and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP. |
| 31 | — | Minor | P2 | To make this consistent with \texttt{use\_cache=True}. on a skipped block we still run the per-layer K/V-producing slice ( ). which costs  --  of a full-layer forward. and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP.. |
| 32 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 33 | — | Minor | P2 | SENTENCE (Line 173, 32 words, 5 clauses) |
| 34 | — | Minor | P2 | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\,\mathcal{L}_{\text{KL}}(\text{skip path}\,\|\,\text{teacher}) + \lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}}(\alpha) , |
| 35 | — | Minor | P2 | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\. \mathcal{L}_{\text{KL}}(\text{skip path}\. \|\. \text{teacher}) + \lambda_{\text{ent}}\. \mathcal{L}_{\text{ent}}(\alpha). |
| 36 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 37 | — | Minor | P2 | SENTENCE (Line 176, 51 words, 5 clauses) |
| 38 | — | Minor | P2 | where the full-path CE is computed on assistant tokens, while the skip-branch KL samples one block at random per step, runs the forward with it skipped, and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped). |
| 39 | — | Minor | P2 | where the full-path CE is computed on assistant tokens. while the skip-branch KL samples one block at random per step. runs the forward with it skipped. and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped).. |
| 40 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 41 | — | Minor | P2 | SENTENCE (Line 186, 45 words, 4 clauses) |
| 42 | — | Minor | P2 | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava},  k steps;   GPU-min on  B), used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts,  k steps), the canonical mix used for every result we report. |
| 43 | — | Minor | P2 | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava}. k steps;   GPU-min on  B). used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts. k steps). the canonical mix used for every result we report.. |
| 44 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 45 | — | Minor | P2 | SENTENCE (Line 186, 59 words, 9 clauses) |
| 46 | — | Minor | P2 | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}, MMMU~\citep{yue2024mmmu}, MMStar~\citep{chen2024mmstar}, AI2D~\citep{kembhavi2016ai2d}, OCRBench~\citep{liu2024ocrbench}, RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal, while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~ ,\, ). |
| 47 | — | Minor | P2 | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}. MMMU~\citep{yue2024mmmu}. MMStar~\citep{chen2024mmstar}. AI2D~\citep{kembhavi2016ai2d}. OCRBench~\citep{liu2024ocrbench}. RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal. while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~. \. ).. |
| 48 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 49 | — | Minor | P2 | SENTENCE (Line 234, 23 words, 5 clauses) |
| 50 | — | Minor | P2 | First, the canonical   partition halves the per-token router count vs  , closing   of the eager-vs-eager gap on its own ( , seq~ , cache, H100/bf16; Tab.~ ). |
| 51 | — | Minor | P2 | First. the canonical   partition halves the per-token router count vs. closing   of the eager-vs-eager gap on its own (. seq~. cache. H100/bf16; Tab.~ ).. |
| 52 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 53 | — | Minor | P2 | SENTENCE (Line 236, 64 words, 5 clauses) |
| 54 | — | Minor | P2 | \paragraph{Two deployment modes.} We separate three speed claims that are easy to conflate: (i)~at the  M from-scratch scale, \reskip{} on \attnres{} reaches   wall-clock with zero benchmark drop (\S , mechanism validation); (ii)~at the Qwen3-VL retrofit scale, the compiled full-depth retrofit runs at   \texttt{base\_compiled} (this section, accuracy gain at iso-cost); (iii)~eager-mode \reskip{} on the retrofit contributes a further  --  standalone (Appendix~ ). |
| 55 | — | Minor | P2 | \paragraph{Two deployment modes.} We separate three speed claims that are easy to conflate: (i)~at the  M from-scratch scale. \reskip{} on \attnres{} reaches   wall-clock with zero benchmark drop (\S. mechanism validation); (ii)~at the Qwen3-VL retrofit scale. the compiled full-depth retrofit runs at   \texttt{base\_compiled} (this section. accuracy gain at iso-cost); (iii)~eager-mode \reskip{} on the retrofit contributes a further  --  standalone (Appendix~ ).. |
| 56 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 57 | — | Minor | P2 | SENTENCE (Line 255, 15 words, 6 clauses) |
| 58 | — | Minor | P2 | Full block-partition sweep, v1 v2 v3 trail, and  -free / informed-init / observer-only baselines are in Appendices~ ,\, ,\, . |
| 59 | — | Minor | P2 | Full block-partition sweep. v1 v2 v3 trail. and  -free / informed-init / observer-only baselines are in Appendices~. \. \. .. |
| 60 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 61 | — | Minor | P2 | SENTENCE (Line 265, 48 words, 5 clauses) |
| 62 | — | Minor | P2 | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT, baseline), \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit,   throughout), and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init,   ramps   on VLA data). |
| 63 | — | Minor | P2 | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT. baseline). \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit. throughout). and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init. ramps   on VLA data).. |
| 64 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 65 | — | Minor | P2 | SENTENCE (Line 328, 30 words, 11 clauses) |
| 66 | — | Minor | P2 | The per-block action-drift sweep that picks   per scale, the cross-modality transfer failure, the per-seed breakdown, the pipeline pitfall, and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~ ,\, ,\, ,\, . |
| 67 | — | Minor | P2 | The per-block action-drift sweep that picks   per scale. the cross-modality transfer failure. the per-seed breakdown. the pipeline pitfall. and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~. \. \. \. .. |
| 68 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 69 | — | Minor | P2 | SENTENCE (Line 335, 31 words, 4 clauses) |
| 70 | — | Minor | P2 | For test-time compute, o1~\citep{openai2024o1}, DeepSeek-R1~\citep{deepseek2025r1}, chain-of-thought~\citep{wei2022chain}, and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed. |
| 71 | — | Minor | P2 | For test-time compute. o1~\citep{openai2024o1}. DeepSeek-R1~\citep{deepseek2025r1}. chain-of-thought~\citep{wei2022chain}. and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed.. |
| 72 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 73 | — | Minor | P2 | SENTENCE (Line 335, 24 words, 6 clauses) |
| 74 | — | Minor | P2 | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive, dehghani2018universal}, CALM~\citep{schuster2022confident}, Mixture-of-Depths~\citep{raposo2024mixture}, LayerSkip~\citep{elhoushi2024layerskip}, and static pruning~\citep{gromov2024unreasonable, men2024shortgpt}. |
| 75 | — | Minor | P2 | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive. dehghani2018universal}. CALM~\citep{schuster2022confident}. Mixture-of-Depths~\citep{raposo2024mixture}. LayerSkip~\citep{elhoushi2024layerskip}. and static pruning~\citep{gromov2024unreasonable. men2024shortgpt}.. |
| 76 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 77 | — | Minor | P2 | SENTENCE (Line 335, 14 words, 4 clauses) |
| 78 | — | Minor | P2 | These add a halting head, exit classifier, router, speculative decoder, or static removal rule. |
| 79 | — | Minor | P2 | These add a halting head. exit classifier. router. speculative decoder. or static removal rule.. |
| 80 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 81 | — | Minor | P2 | SENTENCE (Line 337, 28 words, 4 clauses) |
| 82 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}. |
| 83 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps. while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct. kar2019recurrent. dicarlo2012does}.. |
| 84 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 85 | — | Minor | P2 | SENTENCE (Line 337, 30 words, 5 clauses) |
| 86 | — | Minor | P2 | We take this as a design target, not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical, buschman2007topdown, larkum2013cellular} motivate intrinsic routing, while our implementation is a transformer retrofit. |
| 87 | — | Minor | P2 | We take this as a design target. not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical. buschman2007topdown. larkum2013cellular} motivate intrinsic routing. while our implementation is a transformer retrofit.. |
| 88 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 89 | — | Minor | P2 | SENTENCE (Line 339, 32 words, 6 clauses) |
| 90 | — | Minor | P2 | In VLA, RT-2~\citep{brohan2023rt}, Octo~\citep{team2024octo}, OpenVLA~\citep{kim2024openvla}, and Pi0~\citep{black2024pi0} established the paradigm, while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression. |
| 91 | — | Minor | P2 | In VLA. RT-2~\citep{brohan2023rt}. Octo~\citep{team2024octo}. OpenVLA~\citep{kim2024openvla}. and Pi0~\citep{black2024pi0} established the paradigm. while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression.. |
| 92 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 93 | — | Minor | P2 | SENTENCE (Line 434, 28 words, 5 clauses) |
| 94 | — | Minor | P2 | \paragraph{Over-training signature at  .} Holding ( ,   VLM, fast ramp) fixed and varying only the total step count, MMBench is strictly monotonic in steps:  k   (preserved),  k   ( 14pp),  k   ( 21pp). |
| 95 | — | Minor | P2 | \paragraph{Over-training signature at  .} Holding (. VLM. fast ramp) fixed and varying only the total step count. MMBench is strictly monotonic in steps:  k   (preserved). k   ( 14pp). k   ( 21pp).. |
| 96 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 97 | — | Minor | P2 | SENTENCE (Line 442, 28 words, 4 clauses) |
| 98 | — | Minor | P2 | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc), block~  severe ( ), blocks~ ,  ,   are safest (  to  ). |
| 99 | — | Minor | P2 | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc). block~  severe ( ). blocks~. are safest (  to  ).. |
| 100 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 101 | — | Minor | P2 | SENTENCE (Line 451, 74 words, 1 clauses) |
| 102 | — | Minor | P2 | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| 103 | — | Minor | P2 | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| 104 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 105 | — | Minor | P2 | SENTENCE (Line 510, 30 words, 4 clauses) |
| 106 | — | Minor | P2 | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \,ms at seq   on  H100, so a  --  block-level \attnres{} overhead would push past common  \,Hz /  \,Hz robotic control budgets. |
| 107 | — | Minor | P2 | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \. ms at seq   on  H100. so a  --  block-level \attnres{} overhead would push past common  \. Hz /  \. Hz robotic control budgets.. |
| 108 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 109 | — | Minor | P2 | SENTENCE (Line 758, 67 words, 4 clauses) |
| 110 | — | Minor | P2 | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls, the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward, \S ) tracks the VLM retrofit within   at both partitions, and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved. |
| 111 | — | Minor | P2 | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls. the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward. \S ) tracks the VLM retrofit within   at both partitions. and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved.. |
| 112 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 113 | — | Minor | P2 | SENTENCE (Line 786, 53 words, 4 clauses) |
| 114 | — | Minor | P2 | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at  , but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison, which forces a CPU sync and breaks the captured CUDA graph. |
| 115 | — | Minor | P2 | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at. but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison. which forces a CPU sync and breaks the captured CUDA graph.. |
| 116 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 117 | — | Minor | P2 | SENTENCE (Line 925, 17 words, 5 clauses) |
| 118 | — | Minor | P2 | Sections~ --  and the appendix specify model scales, block partitions, training steps, datasets, calibration rules, and evaluation protocols. |
| 119 | — | Minor | P2 | Sections~ --  and the appendix specify model scales. block partitions. training steps. datasets. calibration rules. and evaluation protocols.. |
| 120 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |

## Pre-Submission Checklist

- [ ] No placeholder text (TODO, FIXME, XXX) — Found on lines: [43]
- [ ] All figures referenced in text — Unreferenced: {'fig:overview', 'fig:reskip_pareto_latency'}
- [ ] All tables referenced in text — Unreferenced: {'tab:vla_seed_variance', 'tab:vla_landscape', 'tab:block_partition', 'tab:reskip_position_ablation', 'tab:mmstar_subcat'}
- [ ] Anonymous submission (blind review check) — Author information detected — verify if blind review required
- [x] Consistent math notation
- [ ] Acronyms defined on first use — Potentially undefined: ['ENDIF', 'ACT', 'ARC', 'SDPA', 'VLM']
