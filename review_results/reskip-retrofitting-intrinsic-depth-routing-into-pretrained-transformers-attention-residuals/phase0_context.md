# Phase 0: Automated Audit Results

**File**: `/home/user01/Minko/reskip2/reskip/paper-codex/main.tex` | **Language**: en | **Mode**: quick-audit
**Venue**: NeurIPS2026
**Generated**: 2026-05-01T05:25:11.313658

## Issue Summary (173 total)
- Critical: 7
- Major: 9
- Minor: 157

## Issues by Module

### BIB

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Check: /home/user01/Minko/reskip2/reskip/paper-codex/main.tex |
| 2 | — | Minor | P2 | PASS |
| 3 | — | Minor | P2 | entries: 0 |
| 4 | — | Minor | P2 | entries: 0 |

### CITATIONS

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 66 | Critical | P0 | Citation stacking: 7 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 2 | 68 | Critical | P0 | Citation stacking: 5 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 3 | 70 | Critical | P0 | Citation stacking: 7 citations clustered in one sentence without individual discussion (section: introduction). Max 2 clustered citations allowed. |
| 4 | 320 | Critical | P0 | Citation stacking: 11 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 5 | 322 | Critical | P0 | Citation stacking: 10 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |
| 6 | 324 | Critical | P0 | Citation stacking: 8 citations clustered in one sentence without individual discussion (section: related). Max 2 clustered citations allowed. |

### DEAI

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Use --analyze for full analysis |

### EXPERIMENT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 229 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 2 | 274 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 3 | 276 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 4 | 330 | Critical | P0 | Conclusion overreaches the reported evidence; avoid universal or guarantee-style claims. |
| 5 | 180 | Minor | P2 | No statistical significance, variance, or confidence information is mentioned. |
| 6 | 757 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 7 | 758 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 8 | 759 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 9 | 778 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 10 | 853 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 11 | 326 | Major | P1 | Discussion may lack depth: low ratio of explanatory/attribution language (10/451 lines). Add causal analysis explaining why results occur. |

### FIGURES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | figures in /home/user01/Minko/reskip2/reskip/paper-codex/main.tex... |
| 2 | — | Minor | P2 | 4 figures. |
| 3 | — | Minor | P2 | Line 83: Image not found: reskip_main_figure_v2.png |
| 4 | — | Minor | P2 | Line 126: Image not found: reskip_routing_importance_vs_ablation.pdf |
| 5 | — | Minor | P2 | Line 631: Image not found: reskip_pareto.pdf |
| 6 | — | Minor | P2 | Line 635: Image not found: reskip_latency.pdf |
| 7 | — | Minor | P2 | ⚠️ Found 4 potential issues. |

### FORMAT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | ============================================================ |
| 2 | — | Minor | P2 | Format Check Report |
| 3 | — | Minor | P2 | ============================================================ |
| 4 | — | Minor | P2 | /home/user01/Minko/reskip2/reskip/paper-codex/main.tex |
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
| 5 | 131 | Minor | P2 | Reference before definition: \ref{tab:reskip_benchmark} at line 131 appears before label definition at line 135 |
| 6 | 162 | Minor | P2 | Unreferenced label: \label{eq:gamma_gate} is never cited in text |
| 7 | 216 | Minor | P2 | Reference before definition: \ref{tab:retrofit_latency} at line 216 appears before label definition at line 220 |
| 8 | 281 | Minor | P2 | Reference before definition: \ref{tab:vla_reskip_pareto} at line 281 appears before label definition at line 285 |
| 9 | 391 | Minor | P2 | Reference before definition: \ref{tab:retrofit_ablation} at line 391 appears before label definition at line 395 |
| 10 | 434 | Minor | P2 | Reference before definition: \ref{tab:data_mix_ablation} at line 434 appears before label definition at line 438 |
| 11 | 521 | Minor | P2 | Unreferenced label: \label{tab:reskip_position_ablation} is never cited in text |
| 12 | 556 | Minor | P2 | Reference before definition: \ref{tab:b1b2_decision_rule} at line 556 appears before label definition at line 561 |
| 13 | 557 | Minor | P2 | Reference before definition: \ref{tab:b1b2_observed_rate} at line 557 appears before label definition at line 585 |
| 14 | 639 | Minor | P2 | Unreferenced label: \label{fig:reskip_pareto_latency} is never cited in text |
| 15 | 663 | Minor | P2 | Reference before definition: \ref{tab:gromov_pruning_2b} at line 663 appears before label definition at line 669 |
| 16 | 668 | Minor | P2 | Reference before definition: \ref{tab:retrofit_pareto} at line 668 appears before label definition at line 703 |
| 17 | 717 | Minor | P2 | Reference before definition: \ref{tab:retrofit_pareto_v1_legacy} at line 717 appears before label definition at line 721 |
| 18 | 741 | Minor | P2 | Reference before definition: \ref{tab:retrofit_latency_full} at line 741 appears before label definition at line 745 |
| 19 | 784 | Minor | P2 | Unreferenced label: \label{tab:block_partition} is never cited in text |
| 20 | 810 | Minor | P2 | Unreferenced label: \label{tab:mmstar_subcat} is never cited in text |
| 21 | 838 | Minor | P2 | Unreferenced label: \label{tab:vla_seed_variance} is never cited in text |
| 22 | 863 | Minor | P2 | Unreferenced label: \label{tab:vla_landscape} is never cited in text |

### SENTENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | SENTENCE (Line 58, 41 words, 5 clauses) |
| 2 | — | Minor | P2 | We build on Attention Residuals, whose routing weights over previous block outputs are computed before the next block executes, and introduce \retrofit{}, a  -gated residual-injection fine-tune that installs this pathway while freezing the base model and adding well under   new parameters. |
| 3 | — | Minor | P2 | We build on Attention Residuals. whose routing weights over previous block outputs are computed before the next block executes. and introduce \retrofit{}. a  -gated residual-injection fine-tune that installs this pathway while freezing the base model and adding well under   new parameters.. |
| 4 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 5 | — | Minor | P2 | SENTENCE (Line 66, 39 words, 5 clauses) |
| 6 | — | Minor | P2 | Kahneman's dual-process framework~\citep{kahneman2011thinking} distinguishes fast and deliberate processing, and biological vision offers a concrete computational analogue: easy stimuli can be recognised through shallow feedforward sweeps, while harder stimuli recruit recurrent and deeper circuits~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}. |
| 7 | — | Minor | P2 | Kahneman's dual-process framework~\citep{kahneman2011thinking} distinguishes fast and deliberate processing. and biological vision offers a concrete computational analogue: easy stimuli can be recognised through shallow feedforward sweeps. while harder stimuli recruit recurrent and deeper circuits~\citep{lamme2000distinct. kar2019recurrent. dicarlo2012does}.. |
| 8 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 9 | — | Minor | P2 | SENTENCE (Line 68, 29 words, 4 clauses) |
| 10 | — | Minor | P2 | Test-time compute scaling~\citep{openai2024o1, deepseek2025r1, snell2024scaling}, chain-of-thought prompting~\citep{wei2022chain}, and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering. |
| 11 | — | Minor | P2 | Test-time compute scaling~\citep{openai2024o1. deepseek2025r1. snell2024scaling}. chain-of-thought prompting~\citep{wei2022chain}. and self-thought methods~\citep{zelikman2024quietstar} spend more compute on hard problems by emitting more reasoning tokens before answering.. |
| 12 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 13 | — | Minor | P2 | SENTENCE (Line 131, 43 words, 5 clauses) |
| 14 | — | Minor | P2 | On a 340M block-\attnres{} transformer ( ,  ,   blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}), \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}, HellaSwag~\citep{zellers2019hellaswag}, and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ). |
| 15 | — | Minor | P2 | On a 340M block-\attnres{} transformer (. blocks; FineWeb-Edu 100BT~\citep{penedo2024fineweb}; \texttt{lm-eval-harness}~\citep{eval-harness}). \reskip{} at   reaches \textbf{  wall-clock at zero benchmark drop} on LAMBADA~\citep{paperno2016lambada}. HellaSwag~\citep{zellers2019hellaswag}. and ARC-E/C~\citep{clark2018arc} (Table~ ; Pareto and latency curves in Appendix~ ).. |
| 16 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 17 | — | Minor | P2 | SENTENCE (Line 155, 55 words, 4 clauses) |
| 18 | — | Minor | P2 | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality, (P2) emits a routing   informative enough to drive \reskip{}, and (P3) trains in a single short fine-tune over off-the-shelf SFT data, with the base frozen. |
| 19 | — | Minor | P2 | The retrofit target is constrained by the problem in the introduction: take a transformer pretrained in the standard-residual regime and produce an \attnres{}-capable model that (P1) preserves benchmark quality. (P2) emits a routing   informative enough to drive \reskip{}. and (P3) trains in a single short fine-tune over off-the-shelf SFT data. with the base frozen.. |
| 20 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 21 | — | Minor | P2 | SENTENCE (Line 168, 37 words, 4 clauses) |
| 22 | — | Minor | P2 | To make this consistent with \texttt{use\_cache=True}, on a skipped block we still run the per-layer K/V-producing slice ( ), which costs  --  of a full-layer forward, and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP. |
| 23 | — | Minor | P2 | To make this consistent with \texttt{use\_cache=True}. on a skipped block we still run the per-layer K/V-producing slice ( ). which costs  --  of a full-layer forward. and skip \texttt{q\_proj} / attention / \texttt{o\_proj} / MLP.. |
| 24 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 25 | — | Minor | P2 | SENTENCE (Line 172, 32 words, 5 clauses) |
| 26 | — | Minor | P2 | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\,\mathcal{L}_{\text{KL}}(\text{skip path}\,\|\,\text{teacher}) + \lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}}(\alpha) , |
| 27 | — | Minor | P2 | \mathcal{L} = \mathcal{L}_{\text{CE}}(\text{full path}) + \lambda_{\text{kl}}\. \mathcal{L}_{\text{KL}}(\text{skip path}\. \|\. \text{teacher}) + \lambda_{\text{ent}}\. \mathcal{L}_{\text{ent}}(\alpha). |
| 28 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 29 | — | Minor | P2 | SENTENCE (Line 175, 51 words, 5 clauses) |
| 30 | — | Minor | P2 | where the full-path CE is computed on assistant tokens, while the skip-branch KL samples one block at random per step, runs the forward with it skipped, and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped). |
| 31 | — | Minor | P2 | where the full-path CE is computed on assistant tokens. while the skip-branch KL samples one block at random per step. runs the forward with it skipped. and pulls the resulting logits toward a frozen pretrained teacher (so the surrogate   is useful both when the block runs and when it is skipped).. |
| 32 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 33 | — | Minor | P2 | SENTENCE (Line 185, 45 words, 4 clauses) |
| 34 | — | Minor | P2 | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava},  k steps;   GPU-min on  B), used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts,  k steps), the canonical mix used for every result we report. |
| 35 | — | Minor | P2 | We compare two data mixes: \textbf{v1} ( /  UltraChat- k~\citep{ding2023ultrachat} + LLaVA-Instruct-VSFT~\citep{liu2023llava}. k steps;   GPU-min on  B). used as a LoRA-comparable narrow-mix control; and \textbf{v3} (  LLaVA-OneVision-Data~\citep{li2024llavaonevision} +   UltraChat +   NuminaMath +   OpenThoughts. k steps). the canonical mix used for every result we report.. |
| 36 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 37 | — | Minor | P2 | SENTENCE (Line 185, 59 words, 9 clauses) |
| 38 | — | Minor | P2 | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}, MMMU~\citep{yue2024mmmu}, MMStar~\citep{chen2024mmstar}, AI2D~\citep{kembhavi2016ai2d}, OCRBench~\citep{liu2024ocrbench}, RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal, while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~ ,\, ). |
| 39 | — | Minor | P2 | Evaluation uses LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} for text and six \texttt{lmms-eval} benchmarks for VLM (MMBench~\citep{liu2023mmbench}. MMMU~\citep{yue2024mmmu}. MMStar~\citep{chen2024mmstar}. AI2D~\citep{kembhavi2016ai2d}. OCRBench~\citep{liu2024ocrbench}. RealWorldQA~\citep{grok2024realworldqa}); text-side \reskip{} eligible sets are selected from per-block static removal. while VLA eligible sets are selected from action-drift sweeps on the trained policy (Appendices~. \. ).. |
| 40 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 41 | — | Minor | P2 | SENTENCE (Line 216, 23 words, 5 clauses) |
| 42 | — | Minor | P2 | First, the canonical   partition halves the per-token router count vs  , closing   of the eager-vs-eager gap on its own ( , seq~ , cache, H100/bf16; Tab.~ ). |
| 43 | — | Minor | P2 | First. the canonical   partition halves the per-token router count vs. closing   of the eager-vs-eager gap on its own (. seq~. cache. H100/bf16; Tab.~ ).. |
| 44 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 45 | — | Minor | P2 | SENTENCE (Line 235, 15 words, 6 clauses) |
| 46 | — | Minor | P2 | Full block-partition sweep, v1 v2 v3 trail, and  -free / informed-init / observer-only baselines are in Appendices~ ,\, ,\, . |
| 47 | — | Minor | P2 | Full block-partition sweep. v1 v2 v3 trail. and  -free / informed-init / observer-only baselines are in Appendices~. \. \. .. |
| 48 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 49 | — | Minor | P2 | SENTENCE (Line 245, 48 words, 5 clauses) |
| 50 | — | Minor | P2 | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT, baseline), \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit,   throughout), and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init,   ramps   on VLA data). |
| 51 | — | Minor | P2 | Three paths share the same head and data: \textbf{Path 0} (stock backbone OFT. baseline). \textbf{Path B} (warm-start from our canonical   v3  k VLM retrofit. throughout). and \textbf{Path C} (same architecture as Path B but \emph{no VLM retrofit}; routers / adapters random-init. ramps   on VLA data).. |
| 52 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 53 | — | Minor | P2 | SENTENCE (Line 313, 30 words, 11 clauses) |
| 54 | — | Minor | P2 | The per-block action-drift sweep that picks   per scale, the cross-modality transfer failure, the per-seed breakdown, the pipeline pitfall, and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~ ,\, ,\, ,\, . |
| 55 | — | Minor | P2 | The per-block action-drift sweep that picks   per scale. the cross-modality transfer failure. the per-seed breakdown. the pipeline pitfall. and a public-VLA landscape table (  / OpenVLA / SpatialVLA on LIBERO) live in Appendices~. \. \. \. .. |
| 56 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 57 | — | Minor | P2 | SENTENCE (Line 320, 31 words, 4 clauses) |
| 58 | — | Minor | P2 | For test-time compute, o1~\citep{openai2024o1}, DeepSeek-R1~\citep{deepseek2025r1}, chain-of-thought~\citep{wei2022chain}, and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed. |
| 59 | — | Minor | P2 | For test-time compute. o1~\citep{openai2024o1}. DeepSeek-R1~\citep{deepseek2025r1}. chain-of-thought~\citep{wei2022chain}. and optimal test-time compute analyses~\citep{snell2024scaling} allocate compute by emitting more reasoning \emph{tokens}; per-token depth remains fixed.. |
| 60 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 61 | — | Minor | P2 | SENTENCE (Line 320, 24 words, 6 clauses) |
| 62 | — | Minor | P2 | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive, dehghani2018universal}, CALM~\citep{schuster2022confident}, Mixture-of-Depths~\citep{raposo2024mixture}, LayerSkip~\citep{elhoushi2024layerskip}, and static pruning~\citep{gromov2024unreasonable, men2024shortgpt}. |
| 63 | — | Minor | P2 | Depth-axis methods include ACT / Universal Transformers~\citep{graves2016adaptive. dehghani2018universal}. CALM~\citep{schuster2022confident}. Mixture-of-Depths~\citep{raposo2024mixture}. LayerSkip~\citep{elhoushi2024layerskip}. and static pruning~\citep{gromov2024unreasonable. men2024shortgpt}.. |
| 64 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 65 | — | Minor | P2 | SENTENCE (Line 320, 14 words, 4 clauses) |
| 66 | — | Minor | P2 | These add a halting head, exit classifier, router, speculative decoder, or static removal rule. |
| 67 | — | Minor | P2 | These add a halting head. exit classifier. router. speculative decoder. or static removal rule.. |
| 68 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 69 | — | Minor | P2 | SENTENCE (Line 322, 28 words, 4 clauses) |
| 70 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}. |
| 71 | — | Minor | P2 | The dual-process distinction~\citep{kahneman2011thinking} maps naturally onto visual processing: easy stimuli can be handled by feedforward sweeps. while harder stimuli recruit recurrent circuits~\citep{lamme2000distinct. kar2019recurrent. dicarlo2012does}.. |
| 72 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 73 | — | Minor | P2 | SENTENCE (Line 322, 30 words, 5 clauses) |
| 74 | — | Minor | P2 | We take this as a design target, not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical, buschman2007topdown, larkum2013cellular} motivate intrinsic routing, while our implementation is a transformer retrofit. |
| 75 | — | Minor | P2 | We take this as a design target. not a cognitive-faithfulness claim: cortical microcircuits and dendritic association~\citep{bastos2012canonical. buschman2007topdown. larkum2013cellular} motivate intrinsic routing. while our implementation is a transformer retrofit.. |
| 76 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 77 | — | Minor | P2 | SENTENCE (Line 324, 32 words, 6 clauses) |
| 78 | — | Minor | P2 | In VLA, RT-2~\citep{brohan2023rt}, Octo~\citep{team2024octo}, OpenVLA~\citep{kim2024openvla}, and Pi0~\citep{black2024pi0} established the paradigm, while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression. |
| 79 | — | Minor | P2 | In VLA. RT-2~\citep{brohan2023rt}. Octo~\citep{team2024octo}. OpenVLA~\citep{kim2024openvla}. and Pi0~\citep{black2024pi0} established the paradigm. while efficiency work has focused mainly on action chunking~\citep{zhao2023learning} and post-hoc compression.. |
| 80 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 81 | — | Minor | P2 | SENTENCE (Line 417, 28 words, 5 clauses) |
| 82 | — | Minor | P2 | \paragraph{Over-training signature at  .} Holding ( ,   VLM, fast ramp) fixed and varying only the total step count, MMBench is strictly monotonic in steps:  k   (preserved),  k   ( 14pp),  k   ( 21pp). |
| 83 | — | Minor | P2 | \paragraph{Over-training signature at  .} Holding (. VLM. fast ramp) fixed and varying only the total step count. MMBench is strictly monotonic in steps:  k   (preserved). k   ( 14pp). k   ( 21pp).. |
| 84 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 85 | — | Minor | P2 | SENTENCE (Line 425, 28 words, 4 clauses) |
| 86 | — | Minor | P2 | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc), block~  severe ( ), blocks~ ,  ,   are safest (  to  ). |
| 87 | — | Minor | P2 | Running the  k Qwen3-VL-2B retrofit (H\_r256\_5k) with a single block statically removed ( ) and measuring LAMBADA at  : block~  is catastrophic (  acc). block~  severe ( ). blocks~. are safest (  to  ).. |
| 88 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 89 | — | Minor | P2 | SENTENCE (Line 434, 74 words, 1 clauses) |
| 90 | — | Minor | P2 | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| 91 | — | Minor | P2 | Three intermediate cells show how the mix landscape partitions: v1 (narrow VL: UltraChat   LLaVA-Instruct-VSFT) is the initial canonical and preserves base within noise at  k steps but \emph{collapses} when trained longer; v2 (aggressive math-CoT:   NuminaMath/OpenThoughts/OpenMath2   VL) recovers MMStar reasoning subtasks on 2B but \emph{crashes AI2D by  pp} (on 2B) and the global VL on 4B; v3 (LLaVA-OneVision-anchored) removes both failure modes and is strictly positive on VL at both scales. |
| 92 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 93 | — | Minor | P2 | SENTENCE (Line 493, 30 words, 4 clauses) |
| 94 | — | Minor | P2 | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \,ms at seq   on  H100, so a  --  block-level \attnres{} overhead would push past common  \,Hz /  \,Hz robotic control budgets. |
| 95 | — | Minor | P2 | At the  B VL scale a stock Qwen3-VL-2B forward already takes  \. ms at seq   on  H100. so a  --  block-level \attnres{} overhead would push past common  \. Hz /  \. Hz robotic control budgets.. |
| 96 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 97 | — | Minor | P2 | SENTENCE (Line 741, 67 words, 4 clauses) |
| 98 | — | Minor | P2 | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls, the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward, \S ) tracks the VLM retrofit within   at both partitions, and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved. |
| 99 | — | Minor | P2 | Tab.~  below shows the full set: the legacy   partition (  blocks at  B) carries a   structural cost over stock Qwen3-VL-2B because each token pays   router calls. the VLA in-backbone forward (the same retrofit run inside the OFT trainer's backbone forward. \S ) tracks the VLM retrofit within   at both partitions. and \texttt{torch.compile} closes the   residual to   but is more dramatic on   where the router count is halved.. |
| 100 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 101 | — | Minor | P2 | SENTENCE (Line 769, 53 words, 4 clauses) |
| 102 | — | Minor | P2 | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at  , but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison, which forces a CPU sync and breaks the captured CUDA graph. |
| 103 | — | Minor | P2 | \paragraph{Skip on top of compile.} Eager-mode \reskip{} contributes a further  --  when used on its own at. but cannot currently be composed with \texttt{torch.compile} in a single forward: the dyn-skip rule requires an \texttt{.item()} on a per-token threshold comparison. which forces a CPU sync and breaks the captured CUDA graph.. |
| 104 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 105 | — | Minor | P2 | SENTENCE (Line 907, 22 words, 4 clauses) |
| 106 | — | Minor | P2 | The abstract, introduction, and conclusions state the main claims and separate the retrofit contribution from calibrated skipping, VLA transfer, and systems characterization. |
| 107 | — | Minor | P2 | The abstract. introduction. and conclusions state the main claims and separate the retrofit contribution from calibrated skipping. VLA transfer. and systems characterization.. |
| 108 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |
| 109 | — | Minor | P2 | SENTENCE (Line 910, 17 words, 5 clauses) |
| 110 | — | Minor | P2 | Sections~ --  and the appendix specify model scales, block partitions, training steps, datasets, calibration rules, and evaluation protocols. |
| 111 | — | Minor | P2 | Sections~ --  and the appendix specify model scales. block partitions. training steps. datasets. calibration rules. and evaluation protocols.. |
| 112 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. |

## Pre-Submission Checklist

- [ ] No placeholder text (TODO, FIXME, XXX) — Found on lines: [43]
- [ ] All figures referenced in text — Unreferenced: {'fig:overview', 'fig:reskip_pareto_latency'}
- [ ] All tables referenced in text — Unreferenced: {'tab:block_partition', 'tab:reskip_position_ablation', 'tab:vla_seed_variance', 'tab:vla_landscape', 'tab:mmstar_subcat'}
- [ ] Anonymous submission (blind review check) — Author information detected — verify if blind review required
- [x] Consistent math notation
- [ ] Acronyms defined on first use — Potentially undefined: ['IF', 'OFT', 'VLA', 'RWQA', 'AND']
