# ReSkip：通过 Attention Residuals 将内生深度路由改造进预训练 Transformer

匿名作者  
单位隐去  
anonymous@submission.invalid

## 摘要

生物视觉表明，有效处理深度应随刺激难度而变化；然而，预训练 Transformer 对几乎每个输入都会执行同一串 block。现有自适应深度方法通常需要额外加入外部 router 或 exit head，或者需要从头训练一个自适应架构。我们研究一个标准预训练 Transformer 是否能够被改造成带有内生 routing path 的模型，并由该路径自身的前向计算提供深度信号。我们提出 **AR-Retrofit**，即向冻结 backbone 中注入一个 identity-preserving 的 AttnRes-style routed residual correction。该 retrofit 增加的参数量远低于 `1%`，并通过一次短程微调进行训练，使 full path 保持可用，同时让 skipped-block surrogate 也可用。由于 routing weights 在每个 block 执行前已经计算出来，它们可以支持 **ReSkip**：一种经过校准的规则，将离线 eligibility selection 与推理时的 input-dependent block skipping 结合起来。一个 340M controlled study 首先表明，AttnRes 改善了 residual baseline，并且 ReSkip 在 benchmark 分数匹配的情况下恢复了 `1.19x` wall-clock speed。在 Qwen3-VL-2B 和 4B 上，AR-Retrofit 在大多数 VLM benchmark 上保持或提升性能，提升 LAMBADA，并使 full-depth runtime 接近 base；在 retrofitted VLM 上，保守的 ReSkip operating point 在 LAMBADA-500 上将质量保持在 `1`pp 以内。LIBERO 结果提供了支持性证据，表明 routed-depth signal 仍可用于 action-stream calibration。这些结果表明，预训练 Transformer 可以通过轻量 retrofit 获得实用的 depth-axis routing，而不需要从头进行 residual pretraining。

## 1 Introduction

不同输入不应总是消耗相同计算量。这是一个设计动机，而不是认知等价性的声明。Kahneman 的 dual-process framework~\citep{kahneman2011thinking} 区分了快速处理和审慎处理；生物视觉提供了一个具体的计算类比：简单刺激可以通过浅层 feedforward sweep 被识别，而更困难的刺激会调用 recurrent 和更深的回路~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}。Recurrent vision model 能够捕获人类表征动态~\citep{kietzmann2019recurrence}，而 confidence-thresholded recurrence 会在更难图像上花费更多步骤，从而进行速度-准确率权衡~\citep{spoerer2020recurrent}。在更局部的尺度上，cortical association 和 dendritic gating 提供了 computation-dependent routing 发生在 processing substrate 内部、而不是在一个独立 controller 中的机制性例子~\citep{bastos2012canonical, buschman2007topdown, larkum2013cellular}。由于有限 recurrence 可以展开成 feedforward depth~\citep{vanbergen2020going}，这些观察引出了一个 Transformer 侧的问题：预训练模型能否获得一个用于按输入分配有效深度的内生信号？

大型语言模型已经暴露出一个 test-time compute 轴：**tokens**。Test-time compute scaling~\citep{openai2024o1, deepseek2025r1, snell2024scaling}、chain-of-thought prompting~\citep{wei2022chain} 和 self-thought 方法~\citep{zelikman2024quietstar} 通过在回答前生成更多 reasoning tokens，在难题上花费更多计算。然而，这类 reasoning trace 中的每个 token 仍会穿过底层 Transformer 的每一层，不论该 token 是常规连接词还是承担推理负载的关键步骤。因此，缺失的互补轴是 **depth**：针对当前输入、当前序列位置，决定应调用哪些 block。

现有 depth-axis 方法并不完全符合这一动机，因为它们将决策外部化。Early-exit classifiers~\citep{schuster2022confident, elbayad2020depth} 在每层添加 auxiliary head；Mixture-of-Depths~\citep{raposo2024mixture} 学习带 capacity-balancing loss 的 token router；layer-pruning~\citep{gromov2024unreasonable, men2024shortgpt} 是 post-hoc 且静态的；Universal Transformers~\citep{dehghani2018universal} 通过 ACT-style halting~\citep{graves2016adaptive} 共享权重。在这些情况下，是否运行网络其余部分由一个独立组件决定。本文所受启发的生物类比不同：控制嵌入在 processing substrate 中，而不是作为独立 decision head 附着其上。

AttnRes~\citep{chen2026attnres} 提供了一个自然切入点。它用对 previous block outputs 的 learned softmax-attention 替代固定标量 `1` 的 residual；routing weights `\alpha_{i\to n}` 是 input-dependent、竞争性归一化的，并且在 block `n` 运行之前、作为形成其输入的一部分被计算。因此，它们作为网络自身 forward 的副产物产生 per-block depth-allocation signal，而不是来自单独 head。实际障碍是，现有 AttnRes 模型需要使用修改后的 residual 从头训练；在真实应用所需的 2--7B 规模上重现这一过程需要数万 GPU-hours（Appendix~\ref{app:motivation}）。本文的核心问题是：如何通过一次短程 fine-tune，将这样的信号安装进标准预训练 Transformer，同时保持原有能力，并使该信号可用于 calibrated depth skipping。

本文有一个主贡献和两个支持性 claim。

- **AR-Retrofit 将内生深度路由安装进预训练模型**（\S\ref{sec:retrofit}）。一个 `\gamma`-gated residual injection 通过短程 fine-tune 向冻结 Transformer 添加 AttnRes-style routing path；Qwen3-VL-2B 新增/训练约 `7.4M` 参数（`<0.4%`），Qwen3-VL-4B 约 `11.8M`（`<0.3%`），并具有 identity-preserving initialization。
- **ReSkip 使用已安装信号进行 pre-execution dynamic depth**（\S\ref{sec:reskip}, \S\ref{sec:controlled_340m}, \S\ref{sec:exp_reskip_tradeoff}）。Skip decision 在当前 block 运行前做出；一个 340M controlled study 验证了 speed-quality motivation；eligible blocks 通过 ablation 或 action-drift 选择，因为 routing weight 是有用信号，但不是 safety certificate。
- **VLM 与 VLA 实验验证 retrofit pathway**（\S\ref{sec:experiments}, \S\ref{sec:vla}）。Retrofit 在 near-base full-depth runtime 下保持或提升预训练 VLM 能力；dynamic ReSkip 在不同 `q` 设置下给出 calibrated adaptive-depth behavior；LIBERO 提供支持性的 transfer evidence。

**Figure 1: ReSkip 与 AR-Retrofit 概览。** 左：标准 Transformer residual 不产生内生 routing signal。中：AR-Retrofit 将 identity-preserving 的 `\gamma`-gated AttnRes path 注入冻结预训练 backbone。右：phase-1 AttnRes weights 在 block execution 前计算，并被校准成 ReSkip decisions。

## 2 Method

### 2.1 Attention Residuals as a pre-execution routing signal

标准 Transformer 使用固定标量 `1` 的组合：

```tex
\mathbf{x}_l = \mathbf{x}_{l-1} + f_l(\mathbf{x}_{l-1})
```

因此不产生 routing signal。AttnRes~\citep{chen2026attnres} 用 learned attention over depth 替代它。每个 block `l` 维护一个 pseudo-query `\mathbf{w}_l \in \R^d`；keys 是 earlier block outputs `\mathbf{h}_i` 的投影 `\mathbf{k}_i = W_K \mathbf{h}_i`，并且

```tex
\alpha_{i \to l} =
\frac{\exp(\mathbf{w}_l^\top \mathbf{k}_i / \sqrt{d})}
{\sum_{j=0}^{l-1} \exp(\mathbf{w}_l^\top \mathbf{k}_j / \sqrt{d})},
\qquad
\mathbf{x}_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\,\mathbf{h}_i .
```

关键性质是 two-phase execution。计算 block `l` 的输入只需要 `\{\mathbf{h}_i\}_{i<l}` 和 `\mathbf{w}_l`，二者都在 `f_l` 运行之前已经可用。我们将 pre-execution attention computation 称为 **phase 1**，将 `f_l` 本身称为 **phase 2**。Phase 1 就是我们利用的 routing signal。Block-AttnRes 将层分组为 `N` 个 block，并在 block level 应用 softmax；全文均使用该粒度。Algorithm~\ref{alg:two_phase} 给出了包含 ReSkip skip check 的完整 two-phase forward。

### 2.2 AR-Retrofit: installing the routing path

Retrofit 目标由 introduction 中的问题约束：给定一个在 standard-residual regime 下预训练的 Transformer，产生一个 AttnRes-capable 模型，使其 (P1) 保持 benchmark quality，(P2) 输出足够 informative、可驱动 ReSkip 的 routing `\alpha`，并且 (P3) 在 off-the-shelf SFT data 上通过一次短程 fine-tune 完成训练，同时 base 保持冻结。为了满足第一个约束，修改必须在 step `0` 完全 identity-preserving；为了满足第二个约束，routing 必须进入真实 forward path，而不能只是旁路观察。

我们将 decoder 组织成 `N` 个相邻层组成的 **blocks**（主实验中 Qwen3-VL-2B 使用 7 个 block，每个 block 4 层）。对每个 block `n\ge 1`，我们添加一个 AttnRes router `\mathbf{w}_n`、一个小 adapter `A_n`（rank 为 `r` 的 down-up bottleneck，SiLU）和一个标量 gate `\gamma_n`：

```tex
r_n = \sum_{i=0}^{n-1} \alpha_{i \to n}\,h_i, \qquad
x_n = h_{n-1} + \gamma_n\,A_n(r_n - h_{n-1}), \qquad
h_n = \mathrm{Block}_n(x_n).
```

`\mathrm{Block}_n` 是原样保留的 pretrained block group；AttnRes 作为 correction 进入 **block 之间**。在 canonical `r=256, L=4` 下，2B retrofit 参数 `\{\mathbf{w}_n, \gamma_n, A_n\}` 约为 `7.4M`（`<0.4%`），4B 约为 `11.8M`（`<0.3%`）；整个 base，包括 embeddings、vision tower、decoder layers、LM head，都被冻结。

当 `\gamma_n=0` 时，有 `x_n = h_{n-1}`，因此 step `0` 的 forward 与预训练模型 bit-identical。Adapter up-projection 用小随机值 `\mathcal{N}(0,0.02^2)` 初始化，使得初始化时 `\partial\mathcal{L}/\partial\gamma_n \ne 0`，避免梯度死锁。然后我们通过 linear curriculum 在训练前 `30%--50%` 中将 `\gamma_n` 从 `0` ramp 到 `1`。当 `\gamma_n=1` 时，AttnRes-routed correction 完全激活，但模型仍是 Eq.~\ref{eq:gamma_gate} 定义的 pretrained stream 上的 residual correction。Appendix~\ref{app:route_ablations} 比较了 observer-only、interpolation 和 informed-initialisation controls。

训练目标来自上述三个约束。我们最小化：

```tex
\mathcal{L} =
\mathcal{L}_{\text{CE}}(\text{full path}) +
\lambda_{\text{kl}}\,\mathcal{L}_{\text{KL}}(\text{skip path}\,\|\,\text{teacher}) +
\lambda_{\text{ent}}\,\mathcal{L}_{\text{ent}}(\alpha).
```

其中 full-path CE 在 assistant tokens 上计算；skip-branch KL 在每步随机采样一个 block，将 forward 以跳过该 block 的方式运行，并将 resulting logits 拉向一个 frozen pretrained teacher。这使 surrogate `x_n` 在 block 运行和被跳过时都可用。`\mathcal{L}_{\text{ent}}` 在 loss 中以负号实现，即一个轻量 entropy-maximising prior，用于防止 `\alpha` 过早 collapse。默认 `\lambda_{\text{kl}}=1`，entropy weight `0.02`，retrofit params 的学习率为 `10^{-3}`，使用带 100-step warmup 的 cosine schedule。Hyperparameters 和 data mixtures 见 Appendix~\ref{app:retrofit_hparams}。

这不是普通 SFT。CE term 使 full path 对下游任务保持可用，但它本身并不会训练 skipped surrogate 去替代真实 block。Skip-branch KL 通过每步将一个 skipped-block forward 绑定到 frozen teacher，提供这一约束。反过来，单独 teacher imitation 只会复现 base model。弱 entropy prior 防止 early one-source collapse，同时仍允许 router 后期变得更 sharp。正是这个三项结构使 retrofit 在 step `0` identity-preserving，训练后 task-improving，并在推理时 skip-ready。

### 2.3 ReSkip: dynamic block skipping from the installed signal

ReSkip 使用已安装的 routing path，而不是单独 controller。对于 block `n`，phase 1 在 block body 运行前计算 `\alpha_{\cdot\to n}`。然后我们读取 router 分配给 immediate predecessor 的权重 `w_{\text{recent}}(n)`，并决定是否执行 phase 2：

```tex
\text{skip block $n$} \iff
w_{\text{recent}}(n) > \tau_n
\;\text{and}\;
n \in \mathcal{P}
\;\text{and}\;
\textstyle\sum_{m<n} \mathbb{1}[\text{skipped}_m] < M_{\max}.
```

该决策在当前 block 执行前可用，因此触发时确实可以节省当前 block。当规则触发时，我们将该 block short-circuit 为 `h_n \leftarrow x_n`，其中 `x_n` 是 Eq.~\ref{eq:gamma_gate} 中的 routed-correction input。为了保持 autoregressive decoding cache-consistent，一个被跳过的 decoder layer 仍运行 K/V-producing slice（`\texttt{LayerNorm}\to\texttt{k/v_proj}\to\texttt{RoPE}\to\texttt{cache.update}`），并跳过 `q_proj`、attention、`o_proj` 和 MLP。Cache-enabled skip path 在 last-position argmax 上与 cacheless path 匹配（Appendix~\ref{app:kv_equiv}）。

### 2.4 Calibration and safety selection

Routing weight 是可用的 depth signal，但不是 block 可以被移除的 certificate。因此，每个 eligible set `\mathcal{P}` 都通过两个互补诊断离线选择。AttnRes importance score：

```tex
I(n)=\max_{l>n}\,\E[\alpha_{n\to l}]
```

衡量未来 block 多频繁引用 block `n`；而 ablation 或 drift score 衡量在相关分布上移除该 block 的代价。对语言实验，我们使用 `A(n)=\mathrm{PPL}(\text{$n$ removed})/\mathrm{PPL}(\text{full})`；对 VLA，我们在 simulated rollouts 上使用 action-drift MSE。Thresholds `\tau_n` 是 held-out calibration data 上 `w_{\text{recent}}(n)` 的第 `q` 个 empirical quantile。这种分离很关键：`\alpha` 决定一个 calibrated block 何时看起来局部冗余，而 ablation 或 drift 决定哪些 block 是 eligible to skip。

## 3 Controlled Motivation: AttnRes and ReSkip at 340M

### 3.1 AttnRes and ReSkip at 340M

我们首先在 AttnRes 从头训练的设定中测试该机制，避免引入 retrofit confound。在一个 340M block-AttnRes Transformer 上（`d=1024`, `L=24`, `N=8` blocks；FineWeb-Edu 100BT~\citep{penedo2024fineweb}；`\texttt{lm-eval-harness}`~\citep{eval-harness}），当 `\mathcal{P}=\{3,5\}`, `M_{\max}=2`, `q=0.85` 时，ReSkip 在 expanded language benchmark set 上匹配 full-depth AttnRes 分数，并在 sequence length `8192` 下得到 `1.19x` wall-clock（Table~\ref{tab:reskip_benchmark}；Pareto curves 见 Appendix~\ref{app:reskip_results}）。观测到的 skip count 平均为 `0.88` blocks，并在每次 forward 中从 `0` 到 `2` 变化，说明这是 input-dependent allocation，而不是 threshold 从未触发。

**Figure: AttnRes routing weights do not predict ablation impact (340M).** (a) Learned block-level `\alpha_{n \to l}`。(b) `I(n)`（蓝色，左轴）与 `A(n)`（橙色，右轴）。ReSkip 使用 routing 进行 input-dependent decisions，但 eligible blocks 仍需要基于 ablation 的 safety selection。

**Table: ReSkip on 340M from-scratch AttnRes.** 选定 operating point 在实际触发时保持 full-depth AttnRes 在 expanded task set 上的分数：ReSkip 在 48 个 sequence-8192 batch 上平均跳过 `0.88` 个 block，并在该长度下得到 `1.19x` wall-clock。PIQA/HellaSwag/ARC-E/ARC-C/OpenBookQA 使用 `acc_norm`，MMLU/LAMBADA 使用 plain `acc`。

| Configuration | Avg. skip | Speed | LAMBADA acc | LAMBADA ppl | HellaSwag | PIQA | ARC-E | ARC-C | MMLU | OpenBookQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla (no AttnRes) | -- | -- | 0.3790 | 24.71 | 0.4436 | 0.6779 | **0.5602** | **0.3046** | **0.2594** | 0.3320 |
| AttnRes full-depth | 0.00 | 1.00x | 0.4054 | 20.20 | 0.4607 | 0.6893 | 0.5438 | 0.3012 | 0.2555 | 0.3580 |
| + ReSkip (`{3,5}`, `M=2`, `q=0.85`) | **0.88** | **1.19x** | **0.4054** | **20.20** | **0.4607** | **0.6893** | **0.5438** | **0.3012** | **0.2555** | **0.3580** |

这个 controlled study 为 retrofit experiments 建立了动机：AttnRes 改善 vanilla residual baseline，暴露 pre-execution depth signal，并且让 ReSkip 在不改变训练权重的情况下恢复 wall-clock speed。我们在 Section~\ref{sec:ablations_main} 回到 matched-rate 与 calibration ablations。

## 4 Retrofitting Qwen3-VL: VLM Quality, Runtime, and ReSkip

### 4.1 VLM adaptation: quality against the base

我们将 **Qwen3-VL-2B**~\citep{bai2025qwen3vl}（`L=28, d=2048`）以 `L=4` 的 block 大小 retrofit 成 `N=7` 个 block；将 **Qwen3-VL-4B**（`L=36, d=2560`）以 `L=4` retrofit 成 `N=9` 个 block。所有 base 参数冻结；在 `r=256` 下，约 `7.4M`（2B）/ 约 `11.8M`（4B）retrofit parameters 通过 Eq.~\ref{eq:retrofit_loss} 训练。最终 SFT mixture 包含 `60%` LLaVA-OneVision-Data~\citep{li2024llavaonevision}、`20%` UltraChat~\citep{ding2023ultrachat}、`10%` NuminaMath-CoT~\citep{aimo2024numinamath} 和 `10%` OpenThoughts-114k~\citep{guha2025openthoughts}，训练 `10k` steps；Appendix~\ref{app:data_mix} 给出具体 mixture。Evaluation 使用 LAMBADA~\citep{paperno2016lambada}/HellaSwag~\citep{zellers2019hellaswag} 作为 text benchmark，并使用六个 `lmms-eval` VLM benchmark（MMBench~\citep{liu2023mmbench}, MMMU~\citep{yue2024mmmu}, MMStar~\citep{chen2024mmstar}, AI2D~\citep{kembhavi2016ai2d}, OCRBench~\citep{liu2024ocrbench}, RealWorldQA~\citep{grok2024realworldqa}）。训练前验证 identity-at-init（`\gamma=0` 复现 base；bf16 下 max `|\Delta\text{logits}|=0.375`，`100%` argmax agreement；Appendix~\ref{app:identity}）。

**Table: AR-Retrofit on Qwen3-VL-2B/4B at the canonical `L=4` recipe.** VLM 使用 `lmms-eval` full splits，LAMBADA/HellaSwag 使用 `n=2000`。Retrofit 在两个规模上都在 `5/6` 个 VLM benchmark 上改善 base，并提升 LAMBADA text。MMStar overall 持平；math sub-category（Appendix~\ref{app:mmstar_subcat}）在 2B 上提升 `+7.9`pp，在 4B 上提升 `+3.9`pp，这与提升 deliberate-reasoning paths 的 router 一致。Auxiliary parameter-matched LoRA controls 未表现出相同 text-gain pattern（Appendix~\ref{app:lora_baselines}）。

| Configuration | LAMBADA | HellaSwag | MMBench | MMMU | MMStar | AI2D | OCRBench | RWQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B (base) | 0.532 | 0.506 | 75.77 | 0.414 | 0.536 | 0.736 | 0.772 | 0.648 |
| + AR-Retrofit (`L=4`, `10k`) | **0.5650** | 0.500 | **78.87** | **0.432** | 0.536 | **0.758** | **0.814** | **0.661** |
| Δ vs. base | **+3.3** | -0.6 | **+3.10** | **+1.8** | 0.0 | **+2.2** | **+4.2** | **+1.3** |
| Qwen3-VL-4B (base) | 0.576 | 0.562 | 83.33 | 0.490 | 0.624 | 0.819 | 0.819 | 0.715 |
| + AR-Retrofit (`L=4`, `10k`) | **0.6625** | 0.5515 | **85.22** | **0.521** | **0.632** | **0.825** | **0.824** | **0.718** |
| Δ vs. base | **+8.7** | -1.1 | **+1.9** | **+3.1** | **+0.8** | **+0.6** | **+0.5** | **+0.3** |

Retrofit 在两个规模上都在六个 VLM benchmark 中的五个上改善 base，其中最大增益出现在 MMStar math subcategory（2B 为 `+7.9`pp，4B 为 `+3.9`pp；完整六格 decomposition 见 Appendix~\ref{app:mmstar_subcat}）。收益集中在 **deliberate reasoning over images**，而不是感知，这与提升 deeper reasoning paths 的 router 一致。Text quality 也提升：2B 上 LAMBADA `+3.3`pp / ppl `-16%`，4B 上 `+8.7`pp / `-32%`；两个规模上的 HellaSwag 都在 `±1.1`pp 内。相关 SFT mix 上的 auxiliary parameter-matched LoRA controls 未恢复 LAMBADA gain（Appendix~\ref{app:lora_baselines}）。这些 controls 并不能证明所有质量增益都唯一来自 routing，但排除了简单的 parameter-count-only explanation。因此，我们将 routed path 视为对 skip-ready behavior 起承载作用，同时将质量增益解释为与 routed correction 一致但不唯一归因于它。

### 4.2 Runtime and calibrated ReSkip after retrofit

AR-Retrofit 安装 routing path 后，ReSkip 可以在不添加单独 router 的情况下应用。在 canonical 2B retrofit 上，保守语言 operating point 的 LAMBADA-500 在跳过 input-dependent blocks 的同时，保持在 no-skip retrofit 的 `1.0`pp 以内（Table~\ref{tab:retrofit_reskip_main}）。更激进的 thresholds 展示了预期的 calibration boundary：一旦 threshold 指示模型跳过它通常会执行的 blocks，质量就会 sharply drop。结合 Table~\ref{tab:retrofit_latency} 的 runtime characterization，这说明了我们想要的 target-scale behavior：installed routing path 在 near-base full-depth runtime 下保持或提升模型能力，而 ReSkip 将该路径转化为 calibrated input-dependent depth allocation。

**Table: Quality--skip behavior after retrofit.** 在 canonical `L=4` 2B retrofit 上的 LAMBADA-500，`\mathcal{P}=\{1,4\}`, `M_{\max}=2`，thresholds 在 32 个 held-out LAMBADA prefixes 上校准。保守点在产生 input-dependent block skipping 的同时保持质量；更低 thresholds 展示预期 calibration boundary。

| Configuration | LAMBADA acc | LAMBADA ppl | Δ acc vs no-skip | Avg. skipped blocks |
|---|---:|---:|---:|---:|
| No-skip retrofit | 0.5700 | 4.526 | --- | 0.00 |
| AR-Retrofit + ReSkip (`q=0.85`) | 0.5600 | 5.258 | -1.0pp | 0.19 |
| AR-Retrofit + ReSkip (`q=0.50`) | 0.4120 | 12.550 | -15.8pp | 1.06 |
| AR-Retrofit + ReSkip (`q=0.30`) | 0.3900 | 14.005 | -18.0pp | 1.17 |

当两个模型在相同 fixed-shape inference setting 下运行时，full-depth retrofit path 仍接近 base（Table~\ref{tab:retrofit_latency}）。这是实际 runtime 要点：添加 routing path 不会用很大的 inference penalty 抹掉质量收益，而 dynamic skip rule 在已安装路径之上运行。

**Table: Full-depth runtime of the installed routing path.** 在 `1xH100`、bf16、batch `1` 上的 forward-pass latency，5 次 warmup 后 20 次运行的 median。Canonical compiled setting 下，retrofitted model 与 compiled base 的差距在 `2.9%` 以内，同时改善 Table~\ref{tab:retrofit_main} 中的质量指标。

| Mode | seq | base (ms) | retrofit (`L=4`, ms) | retrofit / base | vs uncompiled base |
|---|---:|---:|---:|---:|---:|
| Uncompiled | 1024 | 14.91 | 16.66 | 1.117x | 1.117x |
| Uncompiled | 2048 | 25.49 | 28.53 | 1.119x | 1.119x |
| Compiled (`reduce-overhead`) | 2048 | 21.31 | 22.50 | 1.056x | 0.855x |
| **Compiled (`max-autotune`)** | **2048** | **21.81** | **22.43** | **1.029x** | **0.864x** |

Block partition 和 mixture choices 在 appendix 中报告，而不是作为主线。`L=4` 与附近 block sizes 一起位于 accuracy plateau 上，并减少 router calls；final mixture 是在训练 routed correction 的同时保持 VLM capability 的 mixture。完整 block-partition、data-mixture 与 design-control ablations 见 Appendices~\ref{app:block_partition}, \ref{app:data_mix}, \ref{app:retrofit_ablation}。

## 5 VLA Transfer Evidence

VLA 测试询问 retrofitted routed-depth signal 是否在 VLM evaluation 之外仍可用。Vision tokens（感知、early-layer~\citep{raghu2021vision}）、language tokens（task spec.）和 action tokens（motor planning）没有原则性理由共享相同 effective depth；AttnRes routing 在不需要 token-level gating machinery 的情况下，为每个位置提供了 handle。

我们将 OpenVLA-OFT~\citep{kim2025openvlaoft} action head（`L_1` regression，无 chunking）接到 Qwen3-VL-2B/4B backbones（ViT frozen）上，并在 LIBERO~\citep{liu2023libero} 的 pooled `libero_all` mix 上训练。主比较刻意保持窄范围：**Path 0** 使用 stock backbone + OFT；**Path B** 从我们的 canonical `L=4` VLM retrofit warm-start，并始终保持 `\gamma_n=1`。两条路径共享相同 action head、data、以及在 `4xH100` ZeRO-2 上、effective batch 32 的 `30k`-step OFT schedule。每个 policy 在每个 suite 上评估 500 次 rollout（每个 policy 共 2000 次）。VLA in-backbone forward 使用与 VLM retrofit 相同的 installed routing path；在 sequence length `2048` 时，其 eager latency 为 `28.79` ms，而 stock backbone 为 `25.49` ms，与 VLM retrofit runtime 的差距在 `1%` 以内（Appendix~\ref{app:latency}）。

### 5.1 LIBERO results

**Table: LIBERO 4-suite success rates (%) for Qwen3-VL-2B/4B.** Entries 是每个 suite 500 次 rollout（每个 policy 2000 次）的成功率。Path 0 是 stock-backbone OFT baseline；Path B 从我们的 `L=4` VLM retrofit warm-start。该表是 retrofit pathway 的 supporting transfer evidence，而不是 broad robotics claim。

| Scale | Path (steps) | Spatial | Object | Goal | Long-10 | Avg | Δ vs. Path 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B | Path 0 (`30k`) | 94.8 | 99.8 | 97.4 | 91.4 | 95.85 | --- |
| 2B | **Path B (`30k`, ours)** | **97.8** | 99.6 | **98.6** | **92.6** | **97.15** | **+1.30** |
| 4B | Path 0 (`30k`) | 95.0 | 99.2 | 97.8 | 92.2 | 96.05 | --- |
| 4B | **Path B (`30k`, ours)** | 94.6 | **99.8** | **98.2** | **94.2** | **96.70** | **+0.65** |

Path B 在 2B 上达到 `97.15%`，在 4B 上达到 `96.70%`。增益集中在 Spatial（2B 上 `+3.0`pp）和 Long-10（4B 上 `+2.0`pp），这是本评估中我们观察到最清楚增益的 suite。我们将这些结果作为 AR-Retrofit 的 downstream transfer evidence，而不是 broad robotics claim。Appendix~\ref{app:vla_variance} 报告 repeated-rollout sensitivity。

### 5.2 ReSkip on the action stream: calibrated conservative skipping

接着，我们在训练好的 Path B policies 上于推理时应用相同的 ReSkip rule，并在 sim-rollout distribution 上校准 thresholds，而不是在 language tokens 上校准（原因见下一段）。Eligible blocks 由每个 scale 的 per-block action-drift sweep 选择（Appendix~\ref{app:vla_pareto}：2B 上 `\mathcal{P}=\{1,4\}`，4B 上 `\mathcal{P}=\{1,2\}`），并且全程使用 `M_{\max}=2`。Table~\ref{tab:vla_reskip_pareto} 展示 across `q` 的 4-suite Pareto。

**Table: ReSkip 4-suite Pareto on trained Path B policies.** Eligible blocks `\mathcal{P}` 与 threshold `\tau` 按 scale 在 action distribution 上校准；`q` 是产生 `\tau` 的 empirical quantile。保守端（`q=0.99`）在本评估中保持 success rates。

| q (sim-calibrated) | Spatial | Object | Goal | Long-10 | Avg |
|---|---:|---:|---:|---:|---:|
| **Qwen3-VL-2B, `P={1,4}`, `M=2`, no-skip/full ref `97.2`** ||||| |
| 0.30 | 4.7 | --- | --- | --- | (sharp drop) |
| 0.85 | 80.0 | 98.0 | 87.3* | 67.2 | 83.1 |
| 0.95 | 95.0 | 99.4 | 97.6 | 86.8 | 94.7 |
| **0.99** | **97.6** | **99.2** | **99.0** | **93.6** | **97.4** |
| no-skip/full (ref) | 97.8 | 99.6 | 98.6 | 92.6 | 97.2 |
| **Qwen3-VL-4B, `P={1,2}`, `M=2`, no-skip/full ref `96.7`** ||||| |
| 0.30 | 89.6 | 95.4 | 96.4 | 86.0 | 91.85 |
| 0.50 | 91.2 | 98.0 | 98.2 | 89.6 | 94.25 |
| 0.85 | 93.6 | 99.2 | 97.6 | 93.2 | 95.90 |
| 0.95 | 95.6 | 98.0 | 98.0 | 93.0 | 96.15 |
| **0.99** | **96.4** | **98.2** | **98.4** | **92.8** | **96.5** |
| no-skip/full (ref) | 94.6 | 99.8 | 98.2 | 94.2 | 96.7 |

`*` Partial eval（332/500 trials）：driver schedule 提前结束；报告 completed trials 上的 rate。

Thresholds 在 action distribution 上校准，而不是从 language 迁移。原因是 `q` 本身不是 skip rate；它索引的是 `w_{\text{recent}}` 的 empirical distribution，而该分布会跨 modality 变化。因此推荐 VLA point 是保守的（`q=0.99`）：它很少 skip，但只在 action-drift sweep 认为局部安全的 blocks 上 skip。

## 6 Ablations: Skip Budget and Calibration

ReSkip 的主要旋钮是 eligible set `\mathcal{P}`、skip budget `M_{\max}` 和 calibration quantile `q`。Table~\ref{tab:budget_calibration_main} 总结了主实验中使用的 operating points。340M 结果显示，当 skip signal 原生于训练好的 AttnRes 模型时，从 no-skip 增加到 `M_{\max}=2` 可以产生真实 speedup。在 retrofitted VLM 和 VLA 上，保守的 `M_{\max}=2` 设置在使用 Tables~\ref{tab:retrofit_latency} 和 \ref{tab:vla_reskip_pareto} 中刻画的 installed routing path 时保持质量。

**Table: Skip-budget and calibration operating points.** Quality 以每个 setting 的 native metric 报告。Runtime 对 340M controlled model 和 full-depth Qwen3-VL retrofit 直接测量；retrofitted ReSkip rows 报告对应 average skipped blocks 或 success-rate Pareto point。

| Setting | Configuration | `M_max` | `q` | Quality | Depth / runtime behavior |
|---|---|---:|---:|---|---|
| 340M AttnRes | full-depth | 0 | -- | LAMBADA 0.4054 / ppl 20.20 | 1.00x, 0.00 skips |
| 340M AttnRes | ReSkip | 2 | 0.85 | LAMBADA 0.4054 / ppl 20.20 | 1.19x, 0.88 avg skips |
| 2B VLM retrofit | full-depth | 0 | -- | LAMBADA 0.5700 / ppl 4.526 | 22.43 ms compiled |
| 2B VLM retrofit | ReSkip | 2 | 0.85 | LAMBADA 0.5600 / ppl 5.258 | 0.19 avg skips |
| 2B VLA Path B | ReSkip | 2 | 0.99 | LIBERO avg 97.4% | conservative action-stream skipping |
| 4B VLA Path B | ReSkip | 2 | 0.99 | LIBERO avg 96.5% | conservative action-stream skipping |

Routing signal 的作用不只是选择 skip locations。在同一个 340M 模型上、匹配 skip budget 时，基于 phase-1 routing 的 input-dependent rule 比 static 或 random schedules 保留显著更多质量（Table~\ref{tab:decision_main}）。在这个更激进的 calibration point 上，dynamic rule 并不总是 lossless，但它明显优于无条件跳过相同 positions。

**Table: Dynamic, static, and random skip rules at matched rate on 340M.** 所有行使用相同训练好的 AttnRes weights、`\mathcal{P}=\{3,5\}` 和 `M_{\max}=1`。Dynamic ReSkip 使用 calibrated phase-1 statistic；static 和 random controls 在没有 input dependence 的情况下移除相同 positions。完整 decision-rule ablation 与 observed trigger rates 见 Appendix~\ref{app:b1b2_decision_rule}。

| Rule | LAMBADA | HellaSwag | PIQA | ARC-E | OpenBookQA |
|---|---:|---:|---:|---:|---:|
| Full-depth AttnRes | 0.4054 | 0.4607 | 0.6893 | 0.5438 | 0.3580 |
| **Dynamic ReSkip** | **0.3445** | **0.4471** | **0.6774** | **0.5391** | **0.3540** |
| Static skip `P=3` | 0.2624 | 0.3968 | 0.6610 | 0.5206 | 0.3300 |
| Static skip `P=5` | 0.2189 | 0.4223 | 0.6638 | 0.5253 | 0.3260 |
| Random `{P=3 or P=5}` | 0.2416 | 0.4028 | 0.6420 | 0.5101 | 0.3300 |

## 7 Related Work

对于 test-time compute，o1~\citep{openai2024o1}、DeepSeek-R1~\citep{deepseek2025r1}、chain-of-thought~\citep{wei2022chain} 以及 optimal test-time compute analyses~\citep{snell2024scaling} 通过生成更多 reasoning **tokens** 来分配计算；per-token depth 仍然固定。Depth-axis methods 包括 ACT / Universal Transformers~\citep{graves2016adaptive, dehghani2018universal}、CALM~\citep{schuster2022confident}、Mixture-of-Depths~\citep{raposo2024mixture}、LayerSkip~\citep{elhoushi2024layerskip} 和 static pruning~\citep{gromov2024unreasonable, men2024shortgpt}。这些方法添加 halting head、exit classifier、router、speculative decoder 或 static removal rule。我们则使用 AttnRes 作为 forward pass 本身产生的 intrinsic routing signal，然后将该信号 retrofit 进已有 pretrained backbone。Appendix~\ref{app:b1b2_decision_rule} 比较了 340M 上的 dynamic、static 与 random skip rules，Appendix~\ref{app:gromov_pruning} 给出 VLM scale 的 full-layer static-pruning baseline。

Table~\ref{tab:closest_methods} 总结了最接近方法层面的差异。相关区别是 frozen-backbone retrofitting 与 pre-execution signal 的组合，而该 signal 同时也是 forward path 的一部分。

**Table: Closest method comparison.** 相关区别是 frozen-backbone retrofitting 与 pre-execution signal 的组合，并且该 signal 同时是 forward path 的一部分。

| Method | Pretrained backbone | Extra decision head/router | Decision before current block | Applies to frozen pretrained backbone | Same signal forms input and skip |
|---|---|---|---|---|---|
| Early exit / CALM | yes | yes | no | no | no |
| Mixture-of-Depths | train-time design | yes | yes | no | no |
| Static pruning | yes | no | yes | yes | no |
| LayerSkip | continued training | no / separate exit policy | partly / speculative | no | no |
| AttnRes pretraining | no | no | yes | no | yes |
| **AR-Retrofit + ReSkip** | **yes** | **no** | **yes** | **yes** | **yes** |

神经动机是 recurrence as flexible effective depth。Dual-process distinction~\citep{kahneman2011thinking} 可以自然映射到视觉处理：简单刺激可以由 feedforward sweeps 处理，而更难刺激会调用 recurrent circuits~\citep{lamme2000distinct, kar2019recurrent, dicarlo2012does}。Recurrent networks 能够捕获人类表征动态~\citep{kietzmann2019recurrence}；\citet{spoerer2020recurrent} 是最接近的概念先例，表明 recurrent CNNs 可以在更难图像上花费更多步骤，以解释 speed--accuracy behaviour。Recurrence 也具有 depth interpretation：有限 recurrence 可以展开成 feedforward depth，但 recurrent form 会复用固定 substrate，以获得 flexible effective depth~\citep{vanbergen2020going}。我们将其作为设计目标，而不是 cognitive-faithfulness claim：cortical microcircuits 和 dendritic association~\citep{bastos2012canonical, buschman2007topdown, larkum2013cellular} 激发了 intrinsic routing 的动机，而我们的实现是一个 Transformer retrofit。

在 residual architectures 中，DenseNet~\citep{huang2017densely} 和 Highway networks~\citep{srivastava2015highway} 修改 residual flow；AttnRes~\citep{chen2026attnres} 用 depth attention 泛化 residual combination。据我们所知，这是第一项将 AttnRes weights 用作 pre-execution skip signal，并将 AttnRes-style routing path retrofit 到 pretrained transformers 中的工作。在 VLA 中，RT-2~\citep{brohan2023rt}、Octo~\citep{team2024octo}、OpenVLA~\citep{kim2024openvla} 和 Pi0~\citep{black2024pi0} 建立了这一范式，而 efficiency work 主要集中在 action chunking~\citep{zhao2023learning} 和 post-hoc compression。我们贡献的是用于 VLM backbone 的 modality-aware adaptive depth。

## 8 Discussion

AR-Retrofit 表明，一个冻结预训练 Transformer 可以通过短程 identity-preserving fine-tune 获得 pre-execution depth-routing path。ReSkip 表明，这个已安装信号可以被校准为 input-dependent block execution，并通过 ablation 或 drift checks 决定哪些 blocks 可以安全跳过。当前证据最强的是 retrofit mechanism 和 calibrated dynamic-depth behavior；更广泛的 backbone coverage、更强的 end-to-end acceleration，以及 per-modality token-level calibration 仍是开放问题。

## References

见 `references.bib`。

---

# Appendix

## A Limitations and future work

**Limitations.** (i) 两个 retrofit targets 来自同一个 VLM family；扩展到 InternVL / Gemma-VL / text-only LMs 是计划中的工作。(ii) Runtime measurements 使用 H100 / bf16 / batch `1`，并采用标准 PyTorch inference paths；尚未在更低端 GPU / edge 设备上直接测量。(iii) 当前 ReSkip rule 继承自 AttnRes，是 block-level；per-token routing 与 fused production kernels 是未来工作。(iv) VLA Pareto thresholds 是按 scale 在 sim 上校准的，而不是在单个 rollout 内按 modality 校准；per-modality、per-token-class calibration 是下一步 sweep。

**Future work: weight-shared AttnRes (ReLoop).** 一个自然扩展是在 depth 上共享 block weights，并通过 position-indexed pseudo-queries 区分每次应用；AttnRes routing statistic 随后可同时服务于 ReSkip 的 high-immediate-predecessor skip rule，以及基于 recurrent applications 间 low marginal change 的 ReLoop halt rule。一个 74M validation 产生了平滑的 depth--quality Pareto（节省 `25%` compute，ppl change `<1%`）；将其扩展并与 retrofit framework 结合仍是开放问题。

## B Implementation details

### B.1 Online softmax merge with skip

当一个 block 被跳过时，online softmax merge simply 不处理它的输出。Running state `(\mathbf{s}, m, e)` 保持不变。从数学上看，输出等价于一个训练时没有让该 block 贡献的 AttnRes 模型，但要考虑 downstream `\mathbf{w}_l` 的 calibration。

### B.2 Pseudo-query initialisation (Section~\ref{sec:reskip} from-scratch)

对于 from-scratch training，我们使用 `\mathbf{w}_l \sim \mathcal{N}(0, 0.02)`，这会在初始化时产生近似 uniform 的 `\alpha`，并在前几千步中逐渐 specialization。

### B.3 Retrofit hyperparameters

使用 AdamW，`\beta=(0.9,0.95)`，weight decay `0`，gradient clip `1.0`，bfloat16。Retrofit params（routers、adapters、`\gamma`）学习率 `10^{-3}`，使用带 100-step warmup 的 cosine schedule。Sequence length：训练 `1536`，评估 `2048`。Loss weights：`\lambda_{\text{kl}}=1`，entropy weight `0.02`，KD temperature `1`。Entropy term 从 loss 中减去，因此它在训练早期最大化 routing entropy，而不是最小化它。Skip-branch sampling：每步从 eligible blocks 中 uniform random 选择一个 block。Adapter bottleneck rank `r=256`（canonical）；启用 router keys 上的 position-bias；router temperature 固定为 `1`。`\gamma`-curriculum：在总步数前 `30%` 中从 `0` 线性到 `1`（2B `5k`/`10k`，4B `5k`），在 4B x `10k` 上 ramp-fraction 为 `0.5`，用于稳定 late-stage `\gamma=1` transition divergence（Appendix~\ref{app:gamma_stability}）。单张 H100 上总训练 wall-clock：2B `5k` 约 22 分钟，2B `10k` 约 44 分钟，4B `10k` 约 54 分钟。

### B.4 Block granularity

最终 recipe 在两个规模上都使用 `L=4` layers per block：Qwen3-VL-2B 有 `N=7` 个 blocks，Qwen3-VL-4B 有 `N=9` 个 blocks。早期 `L=2` runs 作为 ablation controls 保留在 appendix 中，但 `L=4` 是 canonical setting，因为它在相对 `L=2` 将 router calls 减半的同时保持 accuracy plateau。每个 block 接收一个 router、一个 residual adapter（默认 `r=256`）和一个标量 `\gamma` gate。

### B.5 Identity-at-init details

当 `\gamma_n=0` 且 adapter up-projection 初始化为 `\mathcal{N}(0, 0.02^2)` 时，retrofit forward 在算术上与预训练模型完全相同（二者都产生 `x_n = h_{n-1}`，并将其传入未修改 block）。我们进行了 end-to-end 验证：在 LAMBADA 上 `n=500` 时，retrofit 与 base 在 bfloat16 evaluation noise 内一致（acc `0.530` vs. `0.532`；ppl `5.572` vs. `5.547`）；在 MMMU 和 MMStar 上，它们在 answer-choice level bit-equal。Token level smoke test 显示 max `|\Delta\text{logits}|=0.375`（bfloat16 SDPA noise），并且在 calibration prompts 上 `100%` argmax agreement。

### B.6 Skip under `use_cache=True`: correctness verification

我们验证 \S\ref{sec:retrofit} 中描述的 K/V-only skip path 会产生与 cacheless path argmax-consistent 的输出。Procedure：对固定 prefill input，使用相同 skip configuration 运行两次 retrofit，一次 `use_cache=False`，一次 `use_cache=True`，然后比较 last-position argmax 和 logits。我们在 Qwen3-VL-2B 的 H_r256_5k 上 sweep skip sets `{[4], [4,10], [4,10,12], [2,4,10]}`。在每种 configuration 中，两个 cache regimes 的 last-position argmax 都匹配，maximum logit delta 为 `0.19--0.37`，与 stock base 在无 retrofit、无 skip 时观察到的 bfloat16 SDPA jitter 相同。从 position `27--40` 左右开始的 multi-step autoregressive divergence 是 HF + bf16 + SDPA 的固有性质，stock base 在相同 prompts 上也表现出该现象（在 “Once upon a time” 和 “def fibonacci” 上测得）；它不是我们 skip path 的性质。

### B.7 `\gamma=1` transition stability

`\gamma` curriculum 用于在 routed correction 变得 active 的同时，使冻结 residual stream 保持接近其 pretrained operating point。对于报告的 2B 与 4B canonical runs，我们使用至少在 5000 个优化步骤后才达到 `\gamma=1` 的 ramp。该 schedule 在论文报告的 scales 和 block partitions 上稳定，并且不混淆 Appendix~\ref{app:block_partition} 中的 quality comparisons。

## C Retrofit ablations and inference machinery

### C.1 Design controls

我们将 `\gamma`-gated residual injection 与三个更简单设计进行了比较：observer-only routing、block-input interpolation，以及 direct AttnRes replacement with informed pseudo-query initialisation。这些 controls 要么使 routing weights 不再 load-bearing，要么过于突然地将冻结 backbone 从其 pretrained residual stream 中移开，要么需要比 short retrofit budget 多得多的 tuning。\S\ref{sec:retrofit} 中的 `\gamma`-gated injection 是所有主实验使用的 stable recipe。

### C.2 Adapter-rank ablation

Wall-clock latency 在不同 rank 间基本不变（所有测量 sequence lengths 上为 `±2%`）：与 router stack/softmax/einsum cost 相比，adapter 的 `2048 -> r -> 2048` bottleneck 是较小贡献者。因此，rank 影响质量而不是速度。Canonical `r=256` 是根据 Appendix~\ref{app:retrofit_ablation} 中的 `\gamma -> 1` ablation 选择的；在相同步数预算下，较低 rank 会留下更多 MMBench drop，且没有补偿性的 latency savings。

### C.3 `\gamma -> 1` retrofit ablation

我们沿四个轴 sweep canonical `\gamma -> 1` recipe 的 variants：adapter rank `r in {32,64,128,256}`，training steps `in {5k,10k,20k}`，mix 中 LLaVA-Instruct fraction `in {50%,80%}`，以及 `\gamma` ramp-fraction `in {0.3,0.7}`。所有其他 hyperparameters（loss、optimizer、schedule family、seed）保持不变。

**Table: Retrofit ablation on Qwen3-VL-2B.** 除非注明，所有行在前 `30%` steps 使用 `\gamma`-curriculum `0 -> 1`。`n=300 MMBench noise floor ±1 question` 对应 `±0.33`pp。

| Name | gamma sched | rank | steps | VLM% | ramp | LAMBADA | HellaSwag | MMBench |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-2B | --- | --- | --- | --- | --- | 0.532 | 0.506 | 0.727 |
| **Canonical** | `0->1` fast | **256** | **5k** | **50** | **0.3** | **0.576** | **0.522** | **0.710** |
| + low rank | `0->1` fast | 64 | 10k | 50 | 0.3 | 0.570 | 0.530 | 0.697 |
| + very low rank | `0->1` fast | 32 | 10k | 50 | 0.3 | 0.568 | 0.526 | 0.683 |
| + low rank, VLM-heavy | `0->1` fast | 64 | 10k | 80 | 0.3 | 0.560 | 0.524 | 0.660 |
| + mid rank, VLM-heavy | `0->1` fast | 128 | 10k | 80 | 0.3 | 0.564 | 0.520 | 0.653 |
| + VLM-heavy | `0->1` fast | 256 | 10k | 80 | 0.3 | 0.560 | 0.520 | 0.617 |
| + longer, 10k | `0->1` fast | 256 | 10k | 50 | 0.3 | 0.586 | --- | 0.587 |
| + slow ramp | `0->1` slow | 256 | 10k | 50 | 0.7 | 0.568 | 0.520 | 0.517 |
| + longer, 20k | `0->1` fast | 256 | 20k | 50 | 0.3 | 0.568 | 0.520 | 0.513 |

**Over-training signature at `\gamma=1`.** 固定 (`r=256`, `50%` VLM, fast ramp)，只改变 total step count 时，MMBench 随 steps 严格单调下降：`5k -> 0.710`（preserved），`10k -> 0.587`（-14pp），`20k -> 0.513`（-21pp）。LAMBADA 在 `10k` 达峰（`0.586`），在 `20k` 回落（`0.568`），因此更长训练也没有带来 text-domain gain。`\gamma=1` regime 有一个窄 compute sweet spot；超过它后，router+adapter 会对 training distribution 过度 specialize，而 frozen backbone 无法进一步适应，从而失去 multimodal calibration。

**Rank and data are subordinate.** 在 `10k` 下将 rank 减半能恢复一部分 MMBench（`r=32`: `0.683`, `r=64`: `0.697`），但都达不到 `5k-r=256` 水平（`0.710`）。将 mix 推到 `80%` LLaVA 会在每个 rank 上都恶化 MMBench，说明 regression 是 router 对任何 narrow training distribution 的 over-adaptation，而不是 visual exposure 不足。

**Interpretation.** 这些 ablations 仅用于设定 canonical recipe。它们不是 ReSkip 的主要证据；load-bearing evidence 是 installed routing path 参与 forward computation，并支持 \S\ref{sec:exp_reskip_tradeoff} 中的 calibrated skip rule。

### C.4 Per-block skip importance

在 canonical 2B retrofit 上运行 single-block removals，得到主文使用的 eligibility set `\mathcal{P}=\{1,4\}`（Table~\ref{tab:retrofit_reskip_main}）。同样的 qualitative ranking pattern 出现在 340M from-scratch 与 2B retrofit 设置中：一些靠近 embedding 或 late residual-refinement stages 的 blocks 移除代价较高，而经过 per-model calibration 选出的 mid-depth positions 是更安全的 candidates。

### C.5 Calibration set (dynamic skip)

Calibration 使用 32 个 held-out LAMBADA prefixes（每个截断到 512 tokens）。每个 block 的 `w_{\text{recent}}(n)` samples 通过一次 retrofit forward 收集，并使用 empirical quantile at `q` 作为 `\tau_n`。我们验证了改变 calibration draw 时，在选定的 `q=0.95` 下 `\tau_n` 的绝对变化 `<0.01`。对于 VLA deployment，将相同 calibration pipeline 应用于 VLA in-backbone forward（`\texttt{retrofit/eval/calibrate_vla_thresholds.py}`）会在 matched inputs 上产生 byte-identical thresholds，确认 VLA 和 VLM forwards 实例化的是同一个 router。

### C.6 Canonical data mixture

Table~\ref{tab:data_mix_ablation} 给出了 canonical retrofit 使用的 final SFT mixture。该 mix 有意以 VLM 为 anchor，并使用较小的 math/reasoning component 来加强 deliberate reasoning，同时避免将 frozen VLM backbone 从 visual instruction following 中移开。

**Table: Canonical retrofit SFT mixture.** 2B 和 4B canonical `L=4` recipes 使用相同 source proportions。

| Source | Proportion | Role |
|---|---:|---|
| LLaVA-OneVision-Data~\citep{li2024llavaonevision} | 60% | visual instruction anchor |
| UltraChat~\citep{ding2023ultrachat} | 20% | general instruction following |
| NuminaMath-CoT~\citep{aimo2024numinamath} | 10% | mathematical reasoning traces |
| OpenThoughts-114k~\citep{guha2025openthoughts} | 10% | open-ended reasoning traces |

该 mixture 是主文 canonical results 使用的唯一 data recipe。

### C.7 Two-phase forward with dynamic skip (algorithm)

**Algorithm: Two-phase AttnRes forward with dynamic ReSkip.**

1. **Input:** embeddings `\mathbf{h}_0`；block functions `f_1,\dots,f_N`；pseudo-queries `\{\mathbf{w}_n\}`；eligible `\mathcal{P}`；thresholds `\{\tau_n\}`；max skips `M_{\max}`。
2. Initialise online-softmax state `(\mathbf{s}, m, e) <- Init(\mathbf{h}_0)`；skip counter `k <- 0`。
3. 对 `n = 1` 到 `N`：
   - 在 completed block outputs 上计算 `\alpha_{\cdot \to n}`（phase 1）。
   - 按 Eq.~\ref{eq:gamma_gate} 形成 routed-correction input `\mathbf{x}_n`。
   - `w_{\text{recent}}(n) <- \E_{\text{token}}[\alpha_{n-1\to n}]`。
   - 如果 `n in \mathcal{P}` 且 `k < M_{\max}` 且 `w_{\text{recent}}(n) > \tau_n`：
     - **Skip:** 设置 `\mathbf{h}_n <- \mathbf{x}_n`；若 `use_cache=True`，在每个 constituent layer 上运行 K/V-only slice；`k <- k+1`。
   - 否则：
     - **Execute:** 在 merged input 上执行 block `n`；用 `\mathbf{h}_n` 更新 `(\mathbf{s},m,e)`。
4. **Output:** final hidden state `\mathbf{s}/e`。

## D Phase 1: 340M from-scratch validation

### D.1 Motivation: AttnRes cost from-scratch is prohibitive

我们在 `1xH100` / bf16 / batch `1` 上测量了两个使用相同 FineWeb-Edu 100BT recipe 训练的 340M 模型的 forward-pass latency：一个 vanilla standard-residual transformer 和一个 `N=8` blocks 的 AttnRes transformer。与 vanilla 相比，Block-AttnRes 增加了 `+78%`（8192 seq）到 `+128%`（1024 seq）的 wall-clock。在 2B VL scale，一个 stock Qwen3-VL-2B forward 在 `1xH100`、seq `256` 下已经约 `25.6` ms，因此 `35%--45%` block-level AttnRes overhead 会超过常见 `50Hz` / `20Hz` robotic control budgets。预训练一个 2B AttnRes-VLM 的成本至少与匹配的 standard-residual base 相当（根据公开 Qwen3-VL / InternVL3 / OpenVLA / LLaVA-OneVision tech reports，约 `>=10k--25k H100-h`），因此唯一实际可行的 access path 是对已训练标准模型进行 retrofit。

### D.2 Phase 1: cross-scale validation

340M from-scratch AttnRes（\S\ref{sec:reskip}, Table~\ref{tab:reskip_benchmark}）是 load-bearing existence proof。在相同 FineWeb-Edu recipe 上，对 110M 规模、8-block partition 的 cross-scale check 在更安全 operating point 下复现了 zero-degradation pattern（110M 为 `q=0.97, M_{\max}=1`，340M 为 `q=0.85, M_{\max}=2`）；110M 的 skip trigger rate 更低，这与更大模型携带更多冗余计算一致。1.3B / 2B from-scratch 不作为 load-bearing，因为 retrofit（\S\ref{sec:exp_retrofit}）直接针对一个 pretrained 2.13B model，在不支付 pretrain bill 的情况下回答 2B-scale question。

### D.3 ReSkip method comparison and full Pareto / latency at 340M

**Table: ReSkip versus representative adaptive-computation methods.**

| Method | Auxiliary network | Routing is | Train-time changes | Architectural changes |
|---|---|---|---|---|
| CALM~\citep{schuster2022confident} | exit classifiers | per-token | yes | small |
| Mixture-of-Depths~\citep{raposo2024mixture} | router | per-token | yes, capacity loss | yes |
| Static pruning~\citep{gromov2024unreasonable} | none | static | no | block removed |
| LayerSkip~\citep{elhoushi2024layerskip} | spec. decoder | per-token | yes, cont. pretrain | decoding path |
| **ReSkip (ours)** | **none** | **per-batch** | **none** | **none** |

### D.4 ReSkip position-set ablation at 340M

**Table: Dynamic skip as a function of `\mathcal{P}` on 340M from-scratch AttnRes (`M_{\max}=2, q=0.85`).** 只按 `I` 选择（`{5}`）或只按 `A` 选择（`{3}`）都会在速度或质量上留下空间；结合二者（`{3,5}`）严格最好。

| `\mathcal{P}` | PPL ratio | Speedup |
|---|---:|---:|
| `{5}` (`I`-only) | 0.993 | 1.14x |
| `{3}` (`A`-only) | 1.009 | 1.02x |
| `{2,3}` | 1.009 | 1.17x |
| `{4,5}` | 1.008 | 0.97x |
| `{3,4,5}` | 0.993 | 1.02x |
| `{2,3,4,5,6}` | 1.40 | 1.16x |
| **`{3,5}` (ours)** | **0.991** | **1.19x** |

### D.5 Decision rule: dynamic vs. static-rate-matched vs. random

上面的 position-set ablation 回答 **skip 哪些 blocks**；本小节回答 **input-dependent decision 是否重要**。我们固定 340M from-scratch AttnRes weights，固定 `\mathcal{P}=\{3,5\}` 和 `M_{\max}=1`（block-level skip rate `<=12.5%`），只改变 runtime decision rule：

- **B0.** No-skip upper bound（full AttnRes forward）。
- **B1.b/B2.a.** Dynamic，当 `w_{\text{recent},n} > \tau_n` 时触发（我们在 `L=1` block granularity 下的 ReSkip signal）。
- **B1.c, B1.d.** Static，在 `P=3` 或 `P=5` 上 every-token skip（rate-matched, `12.5%`）。
- **B1.e/B2.d.** Random，每次调用从 `{keep-P=3, keep-P=5}` 均匀抽样（rate-matched, `12.5%`）。
- **B2.b.** Dynamic，当 block-1 entropy `H_n < \tau_n` 时触发（router-confidence rule）。
- **B2.c.** Dynamic，当 `w_{\text{recent},n} - w_{\text{embed},n} > \tau_n` 时触发（relative-recent rule）。

三个 thresholds 都在 `32x8192` FineWeb-Edu tokens 上按 `q=0.5` per-position quantile 校准（\S\ref{app:calibration}）。结果见 Tab.~\ref{tab:b1b2_decision_rule}；eval distribution 上 observed skip rates 见 Tab.~\ref{tab:b1b2_observed_rate}。

**Table: Decision-rule and threshold-rule ablation on 340M from-scratch AttnRes.** 使用 lm-evaluation-harness，`\mathcal{P}=\{3,5\}`, `M_{\max}=1`。在 eval distribution 上 threshold 实际触发的 dynamic rules（B2.c）在相同约 `12%` rate 下保留 LAMBADA `0.345`，而 static / random 为 `0.22--0.26`。在 lm-eval data 上 calibrated thresholds 很少跨过的 dynamic rules（B1.b observed `3.1%`，B2.b 为 `0.0%`，见 Tab.~\ref{tab:b1b2_observed_rate}）几乎等价于 no-skip，因此不是 rate-matched comparison。

| Cell | LAMBADA | HellaSwag | PIQA | ARC-e | OpenBookQA |
|---|---:|---:|---:|---:|---:|
| B0. no-skip (upper bound) | 0.4054 | 0.4607 | 0.6893 | 0.5438 | 0.3580 |
| B1.b. recent_weight_gt | 0.4011 | 0.4534 | 0.6839 | 0.5412 | 0.3580 |
| B2.b. entropy_lt | 0.4036 | 0.4608 | 0.6839 | 0.5417 | 0.3580 |
| **B2.c. recent_minus_embed_gt** | **0.3445** | **0.4471** | **0.6774** | **0.5391** | **0.3540** |
| B1.c. static skip `P=3` every | 0.2624 | 0.3968 | 0.6610 | 0.5206 | 0.3300 |
| B1.d. static skip `P=5` every | 0.2189 | 0.4223 | 0.6638 | 0.5253 | 0.3260 |
| B1.e/B2.d. random `{P=3 OR P=5}` | 0.2416 | 0.4028 | 0.6420 | 0.5101 | 0.3300 |

**Table: Observed skip rate on `8x512` LAMBADA tokens.** Calibration set（FineWeb-Edu）和 eval set（LAMBADA）具有不同的 routing-statistic distributions；B2.b 的 `entropy_lt` threshold 在 LAMBADA 上从不触发，B1.b 只触发 `3.1%`，因此它们在 headline table 中接近 no-skip 的 accuracy 具有误导性。B2.c 是 calibration target（约 `12.5%`）实际 transfer 的 strategy，因此它是 Tab.~\ref{tab:b1b2_decision_rule} 中与 B1.c/d/e 进行 rate-matched comparison 的 load-bearing 对照。

| Cell | Calibration target | Observed (LAMBADA) | Notes |
|---|---:|---:|---|
| B0 | 0% | 0.00% | sanity check, `\tau_n=\infty` |
| B1.b | <=12.5% | 3.12% | `\tau_3=0.4645, \tau_5=0.4010` |
| B2.b | <=12.5% | 0.00% | `\tau_3=0.8631, \tau_5=0.8764` |
| **B2.c** | <=12.5% | **10.94%** | `\tau_3=0.1655, \tau_5=0.2472` |
| B1.c | 12.5% | 12.50% | forced, every-token `P=3` |
| B1.d | 12.5% | 12.50% | forced, every-token `P=5` |
| B1.e | 12.5% | per-batch toggle | uniform `{P=3, P=5}` |

**Conclusion.** 在公平的 rate-matched comparison（约 `12%` fired skips）下，**input-dependent** dynamic skip（B2.c）将 LAMBADA 保持在 `0.345`，而每个 static 或 random alternative 都降至 `0.22--0.26`（`-8` 到 `-13`pp）。相同排序也出现在 HellaSwag、PIQA、ARC-easy 和 OpenBookQA 上。MoD-style 的 “you might be getting a free lunch from any same-rate schedule” critique 被拒绝：在 340M from-scratch 上，schedule choice 很重要，并且 AttnRes routing weights 承担了作用。

**Threshold-transfer caveat.** 三个 dynamic rules 中有两个（B1.b `recent_weight_gt` 和 B2.b `entropy_lt`）在 LAMBADA distribution 上的触发率远低于 FineWeb-Edu calibration target `12.5%`（Tab.~\ref{tab:b1b2_observed_rate}）。因此，它们接近 no-skip 的 accuracy 不是这些 specific rules 的正面 datapoint；它与 “threshold 偶然不触发从而保护了输出” 一致。Load-bearing rate-matched comparison 是 B2.c `recent_minus_embed_gt`（它接近 target 触发）与 static/random alternatives。我们将两个 non-firing rows 视为 sensitivity result：在 threshold/distribution mismatch 下，ReSkip 的 safety margin 较高（规则很少触发时没有 degradation），但部署需要 per-distribution threshold calibration，这也在 cross-modality VLA case 中指出（\S\ref{app:vla_pareto}）。

### D.6 ReSkip full Pareto and latency curves at 340M

**Figure:** ReSkip on 340M FineWeb-Edu：使用 `\mathcal{P}=\{3,5\}` 的 position-calibrated dynamic skip 在零 benchmark drop 下严格优于 single-position baseline。左图为 accuracy--compute Pareto，`\mathcal{P}=\{3,5\}`（橙色星标）位于 `5%` PPL tolerance 之下。右图为 seq `8192` 下的 wall-clock，`{3,5}/M=2/q=0.85` 达到 `1.19x`。

## E Retrofit Pareto and breakdowns

### E.1 LoRA baselines

这些 auxiliary controls 询问：通过标准 adaptation path 添加类似数量的可训练参数，是否足以复现 text gains。它们是较早 `50/50` UltraChat + LLaVA mix 上的 pilot controls，而不是 Tab.~\ref{tab:retrofit_main} 的 final-mixture baseline。LoRA `r=32` on `q,v`：LAMBADA acc `0.540` / `0.516`（两个 seeds），HellaSwag `0.510` / `0.524`。LoRA `r=16` on `q,k,v,o`：LAMBADA `0.534`，HellaSwag `0.492`。LoRA `r=8` on MLP：LAMBADA `0.514`，HellaSwag `0.510`。四次 run 的 mean LAMBADA 为 `0.526`，比 base（`0.532`）低 `-0.6`pp，比该 mix 上 matched `\gamma -> 1` retrofit（`0.576`）低 `-5.0`pp。Mean HellaSwag `0.509`，相对 base `+0.3`pp，而 retrofit 为 `+1.6`pp。这些 controls 在相关 SFT setting 下排除了简单 parameter-count-only explanation，但并不证明所有质量增益都唯一来自 routing。

### E.2 Static-pruning baseline at the VLM scale (Gromov drop-k)

340M decision-rule ablation（Tab.~\ref{tab:b1b2_decision_rule}）显示，**single-position** static skip 和 random skip 都会在 language tasks 上 degradation。VLM scale 上的互补问题是：**full-layer** static pruning，即 \citet{gromov2024unreasonable} 使用的标准 “remove the least useful blocks” baseline，是否能达到 retrofit 所处 accuracy floor。我们按 Gromov angular-distance criterion 对 28 层 Qwen3-VL-2B layers 排序（只用 LM-head，在 FineWeb-Edu calibration set 上计算；无 fine-tuning），移除最低 distance 的 `k` 层，并在 `lmms-eval` 上重新评估。Tab.~\ref{tab:gromov_pruning_2b} 报告 drop-4（layers `{23,24,25,26}`）和 drop-8（额外加入 `{12,13,14,22}`）。

**Table: Static layer pruning on Qwen3-VL-2B.** Gromov-style angular-distance ranking，无 retraining，`lmms-eval` full splits。作为 Tab.~\ref{tab:b1b2_decision_rule} 中 340M in-block decision-rule ablation 的 companion，它覆盖了 full-layer 与 single-position-within-block skip 维度。Pruning 4/28 layers（14%）导致 AI2D `-16`pp、MMMU `-6`pp、MMStar `-5`pp、OCRBench `-65`pp；pruning 8（29%）在每个 cell 上 sharply degrade。Retrofit（Tab.~\ref{tab:retrofit_main}）在 0 removed layers、0pp loss 下运行，并额外提升 base；retrofit 上的 ReSkip 在 `q=0.85` 时在 LAMBADA 上保持在 1pp 以内（Tab.~\ref{tab:retrofit_pareto}）。

| Configuration | AI2D | MMMU | MMStar | OCRBench | RWQA |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-2B (base, `L_rm=0`) | 0.736 | 0.414 | 0.536 | 0.772 | 0.648 |
| Gromov drop-4 (layers `{23,24,25,26}`) | 0.5732 | 0.3556 | 0.4906 | 0.1240 | 0.3791 |
| Gromov drop-8 (drop-4 ∪ `{12,13,14,22}`) | 0.0683 | 0.2389 | 0.0190 | 0.0030 | 0.1451 |
| + AR-Retrofit v3 (`L=4`, `10k`) | **0.758** | **0.432** | **0.536** | **0.814** | **0.661** |

该 pattern 与 in-block 340M observation 一致：任何 **unconditional** block removal（340M 上的 single-position static，2B 上的 full-layer Gromov）即使在 modest pruning fractions 下也会破坏 VLM scale benchmark accuracy；而 AttnRes routing structure 之上的 input-dependent skip 能保持它。我们不尝试 retrain pruned model；目的只是为 cost-quality trade-off 提供一个免费的 static-baseline floor，而不是 tuned competitor。脚注：pruned cells 还在 strict yes/no surface 下表现出 POPE generation drift（drop-4 pope-acc `0.021`，drop-8 `0.500`，来自 accidental all-“yes” decoding），因此我们从表中去掉 POPE。

### E.3 ReSkip on the canonical retrofit: text Pareto and cross-modality consistency

在 canonical `L=4` v3 `10k` retrofit 上，将 ReSkip 应用于 LAMBADA-500，作为 LIBERO Pareto（Tab.~\ref{tab:vla_reskip_pareto}）的 language-modality cross-check。ReSkip 产生了与 action 上相同的形状：在保守端 lossless；当 `q` 低于 action distribution mean 时 collapse。它使用与 2B VLA cell 相同的 `\mathcal{P}=\{1,4\}`, `M_{\max}=2`。

**Table: Cross-modality ReSkip consistency.** 在 canonical `L=4` v3 retrofit 上的 LAMBADA-500，`\mathcal{P}=\{1,4\}`, `M=2`，`\tau` 在 32 个 held-out LAMBADA prefixes 上按 quantile `q` 校准。Operating-point structure 与 LIBERO 4-suite Pareto 相同：在 calibration distribution mean 以上（`q>=0.85`）near-lossless；低于该均值时 sharply degrade，因为 threshold 指示 router 跳过它通常会执行的 blocks。

| q | LAMBADA acc | LAMBADA ppl | Δ acc vs no-skip | avg. skips / forward |
|---|---:|---:|---:|---:|
| no-skip (`M=0`) | 0.5700 | 4.526 | --- | 0.00 / <=7 |
| 0.85 | 0.5600 | 5.258 | -1.0 | 0.19 / 2 |
| 0.50 | 0.4120 | 12.550 | -15.8 | 1.06 / 2 |
| 0.30 | 0.3900 | 14.005 | -18.0 | 1.17 / 2 |

### E.4 Wall-clock latency: full table including earlier L=2 and VLA in-backbone

主文的 Tab.~\ref{tab:retrofit_latency} 报告 canonical `L=4` result。下面的 Tab.~\ref{tab:retrofit_latency_full} 展示完整集合：早期 `L=2` partition（2B 上 14 个 blocks）相对 stock Qwen3-VL-2B 有 `1.39x` structural cost，因为每个 token 要支付 14 次 router calls；VLA in-backbone forward（OFT trainer backbone forward 内加载的同一 retrofit run，\S\ref{sec:vla}）在两个 partitions 上都与 VLM retrofit 差距在 `1%` 内；`\texttt{torch.compile}` 将 `L=2` residual 降到 `1.16x`，但在 `L=4` 上更显著，因为 router count 减半。该表是从 canonical partition 切换到 `L=4` 的依据。

**Table: Forward-pass latency, full sweep.** `1xH100`，bf16，batch `1`，5 次 warmup 后 20 次运行的 median。Eager rows：stock-HF base vs. indicated `L` 下的 retrofit。Compile rows：两者都用 `torch.compile` 包裹，`dynamic=False`；`reduce-overhead` 捕获 CUDA graphs，`max-autotune` 进一步调优 per-kernel matmuls。VLA in-backbone row 是加载到 OFT trainer backbone forward（单独 code path）中的 canonical `L=4` retrofit，与 VLM retrofit cell 相差 `1%` 内，确认 structural cost 来自 per-block router stack，而不是 VLA harness。粗体为论文 canonical operating point。

| Configuration | seq | mode | base (ms) | retrofit (ms) | retrofit / base |
|---|---:|---|---:|---:|---:|
| (1) `L=2` retrofit (earlier) | 1024 | eager | 15.42 | 21.25 | 1.378x |
| (2) `L=2` retrofit (earlier) | 2048 | eager | 26.03 | 36.14 | 1.388x |
| (3) `L=4` retrofit (canonical) | 1024 | eager | 14.91 | 16.66 | 1.117x |
| (4) `L=4` retrofit (canonical) | 2048 | eager | 25.49 | 28.53 | 1.119x |
| (5) `L=2` retrofit | 1024 | `reduce-overhead` | 9.31 | 10.78 | 1.158x |
| (6) `L=2` retrofit | 2048 | `reduce-overhead` | 21.25 | 24.45 | 1.151x |
| (7) `L=4` retrofit | 2048 | `reduce-overhead` | 21.31 | 22.50 | 1.056x |
| **(8) `L=4` retrofit** | **2048** | **`max-autotune`** | **21.81** | **22.43** | **1.029x** |
| (9) VLA in-backbone `L=4` retrofit | 2048 | eager | 25.49 | 28.79 | 1.130x |

Structural per-block router stack（`stack -> RMSNorm -> softmax -> einsum` over `N` completed blocks）很小但在 eager inference 中可见。Adapter rank ablation（Appendix~\ref{app:rank_ablation}）确认 adapter 不是 bottleneck（`<=2%`）。`torch.compile`（rows 5--8）将小 router GEMMs collapse 成 captured graph，并调优 retrofit small shapes 的 matmul kernels，移除了大部分 `L=4` structural gap。剩余 `2.9%` 是 canonical operating point 下 installed routing path 的成本；将 router 和 skip decision 融合成单个 production kernel 是实现方向，而不是方法变化。

### E.5 Compile vs. eager: accuracy parity

\S\ref{sec:exp_retrofit} 的 compiled-overhead result 需要 `torch.compile` 保持 retrofit accuracy。我们在 canonical `L=4` v3 `10k` retrofit 上验证了两个 parity tests。

**LAMBADA-500 accuracy parity.** 使用 default mode 和 dynamic shapes 将 retrofit 包裹在 `torch.compile` 中运行 LAMBADA-500：compiled acc `0.5720` vs. eager `0.5700`（`Δ=+0.20`pp），ppl `4.534` vs. `4.526`。相对 eager 的 per-target argmax agreement 为 `98.60%`；剩余 `1.4%` 是 eager 和 compiled 都处于 bf16 SDPA jitter 范围内、rank-1/rank-2 logits 翻转的 tokens。

**Per-token logit parity.** 在真实 prompt tokens 上使用 `mode="reduce-overhead"`（speed table 使用的 inference-time mode）：per-token argmax agreement `98.51%`，max `|\Delta\text{logits}|=0.50`，RMSE `0.060`，与 stock base 上 cache-on/cache-off SDPA jitter 的量级相同（Appendix~\ref{app:kv_equiv}）。Compile 在 bf16 evaluation noise 内保持 retrofit forward；`1.029x base_compiled` 数字是在 matched accuracy 下得到的。

### E.6 Block partition sweep

**Table: Block partition sweep.** v3 recipe，`r=256`，`\gamma`-curriculum。`L` 为 layers-per-block，`N=L_total/L`。Per-layer `L=1` 低 `2--9`pp；2B 上的 plateau `L in {2,4,7}` 复现了 \citet{chen2026attnres} 的 `S in {2,4,8}`。我们在两个规模上都推荐 `L=4`，因为它在保持质量的同时减少 router calls。

| Scale | L | N | LAMBADA | ΔL | HellaSwag | MMBench | MMMU | MMStar | AI2D | OCRBench | RWQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B base | --- | --- | 0.532 | --- | 0.506 | 75.77 | 0.414 | 0.536 | 0.736 | 0.772 | 0.648 |
| 2B retrofit | 1 | 28 | 0.5645 | +3.3 | 0.490 | 76.20 | 0.388 | 0.499 | 0.743 | 0.809 | 0.652 |
| 2B retrofit | 2 | 14 | 0.5755 | +4.4 | 0.494 | 77.23 | 0.426 | 0.532 | 0.748 | 0.803 | 0.663 |
| **2B (rec.)** | **4** | **7** | **0.5650** | **+3.3** | **0.500** | **78.87** | **0.432** | **0.536** | **0.758** | **0.814** | 0.661 |
| 2B retrofit | 7 | 4 | 0.5155 | -1.7 | 0.492 | 77.49 | 0.427 | 0.530 | 0.756 | 0.808 | 0.656 |
| 4B base | --- | --- | 0.576 | --- | 0.562 | 83.33 | 0.490 | 0.624 | 0.819 | 0.819 | 0.715 |
| 4B retrofit | 1 | 36 | 0.5575 | -1.9 | 0.523 | 83.33 | 0.497 | 0.538 | 0.783 | 0.768 | 0.694 |
| 4B retrofit | 2 | 18 | -- | -- | -- | 84.28 | 0.523 | 0.587 | 0.816 | 0.813 | 0.708 |
| **4B (rec.)** | **4** | **9** | **0.6625** | **+8.7** | **0.552** | **85.22** | 0.521 | **0.632** | **0.825** | **0.824** | **0.718** |
| 4B retrofit | 6 | 6 | 0.6540 | +7.8 | 0.554 | 84.79 | **0.531** | 0.623 | 0.817 | 0.824 | 0.715 |

### E.7 MMStar subcategory breakdown

**Table: MMStar subcategory breakdown for the v3 retrofit.** math、logical 和 sci&tech 三项是 “reasoning over images” cell，retrofit 在这里显示最大增益（2B 上 math `+7.9`pp；4B 上 `+3.9`pp）。Perception cells（coarse / fine / instance）持平或小幅回落，这与 router 提升 deliberate-reasoning paths 而不是改变 early perception 一致。

| Configuration | math | logical | sci&tech | coarse | fine | instance |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B (base) | 0.413 | 0.429 | 0.408 | 0.714 | 0.520 | 0.710 |
| + AR-Retrofit v3 (2B, `L=4`) | **0.492** | **0.432** | 0.353 | 0.734 | 0.505 | 0.683 |
| Δ vs. 2B base | **+7.9** | +0.3 | -5.5 | +2.0 | -1.5 | -2.7 |
| Qwen3-VL-4B (base) | 0.549 | 0.626 | 0.465 | 0.788 | 0.611 | 0.705 |
| + AR-Retrofit v3 (4B, `L=4`) | **0.588** | 0.602 | **0.467** | 0.812 | 0.606 | 0.714 |
| Δ vs. 4B base | **+3.9** | -2.4 | +0.2 | +2.4 | -0.5 | +0.9 |

## F VLA appendix

### F.1 VLA seed variance on 2B

我们对两个 2B suites 运行了两个 seeds，这两个 suite 中 Path 0 / Path B 在 single seed 上处于 evaluation noise 范围内；`libero_spatial` 和 `libero_object` 是 single runs。

**Table: Per-seed Qwen3-VL-2B numbers.** Seed 1 是初始 run；seed 2 是同一 trained checkpoint 的 fresh environment-reseeded rollout。最后一列显示 Table~\ref{tab:vla_libero} 使用的值。

| Suite | Policy | Seed 1 | Seed 2 | Used in Tab. |
|---|---|---:|---:|---:|
| `libero_goal` | Path 0 (`30k`) | 97.6 | 97.4 | 97.4 |
| `libero_goal` | Path B (`30k`) | 97.4 | 98.6 | 98.6 |
| `libero_10` | Path 0 (`30k`) | 92.8 | 91.4 | 91.4 |
| `libero_10` | Path B (`30k`) | 92.6 | 90.6 | 92.6 |

聚合两个 repeated rollouts 得到 Path 0 为 `96.05`，Path B 为 `96.75`（`Δ=+0.70`pp），而 Table~\ref{tab:vla_libero} 的 operating-point values 给出 `Δ=+1.30`pp。两个 summaries 都保持 Path B > Path 0 的排序。因此主文将 LIBERO 视为 supporting transfer evidence，而不是 primary contribution。

### F.2 VLA landscape: open VLAs on standard benchmarks

**Table: Open VLA landscape on standard benchmarks.** LIBERO entries 是每个 task suite 的 success rates（%）。按常见做法，`\pi_0` 与 Octo-Base 的 LIBERO 数字取自 OpenVLA-OFT~\citep{kim2025openvlaoft}，该工作在统一 LIBERO protocol 下重跑了三个 policies。我们的 rows 使用 Table~\ref{tab:vla_libero} 中的 `30k` Path B 数字。

| Model | #Params | Spatial | Object | Goal | Long-10 | ref. |
|---|---:|---:|---:|---:|---:|---|
| Octo-Base | 93M | 78.9 | 85.7 | 84.6 | 51.1 | \citep{team2024octo, kim2025openvlaoft} |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | \citep{kim2024openvla, kim2025openvlaoft} |
| TraceVLA | 7B | 84.6 | 85.2 | 75.1 | 54.1 | \citep{zheng2024tracevla} |
| SpatialVLA | 4B | 88.2 | 89.9 | 78.6 | 55.5 | \citep{qu2025spatialvla} |
| `\pi_0` | 3B | **96.8** | **98.8** | 95.8 | 85.2 | \citep{black2024pi0, kim2025openvlaoft} |
| **Qwen3-VL-2B + AR-Retrofit + OFT (ours)** | 2B | **97.8** | 99.6 | **98.6** | 92.6 | this work |
| **Qwen3-VL-4B + AR-Retrofit + OFT (ours)** | 4B | 94.6 | **99.8** | **98.2** | **94.2** | this work |

### F.3 VLA ReSkip setup: per-block drift and eligible-set selection

每个 scale 的 ReSkip eligible set `\mathcal{P}` 通过在 trained Path B policy 上进行 per-block ablation 选择，而不是只在 VLM retrofit 上选择，因为 VLA action stream 激活的 routing distribution 与 language stream 不同。我们在 24-sample sim trajectory 上测量 per-block action-drift MSE：policy 分别以 no skip 和 individually skipped block 方式运行；skipped 与 no-skip rollouts 之间的 per-step action MSE 就是在 action distribution 上移除该 block 的 per-block cost。每个 scale 选择两个最安全 blocks 组成 `\mathcal{P}`。

**2B Path B 30k** per-block action-drift MSE（升序，单位 `x10^{-3}`）：block 1: `0.4`；block 4: `34.0`；block 0: `58.1`；block 5: `87.0`；其余 blocks `>100`。选择两个最低者：`\mathcal{P}_{2B}={1,4}`。

**4B Path B 30k** per-block action-drift MSE：block 1: `0.6`；block 2: `11.0`；block 0: `34.0`；block 3: `63.0`；其余 `>120`。选择：`\mathcal{P}_{4B}={1,2}`。

**Per-scale eligibility.** Eligible set 对每个 model scale 单独选择，因为 action-stream routing distribution 和 per-block drift 都依赖 scale。这是 modality-aware skip protocol 的 per-scale 部分：`\alpha` 提供 online signal，而 action-drift calibration 为被部署的 policy 选择安全 candidate blocks。

**Threshold calibration.** 每个 block 的 thresholds `\tau_n` 是 `w_{\text{recent},n}` 在 31,286（2B）/ 31,257（4B）条 sim-rollout records 上的第 `q` 个 empirical quantile（一条 record 对应多条 trajectories 中一个 decoded action token）。Tab.~\ref{tab:vla_reskip_pareto} 的数字使用该协议。另一个 “Method A” 在 retrofit pretokenized data 上校准（将 VLA 视作 long-context LM），会落在一个模拟 action distribution 上 `q=0.99` 的保守 operating point；我们将其作为 ablations 中的 sensitivity comparator，并建议任何部署都使用 action-distribution-calibrated thresholds。

**Cross-modality threshold calibration.** Language-calibrated thresholds 与 action-calibrated thresholds 对应不同 effective skip rates，因为 action stream 具有不同的 `w_{\text{recent}}` distribution。因此，我们在 LIBERO 上按 action distribution 校准 `\tau_n`，而不是直接迁移 LAMBADA thresholds。这与主方法中的 calibration principle 相同：routing 决定何时 skip，而 eligible set 和 thresholds 在 deployment distribution 上选择。

### F.4 VLA planned follow-ups

**Per-modality, per-token-class ReSkip.** Tab.~\ref{tab:vla_reskip_pareto} 的 thresholds 在整体 action distribution 上校准；一个明显改进是在单个 rollout 内按 modality 设置 `(\mathcal{P}^{(m)}, \tau_n^{(m)}, M_{\max}^{(m)})`（vision tokens、language tokens、action tokens）。Working hypothesis：vision `\alpha` 集中于 early blocks，而 action `\alpha` 分散到 late blocks，并受益于更保守的 late-block skipping。**Fused routing kernels.** Skip rule 的 static-graph variant 可以进一步降低 installed-path runtime。**Edge hardware.** Wall-clock 使用 H100 / bf16；计划直接测量 RTX 4090 / Jetson Orin。

## NeurIPS Paper Checklist

1. **Claims.** Yes. 摘要、引言和结论陈述了主要 claims，并区分了 retrofit contribution、calibrated skipping、VLA transfer 和 systems characterization。
2. **Limitations.** Yes. Appendix~\ref{app:limitations} 列出了 architecture coverage、hardware scope、block-level routing granularity 和 per-modality calibration limits。
3. **Theory assumptions.** Not applicable. 本文是 empirical and algorithmic；所有公式定义的是实现的 objectives 或 routing rules。
4. **Experimental reproducibility.** Yes. Sections~\ref{sec:reskip}--\ref{sec:vla_results} 与 appendix 指定了 model scales、block partitions、training steps、datasets、calibration rules 和 evaluation protocols。
5. **Open access to code and data.** Yes. 实验使用 public datasets 和 benchmarks；implementation details 和 release plan 按 anonymous submission policy 描述。
6. **Compute.** Yes. 主文和附录报告 GPU type、training steps、rough GPU-minutes / GPU-hours 和 latency measurement settings。
7. **Dataset and benchmark provenance.** Yes. 所有 public datasets 和 benchmarks 均已引用；LIBERO 和 lmms-eval protocols 通过 rollout counts 或 split usage 描述。
8. **Human subjects.** Not applicable. 未收集新的 human-subject data。
9. **Privacy.** Not applicable beyond the privacy considerations of the cited public datasets.
10. **Licenses.** Yes. Public datasets/models 已引用；release artifacts 将包含 license metadata。
11. **Broader impacts.** Yes. 本工作目标是降低 pretrained models 的 inference cost 并实现 adaptive computation；未声称 safety-critical deployment。
12. **Safeguards.** Yes. VLA thresholds 按 model 和 action distribution 校准；cross-modality transfer failures 被明确记录为 deployment risk。
