# AttnRes 的残差强度调制：方法与实验证据

## 1. 研究定位

本文改动的主要目的，是增强 AttnRes 自身的适应能力，而不是训练一个专门服务于跳层的控制器。原始 AttnRes 决定当前 block 如何聚合历史残差；本方法进一步允许模型按 token、按 block 调节这次聚合更新的实际强度。ReSkip 只是在推理阶段复用这一原生状态所获得的附加收益。

因此，方法必须同时满足以下约束：

- 模块属于 AttnRes 的一次适应性训练，并参与每次 Full forward；
- 从预训练基座直接训练，不从任何旧候选继续训练；
- 不包含 skip forward、跳层标签、计算量目标、恢复分支或第二阶段训练；
- ReSkip 不要求超过 Full AttnRes，只要求在给定计算削减下尽量保留 Full 的能力。

代码中的实验名为 Residual Strength Gate（RSG）。论文中建议称为 **Residual-Strength Modulation（RSM）**，以突出它是 AttnRes 表达能力的组成部分，而非外挂 skip gate。

## 2. 方法

设第 `b` 个 AttnRes block 在 token `t` 上产生原有残差更新 `u_b(t)`，输入主干状态为 `h_b(t)`。原始形式为：

```text
x_b(t) = h_b(t) + u_b(t).
```

我们为每个目标 block 加入一个 token-conditioned 标量：

```text
g_b(t) = 1 + 0.5 tanh((v_b^T RMSNorm(u_b(t)) + c_b) / sqrt(d)),
x_b(t) = h_b(t) + g_b(t) u_b(t).
```

其中 `d=2048`。`g_b(t)` 的取值限制在 `[0.5, 1.5]`，只调节已有 AttnRes 更新的幅度，不生成新的残差内容，也不改变原有路由 softmax。

### 2.1 初始化与参数量

所有 `v_b` 和 `c_b` 均初始化为零，因而初始时 `g_b(t)=1`，模型与未加入 RSM 的 AttnRes 完全等价；实测初始最大 logit 差为 `0`。

在 blocks 1--6 上各加入一个 `Linear(2048, 1)`，新增参数为：

```text
6 x (2048 + 1) = 12,294.
```

这约占 2.13B 基座参数的 `0.00058%`，占原 AttnRes 适应参数的 `0.167%`。没有任何 skip-specific 或 external-controller 参数。

### 2.2 训练

候选模型从 `Qwen3-VL-2B-Instruct` 基座独立训练 3,000 步，使用与对照一致的 seed、v3 数据混合、rank-256 adapters 和 gamma 调度。目标函数保持为原 Full AttnRes 适应目标：

```text
L = L_task_CE + L_teacher_KL - 0.02 H_native_route.
```

训练过程中跳层前向次数为零。RSM 没有接收跳层结果、skip KL、top-1 flip、覆盖率或速度信号。

## 3. Full AttnRes 结果

正式评测采用论文相同的六个 VLM 任务和完整测试规模。

| 系统 | AI2D | MMBench | MMMU | MMStar | OCRBench | RealWorldQA | 六任务均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base（论文） | 75.648 | 76.203 | 42.333 | 53.754 | 80.700 | 65.229 | 65.644 |
| Full AttnRes（论文） | 76.522 | 78.952 | 45.111 | 54.923 | 83.800 | 64.837 | 67.357 |
| Full RSM-AttnRes | 76.911 | 78.007 | 46.111 | 56.657 | 83.600 | 66.536 | **67.970** |

Full RSM-AttnRes 相对论文原版 Full AttnRes 提升 `+0.613 pp`，相对 Base 提升 `+2.326 pp`。六个任务中四个提高，MMBench 和 OCRBench 分别下降 `0.945 pp` 与 `0.200 pp`；因此应表述为平均能力的小幅、非一致提升，而不声称所有任务均获益。

## 4. 模块是否真的被 Full 模型使用

在 128 个配对 held-out 样本、24,235 个 assistant tokens 上，RSM 相对同配方原版 AttnRes 的 task CE 变化为 `+0.00187`，teacher KL 变化为 `-0.00245`，两者相加的原训练目标变化为 `-0.00058`。teacher top-1 agreement 从 `90.113%` 提高到 `90.291%`。

把训练后的 RSM 临时替换为恒等因子，会使 `CE+KL` 增加 `0.04643`；把各 block 的 RSM 循环错配，会使其增加 `0.01802`。这说明模型使用了 token/block 对应的调制，而不是简单吸收了一个全局缩放常数。

在同一 held-out 集上，各 block 因子的均值范围为 `0.517--0.732`，标准差范围为 `0.060--0.321`。Block 4 的分布最宽，Blocks 1/2 更集中在较低强度。该结果支持“不同深度和 token 对 AttnRes 更新的需求不同”，但部分 block 接近下界的现象仍需在后续 seed replication 中观察。

## 5. ReSkip：作为推理期附加收益

ReSkip 仍是 token-level 动态跳层。基础条件来自 AttnRes 原生的 `w_recent`；RSM 不另行训练 selector。后验 held-out 分析显示，在 Block 1 中加入 `g_1(t)` 的简单上界条件后：

- 平均 forced-skip 风险从 `0.1813` 降到 `0.1452`；
- 最大 micro-action KL 从 `0.1444` 降到 `0.1146`；
- top-1 flip rate 从 `14.76%` 降到 `12.24%`；
- 覆盖率仅从 `47.24%` 变为 `44.82%`。

固定该规则后，在六任务各 256 样本的实际解码屏测中：

| 系统 | 六任务均值 | 相对 Full | block-equivalents / decode token |
|---|---:|---:|---:|
| Full RSM-AttnRes | 57.391 | 0.000 | 0.000 |
| ReSkip，仅 `w_recent` | 56.892 | -0.498 | 1.845 |
| ReSkip，`w_recent` + RSM 原生因子 | **57.137** | **-0.253** | **1.851** |

两种 ReSkip 的实际计算削减几乎相同，而 RSM 条件收回约 `0.245 pp`。这说明 RSM 因子在该屏测分布上提供了增量安全性信息，但不能由此直接推出完整数据集上的安全性。

随后进行的完整规模评测证实了这一边界：激进策略虽然削减 `1.805 block-equivalents/token`，六任务均值却降至 `64.350`，比 Full 低 `3.620 pp`。其中 OCRBench 从 `83.6` 降到 `64.1`，说明“每任务前 256 样本”对 OCRBench 后段分布没有代表性。该结果作为失败实验保留，不能作为论文主结果。

基于训练前已定义的 forced-skip 风险排序，我们冻结了一次保守收缩：保持所有阈值不变，只移除风险较高的 Blocks 4/5 动作和 Block 1 的 layer offset 2。该版本只进行了一次完整六任务评测，没有再根据结果调整。

| 正式系统 | 六任务均值 | 相对自身 Full | block-equivalents / decode token |
|---|---:|---:|---:|
| Full RSM-AttnRes | 67.970 | 0.000 | 0.000 |
| RSM-ReSkip（保守） | **67.713** | **-0.257** | **0.959** |

保守 RSM-ReSkip 在削减约一个等价 block 的同时，仍比论文原版 Full AttnRes 高 `0.356 pp`，比论文原版 ReSkip 高 `0.599 pp`。这才是当前可以用于论文主表的 ReSkip 结果。激进策略只作为负结果归档。

## 6. 应如何写进论文

建议的主张是：

> We augment AttnRes with a lightweight token-wise residual-strength modulation learned jointly during its ordinary adaptation. The modulation improves the full-compute model and, without skip-aware training or a separate controller, exposes an additional native confidence signal that makes token-level ReSkip safer at inference time.

不建议写成：

- 为 ReSkip 专门训练了一个 gate；
- RSM 能可靠预测任意 block 的 skip risk；
- ReSkip 应超过 Full AttnRes；
- 当前单 seed 结果已经证明跨模型或跨规模普适性。

## 7. 当前结论与边界

当前方案已经达到本轮的两个目标：它在不引入跳层训练的前提下，把 Full AttnRes 的正式六任务均值提高了 `0.613 pp`；保守 token-level ReSkip 又以 `0.257 pp` 的代价削减了 `0.959 block-equivalents/token`。方法本身只有一个清晰机制、12,294 个参数和一次联合适应训练，不需要继续叠加新的判别器或损失项。

论文定稿前仍应补充训练 seed replication，并将正式 ReSkip 全量结果与设备侧速度结果分开报告。前者验证统计稳健性，后者验证节省的 block 计算是否真正转化为 wall-clock 加速。
