Abstract
Pretrained transformers execute nearly the same depth for every input. We study how to retrofit a pretrained
transformer with an intrinsic pre-execution routing signal that can drive dynamic depth decisions without
training a separate router or retraining the model from scratch. We introduce AR-RETROFIT, an identity-
preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with
well under 1% new parameters. The resulting routing weights are computed before each block executes,
enabling RESKIP, a calibrated dynamic block-skipping rule that selects eligible blocks using offline safety
checks and applies input-dependent skipping at inference. On Qwen3-VL-2B and 4B, AR-RETROFIT
preserves or improves most VLM benchmarks and substantially improves LAMBADA, while RESKIP
provides calibrated input-dependent depth allocation on the retrofitted model, with conservative operating
points preserving quality within 1pp on LAMBADA-500. A 340M controlled study shows that the dynamic
routing rule outperforms static or random schedules at matched skip rates, and LIBERO experiments
provide supporting evidence that the same routed-depth signal remains usable for calibrated action-stream
skipping. These results show that pretrained transformers can acquire usable depth-axis routing through a
lightweight retrofit rather than from-scratch residual pretraining.
## 1