Abstract
Pretrained transformers execute every layer for every input. Adding input-dependent depth allocation
typically requires either a separate router / exit head, or rebuilding the residual structure from scratch. We
study a third option: installing an intrinsic depth-routing signal into a standard pretrained transformer
through a short fine-tune. Building on Attention Residuals (ATTNRES) [24], whose softmax routing over
previous block outputs is computed before each block runs, our main contribution is AR-RETROFIT,
an identity-preserving γ-gated residual-injection fine-tune that freezes the base and adds well under 1%
new parameters. On Qwen3-VL-2B and 4B, the retrofit improves the base on 5/6 lmms-eval VLM
benchmarks at both scales, while preserving efficient compiled inference (1.029× base_compiled). A
LIBERO VLA policy warm-started from the retrofit improves a matched OFT baseline at both 2B and 4B.
We further show that the installed routing weights can be calibrated, with a complementary safety check,
into a dynamic block-skip rule (RESKIP) with no auxiliary head, and that the same signal supports adaptive
depth on the action stream of the trained policy. The result is a practical mechanism for installing depth-axis
adaptive computation into already-trained transformers, rather than retraining the residual structure from
scratch.
## 1