Method
Auxiliary network
Routing is
Train-time changes
Architectural changes
CALM [40]
exit classifiers
per-token
yes
small
Mixture-of-Depths [39]
router
per-token
yes, capacity loss
yes
Static pruning [16]
none
static
no
block removed
LayerSkip [13]
spec. decoder
per-token
yes, cont. pretrain
decoding path
RESKIP (ours)
none
per-batch
none
none
D.4
RESKIP position-set ablation at 340M
Table 12: Dynamic skip as a function of P on 340M from-scratch ATTNRES (Mmax=2, q=0.85). Picking by I alone
({5}) or A alone ({3}) leaves speed or quality on the table; combining the two ({3, 5}) is strictly best.
P
PPL ratio
Speedup
{5} (I-only)
0.993
1.14×
{3} (A-only)
1.009
1.02×
{2, 3}
1.009
1.17×
{4, 5}
1.008
0.97×
{3, 4, 5}
0.993
1.02×
{2, 3, 4, 5, 6}
1.40
1.16×
{3, 5} (ours)
0.991
1.19×
D.5
Decision rule: dynamic vs. static-rate-matched vs. random
The position-set ablation above answers which blocks to skip; this subsection answers whether the input-dependent
decision matters at all. We hold the 340M from-scratch ATTNRES weights fixed, fix P={3, 5} and Mmax=1 (block-
level skip rate ≤12.5%), and only vary the runtime decision rule:
• B0. No-skip upper bound (full ATTNRES forward).
• B1.b/B2.a. Dynamic, fire when wrecent,n > τn (our RESKIP signal at L=1 block granularity).
• B1.c, B1.d. Static, every-token skip at P=3 or P=5 (rate-matched, 12.5%).
• B1.e/B2.d. Random, per-call uniform draw from {keep-P=3, keep-P=5} (rate-matched, 12.5%).
16

• B2.b. Dynamic, fire when block-1 entropy Hn < τn (router-confidence rule).
• B2.c. Dynamic, fire when wrecent,n −wembed,n > τn (relative-recent rule).
All three thresholds are calibrated as the q=0.5 per-position quantile on 32×8192 FineWeb-Edu tokens (§C.5). Results
in Tab. 13; observed skip rates on the eval distribution in Tab. 14.
Table 13: Decision-rule and threshold-rule ablation on 340M from-scratch ATTNRES, lm-evaluation-harness with
P={3, 5}, Mmax=1. Dynamic rules whose threshold actually fires on the eval distribution (B2.c) preserve LAMBADA
at 0.345 vs. static / random at the same ∼12% rate (0.22–0.26). Dynamic rules whose calibrated thresholds rarely cross
on lm-eval data (B1.b at 3.1% observed; B2.b at 0.0%, see Tab. 14) are nearly equivalent to no-skip and are not the
rate-matched comparison.
Cell
LAMBADA
HellaSwag
PIQA
ARC-e
OpenBookQA
B0. no-skip (upper bound)
0.4054
0.4607
0.6893
0.5438
0.3580
Dynamic, calibrated q=0.5 on FineWeb-Edu
B1.b. recent_weight_gt
0.4011
0.4534
0.6839
0.5412
0.3580
B2.b. entropy_lt
0.4036
0.4608
0.6839
0.5417
0.3580
B2.c. recent_minus_embed_gt
0.3445
0.4471
0.6774
0.5391
0.3540
Static / random, rate-matched at 12.5%
B1.c. static skip P=3 every
0.2624
0.3968
0.6610
0.5206
0.3300
B1.d. static skip P=5 every
0.2189
0.4223
0.6638
0.5253
0.3260
B1.e/B2.d. random {P=3 OR P=5}
0.2416
0.4028
0.6420
0.5101
0.3300
Table 14: Observed skip rate on 8 × 512 LAMBADA tokens. The calibration set (FineWeb-Edu) and the eval set
(LAMBADA) have different routing-statistic distributions; B2.b’s entropy_lt threshold never fires on LAMBADA
and B1.b fires only 3.1%, so their headline-table accuracy is misleadingly close to no-skip. B2.c is the strategy whose
calibration target (∼12.5%) actually transfers, and is therefore the load-bearing rate-matched comparison against
B1.c/d/e in Tab. 13.
Cell
Calibration target
Observed (LAMBADA)
Notes
B0
0%
0.00%
sanity check, τn=∞
B1.b
≤12.5%
3.12%
τ3=0.4645, τ5=0.4010
B2.b
≤12.5%
0.00%
τ3=0.8631, τ5=0.8764
B2.c
≤12.5%
10.94%
τ3=0.1655, τ5=0.2472
B1.c
12.5%
12.50%
forced, every-token P=3
B1.d
12.5%
12.50%
forced, every-token P=5
B1.e
12.5%
per-batch toggle
uniform {P=3, P=5}