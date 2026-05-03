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
D.3
RESKIP method comparison and full Pareto / latency at 340M
D.4
RESKIP position-set ablation at 340M
D.5
Decision rule: dynamic vs. static-rate-matched vs. random
The position-set ablation above answers which blocks to skip; this subsection answers whether the input-dependent
decision matters at all. We hold the 340M from-scratch ATTNRES weights fixed, fix P={3, 5} and Mmax=1 (block-
level skip rate ≤12.5%), and only vary the runtime decision rule:
• B0. No-skip upper bound (full ATTNRES forward).
• B1.b/B2.a. Dynamic, fire when wrecent,n > τn (our RESKIP signal at L=1 block granularity).
• B1.c, B1.d. Static, every-token skip at P=3 or P=5 (rate-matched, 12.5%).
• B1.e/B2.d. Random, per-call uniform draw from {keep-P=3, keep-P=5} (rate-matched, 12.5%).
• B2.b. Dynamic, fire when block-1 entropy Hn < τn (router-confidence rule).
• B2.c. Dynamic, fire when wrecent,n −wembed,n > τn (relative-recent rule).
All three thresholds are calibrated as the q=0.5 per-position quantile on 32×8192 FineWeb-Edu tokens (§C.5). Results
in Tab. 13; observed skip rates on the eval distribution in Tab. 14.