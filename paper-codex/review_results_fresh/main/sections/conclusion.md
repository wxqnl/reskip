Conclusion.
At a fair rate-matched comparison (∼12% fired skips), input-dependent dynamic skip (B2.c) preserves
LAMBADA at 0.345 while every static or random alternative degrades to 0.22–0.26 (−8 to −13pp). The same ordering
holds on HellaSwag, PIQA, ARC-easy and OpenBookQA. The MoD-style “you might be getting a free lunch from
any same-rate schedule” critique is rejected: at 340M from-scratch, schedule choice matters and the ATTNRES routing
weights carry the load.
Threshold-transfer caveat.
Two of the three dynamic rules (B1.b recent_weight_gt and B2.b entropy_lt)
fire far below the 12.5% FineWeb-Edu calibration target on the LAMBADA distribution (Tab. 14). Their near-no-skip
accuracy is therefore not a positive datapoint for those specific rules; it is consistent with “the threshold protected the
17

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
output by accidentally not firing.” The load-bearing rate-matched comparison is B2.c recent_minus_embed_gt
(which does fire close to target) versus the static/random alternatives. We treat the two non-firing rows as a sensitivity