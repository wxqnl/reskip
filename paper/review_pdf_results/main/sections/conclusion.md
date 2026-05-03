Conclusion.
At a fair rate-matched comparison (∼12% fired skips), input-dependent dynamic skip (B2.c) preserves
LAMBADA at 0.345 while every static or random alternative collapses to 0.22–0.26 (−8 to −13pp). The same ordering
holds on HellaSwag, PIQA, ARC-easy and OpenBookQA. The MoD-style “you might be getting a free lunch from
any same-rate schedule” critique is rejected: at 340M from-scratch, schedule choice matters and the ATTNRES routing
weights carry the load.
Threshold-transfer caveat.
Two of the three dynamic rules (B1.b recent_weight_gt and B2.b entropy_lt)
fire far below the 12.5% FineWeb-Edu calibration target on the LAMBADA distribution (Tab. 12). Their near-no-skip
accuracy is therefore not a positive datapoint for those specific rules; it is consistent with “the threshold protected the
output by accidentally not firing.” The load-bearing rate-matched comparison is B2.c recent_minus_embed_gt
(which does fire close to target) versus the static/random alternatives. We treat the two non-firing rows as a sensitivity