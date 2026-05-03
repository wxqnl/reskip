## Related Work
For test-time compute, o1 [34], DeepSeek-R1 [8], chain-of-thought [46], and optimal test-time compute analyses [41]
allocate compute by emitting more reasoning tokens; per-token depth remains fixed. Depth-axis methods include ACT
/ Universal Transformers [15, 9], CALM [40], Mixture-of-Depths [39], LayerSkip [13], and static pruning [16, 32].
These add a halting head, exit classifier, router, speculative decoder, or static removal rule. We instead use ATTNRES
as an intrinsic routing signal produced by the forward pass itself, then retrofit that signal into an existing pretrained
backbone. Appendix D.5 compares dynamic, static, and random skip rules at 340M, and Appendix E.2 gives a full-layer
static-pruning baseline at the VLM scale.
Table 8 summarises the closest method-level distinction. The relevant distinction is the combination of frozen-
backbone retrofitting and a pre-execution signal that is also part of the forward path.
The neural motivation is recurrence as flexible effective depth. The dual-process distinction [19] maps naturally
onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent
circuits [26, 20, 10]. Recurrent networks capture human representational dynamics [22]; Spoerer et al. [42] are the
closest conceptual antecedent, showing that recurrent CNNs can spend more steps on harder images to explain speed–
accuracy behaviour. Recurrence also has a depth interpretation: finite recurrence can be unrolled into feedforward
7

Table 6: LIBERO 4-suite success rates (%) for Qwen3-VL-2B/4B. Entries are success rates over 500 rollouts per
suite (2,000 per policy). Path 0 is the stock-backbone OFT baseline; Path B warm-starts from our L=4 VLM retrofit.
The table is supporting transfer evidence for the retrofit pathway rather than a broad robotics claim.
Scale
Path (steps)
Spatial
Object
Goal
Long-10
Avg
∆vs. Path 0
2B
Path 0 (30k)
94.8
99.8
97.4
91.4
95.85
—
2B
Path B (30k, ours)
97.8
99.6
98.6
92.6
97.15
+1.30
4B
Path 0 (30k)
95.0
99.2
97.8
92.2
96.05
—
4B
Path B (30k, ours)
94.6
99.8
98.2
94.2
96.70
+0.65
Table 7: RESKIP 4-suite Pareto on the trained Path B policies. Eligible blocks P and threshold τ are calibrated per
scale on the action distribution; q is the empirical quantile that produces τ. The conservative end (q=0.99) preserves
success rates in this evaluation.
q (sim-calibrated)
Spatial
Object
Goal
Long-10
Avg
Qwen3-VL-2B Path B, P={1, 4}, M=2, no-skip ref 96.25
0.30
4.7
—
—
—
(sharp drop)
0.85
80.0
98.0
87.3∗
67.2
83.1
0.95
95.0
99.4
97.6
86.8
94.7
0.99
97.6
99.2
99.0
93.6
97.35
no-skip (ref)
97.4
98.6
98.0
91.0
96.25
Qwen3-VL-4B Path B, P={1, 2}, M=2, no-skip ref 96.25
0.30
89.6
95.4
96.4
86.0
91.85
0.50
91.2
98.0
98.2
89.6
94.25
0.85
93.6
99.2
97.6
93.2
95.90
0.95
95.6
98.0
98.0
93.0
96.15
0.99
96.4
98.2
98.4
92.8
96.45
no-skip (ref)
97.4
98.2
98.0
91.4
96.25
∗Partial eval (332/500 trials) — driver schedule cut short; reported as the rate over completed trials.
depth, but the recurrent form reuses a fixed substrate for flexible effective depth [45]. We take this as a design target,
not a cognitive-faithfulness claim: cortical microcircuits and dendritic association [3, 6, 27] motivate intrinsic routing,
while our implementation is a transformer retrofit.
In residual architectures, DenseNet [18] and Highway networks [43] modify residual flow; ATTNRES [25] gener-
alises residual combination with depth attention. To our knowledge, this is the first work to use ATTNRES weights as a
pre-execution skip signal and to retrofit an ATTNRES-style routing path into pretrained transformers. In VLA, RT-2 [5],
Octo [33], OpenVLA [23], and Pi0 [4] established the paradigm, while efficiency work has focused mainly on action
chunking [51] and post-hoc compression. We contribute modality-aware adaptive depth for the VLM backbone.
## 6