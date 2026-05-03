## Related Work
For test-time compute, o1 [34], DeepSeek-R1 [8], chain-of-thought [45], and optimal test-time compute analyses [41]
allocate compute by emitting more reasoning tokens; per-token depth remains fixed. Depth-axis methods include ACT
/ Universal Transformers [15, 9], CALM [40], Mixture-of-Depths [39], LayerSkip [13], and static pruning [16, 32].
These add a halting head, exit classifier, router, speculative decoder, or static removal rule. We instead use ATTNRES
8

Table 6: RESKIP 4-suite Pareto on the trained Path B policies. Eligible blocks P and threshold τ are calibrated per
scale on the action distribution; q is the empirical quantile that produces τ. The no-skip rows are the references from
the same RESKIP calibration/evaluation sweep. The conservative end (here q=0.99) is the recommended operating
point: +1.10pp 2B / +0.20pp 4B over the same-sweep no-skip backbone, while still saving 5–9% wall-clock per call.
The 4B is graceful all the way to q=0.30 (only −4.4pp); 2B collapses below q=0.85, consistent with the larger model
carrying more depth-redundancy.
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
(collapse)
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
as an intrinsic routing signal produced by the forward pass itself, then retrofit that signal into an existing pretrained
backbone.
The neural motivation is recurrence as flexible effective depth. The dual-process distinction [18] maps naturally
onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent
circuits [25, 19, 10]. Recurrent networks capture human representational dynamics [21]; Spoerer et al. [42] are the
closest conceptual antecedent, showing that recurrent CNNs can spend more steps on harder images to explain speed–
accuracy behaviour. Recurrence also has a depth interpretation: finite recurrence can be unrolled into feedforward
depth, but the recurrent form reuses a fixed substrate for flexible effective depth [44]. We take this as a design target,
not a cognitive-faithfulness claim: cortical microcircuits and dendritic association [2, 5, 26] motivate intrinsic routing,
while our implementation is a transformer retrofit.
In residual architectures, DenseNet [17] and Highway networks [43] modify residual flow; ATTNRES [24] gener-
alises residual combination with depth attention. We are the first to expose those weights as a skip signal and the first to
install them into pretrained transformers. In VLA, RT-2 [4], Octo [33], OpenVLA [22], and Pi0 [3] established the
paradigm, while efficiency work has focused mainly on action chunking [50] and post-hoc compression. We contribute
modality-aware adaptive depth for the VLM backbone.
## 6