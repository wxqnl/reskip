## Related Work
For test-time compute, o1 [34], DeepSeek-R1 [8], chain-of-thought [45], and optimal test-time compute analyses [41]
allocate compute by emitting more reasoning tokens; per-token depth remains fixed. Depth-axis methods include ACT
/ Universal Transformers [15, 9], CALM [40], Mixture-of-Depths [39], LayerSkip [13], and static pruning [16, 32].
These add a halting head, exit classifier, router, speculative decoder, or static removal rule. We instead use ATTNRES
as an intrinsic routing signal produced by the forward pass itself, then retrofit that signal into an existing pretrained
backbone. Appendix D.5 compares dynamic, static, and random skip rules at 340M, and Appendix E.2 gives a full-layer
static-pruning baseline at the VLM scale.
The neural motivation is recurrence as flexible effective depth. The dual-process distinction [18] maps naturally
onto visual processing: easy stimuli can be handled by feedforward sweeps, while harder stimuli recruit recurrent
circuits [25, 19, 10]. Recurrent networks capture human representational dynamics [21]; Spoerer et al. [42] are the
closest conceptual antecedent, showing that recurrent CNNs can spend more steps on harder images to explain speed–
accuracy behaviour. Recurrence also has a depth interpretation: finite recurrence can be unrolled into feedforward
depth, but the recurrent form reuses a fixed substrate for flexible effective depth [44]. We take this as a design target,
8

not a cognitive-faithfulness claim: cortical microcircuits and dendritic association [2, 5, 26] motivate intrinsic routing,
while our implementation is a transformer retrofit.
In residual architectures, DenseNet [17] and Highway networks [43] modify residual flow; ATTNRES [24] gener-
alises residual combination with depth attention. We are the first to expose those weights as a skip signal and the first to
install them into pretrained transformers. In VLA, RT-2 [4], Octo [33], OpenVLA [22], and Pi0 [3] established the
paradigm, while efficiency work has focused mainly on action chunking [50] and post-hoc compression. We contribute
modality-aware adaptive depth for the VLM backbone.
## 6