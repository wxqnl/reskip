# Deep Review Report

**Paper**: `/home/user01/Minko/reskip2/reskip/paper-codex/main.pdf` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-05-01 07:41 | **Venue**: NeurIPS2026
**Artifacts**: `/home/user01/Minko/reskip2/reskip/review_results/main`

## Overall Assessment

Deep review found 1 major, 4 moderate, 0 minor issues. The highest-priority concerns are: Abstract and conclusion claims need explicit evidence traceability; Cross-section numeric consistency should be reconciled.

- **Major**: 1
- **Moderate**: 4
- **Minor**: 0

## Academic Pre-Review Committee

### Editor (Desk Reject Screen)

## Editor Pre-Screen (1-10)

Score: 4.0/10
Verdict: Desk Reject

### Desk-Reject Triggers (if any)
- Abstract and conclusion claims need explicit evidence traceability

### Top 3 Reasons (no hedging)
1. Abstract and conclusion claims need explicit evidence traceability

### Fast Fixes (within 1-2 days)
- Clarify abstract to address abstract and conclusion claims need explicit evidence traceability.
- Clarify abstract to address cross-section numeric consistency should be reconciled.
- Clarify related_work to address novelty claim should be grounded against the closest prior work.

### Reviewer 1 (Theory Contribution)

## Theory Contribution Review

### 3 Fatal Theory Holes
1. (abstract) "A 340M controlled study shows that the dynamic routing rule outperforms static or random schedules at matched skip rates, and LIBERO experiments provide supporting evidence that the same routed-depth signal remains usable for calibrated action-stream skipping." — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.
2. (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Concrete Moves
- Tighten the paper's theoretical positioning in abstract to resolve abstract and conclusion claims need explicit evidence traceability.
- Tighten the paper's theoretical positioning in related_work to resolve novelty claim should be grounded against the closest prior work.

### Reviewer 3 (Literature Dialogue)

## Literature Dialogue Review

### Closest Prior Work Risks
- (related_work) Novelty claim should be grounded against the closest prior work — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Gap Claim Risks
- The claimed gap should be defended more explicitly: Novelty claim should be grounded against the closest prior work.

### Fast Fixes
- Name the closest prior comparator in related_work and explain the real novelty delta.

### Reviewer 2 (Methodology & Transparency)

## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (abstract) "We introduce AR-RETROFIT, an identity- preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with well under 1% new parameters." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (experiment) "Earlier L=2 runs remain in the appendix as ablation controls, but L=4 is the canonical setting because it preserves the accuracy plateau while halving router calls relative to L=2." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.
- (experiment) "Earlier L=2 runs remain in the appendix as ablation controls, but L=4 is the canonical setting because it preserves the accuracy plateau while halving router calls relative to L=2." — The results section reports comparative performance. Confirm whether the paper states the evaluation scope, variance, and fairness conditions tightly enough for a reviewer.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.

### Reviewer 4 (Logic Chain)

## Logic Chain Review

### Breakpoints
- (abstract) "A 340M controlled study shows that the dynamic routing rule outperforms static or random schedules at matched skip rates, and LIBERO experiments provide supporting evidence that the same routed-depth signal remains usable for calibrated action-stream skipping." — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

### Structural Fix Moves
- Add one explicit bridge sentence in abstract so the argument chain closes cleanly.

### Committee Consensus

## Committee Consensus

Overall Score: 4.0/10
Editor Verdict: Desk Reject

### Score Formula
- base 9.0
- minus 1.5 * major (1)
- minus 0.7 * moderate (4)
- minus 0.2 * minor (0)
- floor 1.0
- desk reject cap 4.0

### Top 3 Issues To Fix First
1. Abstract and conclusion claims need explicit evidence traceability
2. Cross-section numeric consistency should be reconciled
3. Comparison protocol should make fairness assumptions explicit

## Paper Summary

# Paper Summary: main

## Research Question
- Pretrained transformers execute nearly the same depth for every input

## Core Thesis
- We introduce AR-RETROFIT, an identity- preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with well under 1% new parameters.

## Headline Claims
- We introduce AR-RETROFIT, an identity- preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with well under 1% new parameters.
- The biological analogue motivating this paper is different: control is embedded in the processing substrate rather than attached as a separate decision head.
- The central problem in this paper is how to install such a signal into standard pretrained transformers through a short fine-tune, while preserving their original capabilities and making the signal usable for calibrated depth skipping.

## Section Map
- abstract (7-22): 192 words
- introduction (23-62): 538 words
- related (552-698): 607 words
- discussion (699-708): 97 words
- experiment (905-1179): 1772 words
- method (1180-1344): 542 words
- conclusion (1345-1358): 149 words
- result (1359-1957): 2684 words

## Closure Targets
- No closure target was extracted automatically.

## Major Issues

### M1: Abstract and conclusion claims need explicit evidence traceability
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: medium
- **Section**: abstract
- **Related Sections**: abstract, results, conclusion
- **Root Cause Key**: `abstract-and-conclusion-claims-need-explicit-evidence-traceability`
- **Quote Verified**: no
- **Quote**: `A 340M controlled study shows that the dynamic routing rule outperforms static or random schedules at matched skip rates, and LIBERO experiments provide supporting evidence that the same routed-depth signal remains usable for calibrated action-stream skipping.`
- **Explanation**: At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

## Moderate Issues

### M1: Cross-section numeric consistency should be reconciled
- **Type**: presentation
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: medium
- **Section**: abstract
- **Related Sections**: abstract, introduction, related
- **Root Cause Key**: `cross-section-numeric-consistency-should-be-reconciled`
- **Quote Verified**: no
- **Quote**: `We introduce AR-RETROFIT, an identity- preserving gated residual injection that adds an ATTNRES-style routing path to a frozen backbone with well under 1% new parameters.`
- **Explanation**: Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.

### M2: Comparison protocol should make fairness assumptions explicit
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: experiment
- **Related Sections**: method, experiment
- **Root Cause Key**: `comparison-protocol-should-make-fairness-assumptions-explicit`
- **Quote Verified**: no
- **Quote**: `Earlier L=2 runs remain in the appendix as ablation controls, but L=4 is the canonical setting because it preserves the accuracy plateau while halving router calls relative to L=2.`
- **Explanation**: Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### M3: Result claims should identify comparison scope and uncertainty
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: experiment
- **Related Sections**: experiment, methods
- **Root Cause Key**: `result-claims-should-identify-comparison-scope-and-uncertainty`
- **Quote Verified**: no
- **Quote**: `Earlier L=2 runs remain in the appendix as ablation controls, but L=4 is the canonical setting because it preserves the accuracy plateau while halving router calls relative to L=2.`
- **Explanation**: The results section reports comparative performance. Confirm whether the paper states the evaluation scope, variance, and fairness conditions tightly enough for a reviewer.

### M4: Novelty claim should be grounded against the closest prior work
- **Type**: claim_accuracy
- **Source**: [LLM] via `prior_art_and_novelty_grounding`
- **Confidence**: low
- **Section**: related_work
- **Related Sections**: related_work, results
- **Root Cause Key**: `novelty-claim-should-be-grounded-against-the-closest-prior-work`
- **Quote Verified**: no
- **Quote**: —
- **Explanation**: The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

## Phase 0 Automated Findings

### [Script] BIB

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Check: /home/user01/Minko/reskip2/reskip/paper-codex/main.pdf |
| --- | Minor | PASS |
| --- | Minor | entries: 2 |
| --- | Minor | entries: 2 |
| --- | Minor | 2 entries missing DOI/URL (Use --online-check to export list) |
| --- | Minor | AI-generated citations have ~40% error rate. Verify entries without DOI/URL using Semantic Scholar API or CrossRef. See references/CITATION_VERIFICATION.md for verification workflow. |

### [Script] DEAI

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Use --analyze for full analysis |

### [Script] EXPERIMENT

| Line | Severity | Issue |
|------|----------|-------|
| 1 | Minor | No efficiency comparison is mentioned; verify whether runtime, memory, or parameter cost should be reported. |

### [Script] GRAMMAR

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | No rule-based issues detected in selected scope. |

### [Script] LOGIC

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | /METHODOLOGY: No rule-based coherence issues detected. |

### [Script] SENTENCES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | SENTENCE (Line 63, 26 words, 4 clauses) |
| --- | Minor | ѫP9δi:\T;v芒E,>]o 醵t.C!iJ9\HDHnnWYY(}Rā@ɆKzBJ_8IKv*XE8$)*# Mw}>EKN@,:Y,qD|v1,VQVPѦK'U&qFb! |
| --- | Minor | ѫP9δi:\T;v芒E. >]o 醵t.C!iJ9\HDHnnWYY(}Rā@ɆKzBJ_8IKv*XE8$)*# Mw}>EKN@. :Y. qD|v1. VQVPѦK'U&qFb!. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 79, 39 words, 4 clauses) |
| --- | Minor | ։ա><VX3W<:<ʹ,qc!	dCpN0XitiaQ<[Z,3sCߗmu/ۉY=lʮsc̥׸	MاEcl"*Hqcvm,9D<u{MU%vecY<*ZY[y8ٚ1Q.OVk-_}{lp^viXγ<zcF|8d]@;1"4n",J)ENѦl4Eju |
| --- | Minor | ։ա><VX3W<:<ʹ. qc!	dCpN0XitiaQ<[Z. 3sCߗmu/ۉY=lʮsc̥׸	MاEcl"*Hqcvm. 9D<u{MU%vecY<*ZY[y8ٚ1Q.OVk-_}{lp^viXγ<zcF|8d]@;1"4n". J)ENѦl4Eju. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 97, 53 words, 4 clauses) |
| --- | Minor | i5T7{ShiRhŢ?e#2[C@y=JR(4ř/!;EfW0VnWZRw+	jk9ŭL7Z[(	cP/axqu2Tg, ~?e@}0`P{9!0L)o9/bX) U%&ܓG5Ĭ=M/FS_QrP^8[OhxЬW,pX<,r4Q!ۅr&C-"C!K찅xq`9Z73ǆip}fYM-Cb,1[Y:2OC41g7. |
| --- | Minor | i5T7{ShiRhŢ?e#2[C@y=JR(4ř/!;EfW0VnWZRw+	jk9ŭL7Z[(	cP/axqu2Tg. ~?e@}0`P{9!0L)o9/bX) U%&ܓG5Ĭ=M/FS_QrP^8[OhxЬW. pX<. r4Q!ۅr&C-"C!K찅xq`9Z73ǆip}fYM-Cb. 1[Y:2OC41g7.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 107, 89 words, 1 clauses) |
| --- | Minor | ^4E=?ӯ٫;uPvl\lDdˉD32ٜ쑢611"Z:>T&gnw?w2+}Po=&=][D<ؘ+fȫ%w1tF5^r7	c\0dؕV83UTv <ghYp6ŉ?HpNn'x`j)tō)̽~胪d2HraG}ڏOmuOq6<8"2HP[F~NQ0H40/JM~lR!ČTʔJ=,rA.d2O5+EЮMECcK8V^gQWlmLhlvufgl)qǺQ=* UCYlBٸk\Jڑ%*<|<Kc&a+<ȋK)C7K@ BwKc	O8Iz$Mƽq%n_b_O\ٽެ}	ar'b lbﾺs{U􅛘?~K_=._# -(|U-Vunwpu9(ofmN5htEŦ09'4<^PinrZ310GeF# |
| --- | Minor | ^4E=?ӯ٫;uPvl\lDdˉD32ٜ쑢611"Z:>T&gnw?w2+}Po=&=][D<ؘ+fȫ%w1tF5^r7	c\0dؕV83UTv <ghYp6ŉ?HpNn'x`j)tō)̽~胪d2HraG}ڏOmuOq6<8"2HP[F~NQ0H40/JM~lR!ČTʔJ=. rA.d2O5+EЮMECcK8V^gQWlmLhlvufgl)qǺQ=* UCYlBٸk\Jڑ%*<|<Kc&a+<ȋK)C7K@ BwKc	O8Iz$Mƽq%n_b_O\ٽެ}	ar'b lbﾺs{U􅛘?~K_=._# -(|U-Vunwpu9(ofmN5htEŦ09'4<^PinrZ310GeF#. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 200, 20 words, 4 clauses) |
| --- | Minor | P5ƌϹ,,	aOaO5GnKޒ"0\eB# 	jFĆ*^6wv!Xq:i3^,0';Qd,""Hԝ\v/[HV@MV |
| --- | Minor | P5ƌϹ. aOaO5GnKޒ"0\eB# 	jFĆ*^6wv!Xq:i3^. 0';Qd. ""Hԝ\v/[HV@MV. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 237, 62 words, 2 clauses) |
| --- | Minor | +XJnDّ'RfKmm bmt2~ydv֗=	N$EV\LTcad,L[v`ja.\|+; dK.>	HXg@F& / H- '#R҂Ðk,r"Ճ) b;P~{}[睼yQ̛[VRC	ɹOWo?jd;ܷiGϺhldmxk(nǭ!<V)l^VV+~1B1<m/<p	K~(Em}Co7lˀoGrrQ3x"~`vњ%>*{B!BIAT RM{2^(W%8ev687sZ#@ Ɖ(h |
| --- | Minor | +XJnDّ'RfKmm bmt2~ydv֗=	N$EV\LTcad. L[v`ja.\|+; dK.>	HXg@F& / H- '#R҂Ðk. r"Ճ) b;P~{}[睼yQ̛[VRC	ɹOWo?jd;ܷiGϺhldmxk(nǭ!<V)l^VV+~1B1<m/<p	K~(Em}Co7lˀoGrrQ3x"~`vњ%>*{B!BIAT RM{2^(W%8ev687sZ#@ Ɖ(h. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 289, 61 words, 4 clauses) |
| --- | Minor | q&ۺGŇ/'(DcPϢXs8I[gZJrѴ֑FjKR~a-A d48Hğe"[9|:V)x1:brbf`&	!I<5J@I^tdTUr&T, y,"c-?fKavZJ;&$6|`3M=lVkbt釤73ж,(g>t0`sI[)ev;xXx{Wop),Dmҝ8sL7׭EsƸXebxjޝFy6)ʹYYx.DD^Uc{:4qEq5!W*#R0qY"Br%ϢIsARsLg[Rq{Z +yQ!a |
| --- | Minor | q&ۺGŇ/'(DcPϢXs8I[gZJrѴ֑FjKR~a-A d48Hğe"[9|:V)x1:brbf`&	!I<5J@I^tdTUr&T.  y. "c-?fKavZJ;&$6|`3M=lVkbt釤73ж. (g>t0`sI[)ev;xXx{Wop). Dmҝ8sL7׭EsƸXebxjޝFy6)ʹYYx.DD^Uc{:4qEq5!W*#R0qY"Br%ϢIsARsLg[Rq{Z +yQ!a. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 325, 46 words, 4 clauses) |
| --- | Minor | ʝ:? A))uwJ˳BaNP]|u,O"xb"SeȪ%;ElEg8L!z􉶋d ("|xfc"u6s,LJDk1d\zs,at|א&S@l*:<a_@S08fiU]F!O)[b33Rl^KΗ2JƑ'hC (piW:f[?Buk>mw^`R,SK RTSA}3S 5hI.D(Yq3!_qh |
| --- | Minor | ʝ:? A))uwJ˳BaNP]|u. O"xb"SeȪ%;ElEg8L!z􉶋d ("|xfc"u6s. LJDk1d\zs. at|א&S@l*:<a_@S08fiU]F!O)[b33Rl^KΗ2JƑ'hC (piW:f[?Buk>mw^`R. SK RTSA}3S 5hI.D(Yq3!_qh. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 341, 15 words, 4 clauses) |
| --- | Minor | M|n!X>`1VT%6NIgCD>/9\Essa ,ۇ,Hm\`2,:d>?].Y,A󀃣A |
| --- | Minor | M|n!X>`1VT%6NIgCD>/9\Essa. ۇ. Hm\`2. :d>?].Y. A󀃣A. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 420, 71 words, 2 clauses) |
| --- | Minor | ÝF}}gM`	Ws:؟ 6#>s1iWRvh\h,UfSV\1@:T:[MA}myE%.cgJGfE>f!i[~ЊYby~3,1"#6SB1?6hP&_jc>  /3g[oObC]5k izeLО׬]n׼/|p}+6I 6Xw赗]	̊Տj pխd_OOݓm#O_~7{V_0G 	sR~?"`fQ[m^!A}mK]y}R]=/yl玿6Mz;~_x9U?Ejخ?:cX]e*衎^?1Ŏۀ(Tw~px߿r|ድ֪9`[lHD04T޶Cf{sSwvG~pՇ^OrfPز |
| --- | Minor | ÝF}}gM`	Ws:؟ 6#>s1iWRvh\h. UfSV\1@:T:[MA}myE%.cgJGfE>f!i[~ЊYby~3. 1"#6SB1?6hP&_jc>  /3g[oObC]5k izeLО׬]n׼/|p}+6I 6Xw赗]	̊Տj pխd_OOݓm#O_~7{V_0G 	sR~?"`fQ[m^!A}mK]y}R]=/yl玿6Mz;~_x9U?Ejخ?:cX]e*衎^?1Ŏۀ(Tw~px߿r|ድ֪9`[lHD04T޶Cf{sSwvG~pՇ^OrfPز. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 440, 147 words, 4 clauses) |
| --- | Minor | !޺Uc_ԔT^Ϯzs_W	_   TJ=cnB]+RڜuQ'-=;c^~3~ֽUԌ[߸ᕕR0}Ţh'mR,=G@o3:{	zG}u}5Aw.?9cȐ&|17A\d\HChkW/޽q"11FM8Q?c%YfKP #5 g㍓ݡ79qKF ixɹ,ؽ&/?̣,vz=E04wP#+Ȏoȇ/x;~འА!o*eOjD*DgmaѼc2Q]+ .Jlq	9hfzBĪ/3MY1Lt8!ϡzjC\Pun\_z_kFu w9tQ4Tթtopg	Fm9ڽ@@Bp!!lƩCIs䐸 ,RƈX#P{2r/#y8Gg^1Z۳Psdp5^s^s._{|[?:f©}jނŇ|_/iLirW!w1G<ٳLo6؃HeH3>lbZb^3ߧ$ƽME|C譢" @+@~l/[߼fr`NChm)1΄?]<lH	9')(dX?^O0D&Uy뮋_\'"WB[ 0gH]@Щh_!Ե`]YD!<|(<DQ)#6F[\ O`<.EP*JJJURhhx{") iO]07=Oz~o\O+o6op]=XsF뱑տ0:Gy=!5)ı7U6U"ƺw3ܭ9sGi	FBOsG+]5<4dd2lXgC)ŢT_Ca\ 5oĤ/% |
| --- | Minor | !޺Uc_ԔT^Ϯzs_W	_   TJ=cnB]+RڜuQ'-=;c^~3~ֽUԌ[߸ᕕR0}Ţh'mR. =G@o3:{	zG}u}5Aw.?9cȐ&|17A\d\HChkW/޽q"11FM8Q?c%YfKP #5 g㍓ݡ79qKF ixɹ. ؽ&/?̣. vz=E04wP#+Ȏoȇ/x;~འА!o*eOjD*DgmaѼc2Q]+ .Jlq	9hfzBĪ/3MY1Lt8!ϡzjC\Pun\_z_kFu w9tQ4Tթtopg	Fm9ڽ@@Bp!!lƩCIs䐸. RƈX#P{2r/#y8Gg^1Z۳Psdp5^s^s._{|[?:f©}jނŇ|_/iLirW!w1G<ٳLo6؃HeH3>lbZb^3ߧ$ƽME|C譢" @+@~l/[߼fr`NChm)1΄?]<lH	9')(dX?^O0D&Uy뮋_\'"WB[ 0gH]@Щh_!Ե`]YD!<|(<DQ)#6F[\ O`<.EP*JJJURhhx{") iO]07=Oz~o\O+o6op]=XsF뱑տ0:Gy=!5)ı7U6U"ƺw3ܭ9sGi	FBOsG+]5<4dd2lXgC)ŢT_Ca\ 5oĤ/%. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 488, 69 words, 0 clauses) |
| --- | Minor | ~#)݋5s=ԋH9`w{(ҩњLg]hQƐ~FV yYgS[uhW>Q4sFF}ƛGiWjHUnc uUY7/{xqB5TwVX2E*xvoZTQ60{s.G1f)=BPT5oan3=K[辎+6@stzp8kjuo9gUE!b~K΂`J`UdD4{|k-bN-"nwY1zk+QeВ֊/Φ  2L~#|}z +wgRagBK/S&k0xUMP/p1j%k_RӽS=Z}T>~Ͼc@6SSC	1Fk* 6	Y~	 Ţ;Mg#:&P 4z`ݩEc݊>U`LcR C3ej =eܲMJ |
| --- | Minor | ~#)݋5s=ԋH9`w{(ҩњLg]hQƐ~FV yYgS[uhW>Q4sFF}ƛGiWjHUnc uUY7/{xqB5TwVX2E*xvoZTQ60{s.G1f)=BPT5oan3=K[辎+6@stzp8kjuo9gUE!b~K΂`J`UdD4{|k-bN-"nwY1zk+QeВ֊/Φ  2L~#|}z +wgRagBK/S&k0xUMP/p1j%k_RӽS=Z}T>~Ͼc@6SSC	1Fk* 6	Y~	 Ţ;Mg#:&P 4z`ݩEc݊>U`LcR C3ej =eܲMJ |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 531, 67 words, 2 clauses) |
| --- | Minor | u,ۯy|Kr[ºu;o/䎕޵?<rwcߺߵt_6?';ZcOmajcFG?_B"y֣TO ɟBڼu}Ư^}ckvON.u,olF`b&Ƨx`u>rmΉw姟|'x?fΩM׿;|&g:'7=;	MiaO9Kz]02b *eMPa]W͆Zo۞~3vo׮صz>6nvV<~yqoףK/|5ol7!31DN}5oz1R#ݑѡ/8}1/5ƠRM6l{Mwxg}Az3wTQ}c*v;]yhG	 #J]qE4@ݍ oZHgh' y9d-("w]vu.ȷuaC |
| --- | Minor | u. ۯy|Kr[ºu;o/䎕޵?<rwcߺߵt_6?';ZcOmajcFG?_B"y֣TO ɟBڼu}Ư^}ckvON.u. olF`b&Ƨx`u>rmΉw姟|'x?fΩM׿;|&g:'7=;	MiaO9Kz]02b *eMPa]W͆Zo۞~3vo׮صz>6nvV<~yqoףK/|5ol7!31DN}5oz1R#ݑѡ/8}1/5ƠRM6l{Mwxg}Az3wTQ}c*v;]yhG	 #J]qE4@ݍ oZHgh' y9d-("w]vu.ȷuaC. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 609, 38 words, 4 clauses) |
| --- | Minor |    JiR1q|!1OکT qͺ-ACQ9KX 1"!9 U22 I)S`WIs)>1x{CǢ}\OY`1%@ ni'	 Q$gy#OX:R]NORv1yj\7},`Q,C,;,vMF.St-[J 64<p1m"t@ Yo? BX.G |
| --- | Minor |    JiR1q|!1OکT qͺ-ACQ9KX 1"!9 U22 I)S`WIs)>1x{CǢ}\OY`1%@ ni'	 Q$gy#OX:R]NORv1yj\7}. `Q. C. ;. vMF.St-[J 64<p1m"t@ Yo? BX.G. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 655, 42 words, 4 clauses) |
| --- | Minor | kZz#  FZj֮^7lv~t;bШჵJw<%Vq+~d'7UC&tFkH&<Pl,CD	-/YUa01AzFeÂS6uΙ#C	` 6x!"pR"6YJ,F,YB%DWv/A, HLJŌL+MRD@2 l-Xm_ֲavCX XVb9Ψ%8N BxޠVb}ԅD |
| --- | Minor | kZz#  FZj֮^7lv~t;bШჵJw<%Vq+~d'7UC&tFkH&<Pl. CD	-/YUa01AzFeÂS6uΙ#C	` 6x!"pR"6YJ. F. YB%DWv/A.  HLJŌL+MRD@2 l-Xm_ֲavCX XVb9Ψ%8N BxޠVb}ԅD. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 721, 61 words, 0 clauses) |
| --- | Minor | `Ei81SCM䂟D*n/6sT"P}Ш]ݖ(9x<;vh"9bFTr}MIk_xdZ}Tt2Dd-9oIW)o5 +D1߽^	*\n\s q vE;:	)N:W>`%7}ٳw<eN`!쳌dRU+ӚEȅJU'cszv|KpA6ozg]ZuKG= X%TX&qVT@"aA٫bI	& a}Y$Rdu@jJ:+ĥ KnEIB{"7I|wI.!Ďر!RUL |
| --- | Minor | `Ei81SCM䂟D*n/6sT"P}Ш]ݖ(9x<;vh"9bFTr}MIk_xdZ}Tt2Dd-9oIW)o5 +D1߽^	*\n\s q vE;:	)N:W>`%7}ٳw<eN`!쳌dRU+ӚEȅJU'cszv|KpA6ozg]ZuKG= X%TX&qVT@"aA٫bI	& a}Y$Rdu@jJ:+ĥ KnEIB{"7I|wI.!Ďر!RUL |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 722, 74 words, 1 clauses) |
| --- | Minor | jPx5s	U3J< \H<cUِPyP)JqfN5!aVRa B"F!@XP@)"JbڈnHkQсLRcF P|`0*͌H+*M ^NT&}2r+H1>֟7P>YnH|bv;ߎ_e>jL%x~ }1Ooj @CW VZu"0V}pVr30Ǚ|ɒ"aV +=y}3@ 󏾴}i6 y(",J+~aݏSWwZu\n6 PZ0q!8)j.cMkAJŴFkwmՒqyB+tvҹ$RĊ@8&1:)	ڦ[ 'ǚT'Ғ Y	+AF5 |
| --- | Minor | jPx5s	U3J< \H<cUِPyP)JqfN5!aVRa B"F!@XP@)"JbڈnHkQсLRcF P|`0*͌H+*M ^NT&}2r+H1>֟7P>YnH|bv;ߎ_e>jL%x~ }1Ooj @CW VZu"0V}pVr30Ǚ|ɒ"aV +=y}3@ 󏾴}i6 y(". J+~aݏSWwZu\n6 PZ0q!8)j.cMkAJŴFkwmՒqyB+tvҹ$RĊ@8&1:)	ڦ[ 'ǚT'Ғ Y	+AF5. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 774, 79 words, 7 clauses) |
| --- | Minor | SVII5,?}_VX ,mq'KP]Gs# J\P3p,qakm܄9Z1q!(D# ]%K6Uˁ q1θxY4gkN9K7)^8cvU"h{1CFQXjB̝R @T(EUzbǲ kV0̘XlfB8n3kc,Dl7Kl7v#  !xzmK֬WƶֶD ,Ӡ= XDQlc,!O	R]&2Xk@"$QZkˎ99]6WNHiA]%,NM<AŰ`fc=Kp^ԃktn'WޓT-50Je*B"TNG(I<""R[K］(XjxN=z/_#e]D'R"႓%WՓRɓ9tH' uEyowoNc=lOϱd\CqN5(?fY#nRE109b`+`K |
| --- | Minor | SVII5. ?}_VX . mq'KP]Gs# J\P3p. qakm܄9Z1q!(D# ]%K6Uˁ q1θxY4gkN9K7)^8cvU"h{1CFQXjB̝R @T(EUzbǲ kV0̘XlfB8n3kc. Dl7Kl7v#  !xzmK֬WƶֶD . Ӡ= XDQlc. !O	R]&2Xk@"$QZkˎ99]6WNHiA]%. NM<AŰ`fc=Kp^ԃktn'WޓT-50Je*B"TNG(I<""R[K］(XjxN=z/_#e]D'R"႓%WՓRɓ9tH' uEyowoNc=lOϱd\CqN5(?fY#nRE109b`+`K. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 783, 69 words, 2 clauses) |
| --- | Minor | lB1hT\D -c*z ӵNTƄ6WG@dQ_56Wݪb"U.kTdn-kVQH~YdBq(Dw}W(Q@dcm$'վ?aI3!*DM#&Ba8uTbBJA7r9fg1l]FdQ-D|i]_,}/=ȥ0sYm K3j\:-%B2a+vv G5mߝ9c<}f^ADVldNm1C<b*_ z੧3l5EŗO.Yvqҍh)ۀE-&gݘJ"|*99W	E"_(A)Wx7+.? REK%D*{nBoR	 (G[,bkl!'؁أ |
| --- | Minor | lB1hT\D -c*z ӵNTƄ6WG@dQ_56Wݪb"U.kTdn-kVQH~YdBq(Dw}W(Q@dcm$'վ?aI3!*DM#&Ba8uTbBJA7r9fg1l]FdQ-D|i]_. }/=ȥ0sYm K3j\:-%B2a+vv G5mߝ9c<}f^ADVldNm1C<b*_ z੧3l5EŗO.Yvqҍh)ۀE-&gݘJ"|*99W	E"_(A)Wx7+.? REK%D*{nBoR	 (G[. bkl!'؁أ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 861, 109 words, 7 clauses) |
| --- | Minor | ACB*Yn'O;C͟P/uB(J	xKpot=jݽh@;ܷ/{z#S_aJDCv76>h;-@.OyyM_xڻ&)IdfT.B!Zޗ|ם5Eg*nI|xsc["	d)xkۚg,gtl(D՟W⹳_׈*gHP]zS?t9ſvk,YtK"\䭉]}[;#Ã/.qtDg{%fXL"J!`l=}xS;CR9,Z &ݻcnpn)B䍱Nw{އzjfG9,rb`ݷꭇLLU9po4)(d7yxI;5,ߛHT^Qg.:bkv9{5 	2,;O~7>IH y޸kqjJğ-|^ǁ茧s3Onl%dyq˃W<?4lLgJvf̡Kwp<M}0b ,f:G.ʿnY5of*3x|o<}N.n%>Bi"`#ݼ	 ~st7/:9'(~_V96e <Ƈmx`鶔@H_`pn:yuw_׻7lj]tƩ=gy_\u{ys[LO |
| --- | Minor | ACB*Yn'O;C͟P/uB(J	xKpot=jݽh@;ܷ/{z#S_aJDCv76>h;-@.OyyM_xڻ&)IdfT.B!Zޗ|ם5Eg*nI|xsc["	d)xkۚg. gtl(D՟W⹳_׈*gHP]zS?t9ſvk. YtK"\䭉]}[;#Ã/.qtDg{%fXL"J!`l=}xS;CR9. Z &ݻcnpn)B䍱Nw{އzjfG9. rb`ݷꭇLLU9po4)(d7yxI;5. ߛHT^Qg.:bkv9{5 	2. ;O~7>IH y޸kqjJğ-|^ǁ茧s3Onl%dyq˃W<?4lLgJvf̡Kwp<M}0b . f:G.ʿnY5of*3x|o<}N.n%>Bi"`#ݼ	 ~st7/:9'(~_V96e <Ƈmx`鶔@H_`pn:yuw_׻7lj]tƩ=gy_\u{ys[LO. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 910, 67 words, 1 clauses) |
| --- | Minor | QfT-t*d ^[@){Kq 1hb0Yў7J:8D@ {Tdl<m	!G?lF4}̄f) (<\(k"@4)~H'MrvU*"B)I ޵bͶr<Է:2\`N12k/Q](.͋HTc }yX<IV29`+I),0cbK6H d0R@(+[@D  ^Ee`}P 2V:yYF윲ԞeւpZ*c Azz1;EY<~dr{cN<*J%3D*Yu{`&"auRC;SFcuʒy`!Zkf. h#7ӇwrArDq[.bUV7G|eM̠cd8 'av2SCLG:gYN |
| --- | Minor | QfT-t*d ^[@){Kq 1hb0Yў7J:8D@ {Tdl<m	!G?lF4}̄f) (<\(k"@4)~H'MrvU*"B)I ޵bͶr<Է:2\`N12k/Q](.͋HTc }yX<IV29`+I). 0cbK6H d0R@(+[@D  ^Ee`}P 2V:yYF윲ԞeւpZ*c Azz1;EY<~dr{cN<*J%3D*Yu{`&"auRC;SFcuʒy`!Zkf. h#7ӇwrArDq[.bUV7G|eM̠cd8 'av2SCLG:gYN. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 985, 38 words, 4 clauses) |
| --- | Minor | S[^mqI6+  ؓ,#,E9i1<d]y9:xp\7J"T *F)&"#yՇ3*?bU L],ֵ]WKKphzIЌWH_o*^dXyRDqjyԪ^GW,@ÆJeMoWII3͐//PK?1/]KWɌuPs}&$%TJAci D-<oyÏW~uC6UO1>Hy |
| --- | Minor | S[^mqI6+  ؓ. #. E9i1<d]y9:xp\7J"T *F)&"#yՇ3*?bU L]. ֵ]WKKphzIЌWH_o*^dXyRDqjyԪ^GW. @ÆJeMoWII3͐//PK?1/]KWɌuPs}&$%TJAci D-<oyÏW~uC6UO1>Hy. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1051, 64 words, 1 clauses) |
| --- | Minor | vj=d~hYGX)!`\U`@O2hy5|0}V7`m_9C'Y32l]Ã³P=qYp#*%{4tsvrKtVH!GDS	("=*6cL6H	SVLjVt߾t²؇=,{T>z#`rMT%ԷPRr }OaA *=tLv =q|Gj@L}5G**6[LU{~1LwBb<h|~ ك6kboxn{c<Lt(;8ɛ}OZ؈ZGȌ08DiwI[QCkׯ*sa#0TٖF)qbxM!#K;n=Se@?y{Tn*pFx<[KHQǹfhR>:YNR\m |
| --- | Minor | vj=d~hYGX)!`\U`@O2hy5|0}V7`m_9C'Y32l]Ã³P=qYp#*%{4tsvrKtVH!GDS	("=*6cL6H	SVLjVt߾t²؇=. {T>z#`rMT%ԷPRr }OaA *=tLv =q|Gj@L}5G**6[LU{~1LwBb<h|~ ك6kboxn{c<Lt(;8ɛ}OZ؈ZGȌ08DiwI[QCkׯ*sa#0TٖF)qbxM!#K;n=Se@?y{Tn*pFx<[KHQǹfhR>:YNR\m. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1057, 39 words, 4 clauses) |
| --- | Minor | ^=Q6Î&g¶Vq)a+w,>;YߊeNTf.pl#M@*:N'TkaՏt6xTىR?*ҫ+nI7\PY뼸yӤY鿅D9om&@>Q,c2\^A^{̨2<#y]^jD,JG!_SV<qTgx7+xlI@;$a`F\Uf䗉{=}[jr,zxWy{Mw!L԰qQﯸ50H`"mje+Th |
| --- | Minor | ^=Q6Î&g¶Vq)a+w. >;YߊeNTf.pl#M@*:N'TkaՏt6xTىR?*ҫ+nI7\PY뼸yӤY鿅D9om&@>Q. c2\^A^{̨2<#y]^jD. JG!_SV<qTgx7+xlI@;$a`F\Uf䗉{=}[jr. zxWy{Mw!L԰qQﯸ50H`"mje+Th. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1169, 72 words, 1 clauses) |
| --- | Minor | &+(1a [7:|:وz[w:0.83OoشaGzhWӽ9~C\hdnPJ(B	m	wEq8;aIH&? [{Q(<}oŻ1iN`w=I9De >lNxػl&ME`G9.e{GY@<Tj#9t]ˎ F.J<dȀ`J@帝ļE]xrACp,Bɉ~>}6ܹsoy[ַezj094EHKpft^J 3Fn&aa<w9B<#qH"ENxzayb z6p$Zmp	vIP-''*(.9:ULO>\y7IÛoe] p'.X0O)519qmySXȹs]vZ]f~9Õ择w7!G{ARd_CDюi?2v"=I |
| --- | Minor | &+(1a [7:|:وz[w:0.83OoشaGzhWӽ9~C\hdnPJ(B	m	wEq8;aIH&? [{Q(<}oŻ1iN`w=I9De >lNxػl&ME`G9.e{GY@<Tj#9t]ˎ F.J<dȀ`J@帝ļE]xrACp. Bɉ~>}6ܹsoy[ַezj094EHKpft^J 3Fn&aa<w9B<#qH"ENxzayb z6p$Zmp	vIP-''*(.9:ULO>\y7IÛoe] p'.X0O)519qmySXȹs]vZ]f~9Õ择w7!G{ARd_CDюi?2v"=I. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1221, 49 words, 5 clauses) |
| --- | Minor | ޏ<W`2%HKxnc`ۉ{ BI];S앜De,( Ea<}}͚5eYa;l ,Xxb6vU_?]<ysm\qg]-8Bٍ}.",+Ê+=q>6^YM^%*]}~_^'O25:WJݳqo~O=3!~ u=x~}хO8&(c~ù%]Rlb76sg0@) {^WUUUUG뚈,KN7>U:1&m答f?cUֺn߾7uTGyСNY,lG |
| --- | Minor | ޏ<W`2%HKxnc`ۉ{ BI];S앜De. ( Ea<}}͚5eYa;l . Xxb6vU_?]<ysm\qg]-8Bٍ}.". +Ê+=q>6^YM^%*]}~_^'O25:WJݳqo~O=3!~ u=x~}хO8&(c~ù%]Rlb76sg0@) {^WUUUUG뚈. KN7>U:1&m答f?cUֺn߾7uTGyСNY. lG. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1228, 14 words, 4 clauses) |
| --- | Minor | ҌQ,oKz 	J BryP"0&)l#sΨ,ᥓ+"+gi,i.,rzIk1a0[ |
| --- | Minor | ҌQ. oKz 	J BryP"0&)l#sΨ. ᥓ+"+gi. i.. rzIk1a0[. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1266, 73 words, 2 clauses) |
| --- | Minor |  938E5^A͌93ȃ8cŉ^QFN3Y~n#f;'l7Ѿ(b\obpF+|tMi'tDGְ_> 6gR:'H6C@rT  0#\ 31(ޔ(3>!RTHvS0$*FZ-=fLN`!DqȇC`	fU:HMᑽ{oA@DkN]3s~o*Ժ.;ݾv `es_ͦAw㎵GΙ{ܹ Pkܹs=X -u΄v2c"qR}hwV62Rd"58؂,=r~X& 6=.e319zU+VyDA׽VZ~_:3MU[66,>N|衇84ٟ7gޓ.~&RED |
| --- | Minor |  938E5^A͌93ȃ8cŉ^QFN3Y~n#f;'l7Ѿ(b\obpF+|tMi'tDGְ_> 6gR:'H6C@rT  0#\ 31(ޔ(3>!RTHvS0$*FZ-=fLN`!DqȇC`	fU:HMᑽ{oA@DkN]3s~o*Ժ.;ݾv `es_ͦAw㎵GΙ{ܹ Pkܹs=X -u΄v2c"qR}hwV62Rd"58؂. =r~X& 6=.e319zU+VyDA׽VZ~_:3MU[66. >N|衇84ٟ7gޓ.~&RED. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1274, 73 words, 0 clauses) |
| --- | Minor | S^b^go<Y5sֻ7M PG*袋?3gN?8?-[zr]Uc^YR(_NA~@!2j}}P>ccsS%L]{ {#fWUU^7vhuwdy;wyf'?ϯn/x|m7{+[??/9y_TvIWU5Dߟl8"i@=N9܈!Q/ =AmqͯzEsx-*̅Ar	d&MP_!&:QF@=ٍtJrM4;~8\B !x!YdQeY3bJHe~؆>QfHFʆ-|P]f;} 8q'.>7i}wu)SWZi\m] xGv`LOSNz p\o:RC5T֤ђ>0R |
| --- | Minor | S^b^go<Y5sֻ7M PG*袋?3gN?8?-[zr]Uc^YR(_NA~@!2j}}P>ccsS%L]{ {#fWUU^7vhuwdy;wyf'?ϯn/x|m7{+[??/9y_TvIWU5Dߟl8"i@=N9܈!Q/ =AmqͯzEsx-*̅Ar	d&MP_!&:QF@=ٍtJrM4;~8\B !x!YdQeY3bJHe~؆>QfHFʆ-|P]f;} 8q'.>7i}wu)SWZi\m] xGv`LOSNz p\o:RC5T֤ђ>0R |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1299, 32 words, 4 clauses) |
| --- | Minor |  6o\[/~E`}߷]sf9s,>A?>6v޿˵1uڔA5IՄJ*˲ٻ_@)TgбW!t0C?4fe"6@3RB/6Cl #%2H}~i5:  s堸ImmCMm6B@͇	,,nO,*x.-(ۂed~p܌%	hm$G44cz4Hb5ߖ(u |
| --- | Minor |  6o\[/~E`}߷]sf9s. >A?>6v޿˵1uڔA5IՄJ*˲ٻ_@)TgбW!t0C?4fe"6@3RB/6Cl #%2H}~i5:  s堸ImmCMm6B@͇. nO. *x.-(ۂed~p܌%	hm$G44cz4Hb5ߖ(u. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1430, 73 words, 0 clauses) |
| --- | Minor | ÏC`iyBrh_NysRAkb#k]&;?@y6+[ &4ȼXx/KG\%b.HYQw{@l]&	#x.`n<QWdyᬐiPuN>Q:m'0vXSZ\%ÁP=\02p0!̂B&D9V8}'[]80?<H٘G۹ۏb tIzۮ"<i9#=	mHsZ轖;@iZot [Zߜ7{x( jR8y5+(=ԌHP=4s}eY*TRP~_t`|w嗿l֬~|~KBk){o?NQn۱[n&s={3IUoBn{ʓ/s0'aNUQfY' Фu+kZheH%g-furd"\e!bE3wq##ʾeR6F XzYw$# |
| --- | Minor | ÏC`iyBrh_NysRAkb#k]&;?@y6+[ &4ȼXx/KG\%b.HYQw{@l]&	#x.`n<QWdyᬐiPuN>Q:m'0vXSZ\%ÁP=\02p0!̂B&D9V8}'[]80?<H٘G۹ۏb tIzۮ"<i9#=	mHsZ轖;@iZot [Zߜ7{x( jR8y5+(=ԌHP=4s}eY*TRP~_t`|w嗿l֬~|~KBk){o?NQn۱[n&s={3IUoBn{ʓ/s0'aNUQfY' Фu+kZheH%g-furd"\e!bE3wq##ʾeR6F XzYw$# |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1447, 26 words, 4 clauses) |
| --- | Minor | gܟ(, hcnȤELy؂16Z[m[wf#fȈxLSka_leYYn9/q$q]Ԍr ~@!B"j &ksܹ Pz0LEt~w,5s uW],^:k,?~e/zɥ^ZUq |
| --- | Minor | gܟ(.  hcnȤELy؂16Z[m[wf#fȈxLSka_leYYn9/q$q]Ԍr ~@!B"j &ksܹ Pz0LEt~w. 5s uW]. ^:k. ?~e/zɥ^ZUq. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1668, 67 words, 2 clauses) |
| --- | Minor | ;ְi3ǲW+^h$fQVI}_,r#4H~Nb]]g鴕)MfZ@6JJizG{E`T.KkHkZ4-0۷]RdTnd-uc-PC%V+hN{zhGk6P-7~uR{v@.q	k9i'=o|oxOyڃ[^t]''?oɟwz绾[>'''dQ'#412.p=.5giYIڂ9_G?rˋoy=1~Kn{f)|S?wO~o;W\q#2ubP,a#r	gz`+6ԩ||~C	MV_ϻ?O9{椴wԧqenTmZ8goٍ; |
| --- | Minor | ;ְi3ǲW+^h$fQVI}_. r#4H~Nb]]g鴕)MfZ@6JJizG{E`T.KkHkZ4-0۷]RdTnd-uc-PC%V+hN{zhGk6P-7~uR{v@.q	k9i'=o|oxOyڃ[^t]''?oɟwz绾[>'''dQ'#412.p=.5giYIڂ9_G?rˋoy=1~Kn{f)|S?wO~o;W\q#2ubP. a#r	gz`+6ԩ||~C	MV_ϻ?O9{椴wԧqenTmZ8goٍ;. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1688, 65 words, 1 clauses) |
| --- | Minor | iKP:կ7w]~,Ɛ t L>9:?|/ſl6h>8ux?}#g./?%7"NaM&bү0:; ) H}d$:Ei>	OW~ѿynO| ֫o/*׿>n%A9x' I ǊSn7~罿G`9R|W7u4\ϻ;hUX%8۹XFG7G<᯸eѧ>yiZVgO| CnW~+v&	SJnD:؄7˲Nн%J?({!x}8ޅL%ҫf#ieQ:zdT(cVcNߗ'EMzyEbbKknQtIިY)UZ~FHA |
| --- | Minor | iKP:կ7w]~. Ɛ t L>9:?|/ſl6h>8ux?}#g./?%7"NaM&bү0:; ) H}d$:Ei>	OW~ѿynO| ֫o/*׿>n%A9x' I ǊSn7~罿G`9R|W7u4\ϻ;hUX%8۹XFG7G<᯸eѧ>yiZVgO| CnW~+v&	SJnD:؄7˲Nн%J?({!x}8ޅL%ҫf#ieQ:zdT(cVcNߗ'EMzyEbbKknQtIިY)UZ~FHA. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1778, 118 words, 4 clauses) |
| --- | Minor | llwD{vM}cp<Q7|Eg.)b  +iSO44>֧Vn%<DT9MjQ&XEEa6nAzf5k^W~}s- )5֫Fi^d0P0;h#+r)Uu~D~­*8sZp&})d-dkW[âPF'B@_Y)WlI$&l^|Bdu[tKQ7:Xцs`rpKɸqcSSU)`E{>vYC5TÌRhk=F>- ~X8L;dj_I:2i%k鋐VDDV6|xN~wz>O~ͼOh嫷Y@3 TjEUΙfi&Nyѣ~)F !#IB((4 l ֜EPUTHuV.BSυՇJ+2p%&="	z(o8#Д0%(,MA4%_-I9(C&CtJ&:<ZWv,Ӳv&D	J[L"	VH@ۜRD)A	r.GhT^	h.ϋaV3"@v;P21,\2Th~MX	D@Z4e	R4+"Jw9B*N8ANxJJ#SI7"圪+|'zsm8vi]e,n4®PB(ZnGWSv\x421@P=W |
| --- | Minor | llwD{vM}cp<Q7|Eg.)b  +iSO44>֧Vn%<DT9MjQ&XEEa6nAzf5k^W~}s- )5֫Fi^d0P0;h#+r)Uu~D~­*8sZp&})d-dkW[âPF'B@_Y)WlI$&l^|Bdu[tKQ7:Xцs`rpKɸqcSSU)`E{>vYC5TÌRhk=F>- ~X8L;dj_I:2i%k鋐VDDV6|xN~wz>O~ͼOh嫷Y@3 TjEUΙfi&Nyѣ~)F !#IB((4 l ֜EPUTHuV.BSυՇJ+2p%&="	z(o8#Д0%(. MA4%_-I9(C&CtJ&:<ZWv. Ӳv&D	J[L"	VH@ۜRD)A	r.GhT^	h.ϋaV3"@v;P21. \2Th~MX	D@Z4e	R4+"Jw9B*N8ANxJJ#SI7"圪+|'zsm8vi]e. n4®PB(ZnGWSv\x421@P=W. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1786, 71 words, 3 clauses) |
| --- | Minor | *̾,j/%,;Ӳ\S'iY`e8sPKkVeȓPi|S6q7h ӟn0QPfTjCwA?#D}meY/4DM3tҐ2lmhj5TR5lIHwĻ.EP.فϺs3GMOG>j5^B?W[';aV^f҆n~-1ZmoFIB7+iL=xc~.ywr	tud[ݲcI-o7:_|S0<>!^uմ^i5MjZV*N"yۓdsݜl662˄8pZV	d%!M)laJ)MiVUZ0Cdm&ʛy>ڞl[ (C1zgi .̞ǋ 踁Vxg5e[5W`J[|jg1S]O/e,W7;f"l/E*1`:#m |
| --- | Minor | *̾. j/%. ;Ӳ\S'iY`e8sPKkVeȓPi|S6q7h ӟn0QPfTjCwA?#D}meY/4DM3tҐ2lmhj5TR5lIHwĻ.EP.فϺs3GMOG>j5^B?W[';aV^f҆n~-1ZmoFIB7+iL=xc~.ywr	tud[ݲcI-o7:_|S0<>!^uմ^i5MjZV*N"yۓdsݜl662˄8pZV	d%!M)laJ)MiVUZ0Cdm&ʛy>ڞl[ (C1zgi .̞ǋ 踁Vxg5e[5W`J[|jg1S]O/e. W7;f"l/E*1`:#m. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1792, 77 words, 0 clauses) |
| --- | Minor | nULs=hӫ82!L>6zǷ%N{ɞikUʢ+OmSoy_s`{t?/xc~W]z:!4%Ē45\YhPXѰ*#I2{yay#Hzt`+9DtjNi">߭\Շ[ffI	O%x53aD5	Z^hLxW@YEֽ9(UƀgUUDĠ Q1nEKry-p~ vؠn0Kݗ&¡<i\een)OaĘ iJWy}-Gcz)eS;K]Xt"ԱG 3[bOP@QnK[p%JIң50I'~?w\Jѡ~ܶX+O Qu1'(̹<Os.t )Vb[T~2?[F/\d 7iuLX}Op$T-5xIȈk.rb} * |
| --- | Minor | nULs=hӫ82!L>6zǷ%N{ɞikUʢ+OmSoy_s`{t?/xc~W]z:!4%Ē45\YhPXѰ*#I2{yay#Hzt`+9DtjNi">߭\Շ[ffI	O%x53aD5	Z^hLxW@YEֽ9(UƀgUUDĠ Q1nEKry-p~ vؠn0Kݗ&¡<i\een)OaĘ iJWy}-Gcz)eS;K]Xt"ԱG 3[bOP@QnK[p%JIң50I'~?w\Jѡ~ܶX+O Qu1'(̹<Os.t )Vb[T~2?[F/\d 7iuLX}Op$T-5xIȈk.rb} * |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1809, 84 words, 4 clauses) |
| --- | Minor | Pok\|ĚP25=xsUut4͜,{E,Qi^pISmкϐGg'40lu[m C	<'tK%."YʥB["5\>.6Ο;߫w+?wʯ|߻ucι.%Y]EN^˭Ʈ(+<۽֔Da+4a0>@AwEZlzhdܞթ.<n[hz^T]SgNߑ.s~R5aIWrx̐8vA2kTטK   <'73g<x_y+<ʻ&,)ŵP5T'!'c6d +곚}.j29llgJ5(4^׹Y^8IOܒ3Kiq(%:(M0O9庼ȩ;lgtlJĬ18U`ruZuԪ1M-4i,gUxҜyw>w/:_MO:>>&ElM$Q%F4Pufz jGF7-XQ~aLF&'iOA5^O쭲4BwCY0eW("_*lɍV{V5Q"-ĝ%#K/洱9 |
| --- | Minor | Pok\|ĚP25=xsUut4͜. {E. Qi^pISmкϐGg'40lu[m C	<'tK%."YʥB["5\>.6Ο;߫w+?wʯ|߻ucι.%Y]EN^˭Ʈ(+<۽֔Da+4a0>@AwEZlzhdܞթ.<n[hz^T]SgNߑ.s~R5aIWrx̐8vA2kTטK   <'73g<x_y+<ʻ&. )ŵP5T'!'c6d +곚}.j29llgJ5(4^׹Y^8IOܒ3Kiq(%:(M0O9庼ȩ;lgtlJĬ18U`ruZuԪ1M-4i. gUxҜyw>w/:_MO:>>&ElM$Q%F4Pufz jGF7-XQ~aLF&'iOA5^O쭲4BwCY0eW("_*lɍV{V5Q"-ĝ%#K/洱9. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1924, 78 words, 4 clauses) |
| --- | Minor | v6e=^SA~Wkq/DG^l3;Rm7[pQ67ASU,[x="-q]0v7"ǒ:+n?V.D+,'zqТj;*xQfGڵ8[y (8NH&=Ҕ\^*Φ|lxr?b3InY,F9;q{8s%-A]"(Tn "Ȼᴚ#qy"y>tCѢW3Qv	UeJ'q'{&w(UZ/4B짡jV')"{؊l^ `E]o^yB7Z$J*ToDcF%ꪰQXDg.rK nJghJD1itBj:55?9+.jd,_ǧ}_+HR@̮Rkr&p_Cf1*Lk鬡NMfgf zGY"@ y.kI"چZ+Om=?׿ |
| --- | Minor | v6e=^SA~Wkq/DG^l3;Rm7[pQ67ASU. [x="-q]0v7"ǒ:+n?V.D+. 'zqТj;*xQfGڵ8[y (8NH&=Ҕ\^*Φ|lxr?b3InY. F9;q{8s%-A]"(Tn "Ȼᴚ#qy"y>tCѢW3Qv	UeJ'q'{&w(UZ/4B짡jV')"{؊l^ `E]o^yB7Z$J*ToDcF%ꪰQXDg.rK nJghJD1itBj:55?9+.jd. _ǧ}_+HR@̮Rkr&p_Cf1*Lk鬡NMfgf zGY"@ y.kI"چZ+Om=?׿. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 1938, 63 words, 3 clauses) |
| --- | Minor | <.3~'=!<E2Y=,?dlQ1gZ._gBf~.V}X6;,-m( ,bFF2r_.;jN+nܧve}0N󰂞sTM]QTcYz\+v3v*/}*u|5 R8W"l0ԓ'c>UE`6!5h"7?G.y  \X%p(.-1;Hz+xbV[yLGbPwaBG	UXx>b5bɯ6V4F$x4SkW_OA}D<ݭCy&wKAۍN2+*cțpGimwh-f5 |
| --- | Minor | <.3~'=!<E2Y=. ?dlQ1gZ._gBf~.V}X6;. -m( . bFF2r_.;jN+nܧve}0N󰂞sTM]QTcYz\+v3v*/}*u|5 R8W"l0ԓ'c>UE`6!5h"7?G.y  \X%p(.-1;Hz+xbV[yLGbPwaBG	UXx>b5bɯ6V4F$x4SkW_OA}D<ݭCy&wKAۍN2+*cțpGimwh-f5. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2034, 42 words, 5 clauses) |
| --- | Minor | 15V*^}uҥIeJi}đGnf'IG?y4oͶG?`3\|(u?;oOjJ19>C ,3߬P6f^ </9O8oL\jyrY3ט7}Ui3f||[isB~m>?, }];׿Ւ$MV@Z[nMG&%*Mo};<؋K/X/~+,Xb宻*NNg@_ߗWUkVc__>{.;wsN8%Y'KRs v vf _!d ϭ0AH,,xH |
| --- | Minor | 15V*^}uҥIeJi}đGnf'IG?y4oͶG?`3\|(u?;oOjJ19>C . 3߬P6f^ </9O8oL\jyrY3ט7}Ui3f||[isB~m>?.  }];׿Ւ$MV@Z[nMG&%*Mo};<؋K/X/~+. Xb宻*NNg@_ߗWUkVc__>{.;wsN8%Y'KRs v vf _!d ϭ0AH. xH. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2105, 63 words, 0 clauses) |
| --- | Minor | êUZLi}|uM*B~.-3	2C~HwH}̳SLWԋxÞrӶ=RFjr:v@eB˖"!Zi"R'{{;p񱱴 H |֝>ڐɀf5`jE쫽"MZk D|r.<sƬnUo*5xFĄuZmskCluQ L%:Dq'Gk4I[擏=}l~c[9@>xHc):y'E5 @ R1+2I^vuף_RLeyDB&2R䠃̔Kv#eZ h[]ɇx< zbr3VѤ]U<_yHBn1&qѿ F@$L+&lU>63 |
| --- | Minor | êUZLi}|uM*B~.-3	2C~HwH}̳SLWԋxÞrӶ=RFjr:v@eB˖"!Zi"R'{{;p񱱴 H |֝>ڐɀf5`jE쫽"MZk D|r.<sƬnUo*5xFĄuZmskCluQ L%:Dq'Gk4I[擏=}l~c[9@>xHc):y'E5 @ R1+2I^vuף_RLeyDB&2R䠃̔Kv#eZ h[]ɇx< zbr3VѤ]U<_yHBn1&qѿ F@$L+&lU>63 |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2112, 62 words, 3 clauses) |
| --- | Minor | $UeBp E*,BͶ<ψ6Gn?,FsמdmE }Z* Cß?\4M=YyL) Gvo'ӧl9w};5cZMnB/vLRTTԦ؈j "BM](pp`@HJUq!#{I3 (1Iv	7[W|_7jYk҈-A:I?3H1i)o!2ߥu=<\~GsDiBAxn (05K}!SRcRADeϯT)ِn	xAPGproV*dԟsYw|xY[MDI/=∇g쫧,~eKr"=ZkbN"{S3Y9VCm8]+ |
| --- | Minor | $UeBp E*. BͶ<ψ6Gn?. FsמdmE }Z* Cß?\4M=YyL) Gvo'ӧl9w};5cZMnB/vLRTTԦ؈j "BM](pp`@HJUq!#{I3 (1Iv	7[W|_7jYk҈-A:I?3H1i)o!2ߥu=<\~GsDiBAxn (05K}!SRcRADeϯT)ِn	xAPGproV*dԟsYw|xY[MDI/=∇g쫧. ~eKr"=ZkbN"{S3Y9VCm8]+. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2220, 96 words, 0 clauses) |
| --- | Minor | O>[uGw܁%덆&tՈ@&dPץ=LQ!DuW(hArD#XstI=Rj`M_vӧ\<mh4}~S##;mݚɬ(s#>rxG @ ?۟<uʠrdtlÆVkU{753Q @\PH&e! jbtlle>4 cd.pkW7ǵ3gͺSj#cC8.Bh=[Z/<?sѱQ)9޺7WWO>nϟ.Z+_[e˗ǗO:O}[o}x~{XʬnmX9>ި7wa!u`oo+ϷmRCRW	boAS10*`\XDJfg_P'W`ty~ZSh]@AQ@R:4iӣnEMJYJ))}W~zH{#J-dl1cƝ=ϜAÒ(>o7駿 k7xk^sZ5:F8pJ!Lr;>imJF;kAP h_f_y1EV&W㩺R 98\*;0[>s`/wd2@SuVR_Q-1FOY;I0iU}.ɕ.twz}^  |
| --- | Minor | O>[uGw܁%덆&tՈ@&dPץ=LQ!DuW(hArD#XstI=Rj`M_vӧ\<mh4}~S##;mݚɬ(s#>rxG @ ?۟<uʠrdtlÆVkU{753Q @\PH&e! jbtlle>4 cd.pkW7ǵ3gͺSj#cC8.Bh=[Z/<?sѱQ)9޺7WWO>nϟ.Z+_[e˗ǗO:O}[o}x~{XʬnmX9>ި7wa!u`oo+ϷmRCRW	boAS10*`\XDJfg_P'W`ty~ZSh]@AQ@R:4iӣnEMJYJ))}W~zH{#J-dl1cƝ=ϜAÒ(>o7駿 k7xk^sZ5:F8pJ!Lr;>imJF;kAP h_f_y1EV&W㩺R 98\*;0[>s`/wd2@SuVR_Q-1FOY;I0iU}.ɕ.twz}^  |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2238, 73 words, 2 clauses) |
| --- | Minor | +մÍA/9<{_vwſ8Fq*"͘n;{x9SA5߿yv)oo8ԯ\÷]~#ܹP  ,`UӤ0f<8e̢W -XaivϮXHRi܅k/q'+qXhL"j#U71j=<]12鲨(jG?_S=2N'a2KT i|\ dRv`:@1ET:wwԱ#PY|Ѿ4%k_0z3 =^VqI!`*3}B.)\Cw{%m1"C9IÒ	=ApC;>bs,XE%n>u.^;zZ1lReˢFxm&Cvd4dw>c@..=!2;a4)GF=jz 	2+gZe@@>/1  R9ļ^&ER@	FtVl.hA (ب$h  Q)RsM { |
| --- | Minor | +մÍA/9<{_vwſ8Fq*"͘n;{x9SA5߿yv)oo8ԯ\÷]~#ܹP  . `UӤ0f<8e̢W -XaivϮXHRi܅k/q'+qXhL"j#U71j=<]12鲨(jG?_S=2N'a2KT i|\ dRv`:@1ET:wwԱ#PY|Ѿ4%k_0z3 =^VqI!`*3}B.)\Cw{%m1"C9IÒ	=ApC;>bs. XE%n>u.^;zZ1lReˢFxm&Cvd4dw>c@..=!2;a4)GF=jz 	2+gZe@@>/1  R9ļ^&ER@	FtVl.hA (ب$h  Q)RsM {. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2250, 67 words, 2 clauses) |
| --- | Minor | >p>Й^~ծ@ʆ}٢/?fҿ~ר).La\hC&o /,V ?rG5aUvVI[z,oc<y)SXDY00'5 ?KAHW&&a|x"EPI9ʤEɧ.l"OlzjqScX泟Adj^XR0G]ĝW]k 	`6Y: *Bsn'1]z  U90ruPt7Fea?Ƥ`H@݉&wh K#_(D)ɑˁ 2o7j+alo'|xn:cuTّ# :D}|-؍L$k0ls_(6u	&Eeh	 O}7k~ֲzo_ojx@P |
| --- | Minor | >p>Й^~ծ@ʆ}٢/?fҿ~ר).La\hC&o /. V ?rG5aUvVI[z. oc<y)SXDY00'5 ?KAHW&&a|x"EPI9ʤEɧ.l"OlzjqScX泟Adj^XR0G]ĝW]k 	`6Y: *Bsn'1]z  U90ruPt7Fea?Ƥ`H@݉&wh K#_(D)ɑˁ 2o7j+alo'|xn:cuTّ# :D}|-؍L$k0ls_(6u	&Eeh	 O}7k~ֲzo_ojx@P. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2368, 100 words, 2 clauses) |
| --- | Minor | r=mWSWd	Seg]{ѫ@zi]rrDCy)ҩwHC'6v+{*n&`lчxۅw;hå5ts+j߾	XPgg<U.94p`Wwp0V>rB&xQY4oh k@֗#?/Cܑ`YFѕ@T}ZV8ΗXY sa%Z͜6(H0sDfgk$z^;{ca)w'U*+]TEu&ގ4Tֿݷy v"O`WǴh,7]v{ ,/+H*ĳ	͚ȪY}GM1PF+%{;.HW>8IGvK\5{(+p˒=aDFmn-8iE/_GY+s!HJlUJ<ǀ{:+ fU̦MexY.jEA.+v{)HT( @eGI%%&K)3)I&Pu){Zb^p=veNcDo+C\C"R paɾI4 MfbB\"		*l |
| --- | Minor | r=mWSWd	Seg]{ѫ@zi]rrDCy)ҩwHC'6v+{*n&`lчxۅw;hå5ts+j߾	XPgg<U.94p`Wwp0V>rB&xQY4oh k@֗#?/Cܑ`YFѕ@T}ZV8ΗXY sa%Z͜6(H0sDfgk$z^;{ca)w'U*+]TEu&ގ4Tֿݷy v"O`WǴh. 7]v{. /+H*ĳ	͚ȪY}GM1PF+%{;.HW>8IGvK\5{(+p˒=aDFmn-8iE/_GY+s!HJlUJ<ǀ{:+ fU̦MexY.jEA.+v{)HT( @eGI%%&K)3)I&Pu){Zb^p=veNcDo+C\C"R paɾI4 MfbB\"		*l. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2463, 86 words, 2 clauses) |
| --- | Minor | =PTΏ8@,`X ]':uP'o?G |5Tl@ c\wtZ:o/;ՍzV0Ru*#k,\&%pF:Od	pqU=@g:0LDQLsq*Zjc82Ɗ8>i0τyiCFpj["X4Z#XFtb잧p3C@&5g?|{JIEd\z LtsR EW#e(HfvZ ^]veh5 [0YoWgZ肋k`߽ѷs.S8.0@h	-)׭p-aSLQzG~2u;#  >OZV^A!Zi~yV7hx\oIPħ^WȃpbTwUaio_F3X^\r~/Cǭ^y4!36hrs!|/܈5~-o'W;!+˿|tC!IknNDltEѨ.uW|t=@H􀭴̳ǭ |
| --- | Minor | =PTΏ8@. `X ]':uP'o?G |5Tl@ c\wtZ:o/;ՍzV0Ru*#k. \&%pF:Od	pqU=@g:0LDQLsq*Zjc82Ɗ8>i0τyiCFpj["X4Z#XFtb잧p3C@&5g?|{JIEd\z LtsR EW#e(HfvZ ^]veh5 [0YoWgZ肋k`߽ѷs.S8.0@h	-)׭p-aSLQzG~2u;#  >OZV^A!Zi~yV7hx\oIPħ^WȃpbTwUaio_F3X^\r~/Cǭ^y4!36hrs!|/܈5~-o'W;!+˿|tC!IknNDltEѨ.uW|t=@H􀭴̳ǭ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2534, 100 words, 3 clauses) |
| --- | Minor | uyU [o;`=R;e<O? C|7xmr#fK?yV+W7YoCy𑝅]]irϔ1]l`xΉ'4_/75U"Qjk;duf_ m7&x/p{|TcZ=(ퟨݒѭNkjW5_V4NћǯA}N!Sz:ޕv	ɐ XcGc<sFsBzD^nM]p.8SC+āvJV)FDVNJ+Ψ+H,hؐah/%!9,I_i?dL>iMapn&jl~-,tD&Eˀ]ˎ޸y"d@gfu|miGsנ$1Hg~RӢ߼~vSPU@`^0ÓTWIj=s<ZBTc!JEԩqK) RGxbDDY</5؏U<%I{s}8jo7uk2@ؾG?oAmbJ6AN DU(EItS/Bz!Q\|&f+BtWyC&zLǤ6Rt |
| --- | Minor | uyU [o;`=R;e<O? C|7xmr#fK?yV+W7YoCy𑝅]]irϔ1]l`xΉ'4_/75U"Qjk;duf_ m7&x/p{|TcZ=(ퟨݒѭNkjW5_V4NћǯA}N!Sz:ޕv	ɐ XcGc<sFsBzD^nM]p.8SC+āvJV)FDVNJ+Ψ+H. hؐah/%!9. I_i?dL>iMapn&jl~-. tD&Eˀ]ˎ޸y"d@gfu|miGsנ$1Hg~RӢ߼~vSPU@`^0ÓTWIj=s<ZBTc!JEԩqK) RGxbDDY</5؏U<%I{s}8jo7uk2@ؾG?oAmbJ6AN DU(EItS/Bz!Q\|&f+BtWyC&zLǤ6Rt. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2538, 23 words, 4 clauses) |
| --- | Minor | } Q,O?}xW* j;³ok.:vA}7{cg{,Eew^8|Q	3zOw]ܹ'L,Pa斯}zh5n]* ! bh,clV9w媝ƍӿIx)' |
| --- | Minor | } Q. O?}xW* j;³ok.:vA}7{cg{. Eew^8|Q	3zOw]ܹ'L. Pa斯}zh5n]* ! bh. clV9w媝ƍӿIx)'. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2541, 47 words, 4 clauses) |
| --- | Minor | Ozj3c1	zwC&qfA[ɪ5Lfw@	:¡h++d.IABJRĒ1|,7wWAA&@˧^DSTo5|0w,J0Y.f؛#AWb0-,y& Gȧճ'Ev #Ե(:cgI?B{ͼ8	wi4h+7NSD؎x P)bus$@(TBT>c1,\W](ŜL)b\z^uB&uJRÛRS('YXiFߪk%Lcكt!EC |
| --- | Minor | Ozj3c1	zwC&qfA[ɪ5Lfw@	:¡h++d.IABJRĒ1|. 7wWAA&@˧^DSTo5|0w. J0Y.f؛#AWb0-. y& Gȧճ'Ev #Ե(:cgI?B{ͼ8	wi4h+7NSD؎x P)bus$@(TBT>c1. \W](ŜL)b\z^uB&uJRÛRS('YXiFߪk%Lcكt!EC. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2547, 81 words, 4 clauses) |
| --- | Minor | 6f:%rF45?կhQ  fW][\UcrB&BvT;AfH,hRdp'm(_}ñfv%BdbU!!Wo:~\{W+y?|GwO;;c- ֭Zeۙ@DqΝNa#>Xd2, >yrà]~R@\Pp=Q_#H2 ([2  wU: p d{W. Ԝd+q "AtL]~UX.=zԍ_:{҈ ON&rиV ,8ո5S2`-849 _t+qX-wՖ3@& {d?Nz7^.;\%h[ױXK:p  I$+V% Q,BλťK)O  JU1!4xr^|MYJ'Q>B83JMX(+e[ܺںHD[oS+֮/9T? |
| --- | Minor | 6f:%rF45?կhQ  fW][\UcrB&BvT;AfH. hRdp'm(_}ñfv%BdbU!!Wo:~\{W+y?|GwO;;c- ֭Zeۙ@DqΝNa#>Xd2.  >yrà]~R@\Pp=Q_#H2 ([2  wU: p d{W. Ԝd+q "AtL]~UX.=zԍ_:{҈ ON&rиV. 8ո5S2`-849 _t+qX-wՖ3@& {d?Nz7^.;\%h[ױXK:p  I$+V% Q. BλťK)O  JU1!4xr^|MYJ'Q>B83JMX(+e[ܺںHD[oS+֮/9T?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2605, 64 words, 2 clauses) |
| --- | Minor | $	8I ui׺V"no8dڌ/т;;n޸iӅGc/Ƽ3ΞvUƍR\9&AV"^*<{MHN RCLJ RD13ɹ_ß~?׸q6mn߰+^߄@GuC I:H HaFH}Zf"Gxg7yOK/^{b+>uАB̉)D;4,{a4s"1BkBU/\V[yԨ`hH@.{o4yWZ?݀ac"ɟjeԧ7k׮N;6xd̿ʕ/}sSi_0[e?TÞR%mQ_#,lHGȓ)2="F!C6aJ |
| --- | Minor | $	8I ui׺V"no8dڌ/т;;n޸iӅGc/Ƽ3ΞvUƍR\9&AV"^*<{MHN RCLJ RD13ɹ_ß~?׸q6mn߰+^߄@GuC I:H HaFH}Zf"Gxg7yOK/^{b+>uАB̉)D;4. {a4s"1BkBU/\V[yԨ`hH@.{o4yWZ?݀ac"ɟjeԧ7k׮N;6xd̿ʕ/}sSi_0[e?TÞR%mQ_#. lHGȓ)2="F!C6aJ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2621, 63 words, 1 clauses) |
| --- | Minor | mS͖ȽoݺZ`	< Iⶶt.f 	Ajl~[o:u]O:>`4HbȸΣSFВRJ	x+% @,c )t!CJ%	ה⋿PG39Ps.JϾˮUbQZVpkY%	)%boHa+M7d'{S`p`GX5hIk$g8n~?%oc?\WXd7:?딓v׾*R"w@ڜiU"d.cJ*Wy}rN{D@)@qkGsz聿cX~0yƔ	'u9q2̬75sjC׎:1{lү1bw=?}u_w]A@TYtyYȱǩ |
| --- | Minor | mS͖ȽoݺZ`	< Iⶶt.f 	Ajl~[o:u]O:>`4HbȸΣSFВRJ	x+% @. c )t!CJ%	ה⋿PG39Ps.JϾˮUbQZVpkY%	)%boHa+M7d'{S`p`GX5hIk$g8n~?%oc?\WXd7:?딓v׾*R"w@ڜiU"d.cJ*Wy}rN{D@)@qkGsz聿cX~0yƔ	'u9q2̬75sjC׎:1{lү1bw=?}u_w]A@TYtyYȱǩ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2657, 62 words, 0 clauses) |
| --- | Minor |  O8` `9"1RB6`mQjA3J+)?]Dt:SnhxwwH#+t^x"k K7VITkkol9m52D`3t|fpJr@SӜ)3BL=j`cC%25MGss1;rAP"D%R9G7n{5Sl.ϼgs'KA I@kG_57sr)<lHJ%Y"`0b.<#UfL]gGqKWWa(+q|;ChÈ hp>ZZ%CH&9n(Xgxgޘ_]Mc Wd+1Fcs2d#]v2Rڷ_ 0Γ:1 |
| --- | Minor |  O8` `9"1RB6`mQjA3J+)?]Dt:SnhxwwH#+t^x"k K7VITkkol9m52D`3t|fpJr@SӜ)3BL=j`cC%25MGss1;rAP"D%R9G7n{5Sl.ϼgs'KA I@kG_57sr)<lHJ%Y"`0b.<#UfL]gGqKWWa(+q|;ChÈ hp>ZZ%CH&9n(Xgxgޘ_]Mc Wd+1Fcs2d#]v2Rڷ_ 0Γ:1 |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2686, 93 words, 3 clauses) |
| --- | Minor | y!?r~o;Cv{gYIB]O^kۮ[~4uj]Yj ɡl>K0 ~&˽1k x2:d0JH(6*N2Y붯~ ߌxΓ3JrO	f̛?l=@g]OWXB=Tjj^=LpynzɈT2Dn-ea"nTSI{Mp~?fs˲ᰪ%d|G[^`Ok~h} J4e	VROXjFףPk-&DpB;֭^ 'OP+K TOP;mh J9Vd}u KY#ƸqT	M7yD^ƽj !5HJ @ "!3K"~K7% RX_Q2uKX 	mHq&ʲK?<\ B B20)%BrXxGfAc{)4k"*WHxf8b,bNqF8,1 Rfp]8c̦>q)6LjBw' H!0/&٘ @ER+0Kj? 鍃s[ I4f <8h,%u |
| --- | Minor | y!?r~o;Cv{gYIB]O^kۮ[~4uj]Yj ɡl>K0 ~&˽1k x2:d0JH(6*N2Y붯~ ߌxΓ3JrO	f̛?l=@g]OWXB=Tjj^=LpynzɈT2Dn-ea"nTSI{Mp~?fs˲ᰪ%d|G[^`Ok~h} J4e	VROXjFףPk-&DpB;֭^ 'OP+K TOP;mh J9Vd}u KY#ƸqT	M7yD^ƽj !5HJ @ "!3K"~K7% RX_Q2uKX 	mHq&ʲK?<\ B B20)%BrXxGfAc{)4k"*WHxf8b. bNqF8. 1 Rfp]8c̦>q)6LjBw' H!0/&٘ @ER+0Kj? 鍃s[ I4f <8h. %u. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2731, 42 words, 4 clauses) |
| --- | Minor | Ԓq&,F­ݝA,	bI*aKul4q1" 	@LI R2C%J+GR,`%8c՘ a3Y'itDJ߼$ZȘ n)[=@	bst%%gQ#Սn#N6kڀ5)B" d n*4x䩓8piHH4*E#7#"#	ĦƋ 0 x>00)vr3u M}UL8__#3\,(A5 ?y4 |
| --- | Minor | Ԓq&. F­ݝA. bI*aKul4q1" 	@LI R2C%J+GR. `%8c՘ a3Y'itDJ߼$ZȘ n)[=@	bst%%gQ#Սn#N6kڀ5)B" d n*4x䩓8piHH4*E#7#"#	ĦƋ 0 x>00)vr3u M}UL8__#3\. (A5 ?y4. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2822, 39 words, 4 clauses) |
| --- | Minor | ZۼABؘUnad<.@U>H`e:=|[}Գۻn|Ӟ&5*ֹp?d1\Bְ⌼h,}YY$a6z&5TpoDԗѐ JX?'\{ЬѺڊkzC\ x9DZsaA:P/~1	u,M#ȒirI,xу:hf^.FvC,"MsD@j |
| --- | Minor | ZۼABؘUnad<.@U>H`e:=|[}Գۻn|Ӟ&5*ֹp?d1\Bְ⌼h. }YY$a6z&5TpoDԗѐ JX?'\{ЬѺڊkzC\ x9DZsaA:P/~1	u. M#ȒirI. xу:hf^.FvC. "MsD@j. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2829, 61 words, 2 clauses) |
| --- | Minor | C[] N1,!1	>3O?x˯O܀a 2(;x~'.ҵGC }{&Dtgaa{_|ɮ{ݫ.{ĕ~Sz"Hð6S<}1`(A!cB7a׮]G>HD~zW~o|K_zƉ'	 2oGX3	1[#Pԭ5y91߀҅n\V]rh&*yand+O[G12\lƥH`"6g	eZ;>;Ư6_&.O-1,~fZx3OH_PA͵bF)NG.OyFK=x&'X50*SXSVMj>c(zB!Q$4 |
| --- | Minor | C[] N1. !1	>3O?x˯O܀a 2(;x~'.ҵGC }{&Dtgaa{_|ɮ{ݫ.{ĕ~Sz"Hð6S<}1`(A!cB7a׮]G>HD~zW~o|K_zƉ'	 2oGX3	1[#Pԭ5y91߀҅n\V]rh&*yand+O[G12\lƥH`"6g	eZ;>;Ư6_&.O-1. ~fZx3OH_PA͵bF)NG.OyFK=x&'X50*SXSVMj>c(zB!Q$4. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2844, 77 words, 1 clauses) |
| --- | Minor | ;Hβ)C	nv*9  :LbNfba2;D4g|˷<a?3n!nҽE/|}`i2l̂cl&rAB*IC*2\߆fðB~?N<ğܳ\צ J%#!1.\L߼n\q_tdc&ao.q7;`D]RO6igۿwq眳{1ؾe5I:K/7}{.}1 cNևgg돝qi>bMf?O.:={Y6'敏ΐtyK@e4ݤ[ǜ~)uy矿s,tb"RB腑(E)'fS:9-fUGL+-6d:Gͬ:Ѩ_d>;4X]V]f	kD)X/x2E6b'\6VI+:Bvu։wׄ)vv_qv敇<z(5NL]#[7R:d |
| --- | Minor | ;Hβ)C	nv*9  :LbNfba2;D4g|˷<a?3n!nҽE/|}`i2l̂cl&rAB*IC*2\߆fðB~?N<ğܳ\צ J%#!1.\L߼n\q_tdc&ao.q7;`D]RO6igۿwq眳{1ؾe5I:K/7}{.}1 cNևgg돝qi>bMf?O.:={Y6'敏ΐtyK@e4ݤ[ǜ~)uy矿s. tb"RB腑(E)'fS:9-fUGL+-6d:Gͬ:Ѩ_d>;4X]V]f	kD)X/x2E6b'\6VI+:Bvu։wׄ)vv_qv敇<z(5NL]#[7R:d. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2913, 89 words, 5 clauses) |
| --- | Minor | ^؟J?I`<;KI#Uv?F*G,ǎÝ D;152!O̆aλy+^o{~%}}{u+^I?O0#1l5\b4Ndxk=T4VgJVLuͅISl/}~Q)vtëF뛙v qa:^+<î&Ӿy,XQS" U='	4B~TT_(@!&ToW_%VJMv+=ђ)1-""r(MP;_	G8S~q4l]L̎cy}f}LO,!mpP]g'IwVrLM+YA)jlX߶*UyʼUR,H5ccKg+!0276)_Y:ҊVg%(>oet cd:׎|]r=bS`{΢%X: ֦P[G=Y(meؕyc㥹<Ъmϻi8383UJ=Dlp"],_^ڮ(xʭ'y3FVԕ |
| --- | Minor | ^؟J?I`<;KI#Uv?F*G. ǎÝ D;152!O̆aλy+^o{~%}}{u+^I?O0#1l5\b4Ndxk=T4VgJVLuͅISl/}~Q)vtëF뛙v qa:^+<î&Ӿy. XQS" U='	4B~TT_(@!&ToW_%VJMv+=ђ)1-""r(MP;_	G8S~q4l]L̎cy}f}LO. !mpP]g'IwVrLM+YA)jlX߶*UyʼUR. H5ccKg+!0276)_Y:ҊVg%(>oet cd:׎|]r=bS`{΢%X: ֦P[G=Y(meؕyc㥹<Ъmϻi8383UJ=Dlp"]. _^ڮ(xʭ'y3FVԕ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2959, 106 words, 8 clauses) |
| --- | Minor | v0sx PɴF̎ԾEb|NC"l̼2#5p8h,`EnĊ%zI,YWxCRﮀ<OM-Ř}F   &vHrNj?Gd~1Օ1ӭ})7c<tdHbt𺼔"|sh P̘iMoֿW'1g03Z(zF02uBkJ0 u Eݠ4N2z4 PRvZ쑙s掸<Ѻ՞>g#f'ފa{2C(	Hr{H 9w~ꛟO{Sql0Xphd01!iV@렕mJ,{z0O-pBaHzg-j< 3A~Nbڼ,j!=J_0,eb#:!oXi`C4bQ]4Myj6Ĩr۶emih<xYjn(SOopyi2Y=~gCŭc]y8iil,BCfClʊ o%*:pP;m.t5sAVPKj*JR#\S++s5P[VيkW=LZF KϮiF,~(kagbg[/¹.;8^|/P p=jg_GN?vN0ŮF4i2oq;˰K6<&d>{":Gr<	\t"9/$O83+_g+P |
| --- | Minor | v0sx PɴF̎ԾEb|NC"l̼2#5p8h. `EnĊ%zI. YWxCRﮀ<OM-Ř}F   &vHrNj?Gd~1Օ1ӭ})7c<tdHbt𺼔"|sh P̘iMoֿW'1g03Z(zF02uBkJ0 u Eݠ4N2z4 PRvZ쑙s掸<Ѻ՞>g#f'ފa{2C(	Hr{H 9w~ꛟO{Sql0Xphd01!iV@렕mJ. {z0O-pBaHzg-j< 3A~Nbڼ. j!=J_0. eb#:!oXi`C4bQ]4Myj6Ĩr۶emih<xYjn(SOopyi2Y=~gCŭc]y8iil. BCfClʊ o%*:pP;m.t5sAVPKj*JR#\S++s5P[VيkW=LZF KϮiF. ~(kagbg[/¹.;8^|/P p=jg_GN?vN0ŮF4i2oq;˰K6<&d>{":Gr<	\t"9/$O83+_g+P. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 2960, 22 words, 4 clauses) |
| --- | Minor | j۬{i\!e$+6L9s+ԋBŚyd	J\{tp=q,i##PNB 	,QCrL	ʞ {, *4b*ljPoj̋:<c3#b7,;eԇdd'U |
| --- | Minor | j۬{i\!e$+6L9s+ԋBŚyd	J\{tp=q. i##PNB 	. QCrL	ʞ {.  *4b*ljPoj̋:<c3#b7. ;eԇdd'U. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3042, 75 words, 1 clauses) |
| --- | Minor | IPŗ8- @dʔ!iW+!PZ8ߵ;nħ>G?+G{q# FVV4 9e0r9~W4s(ruֻXAY'eceoUV ԬJb"8˔&?[).l	Ǟ`!ܻ늾^m5("mdbLFy{E#*gJF*'6\,[ _^ [7:.s7Sqvg-Z7hBqLRT6飯RZ4 QDcA0QZKnڜ-+\{1G#X	M6ȗga4L/ظDâa"]h=ha3i7k0vEpdoM޶)tXK</-DH	T /9|w9b@)`邎 j3D#]MQq"ˑ.!T23-3 ITC_7׺ǯ%?H5&R5&XtmU'a$=qҍ\wXGNP{+ |
| --- | Minor | IPŗ8- @dʔ!iW+!PZ8ߵ;nħ>G?+G{q# FVV4 9e0r9~W4s(ruֻXAY'eceoUV ԬJb"8˔&?[).l	Ǟ`!ܻ늾^m5("mdbLFy{E#*gJF*'6\. [ _^ [7:.s7Sqvg-Z7hBqLRT6飯RZ4 QDcA0QZKnڜ-+\{1G#X	M6ȗga4L/ظDâa"]h=ha3i7k0vEpdoM޶)tXK</-DH	T /9|w9b@)`邎 j3D#]MQq"ˑ.!T23-3 ITC_7׺ǯ%?H5&R5&XtmU'a$=qҍ\wXGNP{+. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3064, 28 words, 4 clauses) |
| --- | Minor | LƐ=Ju܉L(ՉG<LLKT	2BwT,U<VZ,y7H$K`KW#wY⭾u{	N:U']Hʕr,y)"{]عw?"Pq4톷:e+ zFYU CY,ƼH^R |
| --- | Minor | LƐ=Ju܉L(ՉG<LLKT	2BwT. U<VZ. y7H$K`KW#wY⭾u{	N:U']Hʕr. y)"{]عw?"Pq4톷:e+ zFYU CY. ƼH^R. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3077, 82 words, 6 clauses) |
| --- | Minor | Y7~)fxxĎXppvwJ-V+<1YuCsvmUzv#,WyjpMY5In`2< DW{v-)pƙ/-/g	0|NIu]" C&@	{_k<Ga`vTYσb|YEձ,*|bGE{Vvl6򙄶>=K^=?F!j>M-3Ze-%H%̞S,|Zgj|Y#OznM|헯'<Krq/e4,wp1v|5׼}[wۻv/_r/okS5q޾Dc9k̰BH+,2ӿ=䪋 Kӥ1n6oı=3Lj.k;͛IoI|Ցћdv)O}Dͺ\" o~8c14[1^x'{d !xQ!2Iߨb3P߂&:oKoO<BVyƟjZ Dxo7=6R[ 1]VӦ-!|нo,LjO6\x |
| --- | Minor | Y7~)fxxĎXppvwJ-V+<1YuCsvmUzv#. WyjpMY5In`2< DW{v-)pƙ/-/g	0|NIu]" C&@	{_k<Ga`vTYσb|YEձ. *|bGE{Vvl6򙄶>=K^=?F!j>M-3Ze-%H%̞S. |Zgj|Y#OznM|헯'<Krq/e4. wp1v|5׼}[wۻv/_r/okS5q޾Dc9k̰BH+. 2ӿ=䪋 Kӥ1n6oı=3Lj.k;͛IoI|Ցћdv)O}Dͺ\" o~8c14[1^x'{d !xQ!2Iߨb3P߂&:oKoO<BVyƟjZ Dxo7=6R[ 1]VӦ-!|нo. LjO6\x. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3113, 41 words, 4 clauses) |
| --- | Minor | ۝a]C["{ގmD5 &!Dw9SN=pwy:D8 ''PTcTPhBM,++ re5cWBIWj{>ɏ|Ͻ||>BHK ڤ]e˺p4X:l:'LUZLH,X!7;DέͳgĻ5۪!~qt!@@D?'|us{N>{,O&b?!W:89)ŕ3Z (9S!L=F_Eݶʴ Tng,z(R |
| --- | Minor | ۝a]C["{ގmD5 &!Dw9SN=pwy:D8 ''PTcTPhBM. ++ re5cWBIWj{>ɏ|Ͻ||>BHK ڤ]e˺p4X:l:'LUZLH. X!7;DέͳgĻ5۪!~qt!@@D?'|us{N>{. O&b?!W:89)ŕ3Z (9S!L=F_Eݶʴ Tng. z(R. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3167, 66 words, 5 clauses) |
| --- | Minor | {f~Fyi~/FvZ׹:9uu3rUkkǞt-;D M7pCumįN=7 KQ* «Cf[9!QϠitJle \<d&"J P׬9*A20|RWתڃU)W:= @~ĢWrt&^su^%(h sg2LۦO~[J Qrݬ<x4kbmԱ-/ML6҇3Td_,br WƳ?P9s,<a8`PTf^wHDNE6YA,,Wk#٥( <b,D~o/[%pDvdMYϵU(c$4ֿq#+J*[[Y+h?4W(N7AᆽBZ}%*؊R>WUĽNDl6e3̻` |
| --- | Minor | {f~Fyi~/FvZ׹:9uu3rUkkǞt-;D M7pCumįN=7 KQ* «Cf[9!QϠitJle \<d&"J P׬9*A20|RWתڃU)W:= @~ĢWrt&^su^%(h sg2LۦO~[J Qrݬ<x4kbmԱ-/ML6҇3Td_. br WƳ?P9s. <a8`PTf^wHDNE6YA. Wk#٥( <b. D~o/[%pDvdMYϵU(c$4ֿq#+J*[[Y+h?4W(N7AᆽBZ}%*؊R>WUĽNDl6e3̻`. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3189, 44 words, 5 clauses) |
| --- | Minor | @ΪZ9^eTf=;!Udܨb,pTD,i`wǀ;2s@ˈ{vܶᐼy"fh7-,QDL66cZ0/ǽ=-YxLuҮ8a/b)>RI4nb,5ț<+ͩ>^lCRlnQc.c*OrsM,&"c* 1a&ϲ"Iqb<sb2XPA26O+l<dT%U(q_ |
| --- | Minor | @ΪZ9^eTf=;!Udܨb. pTD. i`wǀ;2s@ˈ{vܶᐼy"fh7-. QDL66cZ0/ǽ=-YxLuҮ8a/b)>RI4nb. 5ț<+ͩ>^lCRlnQc.c*OrsM. &"c* 1a&ϲ"Iqb<sb2XPA26O+l<dT%U(q_. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3190, 23 words, 4 clauses) |
| --- | Minor | r!m}m裲Y.ѭěml@S!,!2UM;!!a/p,<}X@,&æM3FDiٰtW*8$F/>K.3Ldz({Twu%[eJCfxh9g,9J`(i\ˢx |
| --- | Minor | r!m}m裲Y.ѭěml@S!. !2UM;!!a/p. <}X@. &æM3FDiٰtW*8$F/>K.3Ldz({Twu%[eJCfxh9g. 9J`(i\ˢx. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3258, 73 words, 3 clauses) |
| --- | Minor | 4W` ޱ%m#C"؎	fLb˗KHz>z@_8@v0&p0	\YyA{[9C2KXWGϋ`T>8`nj R/\I]dxSk 	 jHUmLF@yk\*]gmXi;!1Y	7PC^B%3Ū!Ơ%HAOPsr+6E%X`mFX]ZbGSNOtpE5Fv^7`2]bBeT;<ˉWA GES,Y1`diS,;+]A`,z}6$wW:*lUxP5nPq]13]8n_fѻuqU6ěь{K}+ҩ\V/: '?)8>A/ |
| --- | Minor | 4W` ޱ%m#C"؎	fLb˗KHz>z@_8@v0&p0	\YyA{[9C2KXWGϋ`T>8`nj R/\I]dxSk 	 jHUmLF@yk\*]gmXi;!1Y	7PC^B%3Ū!Ơ%HAOPsr+6E%X`mFX]ZbGSNOtpE5Fv^7`2]bBeT;<ˉWA GES. Y1`diS. ;+]A`. z}6$wW:*lUxP5nPq]13]8n_fѻuqU6ěь{K}+ҩ\V/: '?)8>A/. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3292, 28 words, 4 clauses) |
| --- | Minor | ,NsBYbx$fƈ	J>\L`Wl @D[y\dv<ۏw "| CN>Bf0b,zAR2 :DV{ue7z	vOD#纘j8yX_/Rm]l.LaʐZySque4K6ǺRPUᮆ!ú=R^s,?&>DL,g |
| --- | Minor | . NsBYbx$fƈ	J>\L`Wl @D[y\dv<ۏw "| CN>Bf0b. zAR2 :DV{ue7z	vOD#纘j8yX_/Rm]l.LaʐZySque4K6ǺRPUᮆ!ú=R^s. ?&>DL. g. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3434, 27 words, 4 clauses) |
| --- | Minor | <ep*9,(P#aD[~|@%2908P<IYe נJKRYƆĲdW+~Q,di@y=@o L=,L!@qyJvl~#F/3Ne`POdЌn/pþJL,m=2է P9r?nݾ֧} |
| --- | Minor | <ep*9. (P#aD[~|@%2908P<IYe נJKRYƆĲdW+~Q. di@y=@o L=. L!@qyJvl~#F/3Ne`POdЌn/pþJL. m=2է P9r?nݾ֧}. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3435, 70 words, 0 clauses) |
| --- | Minor | d6uNomË~ϻ_vՏ筭zؑl;sW|Mȇ>}KEWFo/k_ƕ|[g?⑏8gte}?|>_|}_;O}70g?G>/|;kv|k}>_?Ѓ Moyǻ/گ]zw^>Zx	'^p>_y%7x|-h9Ao~+^^x_Wr1yAmٟc?y߻/_/?'}E\7O}.W nye#s79zÿ_K/(ݻ?zыGؽJ ~: OYц͛?ů^o~gi>t\Fns_?{<ꟼo_}yt6`Ϟ=_WOWWn{sO4!m޼۟ӫ?z?kzu?O	6lRl'܄K2=S |
| --- | Minor | d6uNomË~ϻ_vՏ筭zؑl;sW|Mȇ>}KEWFo/k_ƕ|[g?⑏8gte}?|>_|}_;O}70g?G>/|;kv|k}>_?Ѓ Moyǻ/گ]zw^>Zx	'^p>_y%7x|-h9Ao~+^^x_Wr1yAmٟc?y߻/_/?'}E\7O}.W nye#s79zÿ_K/(ݻ?zыGؽJ ~: OYц͛?ů^o~gi>t\Fns_?{<ꟼo_}yt6`Ϟ=_WOWWn{sO4!m޼۟ӫ?z?kzu?O	6lRl'܄K2=S |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3445, 65 words, 0 clauses) |
| --- | Minor | &aLgs^&_8$9sJKIP 'g7mX7꫈ȣ\YvRW=YdE zOzϞaec/ڹsx^/oyدw<}Cu}7=)O~;_nww)?g?k[7nx6Ͼ=xɧ= }ﳞۏ%W?x\]?ſ'o_qc~Q_7xu?'յK~{/GGwO޷uuWz%g?G?wK.d[8t۫O|COgey|±|Ox>_{3]q]r	'3x귾?ayeޣs	}}^Wn\s?yuf~=;>g_l%?7^;cuWUW^~{'>;IO^ʣ; |
| --- | Minor | &aLgs^&_8$9sJKIP 'g7mX7꫈ȣ\YvRW=YdE zOzϞaec/ڹsx^/oyدw<}Cu}7=)O~;_nww)?g?k[7nx6Ͼ=xɧ= }ﳞۏ%W?x\]?ſ'o_qc~Q_7xu?'յK~{/GGwO޷uuWz%g?G?wK.d[8t۫O|COgey|±|Ox>_{3]q]r	'3x귾?ayeޣs	}}^Wn\s?yuf~=;>g_l%?7^;cuWUW^~{'>;IO^ʣ; |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3477, 78 words, 2 clauses) |
| --- | Minor | Ԟ5CMlk(װ{NjLu!w\~@D.H lF]])w<xO=)~ܡiA :D<F# x7{?ذeٴOKKR unܹO"D~>_'/~Ɠt}K^z_'5x8s b\_^}-Vbn;έٽk׮\'O,us}us  :Ǯ.UW6\sU;oFٽ}ܩqw  +6FsqH]cf*|^=s/-mشft %;D: ~Mlyi4С ;@M9`W]cǎ`M!뺕݁_3Yލ7m.--y?{~pЁV 9=}LNt=AB(he^R>YWp[ϼdC̢gTpȟˬqmX`J]j+q%, |
| --- | Minor | Ԟ5CMlk(װ{NjLu!w\~@D.H lF]])w<xO=)~ܡiA :D<F# x7{?ذeٴOKKR unܹO"D~>_'/~Ɠt}K^z_'5x8s b\_^}-Vbn;έٽk׮\'O. us}us  :Ǯ.UW6\sU;oFٽ}ܩqw  +6FsqH]cf*|^=s/-mشft %;D: ~Mlyi4С ;@M9`W]cǎ`M!뺕݁_3Yލ7m.--y?{~pЁV 9=}LNt=AB(he^R>YWp[ϼdC̢gTpȟˬqmX`J]j+q%. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3487, 65 words, 0 clauses) |
| --- | Minor | $tUUZ&-2pjfEEJY_:GC^P.A![L<![rKL`|= b?L&M/c^| };{EDKTD+W7o}[8?zы8gK6Î+l?Go~_v3Q"ιZC /-dyO=O{V6"_Gعha9Ct΁u+ Ѝ:0|{=@ mBhB{ )A~>Mgɳz w: ھfFw˗?|A['	zd} Fˣ[4dIg?s	.1Gx(/ K"T`fD|J 3Z#RKXOA5+neкEJD |
| --- | Minor | $tUUZ&-2pjfEEJY_:GC^P.A![L<![rKL`|= b?L&M/c^| };{EDKTD+W7o}[8?zы8gK6Î+l?Go~_v3Q"ιZC /-dyO=O{V6"_Gعha9Ct΁u+ Ѝ:0|{=@ mBhB{ )A~>Mgɳz w: ھfFw˗?|A['	zd} Fˣ[4dIg?s	.1Gx(/ K"T`fD|J 3Z#RKXOA5+neкEJD |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3494, 68 words, 1 clauses) |
| --- | Minor | I͖V@jkQ!@h9Zھ2(j@n.@q=/vI'~Cx޼i-ܲ(z6&={$Ye^0[J{DhGFtEW3njGY'5[TZ (~#Y" u^TVK%j6Jɬ{WbB f@Tgn,E?6!w5G> 4R/@[16&ʰ"qJf=2H&mC))RZnjf! ׯ#O#	3 lٲet>F>g) W#<R#C `y<^]׾U{u7Mw̎ݻ3۶qhA.8iC}n9k_ʋ.կ~q   s-uqFt;5w f}?Qȕ̓P!z |
| --- | Minor | I͖V@jkQ!@h9Zھ2(j@n.@q=/vI'~Cx޼i-ܲ(z6&={$Ye^0[J{DhGFtEW3njGY'5[TZ (~#Y" u^TVK%j6Jɬ{WbB f@Tgn. E?6!w5G> 4R/@[16&ʰ"qJf=2H&mC))RZnjf! ׯ#O#	3 lٲet>F>g) W#<R#C `y<^]׾U{u7Mw̎ݻ3۶qhA.8iC}n9k_ʋ.կ~q   s-uqFt;5w f}?Qȕ̓P!z. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3533, 75 words, 0 clauses) |
| --- | Minor | >lctyoOn?po|͹ħ>32jqPQ7BDqv&LAsws^*h*q=tQQwY).18a/ċh0YQ@ub4Vi|r&&wqFsa*W9ѹxCu00bp"[ѹ5MCDwy] ēO`u@4u]WUgznRU;urr׫Chȯ fN6Au>	ڗhM/BB$H\75	/#뷼 <O= cVʖ*B	; XyY"}E]s]3Q{LnOL 5DavRBNI墵Œ:F.#yb(rim]-+]׈4ItLsEhdeث|̟:lW)R5bj87vrf[X<񇗾%ZZƢ1`*t[8LrA1-"A9b(:P[ |
| --- | Minor | >lctyoOn?po|͹ħ>32jqPQ7BDqv&LAsws^*h*q=tQQwY).18a/ċh0YQ@ub4Vi|r&&wqFsa*W9ѹxCu00bp"[ѹ5MCDwy] ēO`u@4u]WUgznRU;urr׫Chȯ fN6Au>	ڗhM/BB$H\75	/#뷼 <O= cVʖ*B	; XyY"}E]s]3Q{LnOL 5DavRBNI墵Œ:F.#yb(rim]-+]׈4ItLsEhdeث|̟:lW)R5bj87vrf[X<񇗾%ZZƢ1`*t[8LrA1-"A9b(:P[ |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3680, 79 words, 3 clauses) |
| --- | Minor | `HmZ/,K &AE#lo1lo,^^Ŧ[v#0 `H|:fGY&H7V44dLov <+v]w򕫏|y+3b[@HV~d@AA@h V`	]E\־}ꇾՇom]?8qEՇֵ*4=}WD#Csq&VB#j(vΏc88bb#`KF8)k;/"IZ EґdDA LUEr~Uhzv4(,/VrT I=Ɣ8!!sC[MbݧKIjPF"䝐B:Tf3r\ /k ﷰ*h/v;1YK#s[Xc 8wVJ|DML_:э141t˂eRTWZU0:Γ*QF |
| --- | Minor | `HmZ/. K &AE#lo1lo. ^^Ŧ[v#0 `H|:fGY&H7V44dLov <+v]w򕫏|y+3b[@HV~d@AA@h V`	]E\־}ꇾՇom]?8qEՇֵ*4=}WD#Csq&VB#j(vΏc88bb#`KF8)k;/"IZ EґdDA LUEr~Uhzv4(. /VrT I=Ɣ8!!sC[MbݧKIjPF"䝐B:Tf3r\ /k ﷰ*h/v;1YK#s[Xc 8wVJ|DML_:э141t˂eRTWZU0:Γ*QF. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3722, 53 words, 6 clauses) |
| --- | Minor | @I15ȒSV*d-HP ppFj(^}W-&"4cSb3,JƟXSh)(֙B~vj3KZR/2"anyGDAyr4GLt,Ƴ%NSv(^?^zmm'JUdNxJa,Wc~)JKIh:6BЧSxBoiH'6PGX#BEPI鐒n0#)p",KSԲd/*vdaM]rp6T/r1Q	XC\}șh H8C6$HĮ(lcy!_N#Zu4P,ؘTb%[b3CxD,1 |
| --- | Minor | @I15ȒSV*d-HP ppFj(^}W-&"4cSb3. JƟXSh)(֙B~vj3KZR/2"anyGDAyr4GLt. Ƴ%NSv(^?^zmm'JUdNxJa. Wc~)JKIh:6BЧSxBoiH'6PGX#BEPI鐒n0#)p". KSԲd/*vdaM]rp6T/r1Q	XC\}șh H8C6$HĮ(lcy!_N#Zu4P. ؘTb%[b3CxD. 1. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3725, 18 words, 4 clauses) |
| --- | Minor | qbA%Jqn,	u-#\LACaʸe*Y	",,teOk,djK3:rZRJv򭂌j<Iwqo[Zp2A |
| --- | Minor | qbA%Jqn. u-#\LACaʸe*Y	". teOk. djK3:rZRJv򭂌j<Iwqo[Zp2A. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3730, 62 words, 5 clauses) |
| --- | Minor | r2?OH'֣˧fqN^䮥`R0~, iф)F]=OrͻoMvbM0,-VkMP.{Glۊ^7ȼnRbKV==7$[*pﰦCތ{߽<s{%mD><!&:%>zhECgxĻ)i!Rްב2mXYY?xX=a]Thl+Ѡvߤ~xɔa`	)4]6""5@,4!0A29|㶒K򢳐F2& uvEh\*\To*ć|eC(qT% ;E[G5@,xL#A9`瀈s)Bfwm}(^JNb˵/|\,?>X[GX:ͫE#֫ |
| --- | Minor | r2?OH'֣˧fqN^䮥`R0~. iф)F]=OrͻoMvbM0. -VkMP.{Glۊ^7ȼnRbKV==7$[*pﰦCތ{߽<s{%mD><!&:%>zhECgxĻ)i!Rްב2mXYY?xX=a]Thl+Ѡvߤ~xɔa`	)4]6""5@. 4!0A29|㶒K򢳐F2& uvEh\*\To*ć|eC(qT% ;E[G5@. xL#A9`瀈s)Bfwm}(^JNb˵/|\. ?>X[GX:ͫE#֫. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3753, 67 words, 2 clauses) |
| --- | Minor | л8x)El9PQZGϐs(TϾ*%<r!y% Lrq,W/ߣB@e(>ZmeQ'cbHeQxc~rmeȰhWrpBj `ġL_ZWę=,]܆ţtg1Os^OK*x-lҼ]8o#it<7qwa#ttp+q5m[r$1Ne4̋Fs3;4 J<Kj-c[̈u&ES0!B'B0`Tb!iD]B&1Od Tםa(Ad&&wl_&na_y		+-i*k^@R=(I"~gQI&; r>+_VsK4 bEDż |
| --- | Minor | л8x)El9PQZGϐs(TϾ*%<r!y% Lrq. W/ߣB@e(>ZmeQ'cbHeQxc~rmeȰhWrpBj `ġL_ZWę=. ]܆ţtg1Os^OK*x-lҼ]8o#it<7qwa#ttp+q5m[r$1Ne4̋Fs3;4 J<Kj-c[̈u&ES0!B'B0`Tb!iD]B&1Od Tםa(Ad&&wl_&na_y		+-i*k^@R=(I"~gQI&; r>+_VsK4 bEDż. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3777, 20 words, 4 clauses) |
| --- | Minor | ,{bȘ'NQO,:6zb(MOPɤLO%c?GB䵟I@/޵D#a◢,%_i`eA/ { rȯf+kL-kE(iAy&1KXNug,p!쩍ɍ |
| --- | Minor | . {bȘ'NQO. :6zb(MOPɤLO%c?GB䵟I@/޵D#a◢. %_i`eA/ { rȯf+kL-kE(iAy&1KXNug. p!쩍ɍ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3824, 36 words, 4 clauses) |
| --- | Minor | ׅW,?oMάHnأKU:]3҆YG1#ҚкS.̋H7!ʈdC@x%*Ԋhkj	W#k!0YXrGIdmU]dBJ2B-mԂYE儸-,#}ͶL%ka1LX;'<et)VW)&7dq3'Gu?,w)@#, q9!DXMcz_1 |
| --- | Minor | ׅW. ?oMάHnأKU:]3҆YG1#ҚкS.̋H7!ʈdC@x%*Ԋhkj	W#k!0YXrGIdmU]dBJ2B-mԂYE儸-. #}ͶL%ka1LX;'<et)VW)&7dq3'Gu?. w)@#. q9!DXMcz_1. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 3898, 64 words, 1 clauses) |
| --- | Minor | @`s5?=E\`K5~UNd=Ae3Ӛ+{z)Y#Z\X<'e/{&'1Z4Zky?mfbb2M,c c3w;XvP䥫%y]Uzv#N䒬Jnc&Y@QXi㣮QYkxZYY<=}q 06&79i?8fN6BddmkDXR2*9/v%_R a0!H23C[bp*@!VrRK %-~ȃ$wEV"iCv[FW ;)BW2\%1TPmᐈz+V@U+ГzEl|k&4ܷn}kSЄPWo@G"  !1 4 |
| --- | Minor | @`s5?=E\`K5~UNd=Ae3Ӛ+{z)Y#Z\X<'e/{&'1Z4Zky?mfbb2M. c c3w;XvP䥫%y]Uzv#N䒬Jnc&Y@QXi㣮QYkxZYY<=}q 06&79i?8fN6BddmkDXR2*9/v%_R a0!H23C[bp*@!VrRK %-~ȃ$wEV"iCv[FW ;)BW2\%1TPmᐈz+V@U+ГzEl|k&4ܷn}kSЄPWo@G"  !1 4. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4020, 68 words, 2 clauses) |
| --- | Minor | MFto'hN4Wgx'7Wo~;;|tߣO.zUWCe%BbL&Mn)1 hbB=Z)rre	sPWUQZթyRBe3(HB)3RGw3`^z%qH3gPյK_sBԸZ|D6+z)F,^1Hlp'{Jf%5Uɚ;	sn^QSNM"iѯ*-ӂ0vᕋ:t@>Nت4)e{]~N)oY2y8Ǐ\D/U/+iI5־kzcQHQ|&]	|6qUOwXъ-a35[[vU}bUlZ'gO*%26.l]4"ԣ,([0Np<B~醳c/"D=HuJFW+b. |
| --- | Minor | MFto'hN4Wgx'7Wo~;;|tߣO.zUWCe%BbL&Mn)1 hbB=Z)rre	sPWUQZթyRBe3(HB)3RGw3`^z%qH3gPյK_sBԸZ|D6+z)F. ^1Hlp'{Jf%5Uɚ;	sn^QSNM"iѯ*-ӂ0vᕋ:t@>Nت4)e{]~N)oY2y8Ǐ\D/U/+iI5־kzcQHQ|&]	|6qUOwXъ-a35[[vU}bUlZ'gO*%26.l]4"ԣ. ([0Np<B~醳c/"D=HuJFW+b.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4106, 99 words, 4 clauses) |
| --- | Minor | e28WvkgP{,tR1' )(\@ArBifV2-eYEE^W,oQMp+6*ف5xWEbZ5\Hp]Mƈe2s:gytLQ^Vm0p0bk6VGGlLɾA;>ah"`jݐ> qʄ|D *e)|pp{_4֝}Yï83&&<"	`z%qW[%A(W. .LPXꈷ_H|>zjM'>G}4uĩ矅v6o(/:@}_KuPg@̈PJIyN@R('AJ  $Y蠐6j5y1}O<.iѩ,{:EPY?j:Rb{ΙF_ߪe/vBR䃣Aj@3Dn[sbp5⑱7)C!Aŋ+`~dɡPr(?mR[6۳,z6@-IE/ǺRB43xME#i"+@(nhcNY{+''K懟?omX[e5{0AD;1\IG#bɍ=5ߡ*RJ*ZJ +L۸oA-j|| @W6ݰ##ei5Sea0@j=MR{!!dߐq |
| --- | Minor | e28WvkgP{. tR1' )(\@ArBifV2-eYEE^W. oQMp+6*ف5xWEbZ5\Hp]Mƈe2s:gytLQ^Vm0p0bk6VGGlLɾA;>ah"`jݐ> qʄ|D *e)|pp{_4֝}Yï83&&<"	`z%qW[%A(W. .LPXꈷ_H|>zjM'>G}4uĩ矅v6o(/:@}_KuPg@̈PJIyN@R('AJ  $Y蠐6j5y1}O<.iѩ. {:EPY?j:Rb{ΙF_ߪe/vBR䃣Aj@3Dn[sbp5⑱7)C!Aŋ+`~dɡPr(?mR[6۳. z6@-IE/ǺRB43xME#i"+@(nhcNY{+''K懟?omX[e5{0AD;1\IG#bɍ=5ߡ*RJ*ZJ +L۸oA-j|| @W6ݰ##ei5Sea0@j=MR{!!dߐq. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4139, 31 words, 4 clauses) |
| --- | Minor | /8]p݇6?#勨,	m. =J$"a>0 @!2a8BB SOZ}-O=g>3u=kz勗Q)h, Q_VW%rsAgDWL,ɩ `( 6:_̸L`1flBb#Ŭ/ 1L,>]JzN 	.P` ǕņYx |
| --- | Minor | /8]p݇6?#勨. m. =J$"a>0 @!2a8BB SOZ}-O=g>3u=kz勗Q)h.  Q_VW%rsAgDWL. ɩ `( 6:_̸L`1flBb#Ŭ/ 1L. >]JzN 	.P` ǕņYx. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4323, 26 words, 4 clauses) |
| --- | Minor | 0H#2%,0-{\-C-ـX\,a^by"dI|Q"Ҁ@V(/O pM	609"}bk#W.D#[FYAh|ݖ'=2F~uXW9&"a,.,j) `VslP^|G޳ |
| --- | Minor | 0H#2%. 0-{\-C-ـX\. a^by"dI|Q"Ҁ@V(/O pM	609"}bk#W.D#[FYAh|ݖ'=2F~uXW9&"a. .. j) `VslP^|G޳. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4329, 86 words, 3 clauses) |
| --- | Minor | EkDD0KtkrnF>h"WQTx4(944ٶ C?EBGVJX @Iq;9GzR	Ibh 'pʤv	vů׬}e]4{YQQPdd'r4Ká@ 4@II(Zf]̓{	r򲽦wLGMPR]߸=6q>녗ZGϙ(ɍV_ȳOzs9\yÜ8Ǟxu揻&vz^?pG?*;YGc?檥\BHk篿vtv BRRܱ/a+/NZhک˶=k}߰FFʕs3=Z7ݖG;tpd9W9>_<o>?jcgE*V?'HG+,p&yR ,pҤ,#=n[6/8wuD̄' w]x(ar"2~pfh&'UJVMe.t;KO(R!lw< |
| --- | Minor | EkDD0KtkrnF>h"WQTx4(944ٶ C?EBGVJX @Iq;9GzR	Ibh 'pʤv	vů׬}e]4{YQQPdd'r4Ká@ 4@II(Zf]̓{	r򲽦wLGMPR]߸=6q>녗ZGϙ(ɍV_ȳOzs9\yÜ8Ǟxu揻&vz^?pG?*;YGc?檥\BHk篿vtv BRRܱ/a+/NZhک˶=k}߰FFʕs3=Z7ݖG;tpd9W9>_<o>?jcgE*V?'HG+. p&yR . pҤ. #=n[6/8wuD̄' w]x(ar"2~pfh&'UJVMe.t;KO(R!lw<. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4496, 71 words, 2 clauses) |
| --- | Minor | v^%)p}Iwwn|W5fe';.m]nv,c>^'cM%  EϔÿY޺ݛmA]}u-^P;Z(Pd3*0P\?56gw/k,XTBuqG}Efx{+V+YjIn3D:O9S@RJєe~;KYCL0ލtU}I`	{*׭^?"@cocO9zpa7B?_k<ø柳_'u\y~XƎRIeoTEEKJ BJ aQ2)Dג @" "U[H_~ GZ-C	'u{v3rԲ/ɻ{:kذ}G=]yq/:S>t>$E :;_~-ʁ\:ef̟]0 5<R<|cϮz?ui:i_YJBƖÿ_ ?ாI |
| --- | Minor | v^%)p}Iwwn|W5fe';.m]nv. c>^'cM%  EϔÿY޺ݛmA]}u-^P;Z(Pd3*0P\?56gw/k. XTBuqG}Efx{+V+YjIn3D:O9S@RJєe~;KYCL0ލtU}I`	{*׭^?"@cocO9zpa7B?_k<ø柳_'u\y~XƎRIeoTEEKJ BJ aQ2)Dג @" "U[H_~ GZ-C	'u{v3rԲ/ɻ{:kذ}G=]yq/:S>t>$E :;_~-ʁ\:ef̟]0 5<R<|cϮz?ui:i_YJBƖÿ_ ?ாI. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4533, 21 words, 4 clauses) |
| --- | Minor | k' A	 VKs<,}YpkLBԋ"9ϰ,CmRkZRY%P"K$H@JR(,JmۈH bVRjV B,㓁HBbwoQx7߼;yʼϚs'>  |
| --- | Minor | k' A	 VKs<. }YpkLBԋ"9ϰ. CmRkZRY%P"K$H@JR(. JmۈH bVRjV B. 㓁HBbwoQx7߼;yʼϚs'> . |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4534, 80 words, 2 clauses) |
| --- | Minor | 53F_9"Ƹ&ß@U+I.<#E	dsƵVx9D9YVJxy<E8{-6]tCpg_rRPo#EPۖV:Y^j5l9 )5kP7;ip2Z{,6ƃN0m;{o~ws=e@RVccbՆb_ۙIXL+O+>x́GV0DU|@dL=]v1⅗yyϼ/w3]0cg'=>ﵿ?p:}-F>} vw6>yޱwp&r!U?42v)Rμky'vBY[|9,͛(]w<k]{/u9'׾V	WVoٵq>DDJ=<q;{=WJOAD*3r>td1۠(M%q	 'c9ntgE2-(	53۩=BS% r*ApA]::ưExHX7vY`U9Jy |
| --- | Minor | 53F_9"Ƹ&ß@U+I.<#E	dsƵVx9D9YVJxy<E8{-6]tCpg_rRPo#EPۖV:Y^j5l9 )5kP7;ip2Z{. 6ƃN0m;{o~ws=e@RVccbՆb_ۙIXL+O+>x́GV0DU|@dL=]v1⅗yyϼ/w3]0cg'=>ﵿ?p:}-F>} vw6>yޱwp&r!U?42v)Rμky'vBY[|9. ͛(]w<k]{/u9'׾V	WVoٵq>DDJ=<q;{=WJOAD*3r>td1۠(M%q	 'c9ntgE2-(	53۩=BS% r*ApA]::ưExHX7vY`U9Jy. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4614, 70 words, 1 clauses) |
| --- | Minor | G~^{ƷnY\4k{.%12O#s rF7k&){ExKE1>tZ<ʌ=2Jמ*AEĚ=o7YpU:pRd=6Y`@x̘kfN`rRoևxٞ M6hSŮF9ȴ[+,*F Hw=Knɋ$4+߬Rʢ-[EAX\ `YVe"g_qC8zR SMۻz56Ϙ4@@<m@)kΜ6<iw|h/Y͟<ul>p~ RP^ :yrl;m(JdY	LRoR]w@8QJ 3n	;p%ǜ%D&E)v	"sLI"]w5Vd#gOcnxa{E= ΣWf 0Cd}l(#&*jrb{y6"揬lM |
| --- | Minor | G~^{ƷnY\4k{.%12O#s rF7k&){ExKE1>tZ<ʌ=2Jמ*AEĚ=o7YpU:pRd=6Y`@x̘kfN`rRoևxٞ M6hSŮF9ȴ[+. *F Hw=Knɋ$4+߬Rʢ-[EAX\ `YVe"g_qC8zR SMۻz56Ϙ4@@<m@)kΜ6<iw|h/Y͟<ul>p~ RP^ :yrl;m(JdY	LRoR]w@8QJ 3n	;p%ǜ%D&E)v	"sLI"]w5Vd#gOcnxa{E= ΣWf 0Cd}l(#&*jrb{y6"揬lM. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4659, 30 words, 4 clauses) |
| --- | Minor | Ug  DD9gGgݭOȲ,A,W^yY .mJJDl6hh( R-p,[lD4o<DD	 /Ց;|jA	+ n^d"Oʃ< ^}Kn7JڲH	BH) ,oZSd1Vr(i~D'SmU1KcTQVʲvUZ5;>)\Dm zGKz |
| --- | Minor | Ug  DD9gGgݭOȲ. A. W^yY .mJJDl6hh( R-p. [lD4o<DD	 /Ց;|jA	+ n^d"Oʃ< ^}Kn7JڲH	BH) . oZSd1Vr(i~D'SmU1KcTQVʲvUZ5;>)\Dm zGKz. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4673, 36 words, 4 clauses) |
| --- | Minor | Tva,kh	Cp?ȊLɪ:mE0'9^SniAQ[rbu?"9+AkΟ;*;]fs;/AFWIX P%ΘȐ6l'8q+̤bI<t!aՊk,KsY/I eQD/D@,,Z D.s<+Z."Õ)m2l  V[ |
| --- | Minor | Tva. kh	Cp?ȊLɪ:mE0'9^SniAQ[rbu?"9+AkΟ;*;]fs;/AFWIX P%ΘȐ6l'8q+̤bI<t!aՊk. KsY/I eQD/D@. . Z D.s<+Z."Õ)m2l  V[. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4677, 145 words, 11 clauses) |
| --- | Minor | zN+fF3{{zϘ11(KDT_r"׭]KOr!,K6?uWGe{ϟ||ôiu?QVP`I6l\ǡ)HmEjOx۶qIsf zzz<M7?Xx%Km~=|oq+~t}׬}w_eR"*Y	qE谜p4 dY繾K[Hߥb1 4:)(Ĥ%FhC=÷HD	ԕA^Z֘_Tf*hUA)B髿ղ:,c̙	W``]~t#I9kT@&v:,V~w۲E1*R1< РPQgzUɘy)E׾ 2{tF@pRllFpFKKC"9ȱ$0GOCtb&yA\t_fR8/͔#_ͯ>ˮ}b]M[P'ue"(-PJP R+y	lu\s<q!?]B%K|Q18-K	M\lw5262c8`oIZmivp6,XbI?7"uatB(	Cg&*@8H)POXYze뉭<xDp^Ӂ#96tͧ#N@w2#nSB5%]4#dE:`*qhVL;Nv_@|[,!"6r{+ġiK@u2n7@RN҄IQ@b,s3'3&&E)O,GFE!Ȓ  P@@j\,7Uu,% N!)[EY& f(l`ocSIV ,3]KHY&,iƔ9Ӈe9:6!@A/j!<abvB |
| --- | Minor | zN+fF3{{zϘ11(KDT_r"׭]KOr!. K6?uWGe{ϟ||ôiu?QVP`I6l\ǡ)HmEjOx۶qIsf zzz<M7?Xx%Km~=|oq+~t}׬}w_eR"*Y	qE谜p4 dY繾K[Hߥb1 4:)(Ĥ%FhC=÷HD	ԕA^Z֘_Tf*hUA)B髿ղ:. c̙	W``]~t#I9kT@&v:. V~w۲E1*R1< РPQgzUɘy)E׾ 2{tF@pRllFpFKKC"9ȱ$0GOCtb&yA\t_fR8/͔#_ͯ>ˮ}b]M[P'ue"(-PJP R+y	lu\s<q!?]B%K|Q18-K	M\lw5262c8`oIZmivp6. XbI?7"uatB(	Cg&*@8H)POXYze뉭<xDp^Ӂ#96tͧ#N@w2#nSB5%]4#dE:`*qhVL;Nv_@|[. !"6r{+ġiK@u2n7@RN҄IQ@b. s3'3&&E)O. GFE!Ȓ  P@@j\. 7Uu. % N!)[EY& f(l`ocSIV . 3]KHY&. iƔ9Ӈe9:6!@A/j!<abvB. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4731, 76 words, 1 clauses) |
| --- | Minor | ϛ2%"DVC8l=Uk繚慵 f^ 2ژoͺ;<)Y|lQ{c?]	Q̆BV<W]#p(D&ԮHEvr*@0*V!rWdrUAۆ=9sԝ/1⁚d9V@_rVI"dvߘ*g~r.AXkh7+&-~Ӗ,0Vq`6uY!+-Ξ'}ʶY3dEJ}a͎Y"⛿4/fوLZW髓H@g *( -#o0nj$D*e	y@k|=9s|j҃ʉ֭lbC&XfHG^U41VdAZl>8SD[CP*@IΔfv!CgҹD˼7+7+GҶYmuL@;oFZĝ@pGz*oXiDg_DPhS |
| --- | Minor | ϛ2%"DVC8l=Uk繚慵 f^ 2ژoͺ;<)Y|lQ{c?]	Q̆BV<W]#p(D&ԮHEvr*@0*V!rWdrUAۆ=9sԝ/1⁚d9V@_rVI"dvߘ*g~r.AXkh7+&-~Ӗ. 0Vq`6uY!+-Ξ'}ʶY3dEJ}a͎Y"⛿4/fوLZW髓H@g *( -#o0nj$D*e	y@k|=9s|j҃ʉ֭lbC&XfHG^U41VdAZl>8SD[CP*@IΔfv!CgҹD˼7+7+GҶYmuL@;oFZĝ@pGz*oXiDg_DPhS. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4781, 87 words, 2 clauses) |
| --- | Minor | A{xى+ k5b/q`:n IPhe(2aBc+z@v K_ǏOL4򜈘K0aV={ׇM{kHtJ.;uQggƠ^ē ܖIϺL*dQѝ3{Mmݿry-u&y\x*;xxOvm  x)|&to;w<{'qg|xCDʳ,<@h{UUMhX%]f+LZ`wU5>	"d@JsضݯrA}k֭~gI3#@%aNPUOܦy(p:~}Bgf ")]@oo><near0e_|^Ln1!2ӣ^(PG%]cu<&~4f{cXT)Z&IJIpbpO+j|PΡ@M/Y; H͋س#t֜@'Z,tm۔AUI`VŽR{G= wZ0t4?U cբ9*mw6kB&9T Bg[Rƶ8"s? |
| --- | Minor | A{xى+ k5b/q`:n IPhe(2aBc+z@v K_ǏOL4򜈘K0aV={ׇM{kHtJ.;uQggƠ^ē ܖIϺL*dQѝ3{Mmݿry-u&y\x*;xxOvm  x)|&to;w<{'qg|xCDʳ. <@h{UUMhX%]f+LZ`wU5>	"d@JsضݯrA}k֭~gI3#@%aNPUOܦy(p:~}Bgf ")]@oo><near0e_|^Ln1!2ӣ^(PG%]cu<&~4f{cXT)Z&IJIpbpO+j|PΡ@M/Y; H͋س#t֜@'Z. tm۔AUI`VŽR{G= wZ0t4?U cբ9*mw6kB&9T Bg[Rƶ8"s?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4799, 64 words, 1 clauses) |
| --- | Minor | 1̃4"@?4TMb'^kdaONUd-uڲL͋$2-r|߯SYbވ`ԝKRnA4=IOԇN?þ#k?|Ow{y>6>֦RQ}6!")&󝐱XaXSs.'\}(j>f긚/Vf죔CTzg=s;oG淿sƗl|vӦEuIJ{=Օ&!6MDvJ  R88i^η[ް̃K^vys(PQ:X nouFX`@\тS+>Rݧ5K!8IBPl,!D֯O?9ABE\t.(IM:RCQQ4HwnU2u:4 bQMQ*43=f |
| --- | Minor | 1̃4"@?4TMb'^kdaONUd-uڲL͋$2-r|߯SYbވ`ԝKRnA4=IOԇN?þ#k?|Ow{y>6>֦RQ}6!")&󝐱XaXSs.'\}(j>f긚/Vf죔CTzg=s;oG淿sƗl|vӦEuIJ{=Օ&!6MDvJ  R88i^η[ް̃K^vys(PQ:X nouFX`@\тS+>Rݧ5K!8IBPl. !D֯O?9ABE\t.(IM:RCQQ4HwnU2u:4 bQMQ*43=f. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4870, 86 words, 1 clauses) |
| --- | Minor | H)nvwy]w%̲LUeY4ޚ8`>r)Y'L"	]',*۳u(?3!zf/5N	~jS3ؘ*i*"W)ǫ&]u1 !aB`v&%wHy0qbZ#!P)\  Tr&1^4EPE(N?Sw<[!z]y慧)URx@_]+ˢfYe?f#$) RGfֻ>jaIx\#%fN>.REs2SvSf뎚wlPULLvu>~vs/w%!D eB{TE6k[~x;7|~:"ǧM~G駝"۶f{zJ"VJPсs5܇Fƹ d\D;;fkHl( IJ	[!P;\~F)3ş<ɇ  ʉ	0j }- `a?TЁy )E_Pk|+/}(Dvaa]Fv&·U0Ld9^Ӝ[m |
| --- | Minor | H)nvwy]w%̲LUeY4ޚ8`>r)Y'L"	]'. *۳u(?3!zf/5N	~jS3ؘ*i*"W)ǫ&]u1 !aB`v&%wHy0qbZ#!P)\  Tr&1^4EPE(N?Sw<[!z]y慧)URx@_]+ˢfYe?f#$) RGfֻ>jaIx\#%fN>.REs2SvSf뎚wlPULLvu>~vs/w%!D eB{TE6k[~x;7|~:"ǧM~G駝"۶f{zJ"VJPсs5܇Fƹ d\D;;fkHl( IJ	[!P;\~F)3ş<ɇ  ʉ	0j }- `a?TЁy )E_Pk|+/}(Dvaa]Fv&·U0Ld9^Ӝ[m. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4887, 65 words, 2 clauses) |
| --- | Minor | ;o[7V}+~`SUC+{{ 2l4(Id`/&6Gu??y.geQGF Ft| (4.ZDrK+W!U&p,<ʑ: );}[/ʫ.?k}v>sznSY$'j̜*CEq(@ >7<3QZ"C}``L6+Mx0F1(-6Ĭ՚Z{xF_W) OȄISfAfVI0X\KO!K(ּmnMБ BaZz?PzA["i  g+"Tg2a΀JOqMץb&z]"A#38fORNbیId-am*s2Lepj^,CyűR]D CY |
| --- | Minor | ;o[7V}+~`SUC+{{ 2l4(Id`/&6Gu??y.geQGF Ft| (4.ZDrK+W!U&p. <ʑ: );}[/ʫ.?k}v>sznSY$'j̜*CEq(@ >7<3QZ"C}``L6+Mx0F1(-6Ĭ՚Z{xF_W) OȄISfAfVI0X\KO!K(ּmnMБ BaZz?PzA["i  g+"Tg2a΀JOqMץb&z]"A#38fORNbیId-am*s2Lepj^. CyűR]D CY. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4892, 75 words, 3 clauses) |
| --- | Minor | ({Xa#(9'vy\Z4`K4: H"a.,#eY:oG^|'^xQsEEٖcФ`߽շ޲q<~1/jeW@qYCܹ'!XЉ:YL:-LQU#|a=V-]@P{X,V1|Qrn'}zI'C{ldYB(-쳾xҪV2kl;}7  )AHa p;&~xqA6ޱVX\PS*X j< 3ۥčeVW-mT=2Β{%!ib0:!|RaL/l\F2w|"ɥT4y%2xQ0yr-ѳ62" DX́G:3vCAqZC\o,DAIeSƖ쩢6 P|ls&芸D@iHɒM |
| --- | Minor | ({Xa#(9'vy\Z4`K4: H"a.. #eY:oG^|'^xQsEEٖcФ`߽շ޲q<~1/jeW@qYCܹ'!XЉ:YL:-LQU#|a=V-]@P{X. V1|Qrn'}zI'C{ldYB(-쳾xҪV2kl;}7  )AHa p;&~xqA6ޱVX\PS*X j< 3ۥčeVW-mT=2Β{%!ib0:!|RaL/l\F2w|"ɥT4y%2xQ0yr-ѳ62" DX́G:3vCAqZC\o. DAIeSƖ쩢6 P|ls&芸D@iHɒM. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4908, 90 words, 1 clauses) |
| --- | Minor | )E&:Ņ!R 7aeKE5;օȫS[	묐Ӊ{@͖dRK<">:-_}rKQU$ ksKj؇\{WZg+|.y pLkHNz;Lmׄ[K	iHv=U~0L)NcSN<?nmMLv>D (@Dv;5/՝W/""UE|pjnXU@ԠYij	30ZX3Y.5nYUkCVnݥzYy^.l))q{M&B,Λ˵	?bH#	1ߑ0UdZ"@IԋE:EKk/)eCd3GܲEL]J3֘jbb|ePOI˽}U:Ī}" R>h  d۫hmG@EN2څX(ϳ=OhvYKn&v[țMh%I9wkOIk|7 "u5)8d5Vq9= |
| --- | Minor | )E&:Ņ!R 7aeKE5;օȫS[	묐Ӊ{@͖dRK<">:-_}rKQU$ ksKj؇\{WZg+|.y pLkHNz;Lmׄ[K	iHv=U~0L)NcSN<?nmMLv>D (@Dv;5/՝W/""UE|pjnXU@ԠYij	30ZX3Y.5nYUkCVnݥzYy^.l))q{M&B. Λ˵	?bH#	1ߑ0UdZ"@IԋE:EKk/)eCd3GܲEL]J3֘jbb|ePOI˽}U:Ī}" R>h  d۫hmG@EN2څX(ϳ=OhvYKn&v[țMh%I9wkOIk|7 "u5)8d5Vq9=. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4920, 66 words, 1 clauses) |
| --- | Minor | B])5ҟsxK>df/jς"3Bh#lnl<{ΙNgOVu 恽cՙe>r=JK MgZ6	ӐtJR9bra_u yMPϚ3"4<Tmʾf}$m||!^21'|?J5WohIh-mBD =yc{Cu0lẄ́= E)鋚-F-Ԅ؜|FYũ`HIq3˦xg);ڪɖDI΍u81N#	3go~]p۳鑭cn3.ڃXHJ<R]}7]	'=0ۣN :/i|mOͩ#XJ7v|X9ߎvRb؟LH pI9L{:+xƥkݞّîP8&R/Y:GusI1j4f=&3(, |
| --- | Minor | B])5ҟsxK>df/jς"3Bh#lnl<{ΙNgOVu 恽cՙe>r=JK MgZ6	ӐtJR9bra_u yMPϚ3"4<Tmʾf}$m||!^21'|?J5WohIh-mBD =yc{Cu0lẄ́= E)鋚-F-Ԅ؜|FYũ`HIq3˦xg);ڪɖDI΍u81N#	3go~]p۳鑭cn3.ڃXHJ<R]}7]	'=0ۣN :/i|mOͩ#XJ7v|X9ߎvRb؟LH pI9L{:+xƥkݞّîP8&R/Y:GusI1j4f=&3(, |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4922, 65 words, 0 clauses) |
| --- | Minor | =nm$vLu2FI8Wf1S4V~;Η(L&bj[KK-q.B}y6Ve0{[ozkȝr>'<sO;8y}NӻVM6۸֍wkn8Oُ]W=Ϲg?9O~d{:y ֓*Aaq;ҸqG!_{c pԶe.x8ISgix\͹TTVOvqvv[*MVCx[Z-͋}KkgF5VY1|LqSIw	#g*!>1%`uB[Bpj~䩛UCWdWPm1 ~Z.*ab̚׿sy4mc:߆YQnzh`>^PVV=Ƙ|X |
| --- | Minor | =nm$vLu2FI8Wf1S4V~;Η(L&bj[KK-q.B}y6Ve0{[ozkȝr>'<sO;8y}NӻVM6۸֍wkn8Oُ]W=Ϲg?9O~d{:y ֓*Aaq;ҸqG!_{c pԶe.x8ISgix\͹TTVOvqvv[*MVCx[Z-͋}KkgF5VY1|LqSIw	#g*!>1%`uB[Bpj~䩛UCWdWPm1 ~Z.*ab̚׿sy4mc:߆YQnzh`>^PVV=Ƙ|X |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4926, 63 words, 1 clauses) |
| --- | Minor | .=[k_w%yw>bmTYPL-0[ogy90^;JzF,eUK'f͸be\*i*9͓{c2WYV+Vj7rW%))vGB}T&5M)t5&	&Vg["%[lPTjK G/	9Si$a]|{jQn'%&<&0]GXdo溏r*mk%6uChUɈ	g4n;	E2Uw·W\o{U| D{|uޥ'=;BaxkkYB9ZYr 3ky__+7|[o}(G7's.mLKm-A.3)ESI1UN[u?MXrCW~/; \!B[CWHT |
| --- | Minor | .=[k_w%yw>bmTYPL-0[ogy90^;JzF. eUK'f͸be\*i*9͓{c2WYV+Vj7rW%))vGB}T&5M)t5&	&Vg["%[lPTjK G/	9Si$a]|{jQn'%&<&0]GXdo溏r*mk%6uChUɈ	g4n;	E2Uw·W\o{U| D{|uޥ'=;BaxkkYB9ZYr 3ky__+7|[o}(G7's.mLKm-A.3)ESI1UN[u?MXrCW~/; \!B[CWHT. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 4990, 68 words, 2 clauses) |
| --- | Minor | eѐYIrh~3)ҫ9{.OJ﷎MN<˞{>}߽Ǯ?ieϷAan|2^#l~G9s*pL=J q	u|'5@Ԁ_EViQw~h05&s|^Rf \`cs1N/edOdƖp}S	k\^1"lWNq?ٛSvxfn68ReɥMZRTǸ3]tgp}cuݐ=͜g	T;H@L5+F,(96e,a{DLZ!70<00s"pDڀp1&;qN 3=	X {mwHR#̢rDP͗_=]FEWsZ4C[LV0f 1Vwڸ\v @nqP |
| --- | Minor | eѐYIrh~3)ҫ9{.OJ﷎MN<˞{>}߽Ǯ?ieϷAan|2^#l~G9s*pL=J q	u|'5@Ԁ_EViQw~h05&s|^Rf \`cs1N/edOdƖp}S	k\^1"lWNq?ٛSvxfn68ReɥMZRTǸ3]tgp}cuݐ=͜g	T;H@L5+F. (96e. a{DLZ!70<00s"pDڀp1&;qN 3=	X {mwHR#̢rDP͗_=]FEWsZ4C[LV0f 1Vwڸ\v @nqP-. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5024, 89 words, 3 clauses) |
| --- | Minor | w-j^NrLy0|6{f;ǮQH*1 '}M@O"Qyf`gID/R抡w>dw#ݞl]>Cl DIjď0^'u{Sm( -oGۑ&-P/ f=0.Ld5r6|&Ͳ'2_d.cRJBqa.k q_m9-oE86*:FlU&u;%ݔnnڬDg	rGվ}w_/l{6CO \Ku!4jhS8XsOĎ 93W??>傍Yyl݌)mrFn69!|fZq^g0R.#k%Yvzß:v6)),?)a'0"jú*䙻ee\jiҀ@42*9Y]KH=Ym7-ML4HK0E;6x.19t6&li@i@y}%,v]),R=j_:H]ڋjVJQ@*u]e<.[/X({c}0)lvh+d+8 |
| --- | Minor | w-j^NrLy0|6{f;ǮQH*1 '}M@O"Qyf`gID/R抡w>dw#ݞl]>Cl DIjď0^'u{Sm( -oGۑ&-P/ f=0.Ld5r6|&Ͳ'2_d.cRJBqa.k q_m9-oE86*:FlU&u;%ݔnnڬDg	rGվ}w_/l{6CO \Ku!4jhS8XsOĎ 93W??>傍Yyl݌)mrFn69!|fZq^g0R.#k%Yvzß:v6)). ?)a'0"jú*䙻ee\jiҀ@42*9Y]KH=Ym7-ML4HK0E;6x.19t6&li@i@y}%. v]). R=j_:H]ڋjVJQ@*u]e<.[/X({c}0)lvh+d+8. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5056, 73 words, 2 clauses) |
| --- | Minor | ɣlvBxqV^Ȭ͞@O늚.۱Kc5mHXj#֪HkX_[>kRak{|OYK@ȆҲ/SڰLj#|dY^4Q=i~s6`s-a3@:r&Ω%˟q/1W$uISPb#_N-Ή_dgGEbK#qO?^`mt&N[ÜpY}6Jjt9'rF)ɜ2E=D"_j1"P3I[D/*ۣJL6J[ICtCoѳEp:-v	v*sBRJ,.;YmdT\=4Q^w5FB.~<P3u30ȸTլ+!O}1,A&ǌenHzƭCGڰ9<Mlpֆi3R	3v]1ˍ1a |
| --- | Minor | ɣlvBxqV^Ȭ͞@O늚.۱Kc5mHXj#֪HkX_[>kRak{|OYK@ȆҲ/SڰLj#|dY^4Q=i~s6`s-a3@:r&Ω%˟q/1W$uISPb#_N-Ή_dgGEbK#qO?^`mt&N[ÜpY}6Jjt9'rF)ɜ2E=D"_j1"P3I[D/*ۣJL6J[ICtCoѳEp:-v	v*sBRJ. .;YmdT\=4Q^w5FB.~<P3u30ȸTլ+!O}1. A&ǌenHzƭCGڰ9<Mlpֆi3R	3v]1ˍ1a. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5094, 71 words, 2 clauses) |
| --- | Minor | T6I6dB`V2rRvvLG#Ui:/A>xfQan胬3t7L%Jj,3k:d+;"& ZUVMԺ]fg"Pݜ7R4z^M<ċͩnf^̍P"8GX=>\2_ILg"L丱쇪>՗Ǥ&o_266~L=iJ{uq9vYUO1fOCe[D&>Dv!}DfA%؞.1I}W^79G=4OP"U,*~T ~:ݎA]RɌ΁8LO<n_%_{A98Ƥ3|skvVHG: dXّF+gq'$:p3V*i.|H*s!Hw-PU	tJ2f |
| --- | Minor | T6I6dB`V2rRvvLG#Ui:/A>xfQan胬3t7L%Jj. 3k:d+;"& ZUVMԺ]fg"Pݜ7R4z^M<ċͩnf^̍P"8GX=>\2_ILg"L丱쇪>՗Ǥ&o_266~L=iJ{uq9vYUO1fOCe[D&>Dv!}DfA%؞.1I}W^79G=4OP"U. *~T ~:ݎA]RɌ΁8LO<n_%_{A98Ƥ3|skvVHG: dXّF+gq'$:p3V*i.|H*s!Hw-PU	tJ2f. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5111, 67 words, 3 clauses) |
| --- | Minor | 䚕~/}/O߿ۏ{cgmOЬ5hpHd K5BI{MFC5rCla%_xu4s'(-t9 X30<>2x|\47EЭ`.071JPNhք@ĕ&{aP%}vJ`E FbKál7JIbK1P[~I/KH*0LXYP#sD.17}1L.i&3''oVeMꩤ#A,p@:Z>NQ';V0"8j8DZB%/,eg7ߍP+DkUj<6jFR$&ʍZd,I)O*UPq@r^n{xBʟr6;͹^tC;\o;hp 8b |
| --- | Minor | 䚕~/}/O߿ۏ{cgmOЬ5hpHd K5BI{MFC5rCla%_xu4s'(-t9 X30<>2x|\47EЭ`.071JPNhք@ĕ&{aP%}vJ`E FbKál7JIbK1P[~I/KH*0LXYP#sD.17}1L.i&3''oVeMꩤ#A. p@:Z>NQ';V0"8j8DZB%/. eg7ߍP+DkUj<6jFR$&ʍZd. I)O*UPq@r^n{xBʟr6;͹^tC;\o;hp 8b. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5126, 30 words, 4 clauses) |
| --- | Minor | u&Zᖃ6.]էa|vT*tP?_mkKf!;Quwҝq^8G6ԂdXzY=Ұ,	.h/;{)0:2Tu|k#,,Ph<g,k+Z~[<r's ;|[UڞN ҡ7ZOjA-ӫZȹ{ |
| --- | Minor | u&Zᖃ6.]էa|vT*tP?_mkKf!;Quwҝq^8G6ԂdXzY=Ұ. .h/;{)0:2Tu|k#. Ph<g. k+Z~[<r's ;|[UڞN ҡ7ZOjA-ӫZȹ{. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5159, 101 words, 6 clauses) |
| --- | Minor | ]EBd뢻LQmfllاèlwYWY]d2,^bmE2U4T*[!N*Yj&F~UD7`&x(PsNEUk{y8hk0I{R縐t6S)[Bzm}^],)k̐PRsG*!6`FZ^lKoaZ̥>YJ*rQTDSa-=_bPfOrgŦPqm5"gۙ28T\k7-Y㩐TwA"2yHz,t9/5Ƨ/9c3[ڦ:m(e.gա\P5]ڼ#J13E)|< EU#V0(WdVjLoY 0dJO2a:eЍvKW_я~Q}8Gl!<a?,Va@LJb6 iP2s8%o-ɣȩl#|9^C>eLxxy{f?xNE?V.cc40ץW(aEGXG~,NwbLTH'69p}=a>a6~s4Y\{ahn׮] >؏>/~_g?q+;iϮ.l0xf,V>eNOƪBE\;I9v  0^Gy\@J0`յ}᰽5ڞG<0o9 |
| --- | Minor | ]EBd뢻LQmfllاèlwYWY]d2. ^bmE2U4T*[!N*Yj&F~UD7`&x(PsNEUk{y8hk0I{R縐t6S)[Bzm}^]. )k̐PRsG*!6`FZ^lKoaZ̥>YJ*rQTDSa-=_bPfOrgŦPqm5"gۙ28T\k7-Y㩐TwA"2yHz. t9/5Ƨ/9c3[ڦ:m(e.gա\P5]ڼ#J13E)|< EU#V0(WdVjLoY 0dJO2a:eЍvKW_я~Q}8Gl!<a?. Va@LJb6 iP2s8%o-ɣȩl#|9^C>eLxxy{f?xNE?V.cc40ץW(aEGXG~. NwbLTH'69p}=a>a6~s4Y\{ahn׮] >؏>/~_g?q+;iϮ.l0xf. V>eNOƪBE\;I9v  0^Gy\@J0`յ}᰽5ڞG<0o9. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5165, 84 words, 1 clauses) |
| --- | Minor | *"Tjn8}h*HEfOvrDui|R_0;G	`"hfKW\tBګݓX+/Cvk؇8>3<t6qB]e*VF[*af:Vm*BTSxUmWͬ}21^u6n5T.eWF`1 |_߿sk5g~`Pܭ{?cPχ 'e'lwxI|:}+~GOtF 9Z[]ha]$W}3)nGED[w,xϞJlg{7|6O;uDr^xóݻWF@f;B<st0`=NePNipPc܆5ǋu6_*'6Qz ) G"ZLN ?pI}IoFO3P[arsЀ\ش@`Ea3TN&Sd2?jC {"d	8i`80#y\߯oYgǎ+97pӗoc;s'Up^FK4;p/~K[SOn>ñ϶;{~~?ݻ'oy[g{Cם{޹[ ױ'c|1n |
| --- | Minor | *"Tjn8}h*HEfOvrDui|R_0;G	`"hfKW\tBګݓX+/Cvk؇8>3<t6qB]e*VF[*af:Vm*BTSxUmWͬ}21^u6n5T.eWF`1 |_߿sk5g~`Pܭ{?cPχ 'e'lwxI|:}+~GOtF 9Z[]ha]$W}3)nGED[w. xϞJlg{7|6O;uDr^xóݻWF@f;B<st0`=NePNipPc܆5ǋu6_*'6Qz ) G"ZLN ?pI}IoFO3P[arsЀ\ش@`Ea3TN&Sd2?jC {"d	8i`80#y\߯oYgǎ+97pӗoc;s'Up^FK4;p/~K[SOn>ñ϶;{~~?ݻ'oy[g{Cם{޹[ ױ'c|1n. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5215, 61 words, 1 clauses) |
| --- | Minor | Jv 贘jmq~,\g\لm9ϳf|Fݾ/s&4{ˠef7MO˿|#x67GI.cf#	N?[g>+66+41K-Klfu9B&	O ss!jH5&uDq#V@ pơG1}{WWWaevF&R_H	B cM%ۘB?)A]2:IyW%wR"_^ZxA4(*dDu&A`ctY%"1d5ymJ9qJI9wuQsْfc@<<wΦ9bf3?s"wյy>bv4VN<kؚm=ƌ'Ӯua9?vuLQt5It} |
| --- | Minor | Jv 贘jmq~. \g\لm9ϳf|Fݾ/s&4{ˠef7MO˿|#x67GI.cf#	N?[g>+66+41K-Klfu9B&	O ss!jH5&uDq#V@ pơG1}{WWWaevF&R_H	B cM%ۘB?)A]2:IyW%wR"_^ZxA4(*dDu&A`ctY%"1d5ymJ9qJI9wuQsْfc@<<wΦ9bf3?s"wյy>bv4VN<kؚm=ƌ'Ӯua9?vuLQt5It}. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5301, 78 words, 2 clauses) |
| --- | Minor |  dR[RnFA iLˇibUɓ"ҡ2AIY;NHԒ[&Kd-)Қ*Cg5h=? 0=5xg6󻡪aa׎w_;`i㒃MC,_~i\w6lhouI'n>☍ۖ,nܷo)'29i-@Hhٿ̳:Co@G=ax-fŲv843hV[m7޲嘍?~3sS@7\w`SamMmſhFX:F qh=!AEh le|p͖'|4;K>o~;_?9Ϙ߽}ݚՇn8-wl}p۳䕫W/nغ?\UsH--7~ZM=99"SC TN:@9ϑ[ LLrn{Jsrʝ߹΀Kjk0|o:'Z4ʃC~cC)2miF3u@ |
| --- | Minor |  dR[RnFA iLˇibUɓ"ҡ2AIY;NHԒ[&Kd-)Қ*Cg5h=? 0=5xg6󻡪aa׎w_;`i㒃MC. _~i\w6lhouI'n>☍ۖ. nܷo)'29i-@Hhٿ̳:Co@G=ax-fŲv843hV[m7޲嘍?~3sS@7\w`SamMmſhFX:F qh=!AEh le|p͖'|4;K>o~;_?9Ϙ߽}ݚՇn8-wl}p۳䕫W/nغ?\UsH--7~ZM=99"SC TN:@9ϑ[ LLrn{Jsrʝ߹΀Kjk0|o:'Z4ʃC~cC)2miF3u@. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5312, 63 words, 3 clauses) |
| --- | Minor | HJ:ӥ컰TBHBlC2PxaRhO/3/}8nfvjꗼe^ovߠ TuP'zu6h:RyҢdqOf*,]]n$%wxt oN5DhnF),Iƴ躀}9B0a5S߃I	w):*>"<IԐUzExWLF̚UG?YvU;7o&^/Ic]֬Y,}DKWTUJ/K(kҎOrHǈ=ִdkZ%smq׃ZTGLU;l#o"b*NGJe(cqvB̜K\b- h'rF|YbA\FD]v[3\dS4gK4C |
| --- | Minor | HJ:ӥ컰TBHBlC2PxaRhO/3/}8nfvjꗼe^ovߠ TuP'zu6h:RyҢdqOf*. ]]n$%wxt oN5DhnF). Iƴ躀}9B0a5S߃I	w):*>"<IԐUzExWLF̚UG?YvU;7o&^/Ic]֬Y. }DKWTUJ/K(kҎOrHǈ=ִdkZ%smq׃ZTGLU;l#o"b*NGJe(cqvB̜K\b- h'rF|YbA\FD]v[3\dS4gK4C. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5469, 64 words, 3 clauses) |
| --- | Minor | Cp;IS&q`mi0_|G|_uo^w?GY1	@MPU썔XɃీװZj, D=v7릟lÏ^?^us͠Be;0́дS<()qxq?u^SrrWeZK?@t6PMK阺xڠL6Wc1+%xP,EFz>*dvX	U/L"XUVCv\1':nQ ?,nPa\}el^QI0(IOquJXa0@\!?Q. 0CO@l1@{TbN?Ԁne%!YKLL/<t`%LD KqKlnxđR8q9܅V^F`҅ $J |
| --- | Minor | Cp;IS&q`mi0_|G|_uo^w?GY1	@MPU썔XɃీװZj.  D=v7릟lÏ^?^us͠Be;0́дS<()qxq?u^SrrWeZK?@t6PMK阺xڠL6Wc1+%xP. EFz>*dvX	U/L"XUVCv\1':nQ ?. nPa\}el^QI0(IOquJXa0@\!?Q. 0CO@l1@{TbN?Ԁne%!YKLL/<t`%LD KqKlnxđR8q9܅V^F`҅ $J. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5502, 43 words, 4 clauses) |
| --- | Minor | vL!^?><˷mm}`Gn|Sʢ߮D A@֯N-f2\*evul,ؚB2*O?d;,gXf|C@&@8BRd'fVG];Zc{מt-V[ 7/Ee`fh35R:A?䙡~@-g/;{+smӸUB^("wZ<MvU1aA<X%/ĨXNTh!	ݩ.H]by(CXIh,DaNE;*% , IFu |
| --- | Minor | vL!^?><˷mm}`Gn|Sʢ߮D A@֯N-f2\*evul. ؚB2*O?d;. gXf|C@&@8BRd'fVG];Zc{מt-V[ 7/Ee`fh35R:A?䙡~@-g/;{+smӸUB^("wZ<MvU1aA<X%/ĨXNTh!	ݩ.H]by(CXIh. DaNE;*% . IFu. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5540, 32 words, 4 clauses) |
| --- | Minor |  	L#:1^zmB*zw,$pOLS&&a2q%;n%/DyI#%o=^,rW>0!3Tm"RHLߕ~tOG.)&ۢ8w]	4jD%?q(3V]&*j ,?자1#_^-Y!*7Ȁ-O@9_ C,/zH	l" |
| --- | Minor |  	L#:1^zmB*zw. $pOLS&&a2q%;n%/DyI#%o=^. rW>0!3Tm"RHLߕ~tOG.)&ۢ8w]	4jD%?q(3V]&*j . ?자1#_^-Y!*7Ȁ-O@9_ C. /zH	l". |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5549, 37 words, 4 clauses) |
| --- | Minor | &6xPw.,-`7W:X`t/HBعT8	;zp7ʷ-B0p II'#plWɇJ*csp,vOD PZ1cslL[;BJaPI"Q͍UΟӉe:JJB gǸOpRsy,ݕ|Sh,/*cRLB0;ː-Vb?*ի-<SPzӮa0k4Ơiڶ |
| --- | Minor | &6xPw.. -`7W:X`t/HBعT8	;zp7ʷ-B0p II'#plWɇJ*csp. vOD PZ1cslL[;BJaPI"Q͍UΟӉe:JJB gǸOpRsy. ݕ|Sh. /*cRLB0;ː-Vb?*ի-<SPzӮa0k4Ơiڶ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5649, 44 words, 4 clauses) |
| --- | Minor | |ʈ'T<My)QlY)uЫoLZV8Tej} m^ɿU7%J`^;`0o><sf㚹nv6H%a^ זoAhmx~7{whN>wco`^r[5'Fj7_xq?~PtZZueȂl̒-d_#Grr`ݱܱ^h-[7=S;,F<elBV胑#ؔXlےp8ƑV~6,:,ΒҚ,h(|h-ҞR. |
| --- | Minor | |ʈ'T<My)QlY)uЫoLZV8Tej} m^ɿU7%J`^;`0o><sf㚹nv6H%a^ זoAhmx~7{whN>wco`^r[5'Fj7_xq?~PtZZueȂl̒-d_#Grr`ݱܱ^h-[7=S;. F<elBV胑#ؔXlےp8ƑV~6. :. ΒҚ. h(|h-ҞR.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5670, 51 words, 4 clauses) |
| --- | Minor | if$3*V8Xu)ѭbdQrZ1#Ɇ/Â08IYƔ*d@`D%X6KK1G[Cњ;ӷc˷\qDZh.̬bfrFI"OFQtY/	F| H@۫\h0ɒQRrL,(P&sP s0wXqmT; ZZVҊ' [,};We]_6A(\|Y`Dn~1R(D,h}76cЎ(ec0tS2j&-;Bc#. Y-dklF{<II)/<u#d3l׊|1! |
| --- | Minor | if$3*V8Xu)ѭbdQrZ1#Ɇ/Â08IYƔ*d@`D%X6KK1G[Cњ;ӷc˷\qDZh.̬bfrFI"OFQtY/	F| H@۫\h0ɒQRrL. (P&sP s0wXqmT; ZZVҊ' [. };We]_6A(\|Y`Dn~1R(D. h}76cЎ(ec0tS2j&-;Bc#. Y-dklF{<II)/<u#d3l׊|1!. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5711, 23 words, 4 clauses) |
| --- | Minor | ʓ:1jF9,=MLmaO6駲L`P,,WؐǗtGmy1llͤ;N[2͠vR	c|3ڭ;mBa3# %R#EJ~2斝#6?)},`֍K3#t؞RH |
| --- | Minor | ʓ:1jF9. =MLmaO6駲L`P. WؐǗtGmy1llͤ;N[2͠vR	c|3ڭ;mBa3# %R#EJ~2斝#6?)}. `֍K3#t؞RH. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5849, 68 words, 4 clauses) |
| --- | Minor | puA`R,?֎8OɉM%&фH4nqzxg8p=o|gR&MJ/LU}Wd{;cIgMiRl1S |p04M	;~ӗ*)oH 42|]	ƘIՑly>>|sGIiLb|rʡHj<K,_e9jݧRhIp08f TR	 jyiVE~CNxͨt33u*"jp@?u1\L&"r_DG)n%檝όub,ʈP?O-cbvW{02mAiAcr,(8a~9۰M𺷧Sw!R9+ET	1$.Eo~˫}p05'NabhʅDҸβiG\ LGӒz; |
| --- | Minor | puA`R. ?֎8OɉM%&фH4nqzxg8p=o|gR&MJ/LU}Wd{;cIgMiRl1S |p04M	;~ӗ*)oH 42|]	ƘIՑly>>|sGIiLb|rʡHj<K. _e9jݧRhIp08f TR	 jyiVE~CNxͨt33u*"jp@?u1\L&"r_DG)n%檝όub. ʈP?O-cbvW{02mAiAcr. (8a~9۰M𺷧Sw!R9+ET	1$.Eo~˫}p05'NabhʅDҸβiG\ LGӒz;. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5854, 47 words, 4 clauses) |
| --- | Minor | `"=8U6k,[^f'2,Pjcd-{ ZJ~L-kQ%;3m=|1KYG`we&ȣ,-c/U۪0<8yp`0-$ @!:wܶt~6Mwܽwe7@`~k~ofgg8déG\zRF_=DǙ2;x2Z:84_>=߃̢+V0wur,j;Y]t+bh&Spn]wHFh1yRdJo 9%oT |
| --- | Minor | `"=8U6k. [^f'2. Pjcd-{ ZJ~L-kQ%;3m=|1KYG`we&ȣ. -c/U۪0<8yp`0-$ @!:wܶt~6Mwܽwe7@`~k~ofgg8déG\zRF_=DǙ2;x2Z:84_>=߃̢+V0wur. j;Y]t+bh&Spn]wHFh1yRdJo 9%oT. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5898, 39 words, 4 clauses) |
| --- | Minor | [e;/ ͎ejWʯ<?xEN0FeRf,3(_&,L̊hʅ<SGy<D1,jv=4qCKTZl>>c L`.s-sΨ@Z,vSe^F'LCVo%gγoR EHMr^uffz$33[R~CK=5 Vr~=?~ݚSNtN*PDh	P |
| --- | Minor | [e;/ ͎ejWʯ<?xEN0FeRf. 3(_&. L̊hʅ<SGy<D1. jv=4qCKTZl>>c L`.s-sΨ@Z. vSe^F'LCVo%gγoR EHMr^uffz$33[R~CK=5 Vr~=?~ݚSNtN*PDh	P. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5916, 62 words, 2 clauses) |
| --- | Minor | o=\g;87ɡϾ_HO迌gy 8Zf7)=5<ȃS6o<Tɲe,zq\b;('`A4`,%|aa]ݒEmXl	Ө<tN*ڈA8¸NX8QB}>?K1_\c{:N5}(R@ʴ=a#yX+HnKM}+Yq];.# IP  *dto]={_s߼^ON 	ǷZ`쟺n籮w6no}_7l)H?҂Y~4{\`WZpgDՐB#TxOd|/Sw9VIu2pVYf[הP^jޝԊ*I'<!xI4Ҭphi8T" 	!<?O> |
| --- | Minor | o=\g;87ɡϾ_HO迌gy 8Zf7)=5<ȃS6o<Tɲe. zq\b;('`A4`. %|aa]ݒEmXl	Ө<tN*ڈA8¸NX8QB}>?K1_\c{:N5}(R@ʴ=a#yX+HnKM}+Yq];.# IP  *dto]={_s߼^ON 	ǷZ`쟺n籮w6no}_7l)H?҂Y~4{\`WZpgDՐB#TxOd|/Sw9VIu2pVYf[הP^jޝԊ*I'<!xI4Ҭphi8T" 	!<?O>. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5967, 67 words, 4 clauses) |
| --- | Minor | )!ܼewO?[Y.EϹlտ/~v`;nBlg\×""mHVj52d,ײzAH"!1ϲZ-k<Y`sc5$R2e,s@FB@-YT%Tӂ`3`};&y':rPƳJF2U,쨎ZU7:E53ݵŦj\8tu(,_;&;>̞[jBGhIIwD>  "N?Bb/W˾8aA@!0aM~L?GÌse& 7✱{'F/<W<if͍|x}ނDH}`ǣFN<̋#{}|ہk:sgy3fx큭|92čl۷Y3欘 1: 9"#{_׼|>wV{g!7/P> |
| --- | Minor | )!ܼewO?[Y.EϹlտ/~v`;nBlg\×""mHVj52d. ײzAH"!1ϲZ-k<Y`sc5$R2e. s@FB@-YT%Tӂ`3`};&y':rPƳJF2U. 쨎ZU7:E53ݵŦj\8tu(. _;&;>̞[jBGhIIwD>  "N?Bb/W˾8aA@!0aM~L?GÌse& 7✱{'F/<W<if͍|x}ނDH}`ǣFN<̋#{}|ہk:sgy3fx큭|92čl۷Y3欘 1: 9"#{_׼|>wV{g!7/P>. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 5974, 65 words, 3 clauses) |
| --- | Minor | ذD` A"eOwucq=*C}zg5rk(7]MCfriN+ϲuOG>W~d~όDHrFMxg_~ZDm0!p@ZAڨ} ~HJ2IIQ45"ѓ(TTn`pЄA],b2~3JRVp)h-'0[3`y Ro'%vC,V"c<Yo3O8ty]w߹﮻ZLWfpz{8󸵽GRdӦ]DFXDOp;͛?ēHJ fԏYdD.m=?^q,yXKpCn~X+~ zDƗ`DZV43S\UG۟vbJ_I{%~ρ |
| --- | Minor | ذD` A"eOwucq=*C}zg5rk(7]MCfriN+ϲuOG>W~d~όDHrFMxg_~ZDm0!p@ZAڨ} ~HJ2IIQ45"ѓ(TTn`pЄA]. b2~3JRVp)h-'0[3`y Ro'%vC. V"c<Yo3O8ty]w߹﮻ZLWfpz{8󸵽GRdӦ]DFXDOp;͛?ēHJ fԏYdD.m=?^q. yXKpCn~X+~ zDƗ`DZV43S\UG۟vbJ_I{%~ρ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6042, 67 words, 1 clauses) |
| --- | Minor | ͞ ZR5w3{<I RM{M%g_pshL BILP 0P)HJئ+-@w>oyg>ϚNBV;p=}|K @E=*> e1{vCZc7ǷeK:[eGc*C'lf+#Ћ/FsQ+2-K& Kʗo{,wGs~^5IM~~s׾|ړNnzat($ EA:j<GђŲNe͌T)eՍZ̻ P+_	"AVn~OΙ1wp`Ƭ:C}~2UKI!dj\Jg ɇ-4ӣ!BgsL;{ڪ#6͘SkH" wY1k.y1bkϟTCD8lbNBތKc |
| --- | Minor | ͞ ZR5w3{<I RM{M%g_pshL BILP 0P)HJئ+-@w>oyg>ϚNBV;p=}|K @E=*> e1{vCZc7ǷeK:[eGc*C'lf+#Ћ/FsQ+2-K& Kʗo{. wGs~^5IM~~s׾|ړNnzat($ EA:j<GђŲNe͌T)eՍZ̻ P+_	"AVn~OΙ1wp`Ƭ:C}~2UKI!dj\Jg ɇ-4ӣ!BgsL;{ڪ#6͘SkH" wY1k.y1bkϟTCD8lbNBތKc. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6049, 33 words, 4 clauses) |
| --- | Minor | 8*X:zJW5dYˣP܍QRKء306,2 ]홋B")u1c<c2ŋ$Ps箉Ƿ,[j=7 1L;}zE	,  [:[py @1Lb3.<ӟ޶kۺw[m_֬Ovl?o9E, BJA h6	3j<;{mO |
| --- | Minor | 8*X:zJW5dYˣP܍QRKء306. 2 ]홋B")u1c<c2ŋ$Ps箉Ƿ. [j=7 1L;}zE.   [:[py @1Lb3.<ӟ޶kۺw[m_֬Ovl?o9E.  BJA h6	3j<;{mO. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6071, 72 words, 2 clauses) |
| --- | Minor | {rl#cw9{o6̙=</9^S_2m̎lήhNg~3ߴq΁Fknl aMLurljLc[m|&֏jL81G> iը'&7/t> Z =.QHC?wނO}w=o}dCmܕ/_g=BbYgx>Z}/?L^ϨիiBNyW??Vs^̡(Z6+8sw0q}+Otduy'{ȳD8-VkMY̚B2%Ò@J J ГV̝I^XTL	f//p0,FA	 _T&Q#Q:IuՇz<zUňmjGHb9`VQ.P5BꦅJwR,l`ݠ |
| --- | Minor | {rl#cw9{o6̙=</9^S_2m̎lήhNg~3ߴq΁Fknl aMLurljLc[m|&֏jL81G> iը'&7/t> Z =.QHC?wނO}w=o}dCmܕ/_g=BbYgx>Z}/?L^ϨիiBNyW??Vs^̡(Z6+8sw0q}+Otduy'{ȳD8-VkMY̚B2%Ò@J J ГV̝I^XTL	f//p0. FA	 _T&Q#Q:IuՇz<zUňmjGHb9`VQ.P5BꦅJwR. l`ݠ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6089, 90 words, 3 clauses) |
| --- | Minor | ι8asyY~ P"n'&OFZ'"ޅYa:޽XL4,RDT"U[	+H*EerltH߰~Xej/CƳ4zZvo}S~=sPQ  )5jبCt"Mt*!!"S#ƹ$IR?~9F%JjUc̠@XY@q	PLLJ+Ae !zZjS0[fTcr/,Csf̝-5zm;rv;5gM[y`ΪUǼMܹ#?}EXTbssܾu?@˖_<l4V{[6븓lǾs/l?=zF)	}+8 fL%H_3})t"ӷ6⊰] s	[u졇Bc1	r"@Ѥ E+u&ϱGeO j"Gޞiblq	} E#0 d0";v:@W\q]Ik`\5 K颈.yAZqu'iGmN	MN (ZuN,D.K? |
| --- | Minor | ι8asyY~ P"n'&OFZ'"ޅYa:޽XL4. RDT"U[	+H*EerltH߰~Xej/CƳ4zZvo}S~=sPQ  )5jبCt"Mt*!!"S#ƹ$IR?~9F%JjUc̠@XY@q	PLLJ+Ae !zZjS0[fTcr/. Csf̝-5zm;rv;5gM[y`ΪUǼMܹ#?}EXTbssܾu?@˖_<l4V{[6븓lǾs/l?=zF)	}+8 fL%H_3})t"ӷ6⊰] s	[u졇Bc1	r"@Ѥ E+u&ϱGeO j"Gޞiblq	} E#0 d0";v:@W\q]Ik`\5 K颈.yAZqu'iGmN	MN (ZuN. D.K?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6105, 68 words, 0 clauses) |
| --- | Minor | 0IiADpG>ڻwљgPg	B<.v=odCHnw1Ɛ@Α"幂S`^r2lcL\1mA0@b0SPeyΤv5hJ?%еtH8.33 [1hR/hL@gnC0d	=	xLM;=Na.b16qMJ%I}P~͌-.7yK0cp[Ah8jE	<\-K0NmF*KP b𾥳I]ٚ`I3Qa5yӇPQyWrʇWVCnMS܊;f+Q2#eNa& 0='#dj뮕^WuɌ"]#'W/p98M2CVt<^YQIP:X]˻4D<̀@#POvST}w@՜D?EoL|.|wɾwXsF[<[̀*?z	+SCff'ѩ~ |
| --- | Minor | 0IiADpG>ڻwљgPg	B<.v=odCHnw1Ɛ@Α"幂S`^r2lcL\1mA0@b0SPeyΤv5hJ?%еtH8.33 [1hR/hL@gnC0d	=	xLM;=Na.b16qMJ%I}P~͌-.7yK0cp[Ah8jE	<\-K0NmF*KP b𾥳I]ٚ`I3Qa5yӇPQyWrʇWVCnMS܊;f+Q2#eNa& 0='#dj뮕^WuɌ"]#'W/p98M2CVt<^YQIP:X]˻4D<̀@#POvST}w@՜D?EoL|.|wɾwXsF[<[̀*?z	+SCff'ѩ~ |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6160, 55 words, 4 clauses) |
| --- | Minor | m2A^ǌ4G$S^GI%8pW,Fk>RkaXHa*sDcjA%WXO9d^,UΐmUE"*YB X_BCFM4fßK-j%&]r;x'W2V[gqw氄[;SX 5g [S%.SW[,1,LGn_׮x4\6^۞xS?YagoES=9#ԛj2wɸzMTmYL/Mה֣&8UzvR>[2dLg\|'͘>Wq/D>GKmhI54m+R׺Sqܯ8E;t(>+`W |
| --- | Minor | m2A^ǌ4G$S^GI%8pW. Fk>RkaXHa*sDcjA%WXO9d^. UΐmUE"*YB X_BCFM4fßK-j%&]r;x'W2V[gqw氄[;SX 5g [S%.SW[. 1. LGn_׮x4\6^۞xS?YagoES=9#ԛj2wɸzMTmYL/Mה֣&8UzvR>[2dLg\|'͘>Wq/D>GKmhI54m+R׺Sqܯ8E;t(>+`W. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6282, 76 words, 2 clauses) |
| --- | Minor | BQҎe;!~B6:%23Ќ ;lXE_?B}RJ)DɧQ$(W9#Af)(TwO{B|4@B&DJ (%c=?NYqP@Dly7]wO0 =@\7ÆXҺZP zϭy^Syl<~X,: n^k40\*a>*))=q[J[DUHQnݺk~uԧ]9ȕ&K>x DJ{dӺ,;0g<a?)F`Q*Y{=˂gR.LbcȥhPl99 (lɏ~_3}ܓ"")atxȒD cٌw;ɮn& @DKgH)0֑~wMw\RB)%N"BI ХJX8-t!(z b9p>B   [&Hzq	@p%`d¯1Y |
| --- | Minor | BQҎe;!~B6:%23Ќ ;lXE_?B}RJ)DɧQ$(W9#Af)(TwO{B|4@B&DJ (%c=?NYqP@Dly7]wO0 =@\7ÆXҺZP zϭy^Syl<~X. : n^k40\*a>*))=q[J[DUHQnݺk~uԧ]9ȕ&K>x DJ{dӺ. ;0g<a?)F`Q*Y{=˂gR.LbcȥhPl99 (lɏ~_3}ܓ"")atxȒD cٌw;ɮn& @DKgH)0֑~wMw\RB)%N"BI ХJX8-t!(z b9p>B   [&Hzq	@p%`d¯1Y. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6306, 67 words, 1 clauses) |
| --- | Minor | @BR	Hk۶KWG7n{o~N:<ZU y?yqǂ1&:Ǒ]n[wsf.~`g7RP>C7H]{tG>v л.j(%!,\<Ѓtc#r<@!-Vax'7qbJ6ZW?LZ)1v%wN]Q4aI{tl4 <txv%HBnٱ#?czk)	KL[]~}-*3⨽ɓvUBsĲ}  )@Q薹w>F]؇(ǅB{;1F]=SRXH!ר昡M+::d5dG)=a BS6+U!y+<ořzdv UUȜCvRJԋrN"aE |
| --- | Minor | @BR	Hk۶KWG7n{o~N:<ZU y?yqǂ1&:Ǒ]n[wsf.~`g7RP>C7H]{tG>v л.j(%!. \<Ѓtc#r<@!-Vax'7qbJ6ZW?LZ)1v%wN]Q4aI{tl4 <txv%HBnٱ#?czk)	KL[]~}-*3⨽ɓvUBsĲ}  )@Q薹w>F]؇(ǅB{;1F]=SRXH!ר昡M+::d5dG)=a BS6+U!y+<ořzdv UUȜCvRJԋrN"aE. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6311, 65 words, 0 clauses) |
| --- | Minor | g`[FaFR!0>Qs(w d͖kB &[$O@>\	lXa5sQ]-6Fn5ߩ4e%S'	0+Y>'n\V O	eWYYEû(ڑQ w}̙	 fs`*3Ha]xam˝rgW<X%d*MީG\0 gKoF";NQ(Cw2ꕳ]Wf49biW+	u;#|8Ӊ1"1syntF/|71QA6`!i[0dx)u!:;k&jropMJQY SoM~kss\ vQ6Ds( )e(@!M ۼf>g_;<&ZD |
| --- | Minor | g`[FaFR!0>Qs(w d͖kB &[$O@>\	lXa5sQ]-6Fn5ߩ4e%S'	0+Y>'n\V O	eWYYEû(ڑQ w}̙	 fs`*3Ha]xam˝rgW<X%d*MީG\0 gKoF";NQ(Cw2ꕳ]Wf49biW+	u;#|8Ӊ1"1syntF/|71QA6`!i[0dx)u!:;k&jropMJQY SoM~kss\ vQ6Ds( )e(@!M ۼf>g_;<&ZD |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6343, 25 words, 4 clauses) |
| --- | Minor | !&u0|y9, m/,fDxw!uƑ.nF,F-x\D醐ܴ̐ chnRG/JkD:/*v̆=x=,B :wlG^<Pˮ)֬D	\zt՛^"-[Ȕ#zO~8_KvoﯺW |
| --- | Minor | !&u0|y9.  m/. fDxw!uƑ.nF. F-x\D醐ܴ̐ chnRG/JkD:/*v̆=x=. B :wlG^<Pˮ)֬D	\zt՛^"-[Ȕ#zO~8_KvoﯺW. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6428, 42 words, 4 clauses) |
| --- | Minor |  .\_UYz4i6+riP/-x&ZKSr,q溷W[Q'p#"gC#Pb!)¦/S/jmWrÅDtU,	e4R&Ka`Z"hM=ϏۺHG;N"i8!N81[ݞkivzGB--JmHo亱fH~A,USxkR; C7]oK?,*v|~ʣ{;RJld-ՈY |
| --- | Minor |  .\_UYz4i6+riP/-x&ZKSr. q溷W[Q'p#"gC#Pb!)¦/S/jmWrÅDtU. e4R&Ka`Z"hM=ϏۺHG;N"i8!N81[ݞkivzGB--JmHo亱fH~A. USxkR; C7]oK?. *v|~ʣ{;RJld-ՈY. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6439, 79 words, 1 clauses) |
| --- | Minor | yGyC媋kCL"zd'M;(7":7@'wJD@BH>(j&ic_Uj(Z?ư6?*7ܴNѣL9HGg*4cJAл{Ѹ5;CE =rB	(0z~iSѝ&<laPGV 3O,š"&\g_FF0DD*&z(a4itS2yle6+u&0C\z+{ВEKo_)% RVţ[=Ï8bhP )n19Vs"x?DNl68׭LڸbI9 B/j0L_c r)ˮ6? Va!x&ORxLilxH)H=QVf͞	]qkgq" ;f)B!z;{OyC\J<_KSUIDd~qGHX:w˥ptqql6f1_0 |
| --- | Minor | yGyC媋kCL"zd'M;(7":7@'wJD@BH>(j&ic_Uj(Z?ư6?*7ܴNѣL9HGg*4cJAл{Ѹ5;CE =rB	(0z~iSѝ&<laPGV 3O. š"&\g_FF0DD*&z(a4itS2yle6+u&0C\z+{ВEKo_)% RVţ[=Ï8bhP )n19Vs"x?DNl68׭LڸbI9 B/j0L_c r)ˮ6? Va!x&ORxLilxH)H=QVf͞	]qkgq" ;f)B!z;{OyC\J<_KSUIDd~qGHX:w˥ptqql6f1_0. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6533, 80 words, 2 clauses) |
| --- | Minor | Fg|ɟB<pT=C?f͞]uRE&MꖝF8gM6teS.0aVAiC%*\54;W{(C _SVO%nқ	4YR ۹<2&r7w"eǈ 9i?[NRv"!bZ@(ضpނFS-B)!Bv_^/NJ^Iߟ%,irḽ1^m"Oج;QgH0BL0.%ֵPAd]9AL=ڲsLP |OVqK?np-5TV@cbΓ0/ʍ7@Av:7zƒR}dQ1fCMF# H!˪ەq}=EBޜ9E V~ٳ෿ݵ^8w\M%tї/gx ,Z8B4{Ӗ- xbmzSU{nd;b.[(anD50E{⢈WG ^r}melir_. |
| --- | Minor | Fg|ɟB<pT=C?f͞]uRE&MꖝF8gM6teS.0aVAiC%*\54;W{(C _SVO%nқ	4YR ۹<2&r7w"eǈ 9i?[NRv"!bZ@(ضpނFS-B)!Bv_^/NJ^Iߟ%. irḽ1^m"Oج;QgH0BL0.%ֵPAd]9AL=ڲsLP |OVqK?np-5TV@cbΓ0/ʍ7@Av:7zƒR}dQ1fCMF# H!˪ەq}=EBޜ9E V~ٳ෿ݵ^8w\M%tї/gx . Z8B4{Ӗ- xbmzSU{nd;b.[(anD50E{⢈WG ^r}melir_.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6575, 32 words, 4 clauses) |
| --- | Minor | ~\/{E<y]g·2+x<6Gu,gURIn&uVY֩0okS4*/FL` DX6H֮],q=L7k%#ʁ$1F8A0-+4W="Tk,K^uul.O5C[t+z`z,)fAՒk1ccnJUӷOJ[[]] |
| --- | Minor | ~\/{E<y]g·2+x<6Gu. gURIn&uVY֩0okS4*/FL` DX6H֮]. q=L7k%#ʁ$1F8A0-+4W="Tk. K^uul.O5C[t+z`z. )fAՒk1ccnJUӷOJ[[]]. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6584, 45 words, 4 clauses) |
| --- | Minor | 咕@b$;Tt@aЇ±i"vŭ08׾ۿ4L[Ooy:4,#u5F,[ּpsd`H-_\#SQC,eq-X+QK ːycE@vSPjĔ%d8+e|o&4F]ZT1JdtH}t]5ΑoDGs{AB<G<pcPρ]XqP_#S{]{`0z*ҙgqyi],祥% >kT0/ |
| --- | Minor | 咕@b$;Tt@aЇ±i"vŭ08׾ۿ4L[Ooy:4. #u5F. [ּpsd`H-_\#SQC. eq-X+QK ːycE@vSPjĔ%d8+e|o&4F]ZT1JdtH}t]5ΑoDGs{AB<G<pcPρ]XqP_#S{]{`0z*ҙgqyi]. 祥% >kT0/. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6668, 34 words, 4 clauses) |
| --- | Minor | v<X,̲ @  t:]^^>]o}O>s5}0P~K9| X,2o)JPu=vTM#RXIajj,ᰴAf(ܜNyJČx:SXlN@z2 ׭#'z5e̔W,Ug.nZl 4 e-@dk&v+ In  8 |
| --- | Minor | v<X. ̲ @  t:]^^>]o}O>s5}0P~K9| X. 2o)JPu=vTM#RXIajj. ᰴAf(ܜNyJČx:SXlN@z2 ׭#'z5e̔W. Ug.nZl 4 e-@dk&v+ In  8. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6678, 43 words, 4 clauses) |
| --- | Minor | 3%58^u!OojqǵBsqD&g;EotgJ qrnN/ c	,YRSO,jӃm#P5"T>M/M|Pծ>׊:/,+ѭV.xŒ%*QzM yZpkj!!T<%QxiAxRy<gqAֱhHRo|Nug3N[Z, Ze)u11'Ź3iMU$ |
| --- | Minor | 3%58^u!OojqǵBsqD&g;EotgJ qrnN/ c. YRSO. jӃm#P5"T>M/M|Pծ>׊:/. +ѭV.xŒ%*QzM yZpkj!!T<%QxiAxRy<gqAֱhHRo|Nug3N[Z.  Ze)u11'Ź3iMU$. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6728, 72 words, 4 clauses) |
| --- | Minor | 	ccђIcГ[dJ~ɧ*I&k^L)0kdއSR}1	Ê,n`#,fQ"tK4'@|PY=G``Ŕk`86[Wʩ  J G[HrtgG1\"Ćjj]4g1T52 &gw~(;֮&3cՌ!I`derbi6*P~;BVva=4X'-J#?v"np~{9~v'bx៛X2?VP'pUΫ"yVAOJBȫ1G) K$>sñ_{j!%@ܘZE!hiqs;;`駽߻Go??3?~.ۼyju=˘dn	4 ,(%B(XsVQ+"u)@+[DȉU,mij |
| --- | Minor | 	ccђIcГ[dJ~ɧ*I&k^L)0kdއSR}1	Ê. n`#. fQ"tK4'@|PY=G``Ŕk`86[Wʩ  J G[HrtgG1\"Ćjj]4g1T52 &gw~(;֮&3cՌ!I`derbi6*P~;BVva=4X'-J#?v"np~{9~v'bx៛X2?VP'pUΫ"yVAOJBȫ1G) K$>sñ_{j!%@ܘZE!hiqs;;`駽߻Go??3?~.ۼyju=˘dn	4. (%B(XsVQ+"u)@+[DȉU. mij. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6735, 75 words, 1 clauses) |
| --- | Minor | X+={V3+VT0OMhѳU;s&ֳP+e}1ߕ'X-#"HHR65*ERYgBy~~	AK{aQt&FtbC8g򼍃V]Dd{iHMiND 0~mmK}]re@\-6FU!*1;!b0!2Țxb5wFlZEĆ`T4m ʓK>}>oџ\yǎ+7'~]_+v%O{NT?dr)X0Dz̜i4cfko]RC_{3DyS.3oSprU,Siä18cS{OEq>X$ x!~+/_YY!^T܍x׿E/B8"hT40FǮ!G%.C3>iW5-b aTr5|RYR+[1 |
| --- | Minor | X+={V3+VT0OMhѳU;s&ֳP+e}1ߕ'X-#"HHR65*ERYgBy~~	AK{aQt&FtbC8g򼍃V]Dd{iHMiND 0~mmK}]re@\-6FU!*1;!b0!2Țxb5wFlZEĆ`T4m ʓK>}>oџ\yǎ+7'~]_+v%O{NT?dr)X0Dz̜i4cfko]RC_{3DyS.3oSprU. Siä18cS{OEq>X$ x!~+/_YY!^T܍x׿E/B8"hT40FǮ!G%.C3>iW5-b aTr5|RYR+[1. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6743, 62 words, 2 clauses) |
| --- | Minor | oؕZ|;`/0ഔ'  gQ#ǜ*f\K7asYőʰTbE^0_u}yÆ.6|EY7}O>d˸P1<ϻ"z JAɒ>Q[d= W|=+&eZ9e版{}o fƍΟ+>n:ܲdx\S8x7	V3}ˍ\lo#T!o%WWX>TkaB$GX&UjscZM^1]dYe|f",4u#ax*Dx<y pmw?pg/W^[9D{54<:b~HNI`߫.*o͟U,Ԡ]Q. |
| --- | Minor | oؕZ|;`/0ഔ'  gQ#ǜ*f\K7asYőʰTbE^0_u}yÆ.6|EY7}O>d˸P1<ϻ"z JAɒ>Q[d= W|=+&eZ9e版{}o fƍΟ+>n:ܲdx\S8x7	V3}ˍ\lo#T!o%WWX>TkaB$GX&UjscZM^1]dYe|f". 4u#ax*Dx<y pmw?pg/W^[9D{54<:b~HNI`߫.*o͟U. Ԡ]Q.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6833, 69 words, 3 clauses) |
| --- | Minor | aJodC5"]wV48w"8`%nոR0 ݉*@@ko,L+KAQHO|U̒9Gu*qX+JA("zrg˾>WoYgo~yYuߧ>z{?߽կ~},KTH\<Iu%җ/M\8"j튞UԨ2ɫQ`qje6DLsc BxM5{HJQH1zhrS>l+Qwq:@NUO3%r Q\H}y^xaQCAF8fJ\,'*+x`Y>ꔾ><ZG7uK_ؽؽsڒ@oɳ⬹iz|qY1)]6_`	|E˖sbl=0i|%i{p |
| --- | Minor | aJodC5"]wV48w"8`%nոR0 ݉*@@ko. L+KAQHO|U̒9Gu*qX+JA("zrg˾>WoYgo~yYuߧ>z{?߽կ~}. KTH\<Iu%җ/M\8"j튞UԨ2ɫQ`qje6DLsc BxM5{HJQH1zhrS>l+Qwq:@NUO3%r Q\H}y^xaQCAF8fJ\. '*+x`Y>ꔾ><ZG7uK_ؽؽsڒ@oɳ⬹iz|qY1)]6_`	|E˖sbl=0i|%i{p. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6855, 35 words, 4 clauses) |
| --- | Minor | [Dt,")gw"$'-P2RXCDlΡCJ[J׋.0hi%  tm,0}?y ?):@@MnXK:D?sMX,0H!z`TH9.F4lNd-lUMC<= B'Vi=@Zcw][J&#.r}?5B#e|;z% {.zSyO|ﺗ~KVצ^,P9q<éNA@doꩺ!F1eŤif;y>( |
| --- | Minor | [Dt. ")gw"$'-P2RXCDlΡCJ[J׋.0hi%  tm. 0}?y ?):@@MnXK:D?sMX. 0H!z`TH9.F4lNd-lUMC<= B'Vi=@Zcw][J&#.r}?5B#e|;z% {.zSyO|ﺗ~KVצ^. P9q<éNA@doꩺ!F1eŤif;y>(. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 6906, 67 words, 1 clauses) |
| --- | Minor | oBY&T0	:<fpj}Eg߃@j[^fbnSE F6jۣ: \kP(\\UY5H~3<c_E1(Jmyt @b_@IOR*0_"Ͳt̆3nWC-FcYzI`B,#|d]eR6BRm?uK%k5z텷3uo擶j*90h+Kw*jgX13*_\9f$|HaXvWdtgU<kc(Ϟy)O'QS7E?[	ply=!QĊ!N04(r|)l&[֤Ćȼgj^tJ&F26O(?n*!&xiMp07]w[Oy)fR7GJ |
| --- | Minor | oBY&T0	:<fpj}Eg߃@j[^fbnSE F6jۣ: \kP(\\UY5H~3<c_E1(Jmyt @b_@IOR*0_"Ͳt̆3nWC-FcYzI`B. #|d]eR6BRm?uK%k5z텷3uo擶j*90h+Kw*jgX13*_\9f$|HaXvWdtgU<kc(Ϟy)O'QS7E?[	ply=!QĊ!N04(r|)l&[֤Ćȼgj^tJ&F26O(?n*!&xiMp07]w[Oy)fR7GJ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7142, 53 words, 4 clauses) |
| --- | Minor | ڵKȄĮVJHJD̄ H]Q&)1؞jx#c^ y]g:s1Gڑ-(RdBq-edra6WՃ˪IF_:[ds,<DK7IP]XECܠB2AK,:i	U3> +Ty9;X`L.D}āq?(@d2q ,}>+lH6QR\PHmK)KI@b|5)dܽR<ϦA":پ]^\B R,h.*bVBPy/5;E\vIɷXEёD 2ϳ(S |
| --- | Minor | ڵKȄĮVJHJD̄ H]Q&)1؞jx#c^ y]g:s1Gڑ-(RdBq-edra6WՃ˪IF_:[ds. <DK7IP]XECܠB2AK. :i	U3> +Ty9;X`L.D}āq?(@d2q . }>+lH6QR\PHmK)KI@b|5)dܽR<ϦA":پ]^\B R. h.*bVBPy/5;E\vIɷXEёD 2ϳ(S. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7189, 78 words, 5 clauses) |
| --- | Minor | L{xKb pٲm}ƹs>~Qˍ>Ykܥ?:?y3#ۇV,_YE)L.YƏx?3:R(KRJ)?u?1cccc?E|C, ˲Nίqë'&[sO=u'DYZK<Qߗ1*~Ǜ.9l۞5o{Ձ{/%yuh^ꩵ9sX3^ȲLef[b᧞o66~;_be.XaZMx} 3J7566lKUmk5ħI!5,Gax&br2<?Qb,|2Ppڡ#~V:+cd7+pl}Ŧ?X!uj`	%ɲ(sϦ `;/_Kw5};oO ,Z T^9|H}n6noeALb>QXşSK`+N8(5CA_`Ҥb\q#0QO5%:uh |
| --- | Minor | L{xKb pٲm}ƹs>~Qˍ>Ykܥ?:?y3#ۇV. _YE)L.YƏx?3:R(KRJ)?u?1cccc?E|C.  ˲Nίqë'&[sO=u'DYZK<Qߗ1*~Ǜ.9l۞5o{Ձ{/%yuh^ꩵ9sX3^ȲLef[b᧞o66~;_be.XaZMx} 3J7566lKUmk5ħI!5. Gax&br2<?Qb. |2Ppڡ#~V:+cd7+pl}Ŧ?X!uj`	%ɲ(sϦ `;/_Kw5};oO . Z T^9|H}n6noeALb>QXşSK`+N8(5CA_`Ҥb\q#0QO5%:uh. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7197, 43 words, 4 clauses) |
| --- | Minor | "Ie{,^t3y&H)y]|iWx%`woP ;<xo/Y4<o  J)4ar&'u2L  *}L`Q+wn;]	9[D3ǈ e)eټiGG0B)Jׄ&jffysNm@&,,q:4S\M ~UӸHB,ك4:Vb#Qny#M@2˲ǟC<x%[v |
| --- | Minor | "Ie{. ^t3y&H)y]|iWx%`woP ;<xo/Y4<o  J)4ar&'u2L  *}L`Q+wn;]	9[D3ǈ e)eټiGG0B)Jׄ&jffysNm@&. q:4S\M ~UӸHB. ك4:Vb#Qny#M@2˲ǟC<x%[v. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7204, 114 words, 3 clauses) |
| --- | Minor | o:q{#Q`CoY?䂫=S_xXgnw~߰.R4īҺ❃7{3l{w:~KjP"<tͪ-ǝx{zpۋo6	 ;<گa ;/W V/n~k?gZ,Mo;fWo]q巟z+_u_?zxflٲ+6ˮ{yҹsyڈ!گyFQ*)ټի|r{﹤Rd'u֚>nIN[_;1<2ZNJ D?+L滿_r]9-',ZJr\=K[G.n)r6@<߰y]8?spʿ^r>R@o|p-l8|©ٍ֧?pm<9}jZ2FSe7;e:#C쓙&8"m7 |qU ;تtU"/[jqF&|ieqrk1QCj:3h  |K/.RJS\@Ï ˲`0L>XeY(Pu:f%.괉ϓ1OY{u2fM7z1iխ"4z@I'-$XcY~ fv7b FF&♳CtL!,Yykl:s9H*-Hx;_e˺Xb'i'Al4sB  Ru5k |
| --- | Minor | o:q{#Q`CoY?䂫=S_xXgnw~߰.R4īҺ❃7{3l{w:~KjP"<tͪ-ǝx{zpۋo6	 ;<گa ;/W V/n~k?gZ. Mo;fWo]q巟z+_u_?zxflٲ+6ˮ{yҹsyڈ!گyFQ*)ټի|r{﹤Rd'u֚>nIN[_;1<2ZNJ D?+L滿_r]9-'. ZJr\=K[G.n)r6@<߰y]8?spʿ^r>R@o|p-l8|©ٍ֧?pm<9}jZ2FSe7;e:#C쓙&8"m7 |qU ;تtU"/[jqF&|ieqrk1QCj:3h  |K/.RJS\@Ï ˲`0L>XeY(Pu:f%.괉ϓ1OY{u2fM7z1iխ"4z@I'-$XcY~ fv7b FF&♳CtL!. Yykl:s9H*-Hx;_e˺Xb'i'Al4sB  Ru5k. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7229, 61 words, 3 clauses) |
| --- | Minor | )eC v!ۗlW/:lYQȢ,=`U[ (׼/7mO 2+n~`ݚuhlb?m^~뒣O9>9Qt:gۡy>rUgYxmm[>|~Sٟz9?ITl>~؞K~^{< HIymw%?w[B@e0k	ڝ(YrȊv([PO}kߺd{/΢cc㳦6cw=^-|@T6٣Oo{{㟟u!Lڒ._p#G_eHUU/g/qۿ$RdOsO/o/ߏY6o,ƧOi, Pި3Ff0tIn([yD#C^9`NUk |
| --- | Minor | )eC v!ۗlW/:lYQȢ. =`U[ (׼/7mO 2+n~`ݚuhlb?m^~뒣O9>9Qt:gۡy>rUgYxmm[>|~Sٟz9?ITl>~؞K~^{< HIymw%?w[B@e0k	ڝ(YrȊv([PO}kߺd{/΢cc㳦6cw=^-|@T6٣Oo{{㟟u!Lڒ._p#G_eHUU/g/qۿ$RdOsO/o/ߏY6o. ƧOi. Pި3Ff0tIn([yD#C^9`NUk. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7263, 65 words, 0 clauses) |
| --- | Minor | X+F)S]c{]y+׻[׊{Z^AڴH8BZ\R%M ] 3D(gQn`xKB޽i.Ċ~/V=1hcO>N?M]'k^F|s܅뮹~^ZFq:v'{vzTܴ6sDXf+?_'pR1-!笡%_;L:HCeK:vGyo|;.J-t  CHf[\=gbhW9nܠ;r3 [u[nykӢzo?n	ؒĢ|7^YL{_~weP2lSD!K~qq1>y-Wz؁ H1oΛɵCq٣tSu8#Rg/s#6+? |
| --- | Minor | X+F)S]c{]y+׻[׊{Z^AڴH8BZ\R%M ] 3D(gQn`xKB޽i.Ċ~/V=1hcO>N?M]'k^F|s܅뮹~^ZFq:v'{vzTܴ6sDXf+?_'pR1-!笡%_;L:HCeK:vGyo|;.J-t  CHf[\=gbhW9nܠ;r3 [u[nykӢzo?n	ؒĢ|7^YL{_~weP2lSD!K~qq1>y-Wz؁ H1oΛɵCq٣tSu8#Rg/s#6+? |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7292, 63 words, 4 clauses) |
| --- | Minor | >IAÆuyиs,yzv=rV~oo32wGIRezI  KJ	a5[dfol駝2C-"5=o7=dt:8yy'wi' @<_zvvDd~lK4 bݪ3ǿ99tu, CyA (+픥95۶ G!8"0Es6_kgJDd_~qGMDK˫`Ѣ?\\^yc+lZ}nӆap46ٸil%錐@l?T.߼ݮ+خRIH<k68 g-%7܍>]{[M4J /fB,qjx{܋;[3˂VwޯB,eRv6zV|;.>np~kʌ3μ:hQvQ 8$ |
| --- | Minor | >IAÆuyиs. yzv=rV~oo32wGIRezI  KJ	a5[dfol駝2C-"5=o7=dt:8yy'wi' @<_zvvDd~lK4 bݪ3ǿ99tu. CyA (+픥95۶ G!8"0Es6_kgJDd_~qGMDK˫`Ѣ?\\^yc+lZ}nӆap46ٸil%錐@l?T.߼ݮ+خRIH<k68 g-%7܍>]{[M4J /fB. qjx{܋;[3˂VwޯB. eRv6zV|;.>np~kʌ3μ:hQvQ 8$. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7295, 63 words, 1 clauses) |
| --- | Minor | yԔ!M%e/~^??SD.I--ObUعb%@DpmF<ص㿏:x@BU+w;_:o2 ᧳	UWCMTU|2xȨ{p)cmogfksNNw?`Syq 3V,_'?OW!G8nS\>~`mw:yo_>y]Ғu;o㹭57]ybNV.E/_f^pY3n9uTx~\VwϭO^+#RdtO?/dsmW[6x噏\RIi*eSCq+Ͱ<3|}=9yq4R'+@;;%0hgB--1Yhㅁqmę@ԴD:P?\Q5;8#'Eo |
| --- | Minor | yԔ!M%e/~^??SD.I--ObUعb%@DpmF<ص㿏:x@BU+w;_:o2 ᧳	UWCMTU|2xȨ{p)cmogfksNNw?`Syq 3V. _'?OW!G8nS\>~`mw:yo_>y]Ғu;o㹭57]ybNV.E/_f^pY3n9uTx~\VwϭO^+#RdtO?/dsmW[6x噏\RIi*eSCq+Ͱ<3|}=9yq4R'+@;;%0hgB--1Yhㅁqmę@ԴD:P?\Q5;8#'Eo. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7300, 78 words, 0 clauses) |
| --- | Minor | 4>|CCݻn˗?;ohh~-HGN2w^l+ pnl۸vXq3ň  /5ݯ.?e/Wr@TߘCH (TH!d`b e[8>)}~׿qHدO?ߞ|{8:w7\:y@  )<lf}zwvҵS]ZNN0GE-[|_[I|t'޹sk>tm#}ƛ;TeGa׮W?]sTQæ~orA#:RWD%58O'xgɦ-;vzOf3νl^{owmϊ'~v(+;i+s/xy!3Yyۯ͘vcTI9[|CYx3?}ڤ_z)c˿1z>rx^-Zr孷C9#8hS@mعɏ?hn#Zz{;ﻥ>[()9I4ybc޺Wm]}ovljU|ꍛ3Nk%UvӮ~c2o/r |
| --- | Minor | 4>|CCݻn˗?;ohh~-HGN2w^l+ pnl۸vXq3ň  /5ݯ.?e/Wr@TߘCH (TH!d`b e[8>)}~׿qHدO?ߞ|{8:w7\:y@  )<lf}zwvҵS]ZNN0GE-[|_[I|t'޹sk>tm#}ƛ;TeGa׮W?]sTQæ~orA#:RWD%58O'xgɦ-;vzOf3νl^{owmϊ'~v(+;i+s/xy!3Yyۯ͘vcTI9[|CYx3?}ڤ_z)c˿1z>rx^-Zr孷C9#8hS@mعɏ?hn#Zz{;ﻥ>[()9I4ybc޺Wm]}ovljU|ꍛ3Nk%UvӮ~c2o/r |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7326, 90 words, 2 clauses) |
| --- | Minor | =Y|MGE&mˢҬljlLn467o@ܸn:w*}1: pܣ~Ĝ̤Sj@D9HDו!"@B? pԑO?ZT<ogRs`њ&+R~lּ#ga5ټu7 bs8[![_^}~Vn0W-I)r Hpu;WBLrF dJ:W7Wm7ںmKVlI+ܮ5RiݞyLPVŭq}o?~n%M9NM0ixyv]/sٻ#{&m[0F!z{@oLCگ\ՙ6mwǍѫȶmE:Y`a|ݻ-X,2+6tI]71" ]Ձً7[Y[{֮L .YifKG8D |.08 ʹnض|nV~ђ/;gWT,/dM`[VnL.efSv3:s:ً뛣(զMf˓9^ֈQ!< xqJ.rZu5]@V$ݎМWF? |
| --- | Minor | =Y|MGE&mˢҬljlLn467o@ܸn:w*}1: pܣ~Ĝ̤Sj@D9HDו!"@B? pԑO?ZT<ogRs`њ&+R~lּ#ga5ټu7 bs8[![_^}~Vn0W-I)r Hpu;WBLrF dJ:W7Wm7ںmKVlI+ܮ5RiݞyLPVŭq}o?~n%M9NM0ixyv]/sٻ#{&m[0F!z{@oLCگ\ՙ6mwǍѫȶmE:Y`a|ݻ-X. 2+6tI]71" ]Ձً7[Y[{֮L .YifKG8D |.08 ʹnض|nV~ђ/;gWT. /dM`[VnL.efSv3:s:ً뛣(զMf˓9^ֈQ!< xqJ.rZu5]@V$ݎМWF?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7356, 80 words, 4 clauses) |
| --- | Minor | y/ |H hxoSꁂｕB^V*A5*uU ֧Wv5rTaCS7d-x⢔Zکj!?}˪/W{n5kkwlٱzъ#2v^J=;viդ3ӲUo,Rl֝77um_d Fm\r9kq+u~żt5g/OM3@J1nh?\qʮ3'?v`4bÓt	G  TU,䚛󻋎>7~+/;a@\^iYNMuY&#ny/;/\=qDuu/-;|x)%欕lo܅k4O,M m	  i2.Q9˹)?-\ګ਋8I*H5{ي-kKIRֲ?qՎ<U?[wAt7rNunZZ޷ۤC2)sOJqNk  TUfEUiQmu/cUIMu䦫N;AǋLQ'._nYs8y[pv6]v,\z-T;~^מ}/;#0{o|q)GOU[*A|?[~65 |
| --- | Minor | y/ |H hxoSꁂｕB^V*A5*uU ֧Wv5rTaCS7d-x⢔Zکj!?}˪/W{n5kkwlٱzъ#2v^J=;viդ3ӲUo. Rl֝77um_d Fm\r9kq+u~żt5g/OM3@J1nh?\qʮ3'?v`4bÓt	G  TU. 䚛󻋎>7~+/;a@\^iYNMuY&#ny/;/\=qDuu/-;|x)%欕lo܅k4O. M m	  i2.Q9˹)?-\ګ਋8I*H5{ي-kKIRֲ?qՎ<U?[wAt7rNunZZ޷ۤC2)sOJqNk  TUfEUiQmu/cUIMu䦫N;AǋLQ'._nYs8y[pv6]v. \z-T;~^מ}/;#0{o|q)GOU[*A|?[~65. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7377, 66 words, 1 clauses) |
| --- | Minor | Sֆط_x^L6udҌ0ҿg#`8yQeum{D!%D(A:UxIF*]2ZZ{v!`ũ,^ϱtȈ]* NTT2lK|eE Lt߂V	RB.U>|G߬imˊR'<j٬%In4#JI:קs5 0;AP8q>ymA⠾]N8hxyI))N;e|.#(kTAcȘRw:~^ٜb@fjCG@:x=;}:t|/۶%eE)swZ<\.ׯ3O8*F+H?z#jlmWYD eg0~tTCK4G "O7KVo؞6pPn:԰`16_Lq`-9*W1oжN"DӢvoH2F]z}7 |
| --- | Minor | Sֆط_x^L6udҌ0ҿg#`8yQeum{D!%D(A:UxIF*]2ZZ{v!`ũ. ^ϱtȈ]* NTT2lK|eE Lt߂V	RB.U>|G߬imˊR'<j٬%In4#JI:קs5 0;AP8q>ymA⠾]N8hxyI))N;e|.#(kTAcȘRw:~^ٜb@fjCG@:x=;}:t|/۶%eE)swZ<\.ׯ3O8*F+H?z#jlmWYD eg0~tTCK4G "O7KVo؞6pPn:԰`16_Lq`-9*W1oжN"DӢvoH2F]z}7. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7449, 70 words, 2 clauses) |
| --- | Minor | <b(zӇ%4g&N	WB6vִJ O먍*+^CH(xWd\'u R	X0l>?6:ܲU`Lwp{,Btߏi̸Qp"}qk< Zl7 M}Q'M"=л;wgd%̩fR@s7G0j6L|(t'dPQ IX" vL*ʕ'xƫw1ol4Uw>["KްnݍƎv }kVLg@)"CDG T4!m(0mf/ۺK\],@d4r`KGEfUyCRJI鴹to.Y֘}ƏDB[(@nl-Τ2)t[R@D-TJ6gnfme%i (mbҵKVo8rHTTu0_Wݶ{mʹ1R8B@ͭtlDϘ2 )QHipԔll.G |
| --- | Minor | <b(zӇ%4g&N	WB6vִJ O먍*+^CH(xWd\'u R	X0l>?6:ܲU`Lwp{. Btߏi̸Qp"}qk< Zl7 M}Q'M"=л;wgd%̩fR@s7G0j6L|(t'dPQ IX" vL*ʕ'xƫw1ol4Uw>["KްnݍƎv }kVLg@)"CDG T4!m(0mf/ۺK\]. @d4r`KGEfUyCRJI鴹to.Y֘}ƏDB[(@nl-Τ2)t[R@D-TJ6gnfme%i (mbҵKVo8rHTTu0_Wݶ{mʹ1R8B@ͭtlDϘ2 )QHipԔll.G. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7534, 83 words, 1 clauses) |
| --- | Minor | b0tS`-uz; dke'VϲMRp{Q=[Q@=Gs	Eć"ki2J~@K7oVJ5nB|4qy(iV>݄A/TD~WRa~Vd52v.P+?-W:¼OԤ`M0gh*h,MY!ƅZdG;d#mB2`efq*}ճÛ&̸Ϗ"V @޾γ3O w}};n\Oyd4Bo\ww&K4]]׭yQDw7}?zݓi4qniǨ{8l6 U_p-lȏ;N8h#ƎMW_>tmؼq6}ߍ݁~ܙgx! |W]uK>έ|#=o~;W^qۖ<t7z쎍 o|߹NXYק{tnO|#pkwc_WmM׉ھm:{蚟B箾|uH|ʫ^gl{}?_m4w?g=" ܹ{>ny4	[nmS?Oپqcx4k_ڨL&9Fn~ |
| --- | Minor | b0tS`-uz; dke'VϲMRp{Q=[Q@=Gs	Eć"ki2J~@K7oVJ5nB|4qy(iV>݄A/TD~WRa~Vd52v.P+?-W:¼OԤ`M0gh*h. MY!ƅZdG;d#mB2`efq*}ճÛ&̸Ϗ"V @޾γ3O w}};n\Oyd4Bo\ww&K4]]׭yQDw7}?zݓi4qniǨ{8l6 U_p-lȏ;N8h#ƎMW_>tmؼq6}ߍ݁~ܙgx! |W]uK>έ|#=o~;W^qۖ<t7z쎍 o|߹NXYק{tnO|#pkwc_WmM׉ھm:{蚟B箾|uH|ʫ^gl{}?_m4w?g=" ܹ{>ny4	[nmS?Oپqcx4k_ڨL&9Fn~. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7540, 103 words, 2 clauses) |
| --- | Minor | tܳby?7)-0&c--X)#UIVTB\=8u*=L&f?<;pҥ)T'.,GlrʘQ"mXK6`P zei/.8hS`(j-Fg7́:޲hp,uGbś+UʷĈ;Fܴd<U3v7+]jV+K2j7n Y!4d}Bmȩ܊jU;2gú+W\h+0I.[>%2*@QRT3Ckwͫe޳?v놕0GW\qMnFsdܭN:_o̓g?ЇrЖ@˹߫_Wvɡsnǝm7t}O{Ջ_nXu{ЍG4=ӎ<hl:s]w'K_+~u	Gl}N=xƠ gh4Cl|'Qz?ٲ_Ou;ēpwM<mH3g=ZVw_'n9}x"-M|Dز~gq_=w"r?GѾ}k-/9ts0sO>尭[m+_o]}"t6;Cϻ} ` 8<p=g; iOz1{y#i|G  Gr1h |
| --- | Minor | tܳby?7)-0&c--X)#UIVTB\=8u*=L&f?<;pҥ)T'.. GlrʘQ"mXK6`P zei/.8hS`(j-Fg7́:޲hp. uGbś+UʷĈ;Fܴd<U3v7+]jV+K2j7n Y!4d}Bmȩ܊jU;2gú+W\h+0I.[>%2*@QRT3Ckwͫe޳?v놕0GW\qMnFsdܭN:_o̓g?ЇrЖ@˹߫_Wvɡsnǝm7t}O{Ջ_nXu{ЍG4=ӎ<hl:s]w'K_+~u	Gl}N=xƠ gh4Cl|'Qz?ٲ_Ou;ēpwM<mH3g=ZVw_'n9}x"-M|Dز~gq_=w"r?GѾ}k-/9ts0sO>尭[m+_o]}"t6;Cϻ} ` 8<p=g; iOz1{y#i|G  Gr1h. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7551, 75 words, 1 clauses) |
| --- | Minor | 8~,8jiӚH@RfQrQUF%~b !"cQ1C"s(.7㨳[uAfsVCbRuP[4<BS{ibA0E/~0!z;% (%DBXw˯uTf=Q%XYͩ`ٯd1 :d5缘2Ie#&kDRʁBdfH(<^z]1eC0|-Um1d*L0×	%dmc[ EJse~V%&Rkbq}Ozrk t]ׅ'ZNcD DD\oHD볙^cF0xEGsKY!Z>I캮:L6]_'C9 O{{B\9纮s.fY}Vȍ|wOק>d\WLxk3}&;XŶE0krs@ _;-6A)륟MǰdI:Ay^4-scCR<LSjF |
| --- | Minor | 8~. 8jiӚH@RfQrQUF%~b !"cQ1C"s(.7㨳[uAfsVCbRuP[4<BS{ibA0E/~0!z;% (%DBXw˯uTf=Q%XYͩ`ٯd1 :d5缘2Ie#&kDRʁBdfH(<^z]1eC0|-Um1d*L0×	%dmc[ EJse~V%&Rkbq}Ozrk t]ׅ'ZNcD DD\oHD볙^cF0xEGsKY!Z>I캮:L6]_'C9 O{{B\9纮s.fY}Vȍ|wOק>d\WLxk3}&;XŶE0krs@ _;-6A)륟MǰdI:Ay^4-scCR<LSjF. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7561, 63 words, 1 clauses) |
| --- | Minor | 5=ŴPUERQDt~ǣ)y_Iwq\*ӣ qmw׮Y;dk/M:d$2Zt}\ju]~mm+ =>z)tɤgɁh9"f[ c`1wսD3/FlKթۡȴNeP&p59!> 4\WNamllRc~[L#nT&VJRUiryHNRɞ,BnF1pk bJ>:2g_U^C{vZk )KFD[j1l" qa J{\&0(7pc!DĦ@[YTe:_R;>A4 |
| --- | Minor | 5=ŴPUERQDt~ǣ)y_Iwq\*ӣ qmw׮Y;dk/M:d$2Zt}\ju]~mm+ =>z)tɤgɁh9"f[ c`1wսD3/FlKթۡȴNeP&p59!> 4\WNamllRc~[L#nT&VJRUiryHNRɞ. BnF1pk bJ>:2g_U^C{vZk )KFD[j1l" qa J{\&0(7pc!DĦ@[YTe:_R;>A4. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7570, 27 words, 5 clauses) |
| --- | Minor | *:K,X-yNt<&+0kBdccY)HP,%$c^4uNȄVc|vN4ї,avnjHPVu+PGL`EE,8FL詾}/b+ōdʠL*6m.{H} ؂ŊQ[9,[Ph |
| --- | Minor | *:K. X-yNt<&+0kBdccY)HP. %$c^4uNȄVc|vN4ї. avnjHPVu+PGL`EE. 8FL詾}/b+ōdʠL*6m.{H} ؂ŊQ[9. [Ph. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7572, 86 words, 1 clauses) |
| --- | Minor | `SN $HZt* ȯJ\yы)-nr!Ber>JU%\YE@VPB80ǖ-dRm>P<1өgFYmJ[<ju3Tw5Iȓb]H.y_נdZg\o*R9kQW!\ZW2`nk8l||;0VW3X161=}7Fx9=53JSXfCt/&DjKX0g;a60D.]UV-8Qw(<m! 4UmU=aڠ=ƾ8rnOeW0~יO}䯽G'\C.t BDSUMv{uoxE;y# =!7;7ny9`:Ρs΁K@ yD>	p>o<%rGupރO9>uB2As'k,D@/o>O ߤ"F]|t}_}\ |
| --- | Minor | `SN $HZt* ȯJ\yы)-nr!Ber>JU%\YE@VPB80ǖ-dRm>P<1өgFYmJ[<ju3Tw5Iȓb]H.y_נdZg\o*R9kQW!\ZW2`nk8l||;0VW3X161=}7Fx9=53JSXfCt/&DjKX0g;a60D.]UV-8Qw(<m! 4UmU=aڠ=ƾ8rnOeW0~יO}䯽G'\C.t BDSUMv{uoxE;y# =!7;7ny9`:Ρs΁K@ yD>	p>o<%rGupރO9>uB2As'k. D@/o>O ߤ"F]|t}_}\. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7647, 61 words, 2 clauses) |
| --- | Minor | g._9"-pS	2wu_>Fe'h'DiFEwuY]#%Z]Kj0c^,y㒂#ܲX)H z`)gֿxl\[n2?|oɖ}]}+"^čo{W]o@סCux/~O?>Q2͛V1Y=~,ڵw;uۿnsO"V&KUܶ/y&#7N_{֓ƭۿKC$d<co}4>ٺӏ;㮾o SVa}j٬'"Bz}~s駳n}ݹ1 L>o|SO>fX߻w/{{rF٤s|K㬇q{mC6Ϥ |
| --- | Minor | g._9"-pS	2wu_>Fe'h'DiFEwuY]#%Z]Kj0c^. y㒂#ܲX)H z`)gֿxl\[n2?|oɖ}]}+"^čo{W]o@סCux/~O?>Q2͛V1Y=~. ڵw;uۿnsO"V&KUܶ/y&#7N_{֓ƭۿKC$d<co}4>ٺӏ;㮾o SVa}j٬'"Bz}~s駳n}ݹ1 L>o|SO>fX߻w/{{rF٤s|K㬇q{mC6Ϥ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7701, 64 words, 2 clauses) |
| --- | Minor | J1;M*Mɲql`]H,8*"L5. QbO12ˀ JTx.HQG&qdK|ĥCGi;'K%ϒO&1̊,!sҌ~U>H:q-}uWT'GGvu!.7?|1HρiZH O[֡ب΋)"s_Xk:r^re A*b;욙}enbPZ #7.6Zs^jaDI|^ `F_<Ǫ5sHБnKw.y]ZTh{crTr[ArG榨!%s/6 )Wgb#f8Ԉ㴇ْdReWìΙ(GM O |
| --- | Minor | J1;M*Mɲql`]H. 8*"L5. QbO12ˀ JTx.HQG&qdK|ĥCGi;'K%ϒO&1̊. !sҌ~U>H:q-}uWT'GGvu!.7?|1HρiZH O[֡ب΋)"s_Xk:r^re A*b;욙}enbPZ #7.6Zs^jaDI|^ `F_<Ǫ5sHБnKw.y]ZTh{crTr[ArG榨!%s/6 )Wgb#f8Ԉ㴇ْdReWìΙ(GM O. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7725, 108 words, 3 clauses) |
| --- | Minor | ]x=[wCyꈣD[ ]ރcف}/D^)@[ol.|@zs^:_lU7DnEuʙ#(-*Q{%^"Mpdu,KͺP~mMx2	c&%#9gmVLeby.њ~sPE=3i"[su+rw0g?=7C/y_m=ПW>ky)gs<r{èyglXY{H[uk/yci[`c{#oMW]nx4}9y'>?7wݖcg߁W>p%޷gm'1ݹN8g>;o}W}WO{S_w{VowϺ0O?c_/_}c_~׮_{¯r!{K^כ?w ߻{p;~蝗}\Ug{9'ۿ>ק-ݶo?k6,w߷n<_{,@O|}zc<uҕpS [|s_ǇgK/=GrU7qƷ~~~eƃN8O>jum}ik_]y֣zη_xʩ'Oe7o|[:|kO8g}9e/{uw몛.w{9?i#̦4yҾ3d`Vdkㄙg_Xdq(ee_t ;bYeQgyT |
| --- | Minor | ]x=[wCyꈣD[ ]ރcف}/D^)@[ol.|@zs^:_lU7DnEuʙ#(-*Q{%^"Mpdu. KͺP~mMx2	c&%#9gmVLeby.њ~sPE=3i"[su+rw0g?=7C/y_m=ПW>ky)gs<r{èyglXY{H[uk/yci[`c{#oMW]nx4}9y'>?7wݖcg߁W>p%޷gm'1ݹN8g>;o}W}WO{S_w{VowϺ0O?c_/_}c_~׮_{¯r!{K^כ?w ߻{p;~蝗}\Ug{9'ۿ>ק-ݶo?k6. w߷n<_{. @O|}zc<uҕpS [|s_ǇgK/=GrU7qƷ~~~eƃN8O>jum}ik_]y֣zη_xʩ'Oe7o|[:|kO8g}9e/{uw몛.w{9?i#̦4yҾ3d`Vdkㄙg_Xdq(ee_t ;bYeQgyT. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7739, 83 words, 2 clauses) |
| --- | Minor | +Ije#'BDcwa0",d|f_CC瀈N:;f{rr$FNLQhY1̨zrQ#InJx[C)zkp1ٺ-(heL :<y_QyZG4bŅ}@~%S#VTYB]d469kUjOPwXf&x:7`=6YBHSak)uQ>M\8 !AHAiHQD8ɀg[lR)A\9y:5"=KBJq{  V/Ԋ@>UQcso@#*aE+'gR!@ĈS=Y`UQL֘6{իU3S)kll3H¦͸A܍ul93>i7``CUABCmQ9O[_]0!TIlYl!Vv<Zyu gm1f	<ԙѮј_ߵ`AGR5,QW9 /G<M*{iKZTf_*WOe5} |
| --- | Minor | +Ije#'BDcwa0". d|f_CC瀈N:;f{rr$FNLQhY1̨zrQ#InJx[C)zkp1ٺ-(heL :<y_QyZG4bŅ}@~%S#VTYB]d469kUjOPwXf&x:7`=6YBHSak)uQ>M\8 !AHAiHQD8ɀg[lR)A\9y:5"=KBJq{  V/Ԋ@>UQcso@#*aE+'gR!@ĈS=Y`UQL֘6{իU3S)kll3H¦͸A܍ul93>i7``CUABCmQ9O[_]0!TIlYl!Vv<Zyu gm1f	<ԙѮј_ߵ`AGR5. QW9 /G<M*{iKZTf_*WOe5}. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7750, 66 words, 1 clauses) |
| --- | Minor | G%%6+г=\AsG-X NJ8M] EKvWKjiMkÞҪCZ.Sb<)1X*1BaEE+IXkEJlx^H!bKTn(l =]qCxynR5$hS?176s2n}!FAC"w+^5ֶJ5.l!YI8B+Y(CXZs*lo3e6Ri@F~	]?|">@#К#ZKyd#G	Ѿ]P_H[	hMh7v*E3SY`gq4φ?b"&g+hpҪ1n|/Ě"C,g8\V3.LacbԶ!73i5# }- G |
| --- | Minor | G%%6+г=\AsG-X NJ8M] EKvWKjiMkÞҪCZ.Sb<)1X*1BaEE+IXkEJlx^H!bKTn(l =]qCxynR5$hS?176s2n}!FAC"w+^5ֶJ5.l!YI8B+Y(CXZs*lo3e6Ri@F~	]?|">@#К#ZKyd#G	Ѿ]P_H[	hMh7v*E3SY`gq4φ?b"&g+hpҪ1n|/Ě"C. g8\V3.LacbԶ!73i5# }- G. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7774, 79 words, 1 clauses) |
| --- | Minor | XIL%;h(PQҐWZdb[k\*u/?pC<HvH0IuMwYT`F󧍎6-9I>2@E		ŽkÍVp#㝃zhc0^	'n]j~h0a5Ԅ*3۩ZS:jFF%fQ4Bjsd/EKa_V:6^HK\;\ю觡h8=,8fo&CkZ+I@Gi.J`ؗKh Jrvں>/24nc8\XD%KjFOk~A0w.Kqk]\} 3ri.7:{[%Q6YRhye~b8{p!#9ny\C3Wj]kX-@gXKqȭu$%ª<l 	ʨ`ܡd2~ˆ4ss(|US fQubsv2a@i	lWqS55RCj8]COlbBfU |
| --- | Minor | XIL%;h(PQҐWZdb[k\*u/?pC<HvH0IuMwYT`F󧍎6-9I>2@E		ŽkÍVp#㝃zhc0^	'n]j~h0a5Ԅ*3۩ZS:jFF%fQ4Bjsd/EKa_V:6^HK\;\ю觡h8=. 8fo&CkZ+I@Gi.J`ؗKh Jrvں>/24nc8\XD%KjFOk~A0w.Kqk]\} 3ri.7:{[%Q6YRhye~b8{p!#9ny\C3Wj]kX-@gXKqȭu$%ª<l 	ʨ`ܡd2~ˆ4ss(|US fQubsv2a@i	lWqS55RCj8]COlbBfU. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7832, 86 words, 6 clauses) |
| --- | Minor | bRJР1% (iN!䆓Fשs7RYX|: Z.a	A(=cx =Kf`o3*49D)ZŊ9-C1l@`sA~|Efz}?E5",cA"Vȍ=IRALEk4BB6EK&sAE,)im̻БVa"J|,k<VUQXUnd)mɳfZC5Yȳ2 )E?gWE51E`=h,g<E-ښ;^JQV_1jS2PflU/U#܆,I{A|_	wrWXAo f-o\ٰ ÑpFRH8@@8;zw\sV[6.!E,TX%+Xr1+|IU&f\XtK<UՂDڥͶT߳C9bm!8F5QMAdI.P;*C2-.T;b$%Q>JO4⷟RjM4':̇ |
| --- | Minor | bRJР1% (iN!䆓Fשs7RYX|: Z.a	A(=cx =Kf`o3*49D)ZŊ9-C1l@`sA~|Efz}?E5". cA"Vȍ=IRALEk4BB6EK&sAE. )im̻БVa"J|. k<VUQXUnd)mɳfZC5Yȳ2 )E?gWE51E`=h. g<E-ښ;^JQV_1jS2PflU/U#܆. I{A|_	wrWXAo f-o\ٰ ÑpFRH8@@8;zw\sV[6.!E. TX%+Xr1+|IU&f\XtK<UՂDڥͶT߳C9bm!8F5QMAdI.P;*C2-.T;b$%Q>JO4⷟RjM4':̇. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7852, 66 words, 2 clauses) |
| --- | Minor | 6vlŏ:*[#tX#,hr_Pa!Ir4,x?cKWWw~G{?uk=Ļ Yƭ5/^qwn߾?7o{gnS_r7u+}S HN$ϋsE䥻vefݺ%m "ý!5D/Cei<zr| եHDdθ8WM7m|Ño*<&Ԛ-i@mL6BV\40<1éb-+m6d<(+[_n';wnkw<iyɋ?<|>W:mag}ʋ>폝}wskh]׵skyc@k a8AoH78sT>b|x?O3Sbcj^j[eE2+Di <d%L| |
| --- | Minor | 6vlŏ:*[#tX#. hr_Pa!Ir4. x?cKWWw~G{?uk=Ļ Yƭ5/^qwn߾?7o{gnS_r7u+}S HN$ϋsE䥻vefݺ%m "ý!5D/Cei<zr| եHDdθ8WM7m|Ño*<&Ԛ-i@mL6BV\40<1éb-+m6d<(+[_n';wnkw<iyɋ?<|>W:mag}ʋ>폝}wskh]׵skyc@k a8AoH78sT>b|x?O3Sbcj^j[eE2+Di <d%L|. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7884, 74 words, 5 clauses) |
| --- | Minor | p~MAG]96	1}0J^вS(˨4jLh4n۝&rX^,AE	;l̕Δ(aފ1bs;&Xa.ypƜyx,(tliĠa1g(t&e[ږ m9-˩-K[O~}'|K>ڲ٪ĭ	|iiyv0]?7xt}z;B0#eA,h~U"P0&>xh		A8MuZ:YNb klO(2E%im,E-,DOS֏tPPq ?eiɓamrVBx}ٶ돀 gslƾñ]x9NNWՠ[nmk=߻w}2_fsC\]J	|^y ^:-(Hy<FeyWJ[۞HYvtv]3{wwz??̧|<t |
| --- | Minor | p~MAG]96	1}0J^вS(˨4jLh4n۝&rX^. AE	;l̕Δ(aފ1bs;&Xa.ypƜyx. (tliĠa1g(t&e[ږ m9-˩-K[O~}'|K>ڲ٪ĭ	|iiyv0]?7xt}z;B0#eA. h~U"P0&>xh		A8MuZ:YNb klO(2E%im. E-. DOS֏tPPq ?eiɓamrVBx}ٶ돀 gslƾñ]x9NNWՠ[nmk=߻w}2_fsC\]J	|^y ^:-(Hy<FeyWJ[۞HYvtv]3{wwz??̧|<t. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7916, 38 words, 4 clauses) |
| --- | Minor | *,|}5&XPkVjv@Uޙ: `RY.pi&Nh.tOP<&@8K&r^AMbrof}An,7X\.Ĕڨoƞ>`}Ftbٶ5 nw6_=r23ƠOlۛjv?}w;FAs2ZTqUcF:kfF,erT	32K>>j&0iÙ,HC{p*"(՟90LS@͇w[!o |
| --- | Minor | *. |}5&XPkVjv@Uޙ: `RY.pi&Nh.tOP<&@8K&r^AMbrof}An. 7X\.Ĕڨoƞ>`}Ftbٶ5 nw6_=r23ƠOlۛjv?}w;FAs2ZTqUcF:kfF. erT	32K>>j&0iÙ. HC{p*"(՟90LS@͇w[!o. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 7997, 84 words, 4 clauses) |
| --- | Minor | k0N"R<|xfͼ[n*#nˍܑBM8d=컺[!6(zrf,h؟,`F'Z1=;	V;3hOp^lc/	NJ勗	HHSYɡgyCnֵqnJv8=%>[}VԧT(6Ig:cF s Tis9^3RϜ,+:(jc0؝j޻۳i}XWTFd{Mk+ RMOC&Zqp>d0c_6>P-6L` .X31ZF}B<n5мs+T.i-gV>l̽nܔ.H%27TTQW7r)NSꄵwLzJEg'}gqy8CG4tLw@ѐr^|^qAxY撻K}sgI[Wꔩt`0L[,I.7ӔYT];ٶs}	IxX+&mJgv.z-!]fYOm |
| --- | Minor | k0N"R<|xfͼ[n*#nˍܑBM8d=컺[!6(zrf. h؟. `F'Z1=;	V;3hOp^lc/	NJ勗	HHSYɡgyCnֵqnJv8=%>[}VԧT(6Ig:cF s Tis9^3RϜ. +:(jc0؝j޻۳i}XWTFd{Mk+ RMOC&Zqp>d0c_6>P-6L` .X31ZF}B<n5мs+T.i-gV>l̽nܔ.H%27TTQW7r)NSꄵwLzJEg'}gqy8CG4tLw@ѐr^|^qAxY撻K}sgI[Wꔩt`0L[. I.7ӔYT];ٶs}	IxX+&mJgv.z-!]fYOm. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8020, 69 words, 1 clauses) |
| --- | Minor | iW0k TF7ㄽdzK⢛7K"K|ж]MK!N=ۻ5#KJHfPxQ/jAQ½(a7WqI%<spGF}EDQ?SS?lWwU}/1)aJy󩺔l`36vZqxdj)E^{38g:4Yxp0E0N7t%s9T9z<8?ݟ)0PiA--Jva!@rDx[~,~C*d;璿D2M0_n>AyHiL`nI{Մ'\13]5E@eU&4BBbJԇ{dh!TovLi3TCnɱL\`ą<-Fn:88$R'CaՌ2O^*!T |
| --- | Minor | iW0k TF7ㄽdzK⢛7K"K|ж]MK!N=ۻ5#KJHfPxQ/jAQ½(a7WqI%<spGF}EDQ?SS?lWwU}/1)aJy󩺔l`36vZqxdj)E^{38g:4Yxp0E0N7t%s9T9z<8?ݟ)0PiA--Jva!@rDx[~. ~C*d;璿D2M0_n>AyHiL`nI{Մ'\13]5E@eU&4BBbJԇ{dh!TovLi3TCnɱL\`ą<-Fn:88$R'CaՌ2O^*!T. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8305, 228 words, 3 clauses) |
| --- | Minor | /ColorSpace [/Indexed/DeviceRGB 253 (\375\347%!\220\215\373\347#\037\225\213B@\206;Q\213\361\345\035D\002V\357\345\034\354\345\033:S\213\352\345\032H!s%\203\216\037\241\210>J\211\337\343\030E\004W\335\343\030 \223\214\332\343\031G\021dH#t\320\341\034\315\341\035\312\341\037H&w\302\337#\300\337%\275\337& \222\214AB\207G\020c\262\335-C>\205\260\335/%\204\216G\016a\250\3334H\027i\245\3336%\205\216\235\331;\233\331<F\007Z@E\210\223\327A\220\327CE5\201!\217\215\211\325HH\)yH \206\216\037\224\214\201\323M\323N2d\216:T\214z\321QF4\200w\321SF2~p\317Wl\315Zi\315[H%ve\313^c\313_,s\216^\311b*v\216F0~1g\216X\307eT\305hR\305iN\303k1f\216J\301mH\301nG/}F3E7\201D\277p\037\226\213BA\206@\275r?G\210>I\211=M\212<O\212;\273u\037\227\2139U\2148\271w7[\2156]\2155\267y4a\2153c\2152e\2161\265{0i\216/k\216.\263|-q\216,\261~+u\216*w\216\)\257\(}\216'\255\201&\255\201%\253\202 \252\203\037\237\210"\250\204!\246\205 \244\206\037\242\207\036\234\211!\221\214G\023e&\201\216,q\216H qE\005Y-p\216F\013^\037\240\2104`\215<P\213D9\203)] |
| --- | Minor | /ColorSpace [/Indexed/DeviceRGB 253 (\375\347%!\220\215\373\347#\037\225\213B@\206;Q\213\361\345\035D\002V\357\345\034\354\345\033:S\213\352\345\032H!s%\203\216\037\241\210>J\211\337\343\030E\004W\335\343\030 \223\214\332\343\031G\021dH#t\320\341\034\315\341\035\312\341\037H&w\302\337#\300\337%\275\337& \222\214AB\207G\020c\262\335-C>\205\260\335/%\204\216G\016a\250\3334H\027i\245\3336%\205\216\235\331;\233\331<F\007Z@E\210\223\327A\220\327CE5\201!\217\215\211\325HH\)yH \206\216\037\224\214\201\323M\323N2d\216:T\214z\321QF4\200w\321SF2~p\317Wl\315Zi\315[H%ve\313^c\313_. s\216^\311b*v\216F0~1g\216X\307eT\305hR\305iN\303k1f\216J\301mH\301nG/}F3E7\201D\277p\037\226\213BA\206@\275r?G\210>I\211=M\212<O\212;\273u\037\227\2139U\2148\271w7[\2156]\2155\267y4a\2153c\2152e\2161\265{0i\216/k\216.\263|-q\216. \261~+u\216*w\216\)\257\(}\216'\255\201&\255\201%\253\202 \252\203\037\237\210"\250\204!\246\205 \244\206\037\242\207\036\234\211!\221\214G\023e&\201\216. q\216H qE\005Y-p\216F\013^\037\240\2104`\215<P\213D9\203)]. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8375, 73 words, 0 clauses) |
| --- | Minor | 0A9sfMq5|F?hu:ENQS)u:ENQS)u:ENQS)u:ENQS)uʵ?NQS)u:ENQS)u:ENQS)u:ENQS)u:ENٵxzu:ENQS)u:ENQS)u:ENQS)u:ENQS)u:aBS)u:ENQS)u:ENQS)u:ENQS)u:ENQS)-&n\ϹNQS)u:ENQS)u:ENQS)u:ENQS)u:ENQ<p78i[v:ENQS)u:ENQS)u:ENQS)u:ENQS)u:EdZv:ENQS)u:ENQS)u:ENQS)u:ENQS)u:          k9 |
| --- | Minor | 0A9sfMq5|F?hu:ENQS)u:ENQS)u:ENQS)u:ENQS)uʵ?NQS)u:ENQS)u:ENQS)u:ENQS)u:ENٵxzu:ENQS)u:ENQS)u:ENQS)u:ENQS)u:aBS)u:ENQS)u:ENQS)u:ENQS)u:ENQS)-&n\ϹNQS)u:ENQS)u:ENQS)u:ENQS)u:ENQ<p78i[v:ENQS)u:ENQS)u:ENQS)u:ENQS)u:EdZv:ENQS)u:ENQS)u:ENQS)u:ENQS)u:          k9 |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8422, 70 words, 1 clauses) |
| --- | Minor | úo-~u;{}wnjWúKbI?WJ[Ox{ͼ:v~8YҒ 6z\vgѫϋFϋi:4unIh"-U;NgiRO :;B-eԳúnۼkiZDYgݟM?tߟ^wv3Čhru?ѣ6E~=:8;Ci=ӎ?xCvebVyQ+!e7z}cXZsA+![˅%\<)rٕ47lQÐ-t4^C"rd I9.ڪ*5ir8@2(U ne'$:GGe3W55N"3kQ;3WnG\4MFj(,nae'uQ穝MA"jZ܀8RrjF%FJ׫K8 |
| --- | Minor | úo-~u;{}wnjWúKbI?WJ[Ox{ͼ:v~8YҒ 6z\vgѫϋFϋi:4unIh"-U;NgiRO :;B-eԳúnۼkiZDYgݟM?tߟ^wv3Čhru?ѣ6E~=:8;Ci=ӎ?xCvebVyQ+!e7z}cXZsA+![˅%\<)rٕ47lQÐ-t4^C"rd I9.ڪ*5ir8@2(U ne'$:GGe3W55N"3kQ;3WnG\4MFj(. nae'uQ穝MA"jZ܀8RrjF%FJ׫K8. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8581, 67 words, 3 clauses) |
| --- | Minor | gܬv.$lb	3=o*: ΄"jU4Ma>YziQT(_mLr#)]IYy9ƿ]UO/|\/,WVT=z'&Y@I9۱5{Ŭ0lͶ]~kG?P>oS[T:^_E+ "A-9ڭ3Y&,gG^ubm>z3^yA,txD{.{g3HELM<sw&A91zt C6/O˛=GJ=3``VaqI'L;lɃr&OXsm2Z#oI~Rqu^\LP9izCfe/y{hlU_ƅx*vi}>ލ*[-Seu0nWZbKƓDxs" I0âF2.vs͹/JPدf	i3_FЩ{N |
| --- | Minor | gܬv.$lb	3=o*: ΄"jU4Ma>YziQT(_mLr#)]IYy9ƿ]UO/|\/. WVT=z'&Y@I9۱5{Ŭ0lͶ]~kG?P>oS[T:^_E+ "A-9ڭ3Y&. gG^ubm>z3^yA. txD{.{g3HELM<sw&A91zt C6/O˛=GJ=3``VaqI'L;lɃr&OXsm2Z#oI~Rqu^\LP9izCfe/y{hlU_ƅx*vi}>ލ*[-Seu0nWZbKƓDxs" I0âF2.vs͹/JPدf	i3_FЩ{N. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8630, 69 words, 0 clauses) |
| --- | Minor | B~0]7]_K:	Kh^OgkZNZiXi3-)#G%[xzIj-Vʘ^ݴV7c>:a-	fdJnŮ<hFjZI%8`z\>K^=%䃴vɥ;ZervOC]Cod>UuЭO}_8MolΏ0\3(٨~oCLܺOg%ǲb]9YB5r	[kٛLC7Y 0z/#JMF .D5ߵeoZ5?kOٌ8uq1ح3v|GNjڻH{hXZY0f\Jmj]_]OL&&Su_*6fB<+ojUQN;gl\zǛmook6~ujSK(|I"/OI4bLjWہm+c_aRVc}0Ө=/-LR7t_Z)2m{)1ouZ2~ |
| --- | Minor | B~0]7]_K:	Kh^OgkZNZiXi3-)#G%[xzIj-Vʘ^ݴV7c>:a-	fdJnŮ<hFjZI%8`z\>K^=%䃴vɥ;ZervOC]Cod>UuЭO}_8MolΏ0\3(٨~oCLܺOg%ǲb]9YB5r	[kٛLC7Y 0z/#JMF .D5ߵeoZ5?kOٌ8uq1ح3v|GNjڻH{hXZY0f\Jmj]_]OL&&Su_*6fB<+ojUQN;gl\zǛmook6~ujSK(|I"/OI4bLjWہm+c_aRVc}0Ө=/-LR7t_Z)2m{)1ouZ2~ |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8696, 66 words, 3 clauses) |
| --- | Minor |  AJi^b\gDoܓ'/rVU(S sfkEn&{_dXz;~|x0(;kI_>Xnwn}ȵ/Z1Lq{k'I0+w4ǙL#s?#P83@%t``e&͚̾B[l`_Sz#M`Li;9(ujP7{"r*xBNk}ǅ ;<O)m[Ǒ{+;bqnSTbk	Š&kj,4C'A6#&=ttF6~bWf'#MF*Kv2eG[X,Y?qliS(NFĎK[=AƸvǽ9,7qGduQ卉1zy kN#5;( ?ځP%*Qy֋}Tγ><Y |
| --- | Minor |  AJi^b\gDoܓ'/rVU(S sfkEn&{_dXz;~|x0(;kI_>Xnwn}ȵ/Z1Lq{k'I0+w4ǙL#s?#P83@%t``e&͚̾B[l`_Sz#M`Li;9(ujP7{"r*xBNk}ǅ ;<O)m[Ǒ{+;bqnSTbk	Š&kj. 4C'A6#&=ttF6~bWf'#MF*Kv2eG[X. Y?qliS(NFĎK[=AƸvǽ9. 7qGduQ卉1zy kN#5;( ?ځP%*Qy֋}Tγ><Y. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8713, 78 words, 2 clauses) |
| --- | Minor | a]'><*N'~w͒ǥ`cܩIzzIȅi9<VWvA6YgU]t94dU{q?AY47(lͣfHY:YfKDA!v`K|1C 3(	nhuS\h,첯s)c=i9.Si5n| 6yMk߁q&}>-H4HE_B$JyYՇ:f]}#]Rm\qBk+wx]ysj&%ѽE`LG*mb}yԛúN@Ĕ(^TCfI^ {Zֈ܃Ql knr>_۝SQA8e}6YT22*o_}y"fL.b{'7dD'sRcσEIaWARqi+ܤǌJP9P@8[4=u*Tє,R&05Ջa;"WZ'j. |
| --- | Minor | a]'><*N'~w͒ǥ`cܩIzzIȅi9<VWvA6YgU]t94dU{q?AY47(lͣfHY:YfKDA!v`K|1C 3(	nhuS\h. 첯s)c=i9.Si5n| 6yMk߁q&}>-H4HE_B$JyYՇ:f]}#]Rm\qBk+wx]ysj&%ѽE`LG*mb}yԛúN@Ĕ(^TCfI^ {Zֈ܃Ql knr>_۝SQA8e}6YT22*o_}y"fL.b{'7dD'sRcσEIaWARqi+ܤǌJP9P@8[4=u*Tє. R&05Ջa;"WZ'j.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8749, 119 words, 4 clauses) |
| --- | Minor | ARIFV7/Z%Qz1*߭?0VH ޼~譅P #R>ЀtWixϠaᬺC˚IXyA-N%Rd6u(r3f^ Ht(JW6uX?V"3<jD0,M%Lޭvk!GHkD(;vQ&Q[em=V;6m]Q/ΡxԧP~ye0	-M֭PψZ)ofmԦׇ9Ly D?Ůۺc"#5R۟NG7BJ	cUEyiu:5[l2sYNp1`b?0,Cp#gJm]7"sewͳO<1!Rl6+V/QH"zoOw% ^5xJ0c~ ,8FH@hYBB;D4U'\:&I*,.:&I~h"<9SLC4cRw2@_*zT#`Q`֓qؔZ頸3M"M@1DˀIi LHt"ŜDvX4*	AJ4bG]g~<_ 鱂%xxnhF&8׵YUǜ1ZqWTZA$bgt۩TBCA'@z수eS279n>OFӄTf1AoWZv4z:^PSo5͆^ |
| --- | Minor | ARIFV7/Z%Qz1*߭?0VH ޼~譅P #R>ЀtWixϠaᬺC˚IXyA-N%Rd6u(r3f^ Ht(JW6uX?V"3<jD0. M%Lޭvk!GHkD(;vQ&Q[em=V;6m]Q/ΡxԧP~ye0	-M֭PψZ)ofmԦׇ9Ly D?Ůۺc"#5R۟NG7BJ	cUEyiu:5[l2sYNp1`b?0. Cp#gJm]7"sewͳO<1!Rl6+V/QH"zoOw% ^5xJ0c~ . 8FH@hYBB;D4U'\:&I*. .:&I~h"<9SLC4cRw2@_*zT#`Q`֓qؔZ頸3M"M@1DˀIi LHt"ŜDvX4*	AJ4bG]g~<_ 鱂%xxnhF&8׵YUǜ1ZqWTZA$bgt۩TBCA'@z수eS279n>OFӄTf1AoWZv4z:^PSo5͆^. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8852, 57 words, 7 clauses) |
| --- | Minor | xڭZv8}ۡ֊iMq{|Iglt: S"58Ql J΋%RTծ]Gˣ`,r?Oxrtp ;K+,L<+Q	r.NVm)*QSg-P4UCkTM^Q/l{Q+IK}v˶֐zwCJ5	㍸3&bGUmA5Zv_`?,h!Z{%6=l#,]/m/," ,YB3F]F;<繟a63?Jrdo7m/kjVOr#OXRѴ4u&,E']mZuf[+-7ʽwDEX6T;Y¨jl(|h!N3=مgy]˶= Z |
| --- | Minor | xڭZv8}ۡ֊iMq{|Iglt: S"58Ql J΋%RTծ]Gˣ`. r?Oxrtp ;K+. L<+Q	r.NVm)*QSg-P4UCkTM^Q/l{Q+IK}v˶֐zwCJ5	㍸3&bGUmA5Zv_`?. h!Z{%6=l#. ]/m/. " . YB3F]F;<繟a63?Jrdo7m/kjVOr#OXRѴ4u&. E']mZuf[+-7ʽwDEX6T;Y¨jl(|h!N3=مgy]˶= Z. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8863, 24 words, 4 clauses) |
| --- | Minor | "-ySKK0~Ӕ4P:ӷnDV	NgRht[ozv,ᓹ~8)*ĂV''d(BrP(ss+31d>{2 ?!܈#<<AzavE,\8~_¹Yq9,9"#^,JML']) |
| --- | Minor | "-ySKK0~Ӕ4P:ӷnDV	NgRht[ozv. ᓹ~8)*ĂV''d(BrP(ss+31d>{2 ?!܈#<<AzavE. \8~_¹Yq9. 9"#^. JML']). |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8936, 62 words, 2 clauses) |
| --- | Minor | r[>K?r#vuկ;xJ㞡tSUHj2)-ӄ26J0p~CjPODe6]=mN	Q0g^!WW8pHJْ̺C#kcSq44%Zqٛ@Wjkߋu&l:@HGGg<Vh8N	D,=7VkqGA![z!/īk;CsNrdz)韟vF;G3+m EVm*"hgcjLkK&E-lHON/l6M-otQ.3Nvj*!)q..ID<Y,=dXIEcD	֭ldށVw!z5%3CEv[HH%#M<. |
| --- | Minor | r[>K?r#vuկ;xJ㞡tSUHj2)-ӄ26J0p~CjPODe6]=mN	Q0g^!WW8pHJْ̺C#kcSq44%Zqٛ@Wjkߋu&l:@HGGg<Vh8N	D. =7VkqGA![z!/īk;CsNrdz)韟vF;G3+m EVm*"hgcjLkK&E-lHON/l6M-otQ.3Nvj*!)q..ID<Y. =dXIEcD	֭ldށVw!z5%3CEv[HH%#M<.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8942, 67 words, 1 clauses) |
| --- | Minor | ;3A&{YE&q'aJÉs84؍`9o}j?TL}19 .V5%BWǱTo?"|nMka4%fB?aI7PCCQ>|)	DBW&|FVߛ:tqiQ	}Uz.V'tp,0<P<►<Z`9cpr]7)dغNc]6Ep/J6}J+|J'Mf*oGaO^$ϩȋ8]ޮ<övYۖWD腴>wQ`nWuL'9q%"Ĉd:NH7㖿La]zMN-S'Ϭ9wˌyP)O֙Qgjyf Iqyo)Xںp~b;@"ه}hj^rkOn0QݶZؕ	ݘBf<? |
| --- | Minor | ;3A&{YE&q'aJÉs84؍`9o}j?TL}19 .V5%BWǱTo?"|nMka4%fB?aI7PCCQ>|)	DBW&|FVߛ:tqiQ	}Uz.V'tp. 0<P<►<Z`9cpr]7)dغNc]6Ep/J6}J+|J'Mf*oGaO^$ϩȋ8]ޮ<övYۖWD腴>wQ`nWuL'9q%"Ĉd:NH7㖿La]zMN-S'Ϭ9wˌyP)O֙Qgjyf Iqyo)Xںp~b;@"ه}hj^rkOn0QݶZؕ	ݘBf<?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 8966, 76 words, 4 clauses) |
| --- | Minor | cIK+,_lFr	L3!/G1k'& X@P@_ 8PNg=c3k@5|_Msnf5COwvl Sz2[5Bd>38CBUуL+l:N=(|w#*N}]M@Q׻:zhsNDmb+>I*d9:Z,!=ihD	y z-DEn ];gv yc%H x K၁<	F6ɫN5cwZa&Q#?BXOL=j ~/lx~TЌ97It/04?bTd050m&8}k?7٦}*Ika,S}bVdi1!{ll",6fF4 |
| --- | Minor | cIK+. _lFr	L3!/G1k'& X@P@_ 8PNg=c3k@5|_Msnf5COwvl Sz2[5Bd>38CBUуL+l:N=(|w#*N}]M@Q׻:zhsNDmb+>I*d9:Z. !=ihD	y z-DEn ];gv yc%H x K၁<	F6ɫN5cwZa&Q#?BXOL=j ~/lx~TЌ97It/04?bTd050m&8}k?7٦}*Ika. S}bVdi1!{ll". 6fF4. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9126, 51 words, 4 clauses) |
| --- | Minor | 24Wy2Ca\f/3_,Z#HosHks]d[8NH}koW!{qX>@ڹгCV+SKwBHi,L">H"O&Mt\نsE4ljd&EZv/:hAgshNK/OW/>sUiXH)i&PQFGKݴQ<	{ѱB,ae(խq˨~`튋Z	O{f;ZI<yiS7,JD$Ǣ+bztC`Fnݷ>0C}t |
| --- | Minor | 24Wy2Ca\f/3_. Z#HosHks]d[8NH}koW!{qX>@ڹгCV+SKwBHi. L">H"O&Mt\نsE4ljd&EZv/:hAgshNK/OW/>sUiXH)i&PQFGKݴQ<	{ѱB. ae(խq˨~`튋Z	O{f;ZI<yiS7. JD$Ǣ+bztC`Fnݷ>0C}t. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9160, 63 words, 1 clauses) |
| --- | Minor | ޘ4 QID0=<db߈b/kj߮]KY[0i:}GP"M! Sh>Ӝ+bm!O^hR)MH`2N!^pLJ+gR\*76y;%i;7OKضS+9}u@]^zp:gJ&669NOD!gb^Vl`toDiSK䌺JzNyNq+	>XS2|n%huw>pwpڜmΩ)ed/발;'{Qy{A<a8W6BǉD~dcmaޅŞu,YTL)79=7{lj`sX>myX..)27? |
| --- | Minor | ޘ4 QID0=<db߈b/kj߮]KY[0i:}GP"M! Sh>Ӝ+bm!O^hR)MH`2N!^pLJ+gR\*76y;%i;7OKضS+9}u@]^zp:gJ&669NOD!gb^Vl`toDiSK䌺JzNyNq+	>XS2|n%huw>pwpڜmΩ)ed/발;'{Qy{A<a8W6BǉD~dcmaޅŞu. YTL)79=7{lj`sX>myX..)27?. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9188, 82 words, 3 clauses) |
| --- | Minor | P|_nݏ<8(]	΁sP鄣34b䣓TO&dӃtAŦEL&s!HnE?vAB,1J68G6@qL*AUHi^UlY1o#mz{ʽkX%RXiF	W(M~.zg6h3k7l.fC[@6M@GGx&k9:3VhSǼ+p$	*6u6U_vR,HGw;cӕsosu6{D5G`wIO۲ Ұ;Qcm\šmvǭ(lPHPdC5jWP-*7m4ޑ3&Ob%tģl`dp@r}4RQN~NF/._r%^/SOO0?6\q#u^m`m>iƞ~Tk9kG5L{`MfDY!3ׂ/8JZ(1 r,@#śoB&o[Kj"a9 |
| --- | Minor | P|_nݏ<8(]	΁sP鄣34b䣓TO&dӃtAŦEL&s!HnE?vAB. 1J68G6@qL*AUHi^UlY1o#mz{ʽkX%RXiF	W(M~.zg6h3k7l.fC[@6M@GGx&k9:3VhSǼ+p$	*6u6U_vR. HGw;cӕsosu6{D5G`wIO۲ Ұ;Qcm\šmvǭ(lPHPdC5jWP-*7m4ޑ3&Ob%tģl`dp@r}4RQN~NF/._r%^/SOO0?6\q#u^m`m>iƞ~Tk9kG5L{`MfDY!3ׂ/8JZ(1 r. @#śoB&o[Kj"a9. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9278, 58 words, 4 clauses) |
| --- | Minor | bD@P]>ۙw07s㜕ʟ#m,HI7o?Lc02!UiYy=N6h]qW!˼~S|۔h%hN3<qNY/}5ؿ$Sݓ&ޓ ]X0oݙ/M_=/TKX(7 Kl)ߝpS4ds؉e5_K<D,.1sTاWpfN^gggǇa^НX,nM(t#D9ݛ׶FvZx~NƳ.)JBh؅h	+2	NKp~G|]|a[.2qVgG|!SF a,`+wlEC F͙ |
| --- | Minor | bD@P]>ۙw07s㜕ʟ#m. HI7o?Lc02!UiYy=N6h]qW!˼~S|۔h%hN3<qNY/}5ؿ$Sݓ&ޓ ]X0oݙ/M_=/TKX(7 Kl)ߝpS4ds؉e5_K<D. .1sTاWpfN^gggǇa^НX. nM(t#D9ݛ׶FvZx~NƳ.)JBh؅h	+2	NKp~G|]|a[.2qVgG|!SF a. `+wlEC F͙. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9435, 20 words, 4 clauses) |
| --- | Minor | gZ09I?[	P0%L=^yA,˰ǩq%SZ;AA2?oˤaQ vHe9CdTjqнaٙ,<ysnδi97W6mxδ<NPZ<AU,x, hݿ |
| --- | Minor | gZ09I?[	P0%L=^yA. ˰ǩq%SZ;AA2?oˤaQ vHe9CdTjqнaٙ. <ysnδi97W6mxδ<NPZ<AU. x.  hݿ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9447, 72 words, 1 clauses) |
| --- | Minor | &To7TFdyd5(26H@o}W#/⢰a;\쵐hL0͜;LsARlIJ0=x8Yq2)lfa3 CU 8k&Ɛ{m`'IKIwF;G,9=L"|jO94rpBS7bbF'eTX;ui]60S:xuDhBZؠ=tS/m2Y&2+s2/(e)GrIF2EchR@]/XYw-Eg}hw<r੶YEZ_k^=\v3ФUMIhQ0ބrKb#d!Ixj%.$pZ\*\dVV^d7+EV5Քj;;-6/np.5a<ҼkWsK뮗^1&ڗ_j8g֙^Sz/͔ɀy06	?my{   Et{0Ш!V	1<3Ir~1PN%K묕}. |
| --- | Minor | &To7TFdyd5(26H@o}W#/⢰a;\쵐hL0͜;LsARlIJ0=x8Yq2)lfa3 CU 8k&Ɛ{m`'IKIwF;G. 9=L"|jO94rpBS7bbF'eTX;ui]60S:xuDhBZؠ=tS/m2Y&2+s2/(e)GrIF2EchR@]/XYw-Eg}hw<r੶YEZ_k^=\v3ФUMIhQ0ބrKb#d!Ixj%.$pZ\*\dVV^d7+EV5Քj;;-6/np.5a<ҼkWsK뮗^1&ڗ_j8g֙^Sz/͔ɀy06	?my{   Et{0Ш!V	1<3Ir~1PN%K묕}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9455, 61 words, 8 clauses) |
| --- | Minor | h3	Dq!7B{	Te,>w%]_O7?g.=kw{Hϟ<._DKʌ&#&-fX,f-e䰔-07*&#˱cZ0uj&a!AjMF6^2#Gq 0diN^5N3t+pf`,&5z,IL12i̲l,3gZ,w,N!(IQ0`^(;]una"nK2Jio2Ieɖ,f63'Ix<3L扖h..fflqie<_ooo@帜)'bB+LerK\T[++z[!VVW̍FG`\K8S |
| --- | Minor | h3	Dq!7B{	Te. >w%]_O7?g.=kw{Hϟ<._DKʌ&#&-fX. f-e䰔-07*&#˱cZ0uj&a!AjMF6^2#Gq 0diN^5N3t+pf`. &5z. IL12i̲l. 3gZ. w. N!(IQ0`^(;]una"nK2Jio2Ieɖ. f63'Ix<3L扖h..fflqie<_ooo@帜)'bB+LerK\T[++z[!VVW̍FG`\K8S. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9506, 133 words, 4 clauses) |
| --- | Minor | F&CJS\gZ:]Rj /IA#UyAyN̜l#bE#i?w7)?x#',բFnNi2KD_؃H,zvOfG%dmt.Ci/ѐ4Gg7jTǝՉCQ\dⰯC=`y{U0ߢɴ- Y4B+\koB}h#Q=Ɂ]Ϣ8e΅MLs1=ѓAx^Q(<czA[fI&l,=~s/Fz^cRaq~Lnn12ԲL1+B{X8o%աfyjڠG_睲3ӰlLfo3~-_ߎЯX"t\ތp~AL\g9(!\}ơZ5QZ*QB@pfBi!ЦCh)Ih1h	^4XzrˠCF(B+"h FYBQrhƭv^_]83T.V+pV+ Oτc`kBwdW|÷;ig%"P24	`GA'#MA7iT~4fY lT[mL09cXl&Z	t+*XVǅp>>1<'|L87&<}<3C0>ɛyN^L^/#D򚙼zǽZFNȉy䕵䘑5#/9#_hEk9L:&yv-yF#O[9y*@,#OĐ}cey<jM" |
| --- | Minor | F&CJS\gZ:]Rj /IA#UyAyN̜l#bE#i?w7)?x#'. բFnNi2KD_؃H. zvOfG%dmt.Ci/ѐ4Gg7jTǝՉCQ\dⰯC=`y{U0ߢɴ- Y4B+\koB}h#Q=Ɂ]Ϣ8e΅MLs1=ѓAx^Q(<czA[fI&l. =~s/Fz^cRaq~Lnn12ԲL1+B{X8o%աfyjڠG_睲3ӰlLfo3~-_ߎЯX"t\ތp~AL\g9(!\}ơZ5QZ*QB@pfBi!ЦCh)Ih1h	^4XzrˠCF(B+"h FYBQrhƭv^_]83T.V+pV+ Oτc`kBwdW|÷;ig%"P24	`GA'#MA7iT~4fY lT[mL09cXl&Z	t+*XVǅp>>1<'|L87&<}<3C0>ɛyN^L^/#D򚙼zǽZFNȉy䕵䘑5#/9#_hEk9L:&yv-yF#O[9y*@. #OĐ}cey<jM". |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9620, 76 words, 2 clauses) |
| --- | Minor | b¼=tԉVv7ҘKcOx&g__o]]z^+p\$3G&GBjQ-=wCzЅVzM85*ms]gwGXOttmݜW'"M+e[]juEa#dGϾ#k"Rdwd59vM.0bTl8,3Nr,aj m0tLs7*úR/^d.fG~mgTJYirrNE36rn"ugY-9k*RV#Rה!ϲfH?9-YB]J\kr-Ktl:4Angk˚_~phJנivQn[t'@"dB<63Կ<<;Ԝ>SrW9=i̢6wIɹi؄MNi"&ː~Ԝ55dt32q.4	{{ڄl79oeY\aa(S9c8X)7ޅY5U!!T*^<0h*A |
| --- | Minor | b¼=tԉVv7ҘKcOx&g__o]]z^+p\$3G&GBjQ-=wCzЅVzM85*ms]gwGXOttmݜW'"M+e[]juEa#dGϾ#k"Rdwd59vM.0bTl8. 3Nr. aj m0tLs7*úR/^d.fG~mgTJYirrNE36rn"ugY-9k*RV#Rה!ϲfH?9-YB]J\kr-Ktl:4Angk˚_~phJנivQn[t'@"dB<63Կ<<;Ԝ>SrW9=i̢6wIɹi؄MNi"&ː~Ԝ55dt32q.4	{{ڄl79oeY\aa(S9c8X)7ޅY5U!!T*^<0h*A. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9621, 84 words, 3 clauses) |
| --- | Minor | *]tFNV{{?(9/v>sWZ*T"C}z"<Q8ReZ_\//!O#)wFNnxkq}3z\kDǨ悉i}4ļ>'➭%TA%ܝnJ1<Ә*f	7eLwlKKܙT򣘚B<ݴVmgz^},iϞٗq*Yڈ:"G)aF2[VrF%|췂%ɋX刪x.cᵖe"RwEN7gu!#u][B؅xT|Y>QXoՁ8> /aY	x'o+)oΓoqO7F7xc4N)Z*«Ar,Ј:{,~ǧFc/^~[(t|g|^3	NQgc3	8_)VxZ)'~+RPx"ovv˃8V?GXՃYcWYUhS#>āz>m{=hCCa<c= |
| --- | Minor | *]tFNV{{?(9/v>sWZ*T"C}z"<Q8ReZ_\//!O#)wFNnxkq}3z\kDǨ悉i}4ļ>'➭%TA%ܝnJ1<Ә*f	7eLwlKKܙT򣘚B<ݴVmgz^}. iϞٗq*Yڈ:"G)aF2[VrF%|췂%ɋX刪x.cᵖe"RwEN7gu!#u][B؅xT|Y>QXoՁ8> /aY	x'o+)oΓoqO7F7xc4N)Z*«Ar. Ј:{. ~ǧFc/^~[(t|g|^3	NQgc3	8_)VxZ)'~+RPx"ovv˃8V?GXՃYcWYUhS#>āz>m{=hCCa<c=. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9657, 124 words, 3 clauses) |
| --- | Minor | vt'4XX+঩Oќjj?s{7dNۯǫ?R'-{2頾fQM	hg@|:bF/FT8=h'/qaDPImkx c&z&Zzv؝G(fbNÛ1Uи를	ڵ.k?4ưzG_|Ջ{Ǐ鮂I\A#J!	0x}XIbz{!36]?a):XO1)]\m܀!%T!)qa!FK+VgRy_>yt8;hУU@So\|A`B.ػu^6>|Ӧn{z?9޼rxW#9Bߵk]λ.ի{qxы?LqxHpQܞRbk]*5w0Ѯ"QS$,PeQ,E&İ1\LtLL'&6&nm}XM90&#Gڍ]׺Y=~Y`\=-;wnٲsqUlQשԥ蛥sqz<ฌ<%(<GXa,ru0{4<1z@~K<|ZX a%յJ2Bܬpj1un>^:̢0F|.ѓDs::\9phoԂUQD(L7SC](phӧ1	Oϧ^:yt%:[TYɬzLİaئ`-E!oYHr@6DsXRwQ<NR[[&e\`zNG 'ebvQKh5[Å0k5͚dKf۲ |
| --- | Minor | vt'4XX+঩Oќjj?s{7dNۯǫ?R'-{2頾fQM	hg@|:bF/FT8=h'/qaDPImkx c&z&Zzv؝G(fbNÛ1Uи를	ڵ.k?4ưzG_|Ջ{Ǐ鮂I\A#J!	0x}XIbz{!36]?a):XO1)]\m܀!%T!)qa!FK+VgRy_>yt8;hУU@So\|A`B.ػu^6>|Ӧn{z?9޼rxW#9Bߵk]λ.ի{qxы?LqxHpQܞRbk]*5w0Ѯ"QS$. PeQ. E&İ1\LtLL'&6&nm}XM90&#Gڍ]׺Y=~Y`\=-;wnٲsqUlQשԥ蛥sqz<ฌ<%(<GXa. ru0{4<1z@~K<|ZX a%յJ2Bܬpj1un>^:̢0F|.ѓDs::\9phoԂUQD(L7SC](phӧ1	Oϧ^:yt%:[TYɬzLİaئ`-E!oYHr@6DsXRwQ<NR[[&e\`zNG 'ebvQKh5[Å0k5͚dKf۲. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9696, 69 words, 2 clauses) |
| --- | Minor | >i=+,EqW^8B#l7mw*M{-߽y]6"Zc8O}`Y sH~1\Y"D(֡pc*Pkdn'])3@:-hm,!x䘴\{7꘏qk-[}ꕠT_[Sx%=t;o*EH3mV)@@= [C6lWμéYC.ٜn>K3 &@{F2>MK&EڻǜX_.@[n2EUNS̠>Qt /gJKX_"2%d8.5Bה0Z9Ku~u5j0hK@ ߪAtwVu.|FηމvlLa#j1_zo>G@3To0OB:w}9үHi!m_AqQTY.-H 5A |
| --- | Minor | >i=+. EqW^8B#l7mw*M{-߽y]6"Zc8O}`Y sH~1\Y"D(֡pc*Pkdn'])3@:-hm. !x䘴\{7꘏qk-[}ꕠT_[Sx%=t;o*EH3mV)@@= [C6lWμéYC.ٜn>K3 &@{F2>MK&EڻǜX_.@[n2EUNS̠>Qt /gJKX_"2%d8.5Bה0Z9Ku~u5j0hK@ ߪAtwVu.|FηމvlLa#j1_zo>G@3To0OB:w}9үHi!m_AqQTY.-H 5A. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9704, 49 words, 5 clauses) |
| --- | Minor | [{M'*ٴtlZnܐm,"v!_gGǻ(:Y^SB~NpՍq=YUR­Rʹ22<= +LfQYn&ud,Ց%=ܒ[dHn, @&o& J.OOƕ{DF26@c#dF6.̕6;Bd"q#DB8H)ܰ	dh:q9vCtl+wOlᲭd2,ds*aIF@eH`t v?!IM֓KHRt.iK'>|*I "90.8e9GlN |
| --- | Minor | [{M'*ٴtlZnܐm. "v!_gGǻ(:Y^SB~NpՍq=YUR­Rʹ22<= +LfQYn&ud. Ց%=ܒ[dHn. @&o& J.OOƕ{DF26@c#dF6.̕6;Bd"q#DB8H)ܰ	dh:q9vCtl+wOlᲭd2. ds*aIF@eH`t v?!IM֓KHRt.iK'>|*I "90.8e9GlN. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9790, 68 words, 0 clauses) |
| --- | Minor | Ng/Gzr}3YxF'lbюMD6^O~ȮgU1q)3fMi"1BSy9:ۖr1} ʫb3?U:{wtvsX³լ˥#j	xW>VЌp%#X;$p0z/H)?%c@kPR؛˼ߍ(&ƚGňѶ?(2\e	jތqvp9X:NO5<l"7UQ0:dNTHMr}ؑL0}k/EE{N <h7:Wϳ_k/=	eQ[DDMW~u\n9 {'-R8cMjQD *5nd%L6zW8mu-N!ۤ˩ږ;2EJ\1RSX|?R	K`U>cK1l8̋mQ?%C|W5ot? |
| --- | Minor | Ng/Gzr}3YxF'lbюMD6^O~ȮgU1q)3fMi"1BSy9:ۖr1} ʫb3?U:{wtvsX³լ˥#j	xW>VЌp%#X;$p0z/H)?%c@kPR؛˼ߍ(&ƚGňѶ?(2\e	jތqvp9X:NO5<l"7UQ0:dNTHMr}ؑL0}k/EE{N <h7:Wϳ_k/=	eQ[DDMW~u\n9 {'-R8cMjQD *5nd%L6zW8mu-N!ۤ˩ږ;2EJ\1RSX|?R	K`U>cK1l8̋mQ?%C|W5ot? |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9958, 62 words, 0 clauses) |
| --- | Minor | |3V?M7|YR:'7˫܉4GE~|q+z|ddƤT_TIJc 	zE N)#­/jz1=3Xg2;!¿l*֏-B5~oո|zXԕ`?bTKtI2g)5	{OnʡG"s\#23KN`8}gctDK95 3y\.}Bp$p҉~i6ŕd19F)f[d_EܣP!9Ӝ{44't!04b.DcV#Bq{Y=vj^C8Y3F :jI}D6VzyUz/qZP%N>ťxXl0S놦n10Ӹ{zP@dꁖ~'a";F~p+ |
| --- | Minor | |3V?M7|YR:'7˫܉4GE~|q+z|ddƤT_TIJc 	zE N)#­/jz1=3Xg2;!¿l*֏-B5~oո|zXԕ`?bTKtI2g)5	{OnʡG"s\#23KN`8}gctDK95 3y\.}Bp$p҉~i6ŕd19F)f[d_EܣP!9Ӝ{44't!04b.DcV#Bq{Y=vj^C8Y3F :jI}D6VzyUz/qZP%N>ťxXl0S놦n10Ӹ{zP@dꁖ~'a";F~p+ |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 9973, 38 words, 4 clauses) |
| --- | Minor | lBcEqd+֍%n=GQjSKD(o}sv闦,GBirFFIT(r7vIFI,pQ	`Yr/}.CY'ƍ_T"P`Rbk2/x..QqKKNXػlR{pN)6g#pG߾±S u0x@Ͱm<GgiJP/?a=ǚ,T\牆 |
| --- | Minor | lBcEqd+֍%n=GQjSKD(o}sv闦. GBirFFIT(r7vIFI. pQ	`Yr/}.CY'ƍ_T"P`Rbk2/x..QqKKNXػlR{pN)6g#pG߾±S u0x@Ͱm<GgiJP/?a=ǚ. T\牆. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10051, 62 words, 1 clauses) |
| --- | Minor | GzɖUxN߳^}}Ոٖ2vw4o*400o_8as -<WnXY+ItY1gkR:Ùo=Kǎ|.Ł74EVD^8b?Ug糾ܺЛ1cPMr)njkDĔ1[̵vB\Eٺe*޼]{C2Dt}2"F~n'E^xԕe!^oL龨lutMtꛀv*l~ͧ|k[ʄ+bs)VbD'R*f57鶐nd5c˕i,#VHj~#-fk`#!Q/e⭯bw8)V.>1v16:^u1<(R'dWAYئu2^H'#C`)}P2" |
| --- | Minor | GzɖUxN߳^}}Ոٖ2vw4o*400o_8as -<WnXY+ItY1gkR:Ùo=Kǎ|.Ł74EVD^8b?Ug糾ܺЛ1cPMr)njkDĔ1[̵vB\Eٺe*޼]{C2Dt}2"F~n'E^xԕe!^oL龨lutMtꛀv*l~ͧ|k[ʄ+bs)VbD'R*f57鶐nd5c˕i. #VHj~#-fk`#!Q/e⭯bw8)V.>1v16:^u1<(R'dWAYئu2^H'#C`)}P2". |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10075, 79 words, 4 clauses) |
| --- | Minor | nHt{dN~ЗO*]gp>lO"2:}&fI㌅$;?}{zKKi+Y5F65)	*u}~޾3,LC.l.>EOTw=ߔW_X~	{vs--G)peaQOakX`?cM]Kfi߸LʘvCEL-*p(@sDT/V]ZlРySeAc'lj(k E9^EdJ3»Y[v,Xr/&Eo9U#]:L],|B_N&Ǔ"p}&1BAR|:A[JSC=PӋ#})`C'֗D8<0ʃu}8-k壥)+T)ŪwՕN[+5{fmJBgwVVܙʖΟe4b5h^*Yǽ^z)s޹fho?.],/gY(wl |
| --- | Minor | nHt{dN~ЗO*]gp>lO"2:}&fI㌅$;?}{zKKi+Y5F65)	*u}~޾3. LC.l.>EOTw=ߔW_X~	{vs--G)peaQOakX`?cM]Kfi߸LʘvCEL-*p(@sDT/V]ZlРySeAc'lj(k E9^EdJ3»Y[v. Xr/&Eo9U#]:L]. |B_N&Ǔ"p}&1BAR|:A[JSC=PӋ#})`C'֗D8<0ʃu}8-k壥)+T)ŪwՕN[+5{fmJBgwVVܙʖΟe4b5h^*Yǽ^z)s޹fho?.]. /gY(wl. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10140, 65 words, 1 clauses) |
| --- | Minor | 'ϧ԰v(YhgkD71\6kwaIS[ToM8rrs&[eB?p_;`BEi7͊<}BBsMzmtRa{>&ꝛ]5ѷ*_S2MJ}"M,PNZ|E"8Y؞4KpaX=^Ƥl_3tIrJ4`G-Pj)\28YC󀆪xEV!H. ۲ɇr8 9W^#{~g6?^au`Ա=V?KI^[_Y<vL|6#0xJu&O(s{{v5};DTLZб-Xt^HjcR+niBXp8~P+htYhԈWtfa*U -?nkG{g |
| --- | Minor | 'ϧ԰v(YhgkD71\6kwaIS[ToM8rrs&[eB?p_;`BEi7͊<}BBsMzmtRa{>&ꝛ]5ѷ*_S2MJ}"M. PNZ|E"8Y؞4KpaX=^Ƥl_3tIrJ4`G-Pj)\28YC󀆪xEV!H. ۲ɇr8 9W^#{~g6?^au`Ա=V?KI^[_Y<vL|6#0xJu&O(s{{v5};DTLZб-Xt^HjcR+niBXp8~P+htYhԈWtfa*U -?nkG{g. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10218, 88 words, 3 clauses) |
| --- | Minor | Co1|}F>]uC?,oQT&T[/loU⪮g[!+z3߽ՈD+QrFHMɺL[Jv7zwirBb"rРR՝\3ªޔY.kN2uZlE("BnyAuƗ;_ouoteP	Q74_{Xcv+ΛҰ݇(Ֆ=~}N<;y֎	SXQC\R;F#;[SyU	Iy"<!>XJ	3wRǡu<N'rUP"#'G噣#"ld2OzIC@WC*;?a{>T)drX7ZzVDD8'caG%o`lnmG=r,R|ͷ|37Zd`&wۨ	i"^еKF$Xݦ)0̫|cHntUl2̮UH+iwaLLĊ[i|'Ɍ1Xp(P 1u+mN>pq&,hz &]DIvթTVx{fD6S{b'IY(aASL@i6. |
| --- | Minor | Co1|}F>]uC?. oQT&T[/loU⪮g[!+z3߽ՈD+QrFHMɺL[Jv7zwirBb"rРR՝\3ªޔY.kN2uZlE("BnyAuƗ;_ouoteP	Q74_{Xcv+ΛҰ݇(Ֆ=~}N<;y֎	SXQC\R;F#;[SyU	Iy"<!>XJ	3wRǡu<N'rUP"#'G噣#"ld2OzIC@WC*;?a{>T)drX7ZzVDD8'caG%o`lnmG=r. R|ͷ|37Zd`&wۨ	i"^еKF$Xݦ)0̫|cHntUl2̮UH+iwaLLĊ[i|'Ɍ1Xp(P 1u+mN>pq&. hz &]DIvթTVx{fD6S{b'IY(aASL@i6.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10235, 45 words, 6 clauses) |
| --- | Minor | 7D=)}j1Y2I"4?cٕ0,@L+I,y"5_7MenYA:5@*,Ra`NO+5.dƹ0v[\oxUyQmj`m̠nA0qiKH*M{d NoDnL",PjƺsKr'`n%8N:<yWW9ȏYe[G.^k}G7УP%מ#!$g6+05H,>)aH[,f |
| --- | Minor | 7D=)}j1Y2I"4?cٕ0. @L+I. y"5_7MenYA:5@*. Ra`NO+5.dƹ0v[\oxUyQmj`m̠nA0qiKH*M{d NoDnL". PjƺsKr'`n%8N:<yWW9ȏYe[G.^k}G7УP%מ#!$g6+05H. >)aH[. f. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10323, 46 words, 5 clauses) |
| --- | Minor | Af/ж*ټcT\%mz.Lķ[ Kq];xIQ[2)q[&z,@xنt|u(|3(ڝ,5moIF br=!F_,{>H٥#GZ<ep] M;JJZҷuX\4 KNH9/YFQ k˾cO 'Ba~b Hxr)V}Qq%(JF.uXBXᝐ,ZH{.;*š{vX3#r9 |
| --- | Minor | Af/ж*ټcT\%mz.Lķ[ Kq];xIQ[2)q[&z. @xنt|u(|3(ڝ. 5moIF br=!F_. {>H٥#GZ<ep] M;JJZҷuX\4 KNH9/YFQ k˾cO 'Ba~b Hxr)V}Qq%(JF.uXBXᝐ. ZH{.;*š{vX3#r9. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10362, 61 words, 1 clauses) |
| --- | Minor | 	p|w	wa}f~Ly߉#6"`Ǚbm%Uy y[2`GP]qv_E4/O'A+G}"JL%'+|܎&bSʕ;1`iE^KWq[7s'KXnoY<Z&}f(}Q?⋖w[t}g{l?L*x3nyT&ixnŨr@]1ޓ?ߎ*ݟg.~Ed6 2􈯂=;9֗Qg-w ?\&MwmKh1	>n^eY3^"C`SeՏU=ʎ,eVWuNgB fK|V(>T;Y6VH?K |
| --- | Minor | 	p|w	wa}f~Ly߉#6"`Ǚbm%Uy y[2`GP]qv_E4/O'A+G}"JL%'+|܎&bSʕ;1`iE^KWq[7s'KXnoY<Z&}f(}Q?⋖w[t}g{l?L*x3nyT&ixnŨr@]1ޓ?ߎ*ݟg.~Ed6 2􈯂=;9֗Qg-w ?\&MwmKh1	>n^eY3^"C`SeՏU=ʎ. eVWuNgB fK|V(>T;Y6VH?K. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10507, 37 words, 4 clauses) |
| --- | Minor | *E88~dD؝+&1L7#R<_dTW.*ЍNh:J-,,;f7?[A} wc+Gy}cMPAErOW`9$	DX@]nZ{/DĿzmJtwMYa*y99jOƖ]28]=ztD:ܛЀlj,,-is?jTޔju)0Ɍ@?%B˧N |
| --- | Minor | *E88~dD؝+&1L7#R<_dTW.*ЍNh:J-. ;f7?[A} wc+Gy}cMPAErOW`9$	DX@]nZ{/DĿzmJtwMYa*y99jOƖ]28]=ztD:ܛЀlj. . -is?jTޔju)0Ɍ@?%B˧N. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10576, 80 words, 4 clauses) |
| --- | Minor | vULOFeĊ}?{z񛺡Szڱg1pB'āW~#9ئ#In+.,|UtzT S=9)E*V̵o:IRrk1V9ȧukKK$X0j(4gHOۊ=b%9N &|*&{RSB*y,;WW,(M?jfvYH'T=(=l'L.,~1#Dv{H\2nEKy8&{yQG胓~edx)IZ̑eEƴMcL&Eț?{ϽvmkE|t߮39.ϕ )W4DKYDvLov0vX;FM.α&Nw_g-ka!ى	usG/5]Gb%q}Sbk]nQ3y`I8(sVM?"%%(TJr&\kHz5B*XuթӡwUo.72ꉴ9kP |
| --- | Minor | vULOFeĊ}?{z񛺡Szڱg1pB'āW~#9ئ#In+.. |UtzT S=9)E*V̵o:IRrk1V9ȧukKK$X0j(4gHOۊ=b%9N &|*&{RSB*y. ;WW. (M?jfvYH'T=(=l'L.. ~1#Dv{H\2nEKy8&{yQG胓~edx)IZ̑eEƴMcL&Eț?{ϽvmkE|t߮39.ϕ )W4DKYDvLov0vX;FM.α&Nw_g-ka!ى	usG/5]Gb%q}Sbk]nQ3y`I8(sVM?"%%(TJr&\kHz5B*XuթӡwUo.72ꉴ9kP. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10581, 69 words, 1 clauses) |
| --- | Minor | ?/j,!o7l(CVx?ň		_!r6] _s>h/pVJۨPL|E|qS=kbYZw[z9UQeH569!9n&h:Js;i7fcwܩu9#V@iPZapT@黩.o"06ѐ{m 7(5ٔ{wվj3;1BmbOf?<ɍ>huflxA6)?"-bdWLIW+_^C &W_>H.u:sE.!u0	iz)_)w-eEKLc}=}ce@"7y򅧒	Ȇ9o$DNo?Ł봕_P}5ƣ'Rw	(̅=O`1hHu[L\iUm6ID4[! |
| --- | Minor | ?/j. !o7l(CVx?ň		_!r6] _s>h/pVJۨPL|E|qS=kbYZw[z9UQeH569!9n&h:Js;i7fcwܩu9#V@iPZapT@黩.o"06ѐ{m 7(5ٔ{wվj3;1BmbOf?<ɍ>huflxA6)?"-bdWLIW+_^C &W_>H.u:sE.!u0	iz)_)w-eEKLc}=}ce@"7y򅧒	Ȇ9o$DNo?Ł봕_P}5ƣ'Rw	(̅=O`1hHu[L\iUm6ID4[!. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10624, 65 words, 2 clauses) |
| --- | Minor | xڍP-kp'wwww3.%!kpn#9~_^QӫWzOAM,ndqY l 66N66jj-[oZf+I_yN #+`c @PpP%]mmG =p#P6 _O0s h:YNv`ed1stcqrgxڂm @7G 3G_Pll5f@+`kF,J Ug /_&߳?ق6prt6yۂV@0YA4spsz70u03%Y@F\`Y:Xlh4SYJ:9:A`7?ꓲuZݛ9y|6lAV4a̪ |
| --- | Minor | xڍP-kp'wwww3.%!kpn#9~_^QӫWzOAM. ndqY l 66N66jj-[oZf+I_yN #+`c @PpP%]mmG =p#P6 _O0s h:YNv`ed1stcqrgxڂm @7G 3G_Pll5f@+`kF. J Ug /_&߳?ق6prt6yۂV@0YA4spsz70u03%Y@F\`Y:Xlh4SYJ:9:A`7?ꓲuZݛ9y|6lAV4a̪. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10636, 72 words, 2 clauses) |
| --- | Minor | 5N|0f:,ZqqtzIQ(p9M//G:`ÙYɗeC2s֞2)	QT~5/NB{QQ'LI_ZO5gBlV֪jk3E!D&y:.ԄgH.crPh<-ڒPUSބԆk.A߿rryd/=Z׿kK̶FRqpV9˳F2kxqudPp	dkM),+w1)EIK洣x/ѡ(FpM|Wrx8lfN&X7$<@boh8Sk|"oҙܝ+9z	}Vݓ"y2ZJ*&^~Y-TL'9vaxf-0uٞFgb(ٜY4z=;&VEBd9Y0`]5nPBX19[R^@uoՏ	SyIT'~@;:|%R٠7MH*DK[p&G/BAڤm6Go۪ 4vX%G<xۆNA |
| --- | Minor | 5N|0f:. ZqqtzIQ(p9M//G:`ÙYɗeC2s֞2)	QT~5/NB{QQ'LI_ZO5gBlV֪jk3E!D&y:.ԄgH.crPh<-ڒPUSބԆk.A߿rryd/=Z׿kK̶FRqpV9˳F2kxqudPp	dkM). +w1)EIK洣x/ѡ(FpM|Wrx8lfN&X7$<@boh8Sk|"oҙܝ+9z	}Vݓ"y2ZJ*&^~Y-TL'9vaxf-0uٞFgb(ٜY4z=;&VEBd9Y0`]5nPBX19[R^@uoՏ	SyIT'~@;:|%R٠7MH*DK[p&G/BAڤm6Go۪ 4vX%G<xۆNA. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10662, 85 words, 1 clauses) |
| --- | Minor | ̫'ڠGlͭnG2Bd7zW_kdݹMd3oK0qӨÛg(\uU()g]rk~*/}FbwƓ-ׂEp._8Cl+W{.'N&\1H-I5aW `7p7AlL i+&aj)$7lxžu-􆓧K}iV*-h!J٪T/0z;Vԗ!pv Vov9j[F"S|ALɎanO.#3ҷo}vm/j b1;}^;Hil t_Q2OCر"{|dJ4" o_MSNV~@E{l76|fH}RbnθֱDP3qizlGo2yB=8W+a`_Dsu"1R9wa5tJ>xsS4+rD_t.K8/ŁT+IoNIЃ=IVeY/cO8%累JS1Y~mwP0(,iw. |
| --- | Minor | ̫'ڠGlͭnG2Bd7zW_kdݹMd3oK0qӨÛg(\uU()g]rk~*/}FbwƓ-ׂEp._8Cl+W{.'N&\1H-I5aW `7p7AlL i+&aj)$7lxžu-􆓧K}iV*-h!J٪T/0z;Vԗ!pv Vov9j[F"S|ALɎanO.#3ҷo}vm/j b1;}^;Hil t_Q2OCر"{|dJ4" o_MSNV~@E{l76|fH}RbnθֱDP3qizlGo2yB=8W+a`_Dsu"1R9wa5tJ>xsS4+rD_t.K8/ŁT+IoNIЃ=IVeY/cO8%累JS1Y~mwP0(. iw.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10726, 62 words, 1 clauses) |
| --- | Minor | L\duZ5B&l"nXtѹdQ?;O~]ROHKd{OIO%_S:gFŽ۝ <|-ѷM+uu<72TMi_n]c7x9"xD>D2&ƾ$jm(4S_A{7vܽ_dc2xʿ&m!*m#oD:9HI0!"*YBc%&-3m<˸4>F:O()M<}CH b\k,89g4ȷ5I&搑ovgh%=8Ht=n:oR3TO\SgLƝQKi[ CS%tf<BQWb1Iw^;Ɇ|.l5qD"1 >}2R=?ʐE! |
| --- | Minor | L\duZ5B&l"nXtѹdQ?;O~]ROHKd{OIO%_S:gFŽ۝ <|-ѷM+uu<72TMi_n]c7x9"xD>D2&ƾ$jm(4S_A{7vܽ_dc2xʿ&m!*m#oD:9HI0!"*YBc%&-3m<˸4>F:O()M<}CH b\k. 89g4ȷ5I&搑ovgh%=8Ht=n:oR3TO\SgLƝQKi[ CS%tf<BQWb1Iw^;Ɇ|.l5qD"1 >}2R=?ʐE!. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10772, 40 words, 4 clauses) |
| --- | Minor | ּϕC1>B(jɝQm?HtL՚,{_Q;jW˽;1]K)]Kl6{RnڷF|-eLcVl>-F{ާiK,o}Mmo]WHԪ,ajRM\ckdnĘtCHa㚁T	Kh@QIR8 G/?whN>R䣋*bϰ,|ΔK_Q-U.|dO{뱌p=4Y |
| --- | Minor | ּϕC1>B(jɝQm?HtL՚. {_Q;jW˽;1]K)]Kl6{RnڷF|-eLcVl>-F{ާiK. o}Mmo]WHԪ. ajRM\ckdnĘtCHa㚁T	Kh@QIR8 G/?whN>R䣋*bϰ. |ΔK_Q-U.|dO{뱌p=4Y. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10785, 82 words, 2 clauses) |
| --- | Minor | _׈7P|NlGUfNgH^C{L+Ǔ=F-Tl=(x49\4t|Ոqpwv'~ǭيٵ"IH7lתby{ucׄ2ySg\yǟGm#[_2͵ȍ?bo2w9<~'xW3'1#q},b3e5c.; R*0ub"؉1RWJCcdXPl f6Ssɒsۃ G鲃	hQwmbz.ʜ).5";]w|>8W\.#' )o)؊][Qc[F1U+SHwP\scGgqqjN^Ϟrt^i0w6@@Co4ϴMS5i{ZZ}}xR.$ipiyDc=u-dQuّ+uߏ%=q#lI7!p5'|eC^]N1"p2t+)+A\,3Vc'\қ=|CߺaQ=<V6@(ޅ1(ԮλDB |
| --- | Minor | _׈7P|NlGUfNgH^C{L+Ǔ=F-Tl=(x49\4t|Ոqpwv'~ǭيٵ"IH7lתby{ucׄ2ySg\yǟGm#[_2͵ȍ?bo2w9<~'xW3'1#q}. b3e5c.; R*0ub"؉1RWJCcdXPl f6Ssɒsۃ G鲃	hQwmbz.ʜ).5";]w|>8W\.#' )o)؊][Qc[F1U+SHwP\scGgqqjN^Ϟrt^i0w6@@Co4ϴMS5i{ZZ}}xR.$ipiyDc=u-dQuّ+uߏ%=q#lI7!p5'|eC^]N1"p2t+)+A\. 3Vc'\қ=|CߺaQ=<V6@(ޅ1(ԮλDB. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10850, 62 words, 3 clauses) |
| --- | Minor | {xoyarYo\u/:HLl3Egg'Bj"QkUchΠ,BlGYƛd0NuY|dxyBa NB3e)5&ڜ9B?ꌒJJ4~Վ:9\4WEu`b97_֫dkT}t2P5evGWڷ5n}*	ь&\S2C>+VF9لw&Cn^@-L٩ 2Ǻ*.9{X~l	}=SL3NcHAX#}|B<1+:tQȤ-SKSFEe)O_Kϔx3:16O*P2hF~q \PD]WlmFnɖx*+Ĳh2nxNӣf{ȱq[2mA|e,=^3@A&i+'׽,:qbQ|SYQ1 |
| --- | Minor | {xoyarYo\u/:HLl3Egg'Bj"QkUchΠ. BlGYƛd0NuY|dxyBa NB3e)5&ڜ9B?ꌒJJ4~Վ:9\4WEu`b97_֫dkT}t2P5evGWڷ5n}*	ь&\S2C>+VF9لw&Cn^@-L٩ 2Ǻ*.9{X~l	}=SL3NcHAX#}|B<1+:tQȤ-SKSFEe)O_Kϔx3:16O*P2hF~q \PD]WlmFnɖx*+Ĳh2nxNӣf{ȱq[2mA|e. =^3@A&i+'׽. :qbQ|SYQ1. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10907, 61 words, 2 clauses) |
| --- | Minor | sL7n,_4S.JzA}<=@ƒa1``DPne&Ux9xAh6pKDOYKPY[Ʃ]LsFRڶסED<zMy7^d&=J88N`TCR1z4mXpB؇kbuFnӤl"~NShRܼH/qE"a*8d#&Ʉ^~p;ϫ sϚv#h}־MU1ovɷZVGڽ#Ӿ`~a	՜DVʌB-)cv,ˮɈW\;Z`̴FM\;?<sD NC#O%#CSq!KҴKl1Ba^Ԗ |
| --- | Minor | sL7n. _4S.JzA}<=@ƒa1``DPne&Ux9xAh6pKDOYKPY[Ʃ]LsFRڶסED<zMy7^d&=J88N`TCR1z4mXpB؇kbuFnӤl"~NShRܼH/qE"a*8d#&Ʉ^~p;ϫ sϚv#h}־MU1ovɷZVGڽ#Ӿ`~a	՜DVʌB-)cv. ˮɈW\;Z`̴FM\;?<sD NC#O%#CSq!KҴKl1Ba^Ԗ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 10957, 71 words, 3 clauses) |
| --- | Minor | 9\r^+AɚiYuX\1a)e'l:y%+tZ<k~W@@>u+xv*V=0K,UI,nU0jj{coAC-ޭzO461n 5]'O3hLk10^NXm7MյJ?U}vL_1)7sGxQ]߂qd;o}R8Ǆx|mo}/ҋYMyXN{Cq~_A~)>scBnAdrUFͣ>Wz$:C*?OM`!JV/!@Ru:0~/ xu">kHrǀ8rtIE>[ջy+L	-HyY&WJy8({"1R޵o;䣝I6b{3l-H鮳.x,s٠& |
| --- | Minor | 9\r^+AɚiYuX\1a)e'l:y%+tZ<k~W@@>u+xv*V=0K. UI. nU0jj{coAC-ޭzO461n 5]'O3hLk10^NXm7MյJ?U}vL_1)7sGxQ]߂qd;o}R8Ǆx|mo}/ҋYMyXN{Cq~_A~)>scBnAdrUFͣ>Wz$:C*?OM`!JV/!@Ru:0~/ xu">kHrǀ8rtIE>[ջy+L	-HyY&WJy8({"1R޵o;䣝I6b{3l-H鮳.x. s٠&:. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11029, 77 words, 2 clauses) |
| --- | Minor | ̍w)e Ṟ39CuAgȫWFPg}c'6#ކHKm48&.q*/n,5iw1%72CBlRXxf:j>i]>~*Z9HNvSERx~jGyWs~ꌟX:i@;PxDp©FtKX)C6R="lmYPBtr1ށɆi}|,fZF!o{Ts\\qm]W2R)%%ᙢA39-jOE`{jÁ6\ЈGA};In]/97%ARV}h7<$yt_el޵\Y>л\qb1hmN ՜ÍݧA!XRzp]ԑν2;9lZitewvwh}%l}I/ϸ90Qۗgg?JE962Fkc; 퉙_fr7"ީ'cwg\2;n |
| --- | Minor | ̍w)e Ṟ39CuAgȫWFPg}c'6#ކHKm48&.q*/n. 5iw1%72CBlRXxf:j>i]>~*Z9HNvSERx~jGyWs~ꌟX:i@;PxDp©FtKX)C6R="lmYPBtr1ށɆi}|. fZF!o{Ts\\qm]W2R)%%ᙢA39-jOE`{jÁ6\ЈGA};In]/97%ARV}h7<$yt_el޵\Y>л\qb1hmN ՜ÍݧA!XRzp]ԑν2;9lZitewvwh}%l}I/ϸ90Qۗgg?JE962Fkc; 퉙_fr7"ީ'cwg\2;n. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11058, 72 words, 4 clauses) |
| --- | Minor | E<U:)L=~ĔMڋIC	M=oigĸwIY'O%d}k6'T|Zvȼ>XIwNR-&{,㓛nE4ӡlxFK?Nv+5lؤD2s9'E5Au@k~8K,ʐnZ<2׀ORJoo:!b죍¶U+Iom_/g\=ۀ}7V,[X*UU֡s$fz"%槌鹀wψ5XP4"Epˎxh[eA Uf,ȸUp"~YG_xe& N-nSЕky(O'ZƂdRfN'QCمxaUȎYʤG#4mQ!|L^	Q!fU#РFN6!eVGT7Жo-%b2BMp |
| --- | Minor | E<U:)L=~ĔMڋIC	M=oigĸwIY'O%d}k6'T|Zvȼ>XIwNR-&{. 㓛nE4ӡlxFK?Nv+5lؤD2s9'E5Au@k~8K. ʐnZ<2׀ORJoo:!b죍¶U+Iom_/g\=ۀ}7V. [X*UU֡s$fz"%槌鹀wψ5XP4"Epˎxh[eA Uf. ȸUp"~YG_xe& N-nSЕky(O'ZƂdRfN'QCمxaUȎYʤG#4mQ!|L^	Q!fU#РFN6!eVGT7Жo-%b2BMp. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11300, 60 words, 5 clauses) |
| --- | Minor | cV(>z<9sJN,[MTKقJsL}@:<zrL_t6H>Ƙ>0L`)Xfj+1۾lt`i曔/䛘@&4ΰhrDaiL+W	i=2b[m/:z;|9K{9U,Tb-O%qwAN[:[]V^认#(yj} fFS:.' LL{zU.?.;v8wS!5{1.Ȗ`f+mKˬn<|?V8ʲF4MIU&G˦gtmQ=%t%Ο4ù,_ܐrc[aUM|Ttm}ka!,I/7Ƴm3rsyi-Jԟ9,2];o8we%ϩolĹQܭ		1 |
| --- | Minor | cV(>z<9sJN. [MTKقJsL}@:<zrL_t6H>Ƙ>0L`)Xfj+1۾lt`i曔/䛘@&4ΰhrDaiL+W	i=2b[m/:z;|9K{9U. Tb-O%qwAN[:[]V^认#(yj} fFS:.' LL{zU.?.;v8wS!5{1.Ȗ`f+mKˬn<|?V8ʲF4MIU&G˦gtmQ=%t%Ο4ù. _ܐrc[aUM|Ttm}ka!. I/7Ƴm3rsyi-Jԟ9. 2];o8we%ϩolĹQܭ		1. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11360, 28 words, 4 clauses) |
| --- | Minor | <_PQo⺌S&.6G^_LrP2>mH'G,6ft>!%_miCj?:V,ѣ@g1;V3}bFc%z;-M\LrĔ¤H.f]+eLyFzeӧd){?tËXo>J2NMijՓ#޹,w	PMrͥ, |
| --- | Minor | <_PQo⺌S&.6G^_LrP2>mH'G. 6ft>!%_miCj?:V. ѣ@g1;V3}bFc%z;-M\LrĔ¤H.f]+eLyFzeӧd){?tËXo>J2NMijՓ#޹. w	PMrͥ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11364, 78 words, 0 clauses) |
| --- | Minor | JGʮ'LKxVUm'QH%] Ԏֹm++W7˝[PҞ|9	@*2{8a|zBY	;G@x̡>7y 5yj/>XJsQkZ2˲Z\QE ˃#Еa80bEtD/b`풛5X3;\9Tl%Xaͦ9ODjxEЗ%H|2L 5A_':[Ojm8\\.m1"T7UޙB9:YLiuGhėi^էM=G%97iTX<J폽NKA r=Ofˆ tr1r?8_At:Y>ju[Jh$JcKkU@2~uD9yt0c.i|^32GdBc=D2sV̥ bmz'cźXׇ8{(C	f"eK@D g5oAY;ݮbw&phPPY8>Q5%*=JqĉWaO. |
| --- | Minor | JGʮ'LKxVUm'QH%] Ԏֹm++W7˝[PҞ|9	@*2{8a|zBY	;G@x̡>7y 5yj/>XJsQkZ2˲Z\QE ˃#Еa80bEtD/b`풛5X3;\9Tl%Xaͦ9ODjxEЗ%H|2L 5A_':[Ojm8\\.m1"T7UޙB9:YLiuGhėi^էM=G%97iTX<J폽NKA r=Ofˆ tr1r?8_At:Y>ju[Jh$JcKkU@2~uD9yt0c.i|^32GdBc=D2sV̥ bmz'cźXׇ8{(C	f"eK@D g5oAY;ݮbw&phPPY8>Q5%*=JqĉWaO. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11366, 74 words, 4 clauses) |
| --- | Minor | {֜L[>Fs\8xd/bk2Yb)BgdwZ@x v;v;,L'׳ua|dUû)Bkj)G3@Wb~I(BxH9@B~!--px_#}e/n6}Vkq:caM7J1U/*]R&;I ,Wzrr#D;%}%FE^C'A{{N	mSC;A;\WЕƥcO3i_'-&PjjD:[%(r;Ǽx:'OKsbݼ]߼Kgjhj:޿51j)8s<٠<Xx߁=;8	(|,ܘ`redV,Aڂ3!N_Ʉ|Ąz!'vX9L`ys	}#v]{q͟bÞ\'5Ώ9T֚Ѩ${w`QqKJUTOA&7x/&x |
| --- | Minor | {֜L[>Fs\8xd/bk2Yb)BgdwZ@x v;v;. L'׳ua|dUû)Bkj)G3@Wb~I(BxH9@B~!--px_#}e/n6}Vkq:caM7J1U/*]R&;I. Wzrr#D;%}%FE^C'A{{N	mSC;A;\WЕƥcO3i_'-&PjjD:[%(r;Ǽx:'OKsbݼ]߼Kgjhj:޿51j)8s<٠<Xx߁=;8	(|. ܘ`redV. Aڂ3!N_Ʉ|Ąz!'vX9L`ys	}#v]{q͟bÞ\'5Ώ9T֚Ѩ${w`QqKJUTOA&7x/&x. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11449, 42 words, 4 clauses) |
| --- | Minor | M2S pAAAX*Ex&Z 03ćx[T6\0>0ATPA>Jr=8`y6Xx@[`92FC\*Gs|fA`=V,T]ʧRa̕NVf;p?>s|,f{ĉӠlɡ#l61,`DC][?	|0Arq8pX,`P =<`h@P iLWn. |
| --- | Minor | M2S pAAAX*Ex&Z 03ćx[T6\0>0ATPA>Jr=8`y6Xx@[`92FC\*Gs|fA`=V. T]ʧRa̕NVf;p?>s|. f{ĉӠlɡ#l61. `DC][?	|0Arq8pX. `P =<`h@P iLWn.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11459, 72 words, 2 clauses) |
| --- | Minor | 8`;Ӎeͣ/,TBn{uL~3cGc8eJj lϘh%d-5&T;?dO%7T|fSW>6+S}aNgl~oZ}oe'Rll]\bnzhlSʳӳL鼫l襗F6n֪YC5U̚W֎(g-`_Nɷ{~5.A| 2"-*rwkȌ(ǃS]je)Kh	R7)%,;*uo4kA6~●FI.YUB:IjO䲋-?҅xŝ-o\س]{NcW[ɇY\^تU7lrWO~o-y65O*{8L4\Is:%!&NOl^؝mGƫIi\bzrYjKNQKK_<3N%|MupZŉO+Lt7=˾r4!C^bxZgZ |
| --- | Minor | 8`;Ӎeͣ/. TBn{uL~3cGc8eJj lϘh%d-5&T;?dO%7T|fSW>6+S}aNgl~oZ}oe'Rll]\bnzhlSʳӳL鼫l襗F6n֪YC5U̚W֎(g-`_Nɷ{~5.A| 2"-*rwkȌ(ǃS]je)Kh	R7)%. ;*uo4kA6~●FI.YUB:IjO䲋-?҅xŝ-o\س]{NcW[ɇY\^تU7lrWO~o-y65O*{8L4\Is:%!&NOl^؝mGƫIi\bzrYjKNQKK_<3N%|MupZŉO+Lt7=˾r4!C^bxZgZ. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11495, 65 words, 2 clauses) |
| --- | Minor | 8hǱ+w"uEkچ`4>N⋆]>yZɓQLhnfԓ'FeƏ	^4=T _R޸PD>U'W"PCF8Tk֮>a[75E Lln!.ʧ/1ƛóƒmIgA22=К>`p* ZkS>9O0ݼžq!ѯ茙Q27W<4|5E,\h4i74qd+IK2{8{}B5>kGLdŜS4ʦ	՝{2d`s rUkw#Fu^=z0QS3]͙3A,7d Ν<wʱ^L+=:Z&:9lVqν J! |
| --- | Minor | 8hǱ+w"uEkچ`4>N⋆]>yZɓQLhnfԓ'FeƏ	^4=T _R޸PD>U'W"PCF8Tk֮>a[75E Lln!.ʧ/1ƛóƒmIgA22=К>`p* ZkS>9O0ݼžq!ѯ茙Q27W<4|5E. \h4i74qd+IK2{8{}B5>kGLdŜS4ʦ	՝{2d`s rUkw#Fu^=z0QS3]͙3A. 7d Ν<wʱ^L+=:Z&:9lVqν J!. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11503, 72 words, 2 clauses) |
| --- | Minor | pMzC{T	D;xIyzS?fMJƲ cHܖ05&Eo/nTT(U'Hf% GByn,8-p w󳍥;*ߢ+[2՘l$aѮ?'xyj520N%P.D#/{~xC_ɴŸ1ퟪaG1gyޯ7@<#qӛlXn#CORJ#Pf;gD\]rsn+*Ԝ^Xaq1r`hŷ_5tBKHg7]nEl7S&%39K{EѴK~X8@߶dP?,1W} H> CHgH]aP*tjWT"7j3OFCQ'qk뫲W"IZĢW+?eeJciO`vOp 2&E̾~8K |
| --- | Minor | pMzC{T	D;xIyzS?fMJƲ cHܖ05&Eo/nTT(U'Hf% GByn. 8-p w󳍥;*ߢ+[2՘l$aѮ?'xyj520N%P.D#/{~xC_ɴŸ1ퟪaG1gyޯ7@<#qӛlXn#CORJ#Pf;gD\]rsn+*Ԝ^Xaq1r`hŷ_5tBKHg7]nEl7S&%39K{EѴK~X8@߶dP?. 1W} H> CHgH]aP*tjWT"7j3OFCQ'qk뫲W"IZĢW+?eeJciO`vOp 2&E̾~8K. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11611, 23 words, 4 clauses) |
| --- | Minor | gŉC/6UM=\Yy)C[<;\,2|,o-P{|W?6*z\s6s(sB2O%77yk3,\Wku]4ьF6xd<;)*?&{CDqĜ)`V&|!^O]*Dj,*/=S |
| --- | Minor | gŉC/6UM=\Yy)C[<;\. 2|. o-P{|W?6*z\s6s(sB2O%77yk3. \Wku]4ьF6xd<;)*?&{CDqĜ)`V&|!^O]*Dj. */=S. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11746, 69 words, 4 clauses) |
| --- | Minor | opضWpr4'[x֞~-_g),CMdPB;qm8qiK3ƎWNkZ;ٓ1hY)RC})jʃ@OLlO>rL z.՝l>I5ZF-h9Zm[[BNf~p|6z6)Sfq~{ xrRRIU5]_(NG@kŧͩ>'^m]uۊ+?ߕVY=H,HAݟ;kB7蔐vBd.mC=KJ!J%=ARK DSZp{XlyN?M^9]]e"K۵:)ũvpv,d+p}dyT|V8x	8(D!،1-r{7rbӎJ{|X,󬰩j.t |
| --- | Minor | opضWpr4'[x֞~-_g). CMdPB;qm8qiK3ƎWNkZ;ٓ1hY)RC})jʃ@OLlO>rL z.՝l>I5ZF-h9Zm[[BNf~p|6z6)Sfq~{ xrRRIU5]_(NG@kŧͩ>'^m]uۊ+?ߕVY=H. HAݟ;kB7蔐vBd.mC=KJ!J%=ARK DSZp{XlyN?M^9]]e"K۵:)ũvpv. d+p}dyT|V8x	8(D!،1-r{7rbӎJ{|X. 󬰩j.t. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11768, 37 words, 5 clauses) |
| --- | Minor | Psk;e}r| ,z*IZG/Frmwb%2mk[He <0.=Bol14q.%</@vaj,0Sݦzh	ޢ,U,Z|P\ǫ`׫4ukr3>umJP,*v{	AKtR=biQVfrgAXNP|y1O^=3o}UϜ-O)" |
| --- | Minor | Psk;e}r| . z*IZG/Frmwb%2mk[He <0.=Bol14q.%</@vaj. 0Sݦzh	ޢ. U. Z|P\ǫ`׫4ukr3>umJP. *v{	AKtR=biQVfrgAXNP|y1O^=3o}UϜ-O)". |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11827, 65 words, 2 clauses) |
| --- | Minor | #=㿅VVfV.& s#ۿ=\hkeo@_tV&6o/]ʟAXBHRJ`ew\T=9?0 o:&vN 3'7!.fVM1Yh[7#ӿ?OzFQq17;mSKߪ_sofaf`bR72)3>X֤Z\PUe^J<tG`Ö/ ϗg!&yA~BƙFբ66;~;t*ĭ0 $1<bh|td\8X2#LSdOC}'sRvF-r۲+!C*Sŗ>qex|,sq"?-Ȍ~C*F~tUX+etbOduJAɡ,SIuFN%-(21~(]._^3E>+pC{'^FAA-0GW |
| --- | Minor | #=㿅VVfV.& s#ۿ=\hkeo@_tV&6o/]ʟAXBHRJ`ew\T=9?0 o:&vN 3'7!.fVM1Yh[7#ӿ?OzFQq17;mSKߪ_sofaf`bR72)3>X֤Z\PUe^J<tG`Ö/ ϗg!&yA~BƙFբ66;~;t*ĭ0 $1<bh|td\8X2#LSdOC}'sRvF-r۲+!C*Sŗ>qex|. sq"?-Ȍ~C*F~tUX+etbOduJAɡ. SIuFN%-(21~(]._^3E>+pC{'^FAA-0GW. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11883, 39 words, 4 clauses) |
| --- | Minor | Q/,@>9j^Up9qFP-ѧ10iNT]MQ)9r1> Q.7812k%9XG, V!/^^mS2#C<&D-l<M^ku;;bZ72n]By!dpË~nZ	j .]LFk\ : "WݔwW,>/ kXOGLY#n?*j!lVߑ= ~d,*\	v[i](3dp<w==pa#u{%XnבqR |
| --- | Minor | Q/. @>9j^Up9qFP-ѧ10iNT]MQ)9r1> Q.7812k%9XG. V!/^^mS2#C<&D-l<M^ku;;bZ72n]By!dpË~nZ	j .]LFk\ : "WݔwW. >/ kXOGLY#n?*j!lVߑ= ~d. *\	v[i](3dp<w==pa#u{%XnבqR. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 11937, 66 words, 1 clauses) |
| --- | Minor | C̊k?w ϳ>2?iaܞfrQx]b1:xQsA4Y5yI̗X⬖ro_QqJ@dS`1^ȗCѓ}yRel/3d	QbgM;|zИoR#㙖&=9Fh|rlM]hOthiLp߿L2+a6*yש])_Kk;we'ٞk<9pnL0"6<ܷփlj"(	q9DCUsI5{TΔ"&`iX,<5 8e@B@5͐à\/ic+)LTBB	{E0{ia-h4M.P#J]Zۭ(w-|E\g@4?0 E<WxPMަBiؠ6P$%1= |
| --- | Minor | C̊k?w ϳ>2?iaܞfrQx]b1:xQsA4Y5yI̗X⬖ro_QqJ@dS`1^ȗCѓ}yRel/3d	QbgM;|zИoR#㙖&=9Fh|rlM]hOthiLp߿L2+a6*yש])_Kk;we'ٞk<9pnL0"6<ܷփlj"(	q9DCUsI5{TΔ"&`iX. <5 8e@B@5͐à\/ic+)LTBB	{E0{ia-h4M.P#J]Zۭ(w-|E\g@4?0 E<WxPMަBiؠ6P$%1=. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 12059, 69 words, 3 clauses) |
| --- | Minor | &yC'lS]}JM:Se:JASaڛ}QUYṆoJi I8G2k/Tͣb|6߲V%yޕ)'գɬ%:A~	۵),''g'dN˵ʈzޚyFK%=s<r`G=yA&"*.3?lW{2, PvBR){2ɧ~ȟC)̨VTasڨ[~.}pNvK",yT [m&S@S[ꖑʥG 1U8J ixfEd5B.}aşĥ$KNQ*ެalPRWVΌep>!;)ElOVf"^;ijq=ËP>Ph%l>%Yuuu	'(;v:7&^\Se1ƿ3BДq*W[2!vAmo7xГ}(JűȾxpi{RfRџs^: |
| --- | Minor | &yC'lS]}JM:Se:JASaڛ}QUYṆoJi I8G2k/Tͣb|6߲V%yޕ)'գɬ%:A~	۵). ''g'dN˵ʈzޚyFK%=s<r`G=yA&"*.3?lW{2.  PvBR){2ɧ~ȟC)̨VTasڨ[~.}pNvK". yT [m&S@S[ꖑʥG 1U8J ixfEd5B.}aşĥ$KNQ*ެalPRWVΌep>!;)ElOVf"^;ijq=ËP>Ph%l>%Yuuu	'(;v:7&^\Se1ƿ3BДq*W[2!vAmo7xГ}(JűȾxpi{RfRџs^:. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 12098, 34 words, 5 clauses) |
| --- | Minor | cinK,	>G15lODJG.g[m-S~J,٦] nҏNCTM 7k-O0mH츻+R,P]ѿLX|F V,mi(xMp p59Vha[D[F՘Fip,xM03d(G |
| --- | Minor | cinK. >G15lODJG.g[m-S~J. ٦] nҏNCTM 7k-O0mH츻+R. P]ѿLX|F V. mi(xMp p59Vha[D[F՘Fip. xM03d(G. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |
| --- | Minor | SENTENCE (Line 12520, 101 words, 2 clauses) |
| --- | Minor | 9swDwAOg vm[o[xRL>M_6]4[[61Ďnul[pԠXζwm@f0Mm{_䶑uZ4ī,vƵ6:}o)Y}do',xjVdpI-\UK[ｯiC6ߩM?9R:.Or'w-o|᮲^TN}D( YCvi^çà{5Terhܑv }oWS[C?s=".AĸSE@wRIW夘<gH&eϨ[vТU*ZW=X_k(Fj]ѼuVqGW˜_K{\UCs^=W#46ZtQXŚt"V&W"֒uŃ%F׀Eܒoʏ惣Yj./I2:+xsD1^COxnor\ڳٮ\<6^pFڳ!WC\M7E-kj򢨖("33Wy-TLezp@(.e|p@EzQD:4B@B@xE϶D4t5hfZ=tkIE-Qh<f.2p>8: c2p/KbQ\\(<& |
| --- | Minor | 9swDwAOg vm[o[xRL>M_6]4[[61Ďnul[pԠXζwm@f0Mm{_䶑uZ4ī. vƵ6:}o)Y}do'. xjVdpI-\UK[ｯiC6ߩM?9R:.Or'w-o|᮲^TN}D( YCvi^çà{5Terhܑv }oWS[C?s=".AĸSE@wRIW夘<gH&eϨ[vТU*ZW=X_k(Fj]ѼuVqGW˜_K{\UCs^=W#46ZtQXŚt"V&W"֒uŃ%F׀Eܒoʏ惣Yj./I2:+xsD1^COxnor\ڳٮ\<6^pFڳ!WC\M7E-kj򢨖("33Wy-TLezp@(.e|p@EzQD:4B@B@xE϶D4t5hfZ=tkIE-Qh<f.2p>8: c2p/KbQ\\(<&. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. |

### [Script] VISUAL

| Line | Severity | Issue |
|------|----------|-------|
| --- | Major | (Page 1) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 2) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 3) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 4) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 5) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 6) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 7) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 8) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 9) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 10) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 11) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 12) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 13) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 14) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 15) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 16) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 17) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 18) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 19) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 20) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 21) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 22) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Major | (Page 23) : Content overflows bottom margin (y=734.3pt, page_height=792.0pt) |
| --- | Critical | (Page 2) : Block overlap detected: 785 sq pt |
| --- | Critical | (Page 2) : Block overlap detected: 1284 sq pt |
| --- | Critical | (Page 2) : Block overlap detected: 105 sq pt |
| --- | Critical | (Page 3) : Block overlap detected: 109 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 580 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 493 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 405 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 317 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 229 sq pt |
| --- | Critical | (Page 5) : Block overlap detected: 141 sq pt |
| --- | Minor | (Page 1) : Inconsistent body fonts (7 detected): NimbusRomNo9L-Regu, NimbusMonL-Regu, NimbusRomNo9L-Medi, CMR10, NimbusRomNo9L-ReguItal (+2 more) |

## Decision Signals

- **Committee Score**: 4.0/10
- **Editor Verdict**: Desk Reject
- **Reviewer Recommendation**: Major Revision
- **Issue Bundle**: 1 major / 4 moderate / 0 minor

## Revision Roadmap

### Priority 1

- [ ] Abstract and conclusion claims need explicit evidence traceability ([LLM]; abstract)

### Priority 2

- [ ] Cross-section numeric consistency should be reconciled ([LLM]; abstract)
- [ ] Comparison protocol should make fairness assumptions explicit ([LLM]; experiment)
- [ ] Result claims should identify comparison scope and uncertainty ([LLM]; experiment)
- [ ] Novelty claim should be grounded against the closest prior work ([LLM]; related_work)
