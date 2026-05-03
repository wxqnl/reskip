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
