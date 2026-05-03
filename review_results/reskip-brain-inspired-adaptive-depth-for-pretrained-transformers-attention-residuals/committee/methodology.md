## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (introduction) "The dual-process distinction~kahneman2011thinking contrasts System~1 (fast, automatic) with System~2 (slow, deliberate); the same dichotomy has direct correlates in cortical processing, where easy stimuli are recognised through shallow feedforward sweeps and harder ones recruit recurrent and deeper circuits~lamme2000distinct, kar2019recurrent, kietzmann2019recurrence, spoerer2020recurrent." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (method) "is the canonical partition for both scales: it sits on the accuracy plateau alongside and halves the per-token router count, which is the dominant factor in the inference-cost result below." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.
