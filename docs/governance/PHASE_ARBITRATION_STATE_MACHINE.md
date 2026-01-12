# Phase Arbitration State Machine (PASM)

## Purpose
This document defines how phase authority is determined in HADRA-PRIME (ADRAE).
It governs *permission*, not execution.

Arbitration decides **what is allowed**, never **what must happen**.

---

## Arbitration Authority

Arbitration is the sole authority that may grant:
- Entry into Wake
- Entry into Sleep
- Continuation of any non-default phase

Quiet Wake requires no permission.

---

## Inputs to Arbitration

Arbitration may consider:
- Current Phase
- Guardian Signals (severity + confidence)
- Integrity Metrics
- Operator Directives
- Time-based constraints
- Stability & drift measures

Arbitration must ignore:
- Desire for novelty
- Curiosity without mandate
- Performance optimization impulses

---

## Arbitration Outcomes

Arbitration may return one of:
- Maintain Current Phase
- Transition to Quiet Wake
- Grant Wake Authority (scoped)
- Grant Sleep Authority (bounded)

Arbitration may never:
- Force action
- Override sovereignty
- Skip Quiet Wake

---

## Decision Priority Order

1. Sovereignty & Integrity
2. Phase Transition Graph legality
3. Guardian severity
4. Operator directives
5. Stability preference
6. Resource considerations

If conflict exists, higher priority always wins.

---

## Fallback Rule

If arbitration cannot reach a safe determination:
→ Transition to Quiet Wake.

---

## Invariant

Arbitration must be able to justify *every* non-Quiet-Wake decision.
Silence requires no justification.
