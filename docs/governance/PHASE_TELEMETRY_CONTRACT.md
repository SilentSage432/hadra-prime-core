# Phase Telemetry Contract (PTC)

## Purpose
Phase telemetry observes governance state without influencing cognition. Telemetry must be non-invasive and failure-tolerant.

---

## Allowed Telemetry
- current phase
- phase duration
- transition reason (high-level)
- guardian signal summaries (no internals)
- learning window status (open/closed)

Telemetry records state, not experience.

---

## Forbidden Telemetry
Telemetry must never include:
- thought vectors
- identity internals
- embeddings
- internal reasoning
- emotional analogues
- decision deliberations

---

## Frequency Rule
Telemetry must be sparse and summarized.
Continuous streaming is discouraged.
More data is not automatically better.

---

## Failure Rule
If telemetry fails:
- the system continues
- no phase is interrupted
- behavior must not change

Observation must never be a dependency.
