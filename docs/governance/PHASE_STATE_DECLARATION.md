# Phase State Declaration (PSD)

## Purpose
Introduce a single source of truth for `current_phase` without changing behavior.

**Guarantee:** Runtime cognition continues exactly as today.

---

## 1. What Layer 8 Adds

A single, explicit phase state record that:
- Exists in runtime memory
- Is initialized to QUIET_WAKE by default
- Can be read for telemetry
- Can be written only by governance code (later)
- Does not alter cognition, scheduling, action selection, or memory

**Layer 8 is naming, not controlling.**

---

## 2. Phase Enum (Canonical)

The runtime must use exactly these identifiers (matching docs):
- `QUIET_WAKE`
- `SLEEP`
- `WAKE`

No other values are valid.

---

## 3. Phase State Object (Minimal)

The phase state is a small record:

### Required fields
- `current_phase`: enum
- `entered_at_utc`: ISO-8601 timestamp
- `reason_code`: stable token (e.g., `DEFAULT_ATTRACTOR`)
- `authority_source`: `NONE` | `OPERATOR` | `SCHEDULE` | `FEDERATION` | `GUARDIAN`
- `lock`: optional (null in Layer 8)

### Optional (but recommended)
- `last_transition_id`: UUID
- `time_in_phase_seconds`: computed, not stored

---

## 4. Write Rules (Critical)

**In Layer 8:**
- ✅ Only initialization writes occur
- ✅ Optional: periodic "phase_state" telemetry emits current state
- ❌ No phase transitions
- ❌ No arbitration decisions
- ❌ No guardian influence
- ❌ No gating of actions

**Invariant:** If the phase state ever changes in Layer 8, that is a violation.

---

## 5. Read Rules
- Cognition may read phase state only for telemetry labeling
- Cognition may not branch behavior on phase state (until Layer 10+)

This protects you from "accidental enforcement."

---

## 6. Where This Lives (Conceptual Placement)

To preserve clean boundaries:

### Recommended logical module
- `src/governance/phase_state.py` (Python side)

Or if governance is TypeScript-first in your architecture:
- `src/governance/phase_state.ts`

But since your live cognition loop is Python-driven (`main.py` and `NeuralBridge`), the phase state should be available on the Python side for the first declaration.

---

## 7. Initialization Moment

Phase state initializes at runtime start (process start), not per-step.

**Default initialization:**
- `current_phase = QUIET_WAKE`
- `reason_code = DEFAULT_ATTRACTOR`
- `authority_source = NONE`

This is consistent with your law:

**Quiet Wake is always safe.**

---

## 8. Telemetry During Layer 8 (Allowed)

Layer 8 may emit a minimal record periodically (per Layer 7 spec):

**Example (JSONL):**

```json
{
  "ts": "…",
  "kind": "phase_state",
  "current_phase": "QUIET_WAKE",
  "source": "declared",
  "confidence": 1.0
}
```

No other telemetry is introduced here.

---

## 9. Verification Checklist (Must Pass)

Before moving beyond Layer 8:
- ADRAE behavior unchanged (action distribution stable)
- Loop cadence unchanged
- No new CPU/thermal baseline increase from state tracking
- Phase state remains QUIET_WAKE across hours
- No code branches on phase state exist
- Telemetry contains no forbidden fields

---

## 10. Exit Criteria → Next Layer

Layer 8 is complete when:
- Phase state exists
- It is stable
- It is observed passively
- It has not influenced behavior

Only then do we proceed to:

**Layer 9: Arbitration Dry-Run** (Compute decisions, emit telemetry, enforce nothing)

---

## Status

✅ Layer 8 is now defined with zero enforcement.
