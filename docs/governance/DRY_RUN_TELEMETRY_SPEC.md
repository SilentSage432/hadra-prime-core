# Dry-Run Telemetry Specification (DRTS)

## Purpose
Observe what governance would decide without enforcing it.
Validate alignment with runtime reality before any authority exists.

**Mode:** Read-only observation  
**Enforcement:** None (telemetry only)

---

## 1. Purpose & Guarantees

### Purpose
- Observe what governance would decide without enforcing it.
- Validate alignment with runtime reality before any authority exists.

### Hard Guarantees
- Telemetry is append-only
- Telemetry is never read by runtime
- Telemetry cannot influence cognition
- Telemetry cannot influence scheduling
- Telemetry cannot influence action selection

---

## 2. Activation & Scope

### Activation Flag

```yaml
governance:
  dry_run: true
```

### Scope
- Phase inference
- Transition attempts
- Guardian signals
- Arbitration outcomes

### Out of Scope
- Action execution
- Weight changes
- Loop cadence
- Memory writes
- Identity mutation
- External I/O

---

## 3. Telemetry Channels (Separated by Design)

Dry-run telemetry is split into four isolated streams to prevent accidental coupling.

### 3.1 Phase State Stream

**File:** `telemetry/phase_state.jsonl`

Emits current inferred phase only.

```json
{
  "ts": "2026-01-13T08:12:44.203Z",
  "kind": "phase_state",
  "current_phase": "QUIET_WAKE",
  "source": "inference",
  "confidence": 0.93
}
```

**Rules**
- One record per inference window
- No cognition data
- No decision rationale
- No historical lookups

---

### 3.2 Phase Transition Attempt Stream

**File:** `telemetry/phase_transitions.jsonl`

Emits when a phase would change.

```json
{
  "ts": "2026-01-13T08:14:09.882Z",
  "kind": "phase_transition_attempt",
  "from": "QUIET_WAKE",
  "to": "SLEEP",
  "authority": "TIME_WINDOW_ELAPSED",
  "result": "DENIED",
  "denial_reason": "MIN_DWELL_NOT_MET"
}
```

**Rules**
- Emits only on attempted transitions
- Result is `ALLOWED` | `DENIED`
- Denial reasons must be enumerated
- No retries triggered

---

### 3.3 Guardian Signal Stream

**File:** `telemetry/guardian_signals.jsonl`

Emits normalized guardian outputs.

```json
{
  "ts": "2026-01-13T08:14:09.401Z",
  "kind": "guardian_signal",
  "guardian_id": "IDENTITY_DRIFT",
  "signal_type": "VETO",
  "reason_code": "DRIFT_ABOVE_THRESHOLD",
  "severity": 0.82
}
```

**Rules**
- One signal = one record
- Guardians do not know phase state
- Severity is optional
- No aggregation here

---

### 3.4 Arbitration Outcome Stream

**File:** `telemetry/arbitration_outcomes.jsonl`

Emits final arbitration decision summary.

```json
{
  "ts": "2026-01-13T08:14:09.903Z",
  "kind": "arbitration_outcome",
  "candidate_phase": "SLEEP",
  "final_decision": "REJECTED",
  "blocking_guardians": ["IDENTITY_DRIFT"],
  "authority_used": "NONE"
}
```

**Rules**
- Emitted only when arbitration runs
- No partial data
- No deliberation traces
- No internal scores

---

## 4. Timing & Frequency Constraints

### Emission Cadence
- Phase inference: bounded (e.g., every N cycles)
- Guardian signals: event-driven
- Arbitration: event-driven only

### Hard Limits
- No telemetry emission inside inner cognition loops
- No more than one arbitration outcome per cycle
- No burst logging

---

## 5. Strictly Forbidden Fields (Red Lines)

🚫 **The following must never appear in dry-run telemetry:**
- Thought vectors
- Attention weights (full)
- Embedding tensors
- Memory contents
- Decision chains
- Probability distributions
- Action weights
- Emotional analogues
- Internal scores used by cognition

**Violation of this list blocks enforcement forever.**

---

## 6. Storage & Retention

### Storage
- Local filesystem only
- Append-only JSONL
- Rotation via size/time (operator-defined)

### Retention
- Minimum: 24 hours
- Recommended: 72 hours before enforcement
- Old logs can be archived but never replayed

---

## 7. Validation Queries (Human-Only)

### Allowed analyses (offline):
- Phase dwell duration histograms
- Transition denial counts
- Guardian veto frequency
- Arbitration stability over time

### Not allowed:
- Feeding results back into runtime
- Using telemetry to tune weights
- Automated optimization loops

---

## 8. Exit Criteria (Dry-Run → Enforcement)

Dry-run is considered complete when:
- No phase oscillation observed
- Quiet Wake dominates idle periods
- Guardian vetoes align with expectations
- No unexpected transitions appear
- Telemetry volume remains stable
- Zero forbidden fields detected

Only then may Layer 8 be considered.

---

## 9. Relationship to Quiet Wake Philosophy

Dry-run telemetry watches silence without interrupting it.
- Quiet Wake is not "doing nothing"
- It is being ready without acting
- Telemetry observes without pulling attention

This preserves:
- Sovereignty
- Continuity
- Trust
- Silence as armor

---

## Status

✅ Layer 7 defined.  
No code. No injection. No restart.
