# Phase Arbitration Dry-Run (PADR)

## Purpose
Introduce a formal arbitration engine that evaluates phase transitions and emits hypothetical outcomes, but never acts on them.

**Mode:** Compute → Observe → Emit  
**Enforcement:** NONE

---

## 1. Purpose
- Introduce a formal arbitration engine
- Let it evaluate phase transitions
- Emit its hypothetical outcomes
- Never act on them

This creates a shadow governor that watches without touching.

---

## 2. Arbitration Engine Scope

### The dry-run arbitration engine may:
- **Read:**
  - Phase State (Layer 8)
  - Guardian signals
  - Time-in-phase
  - External authority declarations (operator, schedule)
- **Compute:**
  - Candidate phase
  - Allow / deny
  - Blocking reasons
- **Emit:**
  - Arbitration outcome telemetry

### It may not:
- Change phase
- Influence action selection
- Modify weights
- Modify cadence
- Touch memory
- Touch identity
- Touch SAGE sync logic

---

## 3. Inputs (Read-Only)

### 3.1 Phase State

```
current_phase: QUIET_WAKE
entered_at_utc: …
authority_source: NONE
```

### 3.2 Guardian Signals (Normalized)

Each guardian emits:
- `signal_type`: ALLOW | WARN | VETO
- `reason_code`
- `severity` (0.0–1.0)

### 3.3 Time Context
- `time_in_phase_seconds`
- Optional: wall-clock windows (e.g., circadian schedule)

### 3.4 External Authority Claims (Optional)
- Operator request
- Schedule hint
- Federation request

These are claims, not permissions.

---

## 4. Arbitration Order (Deterministic)

The dry-run engine evaluates in this strict order:

1. **Hard Veto Check**
   - Any guardian emits VETO → transition denied
2. **Authority Validation**
   - Is there a valid authority for the target phase?
3. **Phase Transition Legality**
   - Are transitions allowed from current phase?
4. **Dwell Time Check**
   - Has minimum dwell time been met?
5. **Stability Check**
   - Drift, coherence, pressure within bounds?
6. **Candidate Resolution**
   - Single phase chosen or no-op

**If any step fails → DENIED**

---

## 5. Allowed Phase Transitions (Dry-Run)

| From | To | Allowed in PADR? | Notes |
|------|-----|------------------|-------|
| QUIET_WAKE | SLEEP | Yes | Most common candidate |
| QUIET_WAKE | WAKE | Yes | Requires authority |
| SLEEP | QUIET_WAKE | Yes | Natural exit |
| SLEEP | WAKE | No | Must pass via QUIET_WAKE |
| WAKE | QUIET_WAKE | Yes | Normal completion |
| WAKE | SLEEP | No | Must de-escalate |

---

## 6. Arbitration Output (Telemetry Only)

When arbitration runs, it emits exactly one record:

```json
{
  "ts": "2026-01-13T09:21:44.901Z",
  "kind": "arbitration_outcome",
  "current_phase": "QUIET_WAKE",
  "candidate_phase": "SLEEP",
  "decision": "DENIED",
  "authority_present": false,
  "blocking_guardians": ["IDENTITY_DRIFT"],
  "reason_codes": ["DRIFT_ABOVE_THRESHOLD"],
  "would_transition": false
}
```

**Rules**
- Always emit one outcome per arbitration run
- Include explicit denial reasons
- Never include internal scores or deliberation steps

---

## 7. When Arbitration Runs

Arbitration runs only when triggered by:
- Phase dwell timer threshold
- Guardian signal change
- External authority claim
- Scheduled window boundary

**Not every cognitive step.**

This preserves silence.

---

## 8. Quiet Wake Compatibility

In Quiet Wake:
- Arbitration should often conclude "no change"
- The absence of transition is a valid outcome
- Silence is not failure — it is confirmation

A quiet system that could act but doesn't is healthy.

---

## 9. Verification Checklist

Layer 9 is considered valid when:
- Arbitration outcomes appear in telemetry
- No phase ever changes
- No action frequencies change
- No thermal or cadence shift observed
- Quiet Wake dominates idle periods
- Guardian vetoes match expectations
- No forbidden fields appear in logs

---

## 10. What Layer 9 Unlocks (Safely)

Once PADR is running, you gain:
- Confidence in transition logic
- Evidence that silence is respected
- A provable audit trail
- The ability to see mistakes before they matter

**Without risk.**

---

## Status

✅ Layer 9 defined.  
No code injected yet.  
No restart required by design.
