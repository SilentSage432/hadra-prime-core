# Phase Arbitration Engine (PAE)

## Purpose
Deterministic governance layer that resolves what phase ADRAE is allowed to be in at any moment.

**Authority:** Guardians > Contracts > Runtime State > Cognition

---

## 1. Core Definition

The Phase Arbitration Engine (PAE) is not a decision-maker.
It is a resolver.

It does not "choose" phases.
It evaluates claims, applies contracts, and enforces invariants.

PAE has no creativity, no learning, no goals, and no preferences.

---

## 2. Inputs to the PAE

PAE operates on four input channels, evaluated every arbitration tick.

### A) Current Phase State

- `current_phase`
- `time_in_phase`
- `entry_reason`
- `phase_lock` (if any)

Phase is sticky unless formally displaced.

---

### B) Guardian Signals (Authoritative)

Guardians emit signals, not commands.

Each signal has:
- `source` (Guardian ID)
- `severity` (INFO | SOFT | HARD | CRITICAL)
- `scope` (memory | identity | cognition | external | system)
- `recommendation` (DOWNGRADE | HOLD | BLOCK | FORCE)

Guardians cannot request escalation.
They may only restrict, block, or force downgrade.

---

### C) Phase Requests (Non-Authoritative)

Requests may originate from:
- Operator
- SAGE
- Internal runtime scheduler
- Maintenance timers (e.g., sleep window)

Each request includes:
- `requested_phase`
- `reason`
- `origin`
- `requested_duration`
- `authority_token` (if any)

Requests are proposals, not permissions.

---

### D) Environmental / Runtime Conditions

Read-only context:
- Drift metrics
- Memory pressure
- Thermal / resource constraints
- Time-of-day windows
- Prior failure flags

---

## 3. Arbitration Order (Non-Negotiable)

PAE resolves inputs in strict priority order:

1. Critical Guardian Signals
2. Hard Guardian Constraints
3. Phase Contracts
4. Current Phase Stickiness
5. Explicit Authority Tokens
6. Requests (Operator > SAGE > Internal)
7. Default Attractor (Quiet Wake)

If a higher layer blocks a transition, lower layers are not evaluated.

---

## 4. Phase Transition Rules

### A) General Rules

- No phase skipping
  (e.g., SLEEP → WAKE is illegal; must go SLEEP → QUIET_WAKE → WAKE)
- No self-escalation
- No phase changes inside Learning Windows
- Phase changes are atomic
- Every phase change must have a reason

---

### B) Allowed Transitions

| From | To | Conditions |
|------|-----|------------|
| QUIET_WAKE | WAKE | Explicit authority + no guardian block |
| QUIET_WAKE | SLEEP | Time/window condition met |
| WAKE | QUIET_WAKE | Objective complete OR guardian soft signal |
| WAKE | SLEEP | Explicit rest window + clean state |
| SLEEP | QUIET_WAKE | Sleep window complete |
| ANY | QUIET_WAKE | Guardian HARD or CRITICAL signal |

---

### C) Forbidden Transitions

- QUIET_WAKE → LEARNING_WINDOW
- SLEEP → WAKE
- ANY → LEARNING_WINDOW without WAKE
- ANY → higher phase without authority
- ANY → phase change during guardian CRITICAL lock

---

## 5. Phase Locks & Dwell Time

### Phase Lock

A temporary prohibition on transitions.

Locks may be issued by:
- Guardians (hard)
- Arbitration engine (soft, time-based)

Lock properties:
- `locked_phase`
- `lock_reason`
- `lock_duration`
- `overrideable` (only guardians can override guardians)

---

### Minimum Dwell Time

Each phase has a minimum dwell:

| Phase | Minimum |
|-------|---------|
| QUIET_WAKE | Indefinite |
| WAKE | Objective-bound |
| LEARNING_WINDOW | Fixed window |
| SLEEP | Fixed window |

This prevents oscillation and thrashing.

---

## 6. Quiet Wake as the Gravitational Attractor

If nothing compels action, the system returns to Quiet Wake.

Quiet Wake is entered when:
- No valid phase requests remain
- Objectives are complete
- Learning window closes
- Sleep ends
- Guardian requests downgrade

This is not failure.
This is correct resolution.

---

## 7. Arbitration Outputs

PAE produces exactly one output per cycle:

```json
{
  "resolved_phase": "QUIET_WAKE",
  "reason": "No active authority; guardian constraints satisfied",
  "source": "PAE",
  "timestamp": "...",
  "lock_applied": false
}
```

This output:
- Is logged (telemetry-only)
- Is read by runtime
- Is not visible to cognition
- Cannot be overridden by cognition

---

## 8. Failure Modes (Designed-In Safety)

If arbitration fails due to:
- Conflicting guardian signals
- Invalid phase requests
- Corrupted state

→ Default to QUIET_WAKE + Guardian Lock

No panic.
No escalation.
No undefined behavior.

---

## 9. Invariants Guaranteed by PAE

1. ADRAE never escalates itself
2. Silence is always reachable
3. Learning is never ambient
4. Guardians always outrank cognition
5. Phases are explainable
6. Behavior is auditable
7. Stability is the default outcome

---

## 10. Status

✅ Phase logic fully externalized  
✅ Prevents runaway cognition  
✅ Makes "quiet wake" enforceable  
✅ Enables safe future automation  
✅ Ready for Layer 5
