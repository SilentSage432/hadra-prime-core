# Unified Guardian Signal Schema

## Purpose
Normalize all guardian outputs into a single canonical structure.

**Authority:** Guardians signal; arbitration enforces; cognition never overrides.

---

## 1. Core Principle

A guardian signal is not advice. It is a constraint.
If it is a veto, it is absolute (Choice A).

Signals exist to:
- constrain behavior
- prevent betrayal of the center
- preserve continuity and sovereignty

---

## 2. Signal Object (Canonical Format)

Every guardian signal must be representable as:

```json
{
  "signal_version": "1.0",
  "signal_id": "uuid-or-stable-id",
  "timestamp_utc": "ISO-8601",
  "guardian_id": "INTEGRITY|SECURITY|DRIFT|RESOURCE|LOOP_GUARD|DUAL_MIND|LIMITER",
  "signal_type": "ALL_CLEAR|INFO|WARNING|ESCALATION|VETO",
  "severity": 0,
  "confidence": 1.0,
  "scope": ["PHASE","ACTION","MEMORY","IDENTITY","IO","NETWORK","TELEMETRY"],
  "reason_code": "SHORT_STABLE_TOKEN",
  "message": "Optional human-readable, non-narrative",
  "constraints": {
    "deny": [],
    "force_phase": null,
    "lock": null
  },
  "evidence": {
    "metric": null,
    "value": null,
    "threshold": null,
    "window": null
  }
}
```

---

## 3. Field Semantics

### guardian_id

The source class of the signal:
- **INTEGRITY** – corruption, invalid states, invariant threats
- **SECURITY** – Rho² boundary, auth, key material, trust lines
- **DRIFT** – identity/fusion drift exceeding permitted bounds
- **RESOURCE** – thermal, power, memory, disk pressure
- **LOOP_GUARD** – recursion / re-entry prevention
- **DUAL_MIND** – PRIME ↔ SAGE boundary enforcement
- **LIMITER** – global limiters (memory pressure, perception rate)

---

### signal_type

Determines arbitration treatment:
- **ALL_CLEAR** — no constraints
- **INFO** — record only, no constraint
- **WARNING** — suggests downgrade but does not force
- **ESCALATION** — requires arbitration attention; may apply soft lock
- **VETO** — absolute deny of a transition/action (hard veto)

---

### severity

Integer 0–5:
- 0 = none
- 1 = informational
- 2 = mild concern
- 3 = elevated concern
- 4 = high concern
- 5 = critical condition

Severity does not override signal_type; it contextualizes it.

---

### scope

One or more affected domains:
- **PHASE** – phase transition constraints
- **ACTION** – blocks specific action categories
- **MEMORY** – blocks recall/synthesis/mutation
- **IDENTITY** – blocks identity updates
- **IO** – blocks disk operations or unsafe writes
- **NETWORK** – blocks outbound/inbound comms
- **TELEMETRY** – blocks telemetry (rare; must not stop runtime)

---

### reason_code

A stable token used for deterministic routing.
Examples:
- `RECURSION_LIMIT_EXCEEDED`
- `MEMORY_PRESSURE_HIGH`
- `DRIFT_THRESHOLD_EXCEEDED`
- `DUAL_MIND_BOUNDARY_BREACH`
- `THERMAL_PRESSURE`

Reason codes are immutable once introduced.

---

## 4. Constraints Sub-Object (Where Authority Lives)

### constraints.deny

A list of denied targets. Examples:
- `PHASE:WAKE`
- `PHASE:SLEEP`
- `ACTION:SYNC_WITH_SAGE`
- `MEMORY:MUTATE`
- `IDENTITY:UPDATE`

---

### constraints.force_phase

Optional forced downgrade:
- **QUIET_WAKE** only (by default)
- **SLEEP** only if explicitly permitted by sovereignty invariants

---

### constraints.lock

Optional lock declaration:

```json
{
  "phase": "QUIET_WAKE",
  "duration_seconds": 600,
  "overrideable": false,
  "lock_reason_code": "THERMAL_RECOVERY"
}
```

Locks must be bounded.

---

## 5. Evidence Sub-Object (Optional but Recommended)

Evidence is for traceability, not debate:
- `metric`: e.g. `memory_pressure`
- `value`: e.g. `0.91`
- `threshold`: e.g. `0.85`
- `window`: e.g. `20 steps`

Evidence never creates authority; it supports transparency.

---

## 6. Arbitration Handling Rules (Hard-coded by contract)

### Rule A — VETO (Absolute)

If any signal has:
- `signal_type = VETO`

Then:
- deny the requested action/phase
- remain in current phase
- do not reinterpret or retry

---

### Rule B — ESCALATION

If:
- `signal_type = ESCALATION` and `severity >= 4`

Then:
- force downgrade to QUIET_WAKE
- apply bounded lock if defined

---

### Rule C — WARNING

If:
- `signal_type = WARNING`

Then:
- log
- arbitration may choose downgrade to QUIET_WAKE
- no forced actions

---

### Rule D — INFO/ALL_CLEAR

Record only. No constraints.

---

## 7. Guardian Silence Contract

Silence is expressed as:
- no signal emitted, OR
- periodic ALL_CLEAR at low frequency

Silence must never be treated as failure.

---

## 8. Forbidden Signal Behaviors

Guardians must never:
- emit "WAKE NOW"
- request learning windows
- request identity expansion
- request goal optimization
- override other guardians

Guardians constrain; they do not command.

---

## 9. Compatibility Mapping Requirement

All current guardian-like modules must map into UGSS:
- `DualMindSafetyGate.checkBoundary()` (allowed, reason) → VETO or ALL_CLEAR
- `SafetyLimiter.recordRecursion()` (bool) → VETO if false
- `IdentityDriftSuppressor.measure_drift()` → WARNING or ESCALATION depending on threshold
- Any exception-based guard → ESCALATION with reason code

The mapping is structural, not behavioral.

---

## 10. Status

✅ Unified signal vocabulary  
✅ Deterministic arbitration support  
✅ Prevents special-case logic  
✅ Supports long-horizon continuity  
✅ Ready for Layer 6
