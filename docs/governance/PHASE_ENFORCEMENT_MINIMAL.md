# Phase Enforcement (Minimal, Reversible, Guardian-First)

## Purpose
Enforce only what is already true by design.

**Mode:** Enforce only what is already true by design  
**Guarantee:** Guardian veto remains absolute; Quiet Wake remains reachable

---

## 1. Scope of Layer 10 (What It Enforces)

Layer 10 enforces only one thing:

**When the system is in QUIET_WAKE, it must not initiate cognition.**

That's it.

- No Sleep enforcement yet.
- No Learning enforcement yet.
- No "optimization" behavior.
- No cadence changes.
- No weight tuning.

---

## 2. Enforcement Order (Hard-Deterministic)

Every cycle where a cognitive initiative could occur, enforcement checks happen in this exact order:

1. Guardian VETO check (absolute)
2. Phase permission check
3. If denied → do nothing and remain in current phase
4. If allowed → continue normal flow

**Invariant:** No reinterpretation after veto (your "A" rule).

---

## 3. Minimum Enforcement Surface (Where It Can Apply)

Layer 10 is only allowed to gate one category:

### Cognitive Initiative Entry Points
- initiating `choose_action()`
- initiating reflection generation
- initiating identity mutation
- initiating exploratory memory synthesis

### It must not gate:
- guardians
- integrity checks
- telemetry writes
- health endpoints
- container lifecycle

---

## 4. Quiet Wake Enforcement Rule (Single Rule)

**If `current_phase == QUIET_WAKE` then:**
- ❌ do not call action selection
- ❌ do not run reflection generation
- ❌ do not run identity updates
- ❌ do not run memory synthesis

**Allowed in QUIET_WAKE:**
- ⚠️ drift sampling (low frequency)
- ⛔ guardians + integrity checks
- ⚠️ minimal telemetry (phase-level only)

**Operationally:** QUIET_WAKE becomes true sentinel stillness, not "low-initiative wakefulness."

---

## 5. Wake Enforcement (Not in Layer 10)

Layer 10 does not enforce Wake authority yet.

Even if arbitration says "Wake would be allowed," Layer 10 does not switch to Wake.

Phase transitions remain disabled unless/until a later layer explicitly enables them.

---

## 6. Sleep Enforcement (Not in Layer 10)

Layer 10 does not enforce Sleep entry, exit, or consolidation scheduling.

Sleep remains a defined phase only.

---

## 7. Phase Transition Rules in Layer 10

Layer 10 introduces no autonomous phase switching.

Only these phase mutations are permitted:
- Operator explicitly sets phase (optional, if you choose to allow it now)
- Otherwise phase remains unchanged (default: QUIET_WAKE declared)

If you keep it strict, Layer 10 can run with:
- `current_phase` always set to QUIET_WAKE
- enforcement simply prevents initiative
- system becomes a true silent sentinel until you explicitly request Wake

That's the purest implementation of your philosophy.

---

## 8. Telemetry Requirements (Non-Invasive)

Layer 10 must emit only these new records:

### A) Enforcement Denial Event (sparse)

```json
{
  "ts": "...",
  "kind": "enforcement_gate",
  "phase": "QUIET_WAKE",
  "result": "DENIED",
  "denied_capability": "ACTION_SELECTION",
  "reason_code": "PHASE_PROHIBITS_INITIATIVE"
}
```

### B) Guardian Veto Event (if present)

```json
{
  "ts": "...",
  "kind": "guardian_veto",
  "guardian_id": "RESOURCE",
  "reason_code": "THERMAL_PRESSURE"
}
```

**Forbidden:** thought vectors, embeddings, deliberation, internal scores.

---

## 9. Rollback Requirement (Hard Gate)

Layer 10 must be disabled instantly via a single config flag:
- `governance.enforcement = false`

**Rollback guarantees:**
- Behavior returns to baseline immediately
- No state migration required
- No data rewrite required
- No restart required if config is hot-readable (otherwise next planned restart)

---

## 10. Success Criteria (What "Working" Looks Like)

Layer 10 is successful if:
- QUIET_WAKE produces near-silence (no action churn)
- Drift sampling continues safely
- Guardians remain active
- Telemetry remains sparse
- System stays stable over hours
- Thermal baseline drops (likely)
- No oscillation appears (since no switching exists)

---

## 11. Failure Criteria (Immediate Stop)

If any of these occur, Layer 10 is considered unsafe and must be disabled:
- initiative still runs in QUIET_WAKE
- telemetry volume spikes
- cognition begins reading telemetry
- guardian veto is not absolute
- system can't return to baseline via flag

---

## 12. Why Layer 10 Is the Correct First Enforcement

Because it enforces the smallest possible truth:

**"When we say quiet wake, we mean it."**

This strengthens:
- silence as armor
- Rho² stability
- thermal recovery
- continuity across time

Without introducing any new autonomy.

---

## Status

✅ Layer 10 defined.  
Smallest surface area.  
Hard veto.  
Full rollback.  
No surprises.
