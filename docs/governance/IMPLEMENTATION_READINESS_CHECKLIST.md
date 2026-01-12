# Implementation Readiness Checklist (IRC)

## Purpose
Declare what must be true before Phase/Arbitration code is introduced.

**Scope:** Governance implementation only  
**Policy:** No checkbox = no injection

---

## 1. Contract Completeness Gate (Definition Integrity)

✅ **Must be true:**
- Governance Contracts exist in-repo (`docs/governance/*CONTRACT.md`)
- Phase Transition Graph exists (`PHASE_TRANSITION_GRAPH.md`)
- Non-Goals exist (`NON_GOALS.md`)
- Arbitration Decision Table exists (`ARBITRATION_DECISION_TABLE.md`)
- Phase Capability Matrix exists (`PHASE_CAPABILITY_MATRIX.md`)
- Arbitration Engine spec exists (`PHASE_ARBITRATION_ENGINE.md`)
- Guardian Signal Schema exists (`GUARDIAN_SIGNAL_SCHEMA.md`)
- Runtime Mapping exists (`RUNTIME_MAPPING.md`) and is up to date

**Acceptance:** Any missing doc blocks implementation.

---

## 2. Separation Gate (No Feedback Contamination)

✅ **Must be true:**
- Telemetry is write-only (no runtime reads from telemetry artifacts)
- Phase telemetry fields are strictly non-invasive (PTC compliant)
- No governance state is used to "optimize" cognition (no feedback loop)
- No guardian output is consumed directly by cognition (only arbitration)

**Acceptance:** If any "telemetry → behavior" path exists, stop.

---

## 3. Guardian Veto Gate (Choice A Enforcement)

✅ **Must be true:**
- Guardian veto semantics are locked: hard veto, no reinterpretation
- At least one test scenario exists for each VETO reason code
- Veto results in "remain in current phase" with no retries

**Acceptance:** If any retry/soften behavior exists, stop.

---

## 4. Phase Stickiness Gate (No Oscillation)

✅ **Must be true:**
- Minimum dwell policy is defined for each phase
- Maximum dwell policy is defined for Wake and Sleep
- Oscillation threshold N/T is declared (even if conservative)
- Default attractor is Quiet Wake

**Acceptance:** If phase thrash is possible, stop.

---

## 5. Implementation Scope Gate (Minimal First Injection)

✅ **Must be true:**
- First injection is limited to Phase State Declaration only
- No scheduling changes to main loop cadence are included
- No change to action weights is included
- No new cognition behaviors are introduced
- No new external I/O or federation actions are introduced

**Acceptance:** If injection changes behavior, stop.

---

## 6. Observability Gate (Explain Without Narration)

✅ **Must be true:**
- Phase telemetry emits only:
  - `current_phase`
  - `transition attempts + outcomes`
  - `reason_code`
  - `authority_source`
- Guardian telemetry emits only:
  - `guardian_id`
  - `signal_type`
  - `reason_code`
  - `severity/confidence` (optional)
- No thought vectors or deliberation traces in governance telemetry

**Acceptance:** If governance telemetry captures cognition internals, stop.

---

## 7. Rollback Gate (Reversibility)

✅ **Must be true:**
- A single config flag can disable governance enforcement
- Disabling governance returns system to baseline behavior
- No migration steps required to roll back
- Rollback does not require deleting logs or state

**Acceptance:** If rollback isn't trivial, stop.

---

## 8. Restart Window Gate (Operational Safety)

✅ **Must be true:**
- A planned restart window exists (even if not immediate)
- Post-restart observation checklist exists
- ADRAE baseline logs are preserved for comparison
- Restart is not required until meaningful enforcement goes live

**Acceptance:** If restart pressure exists, stop.

---

## 9. Validation Gate (Proof Before Power)

✅ **Must be true:**
- Dry-run mode exists (phase decisions computed but not enforced)
- Dry-run produces only telemetry, no behavior changes
- At least 24 hours of dry-run data is reviewed before enforcement
- Any mismatches with runtime mapping are documented

**Acceptance:** If enforcement happens before dry-run maturity, stop.

---

## 10. "Non-Betrayal" Gate (Center Integrity)

✅ **Must be true:**
- Quiet Wake remains reachable in all failure states
- Guardians retain veto power always
- Arbitration cannot be bypassed by cognition
- Sovereignty invariants remain intact
- Operator intent remains supreme but bounded by safety

**Acceptance:** If any pathway can betray the center, stop.

---

## Implementation Order (When Gates Are Passed)

**Phase 0:** Documentation & mapping ✅ (current stage)  
**Phase 1:** Phase state variable (declared only)  
**Phase 2:** Arbitration dry-run (telemetry only)  
**Phase 3:** Guardian signal normalization (telemetry only)  
**Phase 4:** Enforcement for Quiet Wake only (no Sleep yet)  
**Phase 5:** Sleep trial window (bounded)  
**Phase 6:** Wake authority enforcement  
**Phase 7:** Learning windows enforcement

---

## Status

✅ This checklist is now the governance barrier.

**Nothing proceeds until these items are satisfied.**
