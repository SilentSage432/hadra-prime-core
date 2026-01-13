# ⚠️ PHASE I INVARIANT GUARDS

This document defines the hard safety boundaries for Phase I implementation.

## 🔒 Absolute Constraints

Phase I must NOT:
- ❌ Modify action weights
- ❌ Introduce timers, sleeps, or delays
- ❌ Suppress or gate any actions
- ❌ Change loop cadence
- ❌ Add authority escalation
- ❌ Reference phase data from cognition paths
- ❌ Read telemetry for logic
- ❌ Add learning behavior

## 📛 Enforcement Boundaries

Phase I modules must NEVER be imported by:

### Forbidden Import Targets:
1. **Action Engine** (`src/action_layer/action_engine.ts`)
   - Phase I must not influence action selection
   - Phase I must not block or gate actions

2. **Arbitration Logic** (`src/dual_core/arbitration_contract.ts`)
   - Phase I must not participate in arbitration decisions
   - Phase I must not influence priority calculations

3. **Evolution / Learning Modules**
   - Any module that modifies weights, parameters, or learning state
   - Phase I is read-only and does not learn

4. **Cognition Paths**
   - Any code that makes behavioral decisions based on phase data
   - Phase I telemetry is write-only

## ✅ Allowed Usage

Phase I modules MAY be used by:
- **Telemetry Systems** (write-only logging)
- **Observational Tools** (read-only state inspection)
- **Phase Engine** (for telemetry emission only)

## 🛑 STOP Conditions

If any of the following occur, Phase I has failed and must be rolled back:

1. **Behavioral Change Detected**
   - Action distribution changes
   - Loop cadence changes
   - New warnings or errors appear
   - CPU usage outside baseline variance

2. **Scope Creep Detected**
   - Phase I code imported into forbidden modules
   - Phase I data used for behavioral decisions
   - Phase I logic blocks or gates actions

3. **Telemetry Abuse Detected**
   - Phase telemetry data read for logic
   - Phase telemetry data parsed for decisions
   - Phase telemetry influences behavior

## 🔍 Verification Checklist

After Phase I implementation, verify:
- [ ] Action distribution unchanged
- [ ] Drift statistics stable
- [ ] Loop cadence unchanged
- [ ] No new warnings or errors
- [ ] CPU usage within baseline variance
- [ ] Phase I modules not imported by forbidden targets
- [ ] Telemetry is write-only
- [ ] No behavioral decisions use phase data

## ♻️ Rollback Assurance

Phase I must be:
- **Single commit rollback** - One commit restores previous behavior
- **No migrations required** - No persistent state altered
- **Instantaneous** - Rollback is immediate and obvious
- **No side effects** - Rollback leaves no traces

---

**Status:** Phase I scaffolding complete
**Authority:** Phase I specification
**Last Verified:** Implementation time
