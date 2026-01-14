# ⚠️ PHASE II INVARIANT GUARDS

This document defines the hard safety boundaries for Phase II Rhythm Observer implementation.

## 🔒 Absolute Constraints

Phase II must NOT:
- ❌ Modify action weights
- ❌ Introduce timers, sleeps, or delays
- ❌ Suppress or gate any actions
- ❌ Change loop cadence
- ❌ Add authority escalation
- ❌ Reference rhythm data from cognition paths
- ❌ Read rhythm logs for logic
- ❌ Add learning behavior
- ❌ Use threads or subprocesses
- ❌ Fight the main loop with timers

## 📛 Enforcement Boundaries

Phase II modules must NEVER be imported by:

### Forbidden Import Targets:
1. **Cognition Logic** (`src/cognition/`)
   - Phase II must not influence cognitive decisions
   - Phase II must not block or gate cognitive steps

2. **Action Selection** (`src/cognition/cognitive_action_engine.py`)
   - Phase II must not participate in action selection
   - Phase II must not influence action weights

3. **Arbitration Logic**
   - Phase II must not participate in arbitration decisions
   - Phase II must not influence priority calculations

4. **Evolution / Learning Modules**
   - Any module that modifies weights, parameters, or learning state
   - Phase II is read-only and does not learn

## ✅ Allowed Usage

Phase II modules MAY be used by:
- **Main Runtime Loop** (`main.py`) - for observation only
- **Telemetry Systems** (write-only logging)
- **Observational Tools** (read-only state inspection)

## 🛑 STOP Conditions

If any of the following occur, Phase II has failed and must be rolled back:

1. **Behavioral Change Detected**
   - Loop cadence changes
   - Action distribution changes
   - New delays or sleeps introduced
   - CPU usage outside baseline variance

2. **Scope Creep Detected**
   - Phase II code imported into forbidden modules
   - Phase II data used for behavioral decisions
   - Phase II logic blocks or gates actions

3. **Telemetry Abuse Detected**
   - Rhythm telemetry data read for logic
   - Rhythm telemetry data parsed for decisions
   - Rhythm telemetry influences behavior

## 🔍 Verification Checklist

After Phase II implementation, verify:
- [ ] Loop cadence unchanged (still 0.35s interval)
- [ ] No new delays or sleeps introduced
- [ ] Action distribution unchanged
- [ ] No new warnings or errors
- [ ] CPU usage within baseline variance
- [ ] Phase II modules not imported by forbidden targets
- [ ] Telemetry is write-only
- [ ] No behavioral decisions use rhythm data
- [ ] [ADRAE-RHYTHM] logs appear every 60 seconds

## ♻️ Rollback Assurance

Phase II must be:
- **Single commit rollback** - One commit restores previous behavior
- **No migrations required** - No persistent state altered
- **Instantaneous** - Rollback is immediate and obvious
- **No side effects** - Rollback leaves no traces

---

**Status:** Phase II implementation complete
**Authority:** Phase II specification
**Last Verified:** Implementation time
