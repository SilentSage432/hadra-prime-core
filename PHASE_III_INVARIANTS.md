# Phase III Invariants

**Purpose:** This document defines the non-negotiable invariants that Phase III must maintain. Any code changes that violate these invariants must be rejected.

---

## 🔒 Non-Negotiable Invariants (Must Not Be Violated)

Phase III must remain **purely observational**. It may **never influence cognition**.

### 1. ❌ No Changes to Action Weights

- Phase III must not modify `CognitiveActionEngine.action_weights`
- Phase III must not alter action selection probabilities
- Phase III must not read action weights for feedback loops

**Verification:** Search codebase for references to `action_weights` from Phase III modules.

---

### 2. ❌ No Changes to Loop Cadence

- Phase III must not modify `main.py` loop `interval` parameter
- Phase III must not inject `time.sleep()` calls
- Phase III must not add blocking operations
- Phase III must not create new timers or schedulers

**Verification:** Phase III code should only use `time.monotonic()` for time checks, never sleep.

---

### 3. ❌ No Blocking, Gating, or Delaying Actions

- Phase III must not prevent any action from executing
- Phase III must not add conditional gates in cognitive paths
- Phase III must not delay the main loop

**Verification:** Phase III observers should be called **after** cognitive steps complete, never before.

---

### 4. ❌ No Feedback into Core Systems

Phase III must never:
- Import or call methods from `NeuralBridge` (except read-only state access)
- Import or call methods from `CognitiveLoopOrchestrator`
- Import or call methods from `CognitiveActionEngine`
- Write to any cognitive state (fusion vectors, attention vectors, identity vectors)
- Modify drift calculations or coherence metrics

**Verification:** Phase III should only read `output` dict from `bridge.cognitive_step()`, never call bridge methods.

---

### 5. ✅ Read-Only Observation

Phase III is **allowed** to:
- Read the `output` dict from each cognitive step
- Read Phase II rhythm payloads (preferred input)
- Read existing runtime state for metrics (drift, coherence, action counts)
- Use `time.monotonic()` and `time.time()` for time tracking

**Verification:** All Phase III methods should be marked with read-only constraints in docstrings.

---

### 6. ✅ Low-Frequency, Append-Only Writes

Phase III is **allowed** to:
- Write JSONL records to `/data/observations/phase3_rotation_windows.jsonl` (on window open/close)
- Write JSONL records to `/data/observations/phase3_rotation_baselines.jsonl` (every 10 minutes)
- Emit console logs with `[ADRAE-ROTATION-WINDOW]` and `[ADRAE-ROTATION-BASELINE]` tags

**Verification:** 
- Writes must be append-only (use `"a"` mode, never `"w"` mode)
- Baseline writes must be rate-limited to 10-minute intervals
- Window writes must only occur on open/close events

---

### 7. ✅ Restart Safety

Phase III must:
- Survive restarts without corruption
- Reinitialize cleanly on restart (no state persistence required for basic operation)
- Not require migrations or schema changes
- Be fully removable in one commit revert

**Verification:** Phase III should not depend on pre-existing files. If files don't exist, it should create them on first write.

---

## 🚫 Explicitly Out of Scope (Phase III Must Not Do)

Phase III must **NOT**:
- Implement Sleep or Wake authority
- Implement phase arbitration or enforcement
- Implement learning windows or cognitive adaptation
- Implement guardian aggregation
- Implement timers, dwell logic, or behavioral triggers
- Parse or reuse telemetry data for logic decisions
- Feed telemetry back into cognition paths

---

## 📁 Module Boundaries

### Phase III Module: `src/telemetry/phase3_rotation_observer.py`

**Allowed Imports:**
- `time` (for monotonic time tracking)
- `json` (for JSONL persistence)
- `os` (for file paths)
- `datetime` (for timestamps)
- `collections` (for data structures)

**Forbidden Imports:**
- `src.neural.neural_bridge` (except read-only access via method parameters)
- `src.cognition.cognitive_action_engine`
- `src.cognition.cognitive_loop_orchestrator`
- Any module that makes behavioral decisions

---

## 🔍 Invariant Verification Checklist

Before committing Phase III code, verify:

- [ ] No `action_weights` modifications
- [ ] No `time.sleep()` calls in Phase III code
- [ ] No blocking operations
- [ ] No imports of cognitive action engines
- [ ] No calls to `bridge.cognitive_step()` from Phase III
- [ ] All writes are append-only (`"a"` mode)
- [ ] Baseline writes are rate-limited to 10-minute intervals
- [ ] Window writes only occur on open/close events
- [ ] Phase III observers called **after** cognitive steps
- [ ] Phase III can be removed in one commit revert
- [ ] No behavioral changes to existing code paths

---

## 📝 Rollback Safety

Phase III code must be:
- **Additive:** New module + small hook points in existing code
- **Removable:** Fully removable in one commit revert
- **Non-migratory:** No migrations or schema changes
- **Non-persistent:** No changes to persisted memory format

**Rollback Procedure:**
1. Remove `src/telemetry/phase3_rotation_observer.py`
2. Remove Phase III imports from `main.py`
3. Remove Phase III hook from `src/observers/rhythm_observer.py`
4. Revert Phase III initialization in `main.py.__init__`
5. Revert Phase III observer calls in `main.py.start()`

This should restore pre-Phase III behavior immediately.

---

## ⚠️ Violation Reporting

If you see Phase III code that:
- Modifies action weights
- Injects sleeps or delays
- Gates or blocks actions
- Feeds back into cognition

**STOP** and report the violation before proceeding.
