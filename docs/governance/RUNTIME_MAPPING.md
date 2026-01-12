# Runtime Mapping: Governance Contracts → Current ADRAE Runtime

## Purpose
Map the governance contracts to current runtime behavior (code + logs). Read-only. No behavioral changes.

## Evidence Sources
- Code: file paths + function names + line anchors (best-effort)
- Logs: prime_runtime.log patterns + representative excerpts
- Constraints: current runtime loop cadence and container orchestration

---

## A) Phase Governance Contract (PGC) Mapping

### Default resolution → Quiet Wake
- Contract says: default is Quiet Wake when no authority exists.
- Current runtime reality:
  - System runs continuous cognitive steps on a timer (main loop).
  - No explicit phase variable observed yet.
- Interpretation:
  - "Quiet Wake" currently manifests as *low-initiative cycles* inside continuous wakefulness.
- Evidence:
  - **Cadence defined:** `main.py:27-28` - `loop_interval=0.35` seconds
  - **Main loop:** `main.py:41` - `while self.running:` with `bridge.cognitive_step()` at line 44
  - **Sleep interval:** `main.py:108` - `time.sleep(self.loop_interval)` between steps
  - **Action selection:** `src/cognition/cognitive_action_engine.py:70` - `choose_action()` uses weighted probabilities
  - **Safety guards:** Multiple recursion guards prevent escalation (see Section E)

### Preemption and authority declarations
- What exists now:
  - **SafetyGuard** (TypeScript: `src/safety/safety_guard.ts`) - recursion and memory pressure checks
  - **PRIME_LOOP_GUARD** (TypeScript: `src/shared/loop_guard.ts`) - prevents recursive entry
  - **DualMindSafetyGate** (TypeScript: `src/safety/dual_mind_safety_gate.ts`) - cross-mind recursion protection
  - **SafetyLimiter** (TypeScript: `src/safety/safety_limiter/limiter.ts`) - recursion counter, memory pressure, perception rate limits
  - **IdentityDriftSuppressor** (Python: `src/identity/identity_drift_suppressor.py:18`) - identity drift correction
- What is missing:
  - explicit arbitration object
  - explicit phase claims with reasons

### No-oscillation
- What exists now:
  - **Loop guard:** `PRIME_LOOP_GUARD.enter()` / `exit()` prevents recursive entry (`src/kernel/index.ts:1183`)
  - **Event-driven protection:** TypeScript cognitive loop is event-driven, not timer-based (`src/kernel/cognitive_loop.ts:17`)
  - **Cooldown mechanism:** `src/kernel/cognitive_loop.ts:22-32` - 250ms cooldown window
- What is missing:
  - phase dwell timers (explicit)

---

## B) Quiet Wake Contract (QWC) Mapping

### Allowed (passive awareness) present?
- **Drift sampling:** YES - `src/identity/identity_drift_suppressor.py:36` - `measure_drift()` method exists
- **Integrity checks:** YES - Multiple safety guards (SafetyGuard, PRIME_LOOP_GUARD) perform integrity checks
- **Passive telemetry:** YES - JSONL logging to `prime_runtime.log` (phase-level data, not thought internals)

### Prohibited (initiative) currently controlled how?
- **Reflection frequency governed by:** Action weights (`src/cognition/cognitive_action_engine.py:56`) - `generate_reflection: 0.30` (30% probability)
- **Identity mutation governed by:** Action weights (`src/cognition/cognitive_action_engine.py:60`) - `update_identity: 0.10` (10% probability) + `IdentityDriftSuppressor` enforces max_drift=0.15
- **Memory synthesis governed by:** Action weights (`src/cognition/cognitive_action_engine.py:54`) - `retrieve_memory: 0.25` (25% probability)

### Break conditions (exit) currently represented by:
- **External input:** Not explicitly mapped (operator input would trigger action selection)
- **Drift threshold:** `IdentityDriftSuppressor.max_drift=0.15` (`src/identity/identity_drift_suppressor.py:27`)
- **Guardian escalation:** SafetyGuard signals recursion/memory pressure (`src/safety/safety_guard.ts`)

---

## C) Sleep Contract (SC) Mapping
- Current state: **No explicit Sleep phase.**
- Closest analogs:
  - **Reduced action selection periods:** Action weights can be modified by path shaping (`src/cognition/cognitive_loop_orchestrator.py:910-923`)
  - **Low novelty + bounded drift windows:** Drift is continuously monitored and bounded (`IdentityDriftSuppressor.max_drift=0.15`)
  - **Consolidation processes:** Memory metabolism occurs in each cognitive step (`src/cognition/cognitive_loop_orchestrator.py`)
- Missing:
  - Explicit "Sleep → Quiet Wake" transition rule enforcement
  - No explicit Sleep entry/exit conditions
  - No bounded sleep window mechanism
  - Sleep can only be entered from Quiet Wake (contract requirement) - **not yet enforced**

---

## D) Wake Contract (WC) Mapping

### Wake requires explicit authority
- Current state:
  - Actions are selected probabilistically via weighted random choice
  - No explicit "authority" check before action selection
- Evidence:
  - **Action selection:** `src/cognition/cognitive_action_engine.py:70` - `choose_action()` uses `random.choices()` with weights
  - **Action weights:** `src/cognition/cognitive_action_engine.py:52-68` - Base weights defined (sync_with_sage: 0.10 = 10%)
  - **sync_with_sage occurrences:** Only 6 references total:
    - `src/cognition/cognitive_action_engine.py:22,62,255` - Action definition and execution
    - `src/cognition/cognitive_loop_orchestrator.py:61,1167` - Intent emission hook (observational only)
    - `src/cognition/supervisory_control_network.py:43` - Supervisory override (weight=1.0 when active)
  - **Mode overrides:** `src/cognition/cognitive_action_engine.py:85-90` - `enter_adaptive_evolution` gated by `ready_for_adaptive_evolution` flag

### Wake bounded by objective/scope/exit
- Current state:
  - Not formalized as a governance envelope
  - Actions execute and return outputs, but no explicit "objective complete" signal
- Evidence:
  - **Action outputs bounded:** Each action returns a result, but no explicit scope/exit conditions
  - **Safety guards prevent recursion storms:** `PRIME_LOOP_GUARD`, `SafetyGuard.limiter`, recursion limits prevent unbounded execution
  - **Event-driven TypeScript loop:** `src/kernel/cognitive_loop.ts:17` - Event-driven, not always-on

---

## E) Learning Windows Contract (LWC) Mapping
- Current state:
  - Evolution activation gates exist
  - "enter_adaptive_evolution" is a special action that requires explicit conditions
- Evidence:
  - **Evolution gate:** `src/cognition/cognitive_action_engine.py:83-90` - Checks `bridge.ready_for_adaptive_evolution` and `not bridge.evolution.active`
  - **Activation conditions:** Requires stability threshold to be reached (`bridge.ready_for_adaptive_evolution`)
  - **Bounded learning:** Evolution engine has `active` flag that prevents re-entry while active
  - **Missing:** No explicit "learning window" with duration/scope/exit conditions defined

---

## F) Guardian Signaling Contract (GSC) Mapping
- Current state:
  - Multiple guardian analogs exist (SafetyGuard, DualMindSafetyGate, SafetyLimiter, IdentityDriftSuppressor)
- Evidence:
  - **SafetyGuard** (`src/safety/safety_guard.ts`):
    - `preCognitionCheck()` - Blocks cognition if recursion/memory pressure too high
    - `snapshot()` - Returns recursion depth, memory pressure, perception rate
    - Used throughout TypeScript codebase (63+ references)
  - **DualMindSafetyGate** (`src/safety/dual_mind_safety_gate.ts:9`):
    - `checkBoundary()` - Prevents cross-mind recursion (PRIME ↔ SAGE)
    - Returns `{allowed: bool, reason: string}` - Signal-like structure
  - **SafetyLimiter** (`src/safety/safety_limiter/limiter.ts`):
    - Tracks recursion count, memory pressure, perception rate
    - `recordRecursion()` - Returns boolean (signal-like)
    - `memoryAllowed()`, `perceptionAllowed()` - Gate checks
  - **IdentityDriftSuppressor** (`src/identity/identity_drift_suppressor.py:36`):
    - `measure_drift()` - Measures identity drift (signal-like)
    - `suppress_drift()` - Corrects drift if exceeds threshold
- Missing:
  - Unified "signal schema" output (each guardian uses different formats)
  - No centralized guardian signal aggregation
  - Signals are used directly, not passed through arbitration layer

---

## G) Phase Telemetry Contract (PTC) Mapping
- Current state:
  - JSONL runtime logging exists (`prime_runtime.log`)
  - Telemetry rollups exist (sovereign-core integration)
- Evidence:
  - **What is recorded** (from log sample):
    - Action names (`"action": "update_identity"`)
    - Drift metrics (`"drift": {"latest_drift": 0.0036, "avg_drift": 0.121}`)
    - Coherence scores (`"coherence": 1.0`)
    - Fusion/attention vectors (previews only)
    - Identity coherence (`"identity_coherence": {"similarity_to_baseline": 0.993}`)
    - Workspace state, goals, personality metrics
  - **What is NOT recorded** (good):
    - No thought vectors (only previews)
    - No internal reasoning chains
    - No decision deliberations
    - No emotional analogues (only metrics)
  - **Logging mechanism:** `src/persistence/log_writer.py` - `LogWriter.write()` method
- Risk check:
  - **Telemetry does not feed back into cognition:** CONFIRMED
    - Logs are write-only (append to file)
    - No code paths found that read from `prime_runtime.log` for decision-making
    - Telemetry is observational only

---

## Summary: What Already Holds vs What Is Future Work

### Already Holds (by behavior)
- **Restraint exists:** Action diversity maintained (sync_with_sage only 10% weight, not dominant)
- **Silence is competence:** System doesn't escalate when no external input (low-initiative cycles continue)
- **Safety outranks action:** Multiple safety guards prevent recursion storms and unsafe states
- **Drift is bounded:** IdentityDriftSuppressor enforces max_drift=0.15 threshold
- **Telemetry is non-invasive:** Logs are write-only, no feedback loops into cognition

### Already Holds (by enforcement)
- **Recursion protection:** PRIME_LOOP_GUARD, SafetyGuard.limiter prevent recursive entry
- **Memory pressure checks:** SafetyLimiter.memoryAllowed() gates memory operations
- **Identity continuity:** IdentityDriftSuppressor corrects drift above threshold
- **Cross-mind safety:** DualMindSafetyGate prevents PRIME ↔ SAGE recursion
- **Event-driven TypeScript loop:** No auto-looping timers (disabled to prevent recursion storms)

### Missing (by design, deferred)
- **No explicit phase state:** No QUIET_WAKE/SLEEP/WAKE enum or variable
- **No explicit arbitration engine:** Action selection is probabilistic, not arbitration-based
- **Sleep is not a declared phase:** Only implicit "low initiative" periods, no Sleep → Quiet Wake transition rule
- **No phase dwell timers:** No minimum time-in-phase requirements
- **No unified guardian signal schema:** Each guardian uses different formats
- **No learning window formalization:** Evolution activation exists but not as bounded "window" with duration/scope
- **No explicit authority declarations:** Actions selected without "why this action now" reasoning
