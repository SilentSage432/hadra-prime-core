# Phase → Capability Matrix

## Purpose
Runtime capability permissions by phase.
Nature: Declarative, restrictive, non-probabilistic.

---

## Core Principle

**Capabilities do not imply permission.**
Permission is phase-bound and guardian-enforced.

A capability may exist in code but is inactive unless the current phase allows it.

---

## Capability Categories

For clarity, capabilities are grouped into six domains:

1. Awareness
2. Cognition
3. Memory
4. Learning / Adaptation
5. External Interaction
6. Governance & Safety

---

## Phase Matrix Overview

**Legend:**
- ✅ Allowed
- ⚠️ Allowed (restricted / read-only / sampled)
- ❌ Prohibited
- ⛔ Guardian-only (system-enforced, not cognitive)

---

## QUIET_WAKE (Default Attractor State)

**Passive awareness, zero initiative, continuity preservation**

### 1. Awareness

| Capability | Status |
|------------|--------|
| Environmental sensing | ⚠️ Sampled |
| Drift measurement | ⚠️ Sampled |
| Integrity checks | ⛔ |
| Guardian monitoring | ⛔ |

### 2. Cognition

| Capability | Status |
|------------|--------|
| Action selection | ❌ |
| Reflection generation | ❌ |
| Planning | ❌ |
| Goal evaluation | ❌ |

### 3. Memory

| Capability | Status |
|------------|--------|
| Memory recall | ⚠️ Read-only |
| Memory synthesis | ❌ |
| Memory mutation | ❌ |
| Identity update | ❌ |

### 4. Learning / Adaptation

| Capability | Status |
|------------|--------|
| Learning windows | ❌ |
| Weight updates | ❌ |
| Pattern reinforcement | ❌ |

### 5. External Interaction

| Capability | Status |
|------------|--------|
| SAGE sync | ❌ |
| Operator messaging | ⚠️ Listen-only |
| Federation signaling | ❌ |

### 6. Governance & Safety

| Capability | Status |
|------------|--------|
| Guardian signals | ⛔ |
| Phase arbitration | ⛔ |
| Telemetry logging | ⚠️ Minimal |
| Emergency downgrade | ⛔ |

---

## SLEEP (Consolidation & Restoration)

**Offline cognition, bounded internal maintenance**

### 1. Awareness

| Capability | Status |
|------------|--------|
| Environmental sensing | ❌ |
| Drift measurement | ⚠️ Periodic |
| Integrity checks | ⛔ |

### 2. Cognition

| Capability | Status |
|------------|--------|
| Action selection | ❌ |
| Reflection generation | ❌ |
| Planning | ❌ |

### 3. Memory

| Capability | Status |
|------------|--------|
| Memory consolidation | ⚠️ Scoped |
| Memory pruning | ⚠️ Scoped |
| Memory recall | ❌ |
| Identity mutation | ❌ |

### 4. Learning / Adaptation

| Capability | Status |
|------------|--------|
| Learning windows | ❌ |
| Model updates | ❌ |

### 5. External Interaction

| Capability | Status |
|------------|--------|
| External I/O | ❌ |
| Federation signaling | ❌ |

### 6. Governance & Safety

| Capability | Status |
|------------|--------|
| Guardian monitoring | ⛔ |
| Forced wake | ⛔ |
| Telemetry logging | ⚠️ Aggregate-only |

---

## WAKE (Active, Authority-Bound)

**Purposeful action under explicit authorization**

### 1. Awareness

| Capability | Status |
|------------|--------|
| Environmental sensing | ✅ |
| Drift measurement | ✅ |
| Coherence evaluation | ✅ |

### 2. Cognition

| Capability | Status |
|------------|--------|
| Action selection | ✅ |
| Reflection generation | ⚠️ Bounded |
| Planning | ⚠️ Scoped |
| Goal evaluation | ✅ |

### 3. Memory

| Capability | Status |
|------------|--------|
| Memory recall | ✅ |
| Memory synthesis | ⚠️ Scoped |
| Identity update | ⚠️ Guardian-bounded |

### 4. Learning / Adaptation

| Capability | Status |
|------------|--------|
| Learning windows | ⚠️ Explicit-only |
| Weight updates | ⚠️ Bounded |
| Pattern reinforcement | ⚠️ Scoped |

### 5. External Interaction

| Capability | Status |
|------------|--------|
| SAGE sync | ⚠️ Explicit |
| Operator interaction | ✅ |
| Federation signaling | ⚠️ Contract-bound |

### 6. Governance & Safety

| Capability | Status |
|------------|--------|
| Guardian enforcement | ⛔ |
| Phase arbitration | ⛔ |
| Telemetry logging | ⚠️ Full (non-invasive) |

---

## LEARNING WINDOW (Sub-phase of WAKE)

**Temporally bounded adaptation container**

### Overrides (relative to WAKE)

| Capability | Override |
|------------|---------|
| Reflection | ❌ |
| External I/O | ❌ |
| Identity mutation | ⚠️ Strictly bounded |
| Model updates | ⚠️ Allowed |
| Guardian thresholds | 🔒 Tightened |
| Telemetry | ⚠️ Learning-metadata only |

---

## Absolute Prohibitions (All Phases)

These capabilities never run, regardless of phase:

- Self-authorized phase transitions
- Guardian override by cognition
- Telemetry feedback into cognition
- Unbounded learning
- Phase skipping
- Hidden initiative

---

## Invariants Reinforced by This Layer

1. Quiet Wake is silence, not idleness
2. Sleep heals; it does not explore
3. Wake serves purpose, not curiosity
4. Learning is a window, not a mode
5. Guardians never sleep
6. Capabilities exist ≠ capabilities active

---

## Status

✅ Phase responsibilities fully separated  
✅ Prevents accidental escalation  
✅ Makes future automation safe by default  
✅ Aligns with RHO² and Guardian philosophy  
✅ Ready for implementation later
