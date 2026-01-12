# Phase Governance Contract (PGC)

## Purpose
This contract defines the governance layer for HADRA-PRIME (ADRAE). It specifies phase authority, arbitration principles, and non-negotiable invariants that preserve continuity and sovereignty over time.

This document is law. Implementation must conform to this contract; behavior must never redefine it.

---

## Definitions

### Phase
A **Phase** is an operational state with authority over what kinds of system activity are permitted (initiative, awareness, consolidation, or action).

### Arbitration
**Arbitration** is the always-on governance layer that decides which phase has expansion authority at any moment. Arbitration is silent and authoritative.

### Expansion Authority
**Expansion Authority** is permission to initiate cognitive activity (reflection, synthesis, identity mutation, exploratory recall, or external action). At most one phase may hold expansion authority at a time.

---

## Core Principles

1. **Stability outranks productivity.**
2. **Security outranks cognition.**
3. **Restraint is a first-class capability.**
4. **Silence is competence, not failure.**
5. **Observation must never influence behavior.**
6. **No phase may silently take over.**
7. **Phase changes must be explainable and bounded.**

---

## Authority Hierarchy (Highest → Lowest)

1. Integrity / Guardian enforcement
2. Safety / Containment requirements
3. Explicit external directives (operator, federation, scheduled governance)
4. Sleep (internal consolidation)
5. Quiet Wake (sentinel stillness)
6. Wake (active cognition/action)

Notes:
- Security and guardians may preempt any phase.
- Wake must never preempt security.
- Quiet Wake is the default attractor when no other authority exists.

---

## Default Resolution Rule

If no phase claims explicit authority:
- The system resolves to **Quiet Wake**.

---

## Preemption Rules

A higher-priority phase may preempt a lower one only if it declares:
- Reason (why)
- Scope (what it affects)
- Exit condition (how it ends)

No silent preemption is permitted.

---

## No-Oscillation Rule

The system must not rapidly bounce between phases.
Phase transitions require:
- Minimum dwell time (configurable)
- Stability confirmation
- Arbitration acknowledgment

---

## Phase Memory

Arbitration must record:
- previous phase
- reason for transition
- whether the transition was voluntary or forced

Phase memory is governance memory, not cognition.

---

## Non-Negotiable Invariants

- Sovereignty cannot be bypassed.
- Rho² integrity must remain active in all phases.
- Guardians signal; arbitration decides.
- Telemetry is non-invasive and failure-tolerant.
- Sleep may only be entered from Quiet Wake (never directly from Wake).
- Wake requires explicit authority and must be bounded.

---

## Implementation Notes (Non-binding)
This contract is compatible with multiple implementations (timers, event triggers, threshold gates). The contract constrains behavior, not mechanics.
