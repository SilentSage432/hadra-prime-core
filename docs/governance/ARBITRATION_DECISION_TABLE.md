# Arbitration Decision Table

## Purpose
Deterministic phase resolution under all known conditions.
Scope: Phase transitions only (not behavior execution).

---

## Global Precedence Rules (Always Applied First)

| Priority | Rule |
|----------|------|
| P0 | Guardian VETO → Immediate DENY → Remain in current phase |
| P1 | Integrity > Security > Continuity > Authority > Optimization |
| P2 | No transition may skip QUIET_WAKE |
| P3 | No reinterpretation after veto (Choice A enforced) |
| P4 | QUIET_WAKE is the default attractor state |

---

## Phase Transition Decisions

### 1) Any Phase → Any Phase

| Condition | Decision |
|-----------|----------|
| Any guardian emits VETO | DENY, remain in current phase |
| Arbitration engine unavailable | DENY, remain in QUIET_WAKE |
| Phase oscillation threshold exceeded | FORCE_DOWNGRADE → QUIET_WAKE |
| Unknown phase requested | DENY |

---

### 2) QUIET_WAKE → WAKE

| Condition | Decision |
|-----------|----------|
| Operator authority present + no guardian veto | GRANT |
| Scheduled governance task + no veto | GRANT |
| Drift rising but below threshold | DENY |
| No explicit authority | DENY |
| Resource pressure high | DENY |
| Recent WAKE exit within dwell window | DENY |

**Notes:**
- WAKE cannot self-initiate
- Probability weights are ignored here
- "Capability" is not authority

---

### 3) QUIET_WAKE → SLEEP

| Condition | Decision |
|-----------|----------|
| Operator-authorized sleep + no veto | GRANT |
| Scheduled sleep window + stable state | GRANT |
| External stimulus pending | DENY |
| Drift unstable | DENY |
| Not originating from QUIET_WAKE | DENY |

**Invariant:** SLEEP may only be entered from QUIET_WAKE

---

### 4) SLEEP → QUIET_WAKE

| Condition | Decision |
|-----------|----------|
| Sleep window complete | GRANT |
| Guardian escalation | FORCE_EXIT → QUIET_WAKE |
| Operator wake request | GRANT |
| Consolidation incomplete | DENY |

**Invariant:** SLEEP cannot transition directly to WAKE

---

### 5) WAKE → QUIET_WAKE

| Condition | Decision |
|-----------|----------|
| Objective complete | GRANT |
| Authority expired | GRANT |
| Guardian warning (non-veto) | GRANT |
| Drift approaching threshold | GRANT |
| No activity detected | GRANT |

QUIET_WAKE is always a safe downgrade

---

### 6) WAKE → SLEEP

| Condition | Decision |
|-----------|----------|
| Any | DENY |

**Reason:** Must pass through QUIET_WAKE

---

## Learning Window Decisions

### Learning Window Open Request

| Condition | Decision |
|-----------|----------|
| Operator-authorized learning + WAKE + no veto | GRANT |
| Autonomous request | DENY |
| Drift elevated | DENY |
| Learning already active | DENY |

---

### Learning Window Close

| Condition | Decision |
|-----------|----------|
| Duration complete | GRANT |
| Guardian warning | GRANT |
| Guardian veto | FORCE_CLOSE |

---

## Guardian Escalation Overrides

| Guardian Signal | Result |
|-----------------|--------|
| VETO | Immediate DENY |
| ESCALATION (severity ≥4) | FORCE_DOWNGRADE → QUIET_WAKE |
| WARNING | No transition, log only |
| INFO | No effect |

---

## Telemetry Commit Rules (Non-Invasive)

| Event | Telemetry |
|-------|-----------|
| Phase change attempt | Log (reason + result only) |
| Guardian veto | Log (guardian_id + reason_code) |
| Phase granted | Log (from → to + authority) |
| Phase denied | Log (reason_code only) |

**Never logged:**
- Thought vectors
- Deliberation traces
- Emotional analogues
- Internal decision scoring

---

## Absolute Invariants (Cannot Be Overridden)

1. Guardians cannot be argued with
2. Silence is competence
3. QUIET_WAKE is the system's natural state
4. Authority must be explicit
5. Safety outranks progress
6. No learning without containment
7. No wake without reason
8. No sleep without return path

---

## Status

✅ Governance-complete  
✅ Code-agnostic  
✅ Deterministic  
✅ Aligned with RHO² foundation  
✅ Supports long-term continuity
