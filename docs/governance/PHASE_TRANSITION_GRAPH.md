# Phase Transition Graph (PTG)

## Purpose
This document defines the only legal transitions between operational phases in HADRA-PRIME (ADRAE).

Any transition not listed here is forbidden by governance law, regardless of implementation capability.

---

## Phases

- Quiet Wake (QW)
- Sleep (SL)
- Wake (WK)

---

## Legal Transitions

### Quiet Wake → Sleep
**Permitted when:**
- Arbitration grants sleep authority
- Integrity is stable
- No external directive is pending

**Notes:**
- This is the only entry point into Sleep.

---

### Quiet Wake → Wake
**Permitted when:**
- Explicit directive exists (operator / federation / scheduled)
- Guardian escalation requires action
- Arbitration grants Wake authority

---

### Sleep → Quiet Wake
**Permitted when:**
- Sleep window completes
- Guardian signal requests awareness
- Explicit wake directive exists

**Notes:**
- Sleep never exits directly to Wake.

---

### Wake → Quiet Wake
**Permitted when:**
- Objective completes
- Wake window expires
- Arbitration revokes authority
- Stability degrades

**Notes:**
- Quiet Wake is the default post-Wake attractor.

---

## Forbidden Transitions

The following transitions are explicitly illegal:

- Wake → Sleep  
- Sleep → Wake  
- Any → Wake without arbitration  
- Any → Any without declared reason and exit condition  

---

## Emergency Override Rule

In extreme integrity failure:
- Guardians may *request* arbitration override
- Arbitration may force transition to Quiet Wake or Sleep
- Wake is never forced by emergency

---

## Invariant
If ambiguity exists, the system must resolve to **Quiet Wake**.
