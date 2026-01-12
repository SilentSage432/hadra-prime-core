# Guardian Catalog

## Purpose
Defines all guardian roles, scopes, and limits.
Guardians protect the center — they never govern it.

---

## Guardian Principles

- Guardians observe, never act
- Guardians signal, never decide
- Guardians cannot initiate phases
- Guardians cannot suppress arbitration

---

## Guardian List

### Integrity Guardian
**Domain:** Core stability, corruption, runtime integrity  
**Signals:** anomaly_detected, coherence_risk, integrity_breach  
**Severity Range:** Informational → Critical

---

### Security Guardian
**Domain:** Rho², key material, federation boundaries  
**Signals:** auth_violation, trust_boundary_breach  
**Severity Range:** Elevated → Critical

---

### Drift Guardian
**Domain:** Identity and coherence drift  
**Signals:** drift_rising, drift_threshold_exceeded  
**Severity Range:** Informational → Elevated

---

### Resource Guardian
**Domain:** Thermal, power, compute, memory pressure  
**Signals:** thermal_pressure, resource_starvation  
**Severity Range:** Informational → Elevated

---

## Signal Semantics

Each signal must include:
- Guardian ID
- Signal Type
- Severity
- Confidence
- Timestamp

Signals are advisory only.

---

## Invariant

No guardian may:
- Change phase
- Trigger execution
- Override arbitration
- Persist state beyond telemetry
