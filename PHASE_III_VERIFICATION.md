# Phase III Verification Guide

**Purpose:** This document provides verification steps to confirm Phase III is operating correctly and maintaining behavioral invariants.

---

## 🎯 Expected Results in Logs

### Within 3 Minutes

You should see:
- Either **nothing** (good - no window opened yet)
- Or a `[ADRAE-ROTATION-WINDOW] OPEN` log if stability conditions are met

**Expected Log Signature:**
```
[ADRAE-ROTATION-WINDOW] OPEN {"timestamp": "...", "uptime_seconds": 180, "steps_per_min": 0, "confidence": 0.95}
```

---

### Within 10 Minutes

You should see the first baseline snapshot:

**Expected Log Signature:**
```
[ADRAE-ROTATION-BASELINE] {"timestamp": "...", "uptime_seconds": 600, "ewma_steps_per_min": 2.5, "top_5_action_proportions": {...}, "total_windows_detected": 0, "average_window_duration_sec": null, "last_restart_settle_estimate_sec": null}
```

---

### Within 30-90 Minutes

You should see:
- Window `CLOSE` events as conditions change naturally
- Additional baseline snapshots every 10 minutes

**Expected Log Signatures:**

Window Close:
```
[ADRAE-ROTATION-WINDOW] CLOSE {"duration_sec": 2100.5, "mean_steps": 1.2, "action_mix": {"retrieve_memory": 45, "analyze_drift": 12}}
```

Baseline (every 10 minutes):
```
[ADRAE-ROTATION-BASELINE] {"timestamp": "...", "uptime_seconds": 1200, "ewma_steps_per_min": 2.8, "top_5_action_proportions": {...}, "total_windows_detected": 1, "average_window_duration_sec": 2100.5, "last_restart_settle_estimate_sec": null}
```

---

## 📁 File Persistence Verification

### JSONL Files Should Exist

Check for JSONL files in `/data/observations/`:

```bash
ls -la /data/observations/phase3_rotation_windows.jsonl
ls -la /data/observations/phase3_rotation_baselines.jsonl
```

### File Format Verification

**Rotation Windows (`phase3_rotation_windows.jsonl`):**
- Each line is a valid JSON object
- Each object contains: `start_timestamp`, `end_timestamp`, `duration_sec`, `mean_steps_per_min`, `peak_steps_per_min`, `action_histogram`
- Only windows meeting minimum duration (30 minutes) are recorded

**Rotation Baselines (`phase3_rotation_baselines.jsonl`):**
- Each line is a valid JSON object
- Each object contains: `timestamp`, `uptime_seconds`, `ewma_steps_per_min`, `top_5_action_proportions`, `total_windows_detected`, `average_window_duration_sec`
- Baseline snapshots appear approximately every 10 minutes (600 seconds)

### Append-Only Verification

Verify files are append-only (new entries appended, never overwritten):

```bash
# Get line count
wc -l /data/observations/phase3_rotation_windows.jsonl
# Wait 10 minutes
# Verify line count increased (or remained same if no windows)
wc -l /data/observations/phase3_rotation_windows.jsonl
```

---

## 🔍 Behavioral Invariant Verification

### Action Distribution Unchanged

**Before Phase III:**
- Record action distribution over 1 hour
- Count actions per minute from `[ADRAE-RHYTHM]` logs

**After Phase III:**
- Record action distribution over 1 hour
- Verify no significant change in action frequencies

**Verification Command:**
```bash
# Extract actions from Phase II rhythm logs
grep "\[ADRAE-RHYTHM\]" prime_runtime.log | jq -r '.telemetry.actions_last_minute' | sort | uniq -c
```

**Expected:** Action distribution should remain stable (within 5% variance).

---

### Cognitive Steps per Minute Unchanged

**Before Phase III:**
- Count `cognitive_steps_last_minute` from `[ADRAE-RHYTHM]` logs

**After Phase III:**
- Count `cognitive_steps_last_minute` from `[ADRAE-RHYTHM]` logs

**Verification Command:**
```bash
# Extract cognitive steps from Phase II rhythm logs
grep "\[ADRAE-RHYTHM\]" prime_runtime.log | jq '.telemetry.cognitive_steps_last_minute' | awk '{sum+=$1; count++} END {print sum/count}'
```

**Expected:** Average cognitive steps per minute should remain stable (within 10% variance).

---

### No Disk Write Bursts

**Verification:** Check that disk writes are low-frequency and append-only:

```bash
# Monitor file write timestamps
watch -n 60 'ls -lht /data/observations/phase3_rotation_baselines.jsonl | head -5'
```

**Expected:**
- Baselines file should update approximately every 10 minutes (not every minute)
- Windows file should only update on window open/close events (infrequent)

---

## 🧪 Phase III Operation Tests

### Test 1: Window Detection

**Condition:** System should enter QUIET_WAKE state with:
- `inferred_state == "QUIET_WAKE"`
- `cognitive_steps_last_minute <= 3`
- `coherence >= 0.99`
- `avg_drift <= 0.15`

**Expected:** Window should open after 3 consecutive stable samples (3 minutes).

**Verification:**
```bash
# Watch for window open log
tail -f prime_runtime.log | grep "\[ADRAE-ROTATION-WINDOW\] OPEN"
```

---

### Test 2: Window Closure

**Condition:** System should exit QUIET_WAKE state or exceed thresholds.

**Expected:** Window should close after 2 consecutive unstable samples (2 minutes).

**Verification:**
```bash
# Watch for window close log
tail -f prime_runtime.log | grep "\[ADRAE-ROTATION-WINDOW\] CLOSE"
```

---

### Test 3: Baseline Emission

**Expected:** Baseline should emit every 10 minutes (600 seconds).

**Verification:**
```bash
# Extract baseline timestamps
grep "\[ADRAE-ROTATION-BASELINE\]" prime_runtime.log | jq -r '.timestamp' | while read ts; do echo "$(date -d "$ts" +%s)"; done | awk 'NR>1 {print $1-prev} {prev=$1}'
```

**Expected:** Time differences between baselines should be approximately 600 seconds (±60 seconds tolerance).

---

### Test 4: Restart Safety

**Test Procedure:**
1. Start ADRAE and let it run for 15 minutes
2. Stop ADRAE (CTRL+C or kill)
3. Restart ADRAE
4. Verify Phase III reinitializes cleanly

**Expected:**
- Phase III should reinitialize without errors
- Baseline snapshots should continue from where they left off
- Window detection should resume correctly

**Verification:**
```bash
# Check for Phase III errors on restart
grep "\[PHASE-III-ERROR\]" prime_runtime.log
```

**Expected:** No errors should appear.

---

## 🚨 Error Detection

### Common Errors to Watch For

1. **Phase III observer not initialized:**
   ```
   [PHASE-III-ERROR] Rotation observer failed: ...
   ```

2. **JSONL write failures:**
   ```
   [PHASE-III-ERROR] Failed to write window record: ...
   [PHASE-III-ERROR] Failed to write baseline record: ...
   ```

3. **Import errors:**
   ```
   ModuleNotFoundError: No module named 'src.telemetry.phase3_rotation_observer'
   ```

### Verification Commands

```bash
# Check for Phase III errors
grep "\[PHASE-III-ERROR\]" prime_runtime.log

# Check for import errors at startup
grep -i "phase.*iii\|rotation.*observer" prime_runtime.log | grep -i error

# Verify Phase III is being called
grep "\[ADRAE-ROTATION" prime_runtime.log | tail -20
```

---

## ✅ Verification Checklist

Before declaring Phase III successful, verify:

- [ ] Window open logs appear when conditions are met (within 3 minutes if stable)
- [ ] Window close logs appear when conditions change (within 30-90 minutes)
- [ ] Baseline snapshots appear every 10 minutes (starting at 10 minutes uptime)
- [ ] JSONL files exist in `/data/observations/`
- [ ] JSONL files contain valid JSON (one object per line)
- [ ] JSONL files are append-only (no overwrites)
- [ ] Action distribution unchanged (within 5% variance)
- [ ] Cognitive steps per minute unchanged (within 10% variance)
- [ ] No disk write bursts (baselines every 10 min, windows on open/close)
- [ ] No Phase III errors in logs
- [ ] Phase III survives restarts cleanly
- [ ] Phase III can be removed via one commit revert (test rollback)

---

## 📊 Baseline Statistics Example

After 1 hour of operation, baseline snapshot might look like:

```json
{
  "timestamp": "2024-01-15T12:00:00.000Z",
  "uptime_seconds": 3600,
  "ewma_steps_per_min": 2.8,
  "top_5_action_proportions": {
    "retrieve_memory": 0.45,
    "analyze_drift": 0.25,
    "generate_reflection": 0.15,
    "update_identity": 0.10,
    "propose_thoughts": 0.05
  },
  "total_windows_detected": 2,
  "average_window_duration_sec": 1800.5,
  "last_restart_settle_estimate_sec": null
}
```

**Interpretation:**
- EWMA steps/min: 2.8 (low, indicates QUIET_WAKE dominance)
- Top action: `retrieve_memory` (45% of actions)
- Windows detected: 2 (system entered stable QUIET_WAKE twice)
- Average window duration: 30 minutes (meets minimum threshold)

---

## 🔄 Rollback Verification

To verify Phase III can be cleanly removed:

1. **Stop ADRAE**
2. **Remove Phase III code:**
   - Delete `src/telemetry/phase3_rotation_observer.py`
   - Remove Phase III imports from `main.py`
   - Remove Phase III hooks from `src/observers/rhythm_observer.py`
3. **Restart ADRAE**
4. **Verify:** System operates normally without Phase III

**Expected:** No errors, system behavior identical to pre-Phase III state.

---

## 📝 Notes

- Phase III logs are **observational only** - they do not affect system behavior
- JSONL files may grow over time - consider archival/rotation policies for long-running systems
- Window detection is conservative (requires 3 consecutive stable samples) to prevent chatter
- Baseline emission is rate-limited to 10 minutes to keep disk IO low
