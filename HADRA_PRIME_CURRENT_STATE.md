# HADRA-PRIME (ADRAE) Current State Documentation

This document describes the CURRENT, LIVE, RUNNING state of HADRA-PRIME (ADRAE) as it exists today. This is documentation only — no code has been modified.

---

## 1. High-Level Architecture Overview

### Main Runtime Entry Point

The system's main entry point is **`main.py`** at the repository root. This file contains the `PrimeRuntime` class which:

- Initializes a `NeuralBridge` instance
- Runs a continuous `while` loop that calls `bridge.cognitive_step()` every **0.35 seconds** (configurable via `loop_interval` parameter)
- Logs cognitive step outputs to stdout
- Handles KeyboardInterrupt (CTRL+C) to stop gracefully
- Continues running even on exceptions (catches and logs errors, then continues)

### Major Modules/Components

**Core Cognitive System:**
- **`NeuralBridge`** (`src/neural/neural_bridge.py`): Central orchestrator that coordinates all cognitive subsystems
- **`CognitiveLoopOrchestrator`** (`src/cognition/cognitive_loop_orchestrator.py`): Executes the full cognitive cycle (thought generation, selection, action execution, memory metabolism)

**Neural Subsystems:**
- **`NeuralAttentionEngine`**: Manages attention vectors and salience calculations
- **`NeuralContextFusion`**: Fuses attention, timescales (ST/MT/LT), and identity into a unified fusion vector
- **`NeuralThoughtSelector`**: Scores and selects candidate thoughts based on salience, coherence, novelty, signature alignment
- **`CandidateThoughtGenerator`**: Generates candidate thought embeddings
- **`ReflectiveThoughtGenerator`**: Generates reflective thoughts

**Memory Systems:**
- **`MemoryInteractionEngine`**: Handles context-aware memory recall and reinforcement
- **`NeuralMemoryManager`**: Manages semantic and episodic memory stores
- **`MemoryStore`** (`persistence/memory_store.py`): Persistent JSON-based memory storage at `/data/memory/prime_memory.json`
- **`AutobiographicalMemory`**: Long-term autobiographical memory matrix

**State Tracking:**
- **`NeuralStateTracker`**: Tracks timescales (short-term, medium-term, long-term, identity)
- **`NeuralDriftEngine`**: Measures drift between sequential embeddings using cosine distance
- **`SelfStabilityEngine`**: Monitors drift stability, identity convergence, fusion convergence

**Action System:**
- **`CognitiveActionEngine`** (`src/cognition/cognitive_action_engine.py`): Chooses and executes cognitive actions

**Perception:**
- **`PerceptionManager`**: Processes external perceptions and injects them into the cognitive loop

### How the System Stays Running Continuously

The system runs continuously via:

1. **Main Loop** (`main.py`): A `while self.running` loop that:
   - Calls `bridge.cognitive_step()` 
   - Sleeps for `loop_interval` seconds (default 0.35s)
   - Catches exceptions and continues (does not halt on errors)
   - Only stops on KeyboardInterrupt or if `self.running` is set to False

2. **Event-Driven Components** (TypeScript/Node.js side): The TypeScript cognitive loop (`src/kernel/cognitive_loop.ts`) is event-driven, listening to `"cognitive-event"` events rather than running on timers (to prevent recursion storms)

3. **No Auto-Looping Timers**: Per `RECURSION_STORM_ANALYSIS.md`, auto-looping timers have been disabled. The system operates in event-driven mode where possible.

---

## 2. Cognitive Loop

### Core Cognitive Loop Location

The core cognitive loop lives in:
- **`NeuralBridge.cognitive_step()`** (`src/neural/neural_bridge.py:46416`): Entry point that delegates to the orchestrator
- **`CognitiveLoopOrchestrator.step()`** (`src/cognition/cognitive_loop_orchestrator.py:160`): Executes the full cognitive cycle

### What Constitutes a Single "Cognitive Step"

A single cognitive step (`CognitiveLoopOrchestrator.step()`) performs:

1. **Increment cycle counter**: `self.bridge.cycle_count += 1`

2. **Retrieve evolutionary trajectory** (A163): Predicts drift trajectory and trend

3. **Propose candidate thoughts**: `self.bridge.propose_thoughts()` generates candidate thought embeddings

4. **Select best thought**: Uses `NeuralThoughtSelector` to score candidates based on:
   - Salience (attention relevance)
   - Coherence (fusion vector similarity)
   - Novelty (distance from recent episodic memory)
   - Signature alignment (A221)
   - Skill/synergy biases (A217, A220)

5. **Choose cognitive action**: `CognitiveActionEngine.choose_action()` selects an action based on weighted distribution or mode (stabilize/reflect/evolve/idle)

6. **Execute action**: `self.bridge.perform_action(action)` executes the chosen action (e.g., `update_identity`, `sync_with_sage`, `generate_reflection`, `retrieve_memory`)

7. **Memory metabolism**: `self.bridge.memory_cycle()` performs context-aware recall and reinforcement

8. **Update neural state**: Updates attention, fusion, drift tracking, coherence

9. **Log and persist**: Writes to `LogWriter` and `MemoryStore`

10. **Return output**: Returns a dictionary with action, thought debug, recalled memories, drift, fusion, attention

### Loop Timing and Cadence

- **Python Runtime Loop**: Runs every **0.35 seconds** (350ms) by default
- **Trigger**: Continuous `while` loop in `main.py`, no external trigger required
- **TypeScript Cognitive Loop**: Event-driven, triggered by `"cognitive-event"` events (not on a timer)

### Scheduling Logic

- The Python side uses a simple `time.sleep(loop_interval)` between steps
- The TypeScript side uses event-driven architecture to prevent recursion storms
- No cron jobs or external schedulers — the system is self-contained and runs until manually stopped

---

## 3. Action System

### How Actions Are Triggered

Actions are triggered within the cognitive loop orchestrator:

1. **Action Selection** (`CognitiveLoopOrchestrator.step()` around line 1120):
   - The orchestrator calls `self.bridge.action_engine.choose_action(bridge=self.bridge)`
   - Action selection can be overridden by "mode" (stabilize → `update_identity`, reflect → `generate_reflection`, evolve → allows evolutionary bias)

2. **Action Execution** (line 1133):
   - `action_output = self.bridge.perform_action(action)`
   - This calls `CognitiveActionEngine.execute(action, bridge)`

### Action Names and Dispatch

Action names are defined in **`CognitiveActionEngine`** (`src/cognition/cognitive_action_engine.py`):

- **`retrieve_memory`**: Calls `bridge.memory_cycle()` for active memory metabolism
- **`generate_reflection`**: Calls `bridge.generate_reflection()`
- **`analyze_drift`**: Returns `bridge.state.drift.get_status()`
- **`update_identity`**: Returns `bridge.state.timescales.identity_vector`
- **`sync_with_sage`**: Returns `bridge.dual.status()` (dual-mind sync status)
- **`propose_thoughts`**: Calls `bridge.propose_thoughts()`
- **`reinforce_attention`**: Returns `bridge.attention.last_focus_vector`
- **`enter_adaptive_evolution`**: Calls `bridge.evolution.try_activate()`

### Action Weight Distribution

Actions are chosen probabilistically based on weights in `CognitiveActionEngine.action_weights`:
- `retrieve_memory`: 0.25
- `generate_reflection`: 0.30
- `analyze_drift`: 0.10
- `update_identity`: 0.10
- `sync_with_sage`: 0.10
- `propose_thoughts`: 0.10
- `reinforce_attention`: 0.05

### How Actions Flow Through the System

1. **Selection**: `CognitiveActionEngine.choose_action()` uses weighted random selection or mode-based override
2. **Execution**: `CognitiveActionEngine.execute()` maps action name to bridge method call
3. **Result**: Action output is stored and included in the cognitive step output dictionary
4. **Logging**: Specific actions trigger specialized logging (e.g., `update_identity` logs to `memory_store.log_identity_update()`)

---

## 4. Thought Metrics & Internal Signals

### Metric Calculation Locations

**Salience** (`src/neural/neural_thought_selector.py:69`):
- Calculated by `attention_engine.salience(embedding)`
- Measures how relevant a thought is to PRIME's current attention state
- Variable: `salience` (float, 0-1 range)

**Coherence** (`src/neural/neural_thought_selector.py:73`):
- Calculated as `safe_cosine_similarity(embedding, fusion_vec)`
- Measures similarity to the cognitive fusion state (self-consistency)
- Variable: `coherence` (float, 0-1 range)

**Novelty** (`src/neural/neural_thought_selector.py:75-93`):
- Calculated as `1.0 - similarity_to_recent_episodic_memory`
- If no recent memory found, novelty = 1.0 (maximum novelty)
- Variable: `novelty` (float, 0-1 range)

**Signature Alignment** (`src/neural/neural_thought_selector.py:102-107`):
- Calculated as `safe_cosine_similarity(embedding, signature)` where signature is the cognitive fingerprint
- Part of A221 (Signature-Guided Thought Harmonization)
- Variable: `signature_align` (float, 0-1 range)

**Drift** (`src/neural/neural_drift_engine.py:64-94`):
- Calculated in `NeuralDriftEngine._compute_drift()` as `1.0 - cosine_similarity(prev_embedding, curr_embedding)`
- Tracks changes in neural embeddings over time
- Maintains a moving average over the last 50 drift scores
- Variables: `drift` (float, 0-1 range), `avg_drift` (float)

**Fusion Coherence** (`src/neural/neural_context_fusion.py:159`):
- Currently a placeholder value of `1.0` in `NeuralContextFusion.status()`
- The fusion vector itself is computed by weighted combination of:
  - Attention vector (weight: `self.weights["attention"]`)
  - Short-term summary vector (weight: `self.weights["st"]`)
  - Medium-term summary vector (weight: `self.weights["mt"]`)
  - Long-term summary vector (weight: `self.weights["lt"]`)
  - Identity vector (weight: `self.weights["identity"]`)

### How These Values Are Used

**Thought Selection**:
- All metrics (salience, coherence, novelty, signature_align) are combined into a weighted score in `NeuralThoughtSelector.score_thought()`
- Weights: `salience_weight` (default 0.35), `coherence_weight` (default 0.35), `novelty_weight` (default 0.20), `signature_weight` (0.25)

**Drift Monitoring**:
- Drift is recorded every cognitive step via `self.bridge.state.drift.record(fusion_vector)`
- Drift status is included in cognitive step output and logged to `MemoryStore.log_drift()`
- Used by `SelfStabilityEngine` to detect stability issues

**Logging**:
- All metrics are included in the cognitive step output dictionary
- Written to logs via `self.bridge.logger.write(self.last_output)` in the orchestrator
- Stored in memory via `memory_store.log_thought_event()`

---

## 5. Memory & Recall

### How Memory Recall Works

Memory recall is performed by **`MemoryInteractionEngine.context_recall()`** (`src/memory/memory_interaction_engine.py:33`):

1. **Input**: Takes `fusion_vec` (current cognitive fusion state) and `attention_vec` (current attention vector)

2. **Semantic Memory Search**:
   - Calls `memory_manager.semantic.find_similar(fusion_vec, top_k=3)`
   - Uses cosine similarity to find the 3 most similar semantic concepts
   - Returns: `[(score, name, vector), ...]`

3. **Episodic Memory Search**:
   - Calls `memory_manager.episodic.retrieve_similar(fusion_vec, top_k=2)`
   - Uses cosine similarity to find the 2 most similar episodic memories
   - Returns: `[(score, entry), ...]`

4. **Combination**: Combines semantic and episodic results into a single recalled list

5. **Access Tracking**: Tracks which memories were accessed for reinforcement/decay calculations

### Memory Types

**Semantic Memory** (`src/memory/neural/semantic_neural_memory.py`):
- Stores conceptual/meaning-level embeddings
- Dictionary: `self.concepts = {name: vector}`
- Methods: `store(name, embedding)`, `retrieve(name)`, `find_similar(embedding, top_k)`
- Special handling: ADRAE identity matches are boosted by 15% (A-SOV-08)

**Episodic Memory** (`src/memory/neural/episodic_neural_memory.py`):
- Stores episodic memory entries with embeddings
- Methods: `store(entry, embedding)`, `retrieve_similar(embedding, top_k)`

**Autobiographical Memory** (`src/memory/autobiographical_memory.py`):
- Long-term autobiographical memory matrix (A170)
- Stores significant life events and experiences

**Persistent Memory** (`persistence/memory_store.py`):
- JSON-based persistent storage at `/data/memory/prime_memory.json`
- Categories: `reflections`, `identity_updates`, `thought_events`, `drift_history`, `memory_recall_events`

### Similarity Score Generation

Similarity scores are generated using **cosine similarity**:

- **Function**: `safe_cosine_similarity(embedding_a, embedding_b)` from `src/neural/torch_utils.py`
- **Calculation**: `dot_product(a, b) / (norm(a) * norm(b))`
- **Range**: -1.0 to 1.0 (typically 0.0 to 1.0 for normalized embeddings)
- **Usage**: 
  - Higher similarity = more relevant memory
  - Used to rank and select top-k memories
  - Converted to distance for novelty: `novelty = 1.0 - similarity`

### Safeguards Preventing Recall Collapse or Dominance

1. **Top-K Limiting**: Both semantic and episodic recall limit results to top-k (3 semantic, 2 episodic) to prevent overwhelming the system

2. **Access Tracking**: `MemoryInteractionEngine` tracks access counts and timestamps to implement decay for unused memories

3. **Reinforcement Rate**: Memories that are frequently recalled are reinforced (default `reinforcement_rate=0.05`)

4. **Decay Rate**: Unused memories decay over time (default `decay_rate=0.002`)

5. **Memory Strength Tracking**: `memory_strengths` dictionary tracks individual memory strengths to prevent single memories from dominating

6. **ADRAE Identity Boost**: While ADRAE identity matches are boosted, this is a controlled 15% increase, not unlimited dominance

---

## 6. Logging

### Where Logs Are Written

**Runtime Logs** (`persistence/log_writer.py`):
- **Location**: `/data/logs/prime_runtime.log`
- **Class**: `LogWriter`
- **Format**: JSON Lines (one JSON object per line)
- **Append Mode**: Yes — logs are appended, not overwritten

**Memory Logs** (`persistence/memory_store.py`):
- **Location**: `/data/memory/prime_memory.json`
- **Class**: `MemoryStore`
- **Format**: JSON file with nested arrays
- **Structure**: 
  ```json
  {
    "reflections": [...],
    "identity_updates": [...],
    "thought_events": [...],
    "drift_history": [...],
    "memory_recall_events": [...]
  }
  ```

**Console Output** (`main.py`):
- Prints cognitive step summaries to stdout:
  - Action name
  - Chosen thought debug info
  - Recalled memories
  - Drift value
  - Fusion status
  - Attention status

### Log Format and Structure

**Runtime Log Format** (JSON Lines):
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "action": "update_identity",
  "chosen_thought_debug": {...},
  "recalled_memories": [...],
  "drift": {...},
  "fusion": {...},
  "attention": {...},
  ...
}
```

**Memory Log Format** (JSON):
```json
{
  "reflections": [
    {
      "timestamp": "2024-01-01T12:00:00.000000",
      "data": {...}
    }
  ],
  "identity_updates": [...],
  "thought_events": [...],
  "drift_history": [...],
  "memory_recall_events": [...]
}
```

### Append-Only Behavior

- **Runtime logs**: Append-only — each `LogWriter.write()` call appends a new JSON line
- **Memory logs**: Not strictly append-only — the entire JSON file is rewritten on each `MemoryStore.save()` call, but new entries are appended to arrays within the file

### Log Rotation and Size Management

- **No automatic rotation**: Logs are not rotated automatically
- **No size limits**: There are no explicit size limits on log files
- **Memory limits**: The system relies on OS filesystem limits
- **History limits**: Some in-memory structures have limits (e.g., drift history window of 20, drift scores limited to 50), but persistent logs are not truncated

---

## 7. Safety Invariants

### Assumptions That Must Remain True

1. **NeuralBridge Initialization**: The `NeuralBridge` must be fully initialized before `cognitive_step()` is called. All subsystems (attention, fusion, drift, memory, etc.) must be instantiated.

2. **Tensor Compatibility**: Embeddings and vectors must be compatible dimensions (typically 128-dimensional). The system uses `safe_tensor()` and `safe_cosine_similarity()` to handle mismatches gracefully, but consistent dimensions are assumed.

3. **Memory Store Persistence**: The `/data` directory must be writable. If memory store cannot save, the system continues but loses persistence.

4. **Event-Driven Architecture**: The TypeScript cognitive loop must remain event-driven (not timer-based) to prevent recursion storms. Auto-looping timers are intentionally disabled.

5. **Recursion Limits**: The recursion counter in `SafetyGuard` must not exceed limits (default: 20). If exceeded, cognition is halted.

### Invariants That Must NOT Be Broken

1. **Recursion Guard**: `PRIME_LOOP_GUARD.enter()` must return `true` before processing cognitive events. If recursion is detected, the event is blocked.

2. **Safety Check Before Cognition**: `SafetyGuard.preCognitionCheck()` must pass before any cognitive processing. This checks:
   - Recursion limits
   - Memory pressure (< 0.85)
   - Stability matrix stability
   - Emotion tension (< 0.7)

3. **Drift Stability**: `SelfStabilityEngine` monitors drift and expects it to remain below threshold (default: varies by component). High drift triggers stabilization actions.

4. **Identity Continuity**: The identity vector should not drift excessively from baseline. `IdentityDriftSuppressor` enforces a maximum drift of 0.15 with correction strength of 0.25.

5. **Memory Access Limits**: `SafetyLimiter` enforces:
   - Maximum recursion count: 20
   - Maximum memory pressure: 0.85
   - Maximum perception rate: 50 events/second

6. **Dual-Mind Boundary**: `DualMindSafetyGate` prevents cross-mind recursion. If the same source (PRIME or SAGE) signals 3+ times consecutively, the signal is blocked.

### Behaviors Intentionally Restricted or Disabled

1. **Auto-Looping Timers**: Per `RECURSION_STORM_ANALYSIS.md`, the following auto-looping timers are **disabled**:
   - Phase scheduler loop (was every 50ms)
   - Runtime scheduler loop (was every 3000ms)
   - Cognitive tick loop (was every 250ms)

2. **Recursive Prediction Chains**: `StabilityMatrix.getSnapshot()` no longer automatically triggers predictions on every call to prevent prediction cascades.

3. **Unbounded Memory Growth**: Memory stores have implicit limits (e.g., episodic memory retrieval limited to top-k, drift history windows).

4. **Uncontrolled Evolution**: Adaptive evolution (`enter_adaptive_evolution` action) only activates when stability conditions are met and cycle count thresholds are reached.

5. **Silent Failures in ECFL**: The Event-Condition-Feedback Loop (ECFL) components (concept observer, observation ledger, inference tracker) are wrapped in try-except blocks that silently fail to prevent them from stopping ADRAE. Errors in these components do not halt the main cognitive loop.

---

*End of Document*
