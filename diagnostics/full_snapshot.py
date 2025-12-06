"""
Full PRIME Mind Snapshot Diagnostic Tool
----------------------------------------
Provides a complete snapshot of PRIME's cognitive state.
"""

import sys
import os

# Add project root to path
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from src.neural.neural_bridge import NeuralBridge

bridge = NeuralBridge()

print("\n=== FULL PRIME SNAPSHOT ===\n")

print("📊 INITIAL STATE:")
print("  Seed embeddings:", len(bridge.state.seed_embeddings))
print("  Last perception:", "exists" if bridge.state.last_perception else "None")
print("  Task embeddings:", len(getattr(bridge.state, "task_embeddings", [])))
print("  Memory store events:", len(bridge.memory_store.data.get("thought_events", [])))

print("\n🧠 NEURAL STATE:")
print("  Last embedding:", "exists" if bridge.state.last_embedding is not None else "None")
print("  Fusion vector:", "exists" if bridge.fusion.last_fusion_vector is not None else "None")
print("  Attention vector:", "exists" if bridge.attention.last_focus_vector is not None else "None")
print("  Identity vector:", "exists" if bridge.state.timescales.identity_vector is not None else "None")

print("\n🔄 DRIFT STATUS:")
drift_status = bridge.state.drift.get_status()
print("  ", drift_status)

print("\n💭 THOUGHT CANDIDATES:")
candidates = bridge.propose_thoughts()
print("  Candidate count:", len(candidates))

print("\n--- Running a cognition step ---")
try:
    output = bridge.cognitive_step()
    
    print("\n📋 COGNITION OUTPUT:")
    print("  Action:", output.get("action"))
    print("  Candidates processed:", len(candidates))
    print("  Thought debug keys:", list(output.get("chosen_thought_debug", {}).keys()) if output.get("chosen_thought_debug") else "None")
    print("  Recalled memories:", len(output.get("recalled_memories", [])))
    print("  Active task:", output.get("active_task"))
    
    print("\n📈 STATE AFTER STEP:")
    print("  Fusion:", "exists" if bridge.fusion.last_fusion_vector is not None else "None")
    print("  Attention:", "exists" if bridge.attention.last_focus_vector is not None else "None")
    print("  Drift history length:", len(bridge.state.drift.history) if hasattr(bridge.state.drift, "history") else "unknown")
    
except Exception as e:
    print(f"\n❌ ERROR during cognitive step: {e}")
    import traceback
    traceback.print_exc()

print("\n=== END SNAPSHOT ===")

