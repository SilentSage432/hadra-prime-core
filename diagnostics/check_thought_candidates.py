"""
Thought Generation Candidate Diagnostic Tool
-------------------------------------------
Checks if thought generator produces candidate vectors.
"""

import sys
import os

# Add project root to path
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from src.neural.neural_bridge import NeuralBridge

print("=== THOUGHT GENERATION DIAGNOSTIC ===")

bridge = NeuralBridge()

# Get candidates using propose_thoughts
candidates = bridge.propose_thoughts()

print("\nNumber of candidate vectors:", len(candidates))

if len(candidates) == 0:
    print("\n❌ NO CANDIDATES — embedding chain is broken.")
    print("\nDebugging info:")
    print("  Seed embeddings count:", len(getattr(bridge.state, "seed_embeddings", [])))
    print("  Last perception:", "exists" if bridge.state.last_perception else "None")
    print("  Task embeddings count:", len(getattr(bridge.state, "task_embeddings", [])))
    print("  Fusion vector:", "exists" if bridge.fusion.last_fusion_vector is not None else "None")
    print("  Attention vector:", "exists" if bridge.attention.last_focus_vector is not None else "None")
    print("  Identity vector:", "exists" if bridge.state.timescales.identity_vector is not None else "None")
else:
    print("\n✅ Candidates generated!")
    print("\nSample candidate inspection:")
    sample = candidates[0]
    if hasattr(sample, "shape"):
        print("  shape:", sample.shape)
        print("  type:", type(sample))
    else:
        try:
            print("  type:", type(sample))
            if hasattr(sample, "__len__"):
                print("  length:", len(sample))
            else:
                print("  value:", sample)
        except Exception as e:
            print("  Cannot inspect candidate format:", e)
    
    print(f"\nTotal candidates: {len(candidates)}")
    print("  - From generator.propose():", len(bridge.generator.propose(
        bridge.fusion.last_fusion_vector,
        bridge.attention.last_focus_vector,
        bridge.state.timescales.identity_vector
    )))
    print("  - Seed embeddings:", len(getattr(bridge.state, "seed_embeddings", [])))
    print("  - Perception:", 1 if bridge.state.last_perception else 0)
    print("  - Tasks:", len(getattr(bridge.state, "task_embeddings", [])))

print("\n=== END THOUGHT CANDIDATE CHECK ===")

