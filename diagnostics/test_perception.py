"""
Perception Injection Diagnostic Tool
------------------------------------
Tests if perception injection creates valid embeddings.
"""

import sys
import os

# Add project root to path
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from src.neural.neural_bridge import NeuralBridge

print("=== PERCEPTION DIAGNOSTIC ===")

bridge = NeuralBridge()

sample = "This is a test perception for HADRA-PRIME diagnostics."
p = bridge.inject_perception(sample)

print("\nInjected perception:")
print(p)

state = bridge.state.last_perception

if state:
    print("\nPerception state snapshot:")
    print("  text:", state.get("text", "NO TEXT"))
    emb = state.get("embedding")
    if emb is not None:
        print("  embedding type:", type(emb))
        if hasattr(emb, "shape"):
            print("  embedding shape:", emb.shape)
        elif hasattr(emb, "__len__"):
            print("  embedding length:", len(emb))
        else:
            print("  embedding:", emb)
    else:
        print("  embedding: None")
else:
    print("\n❌ No perception state found!")

print("\n=== END PERCEPTION CHECK ===")

