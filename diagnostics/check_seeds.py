"""
Seed Embedding Diagnostic Tool
-------------------------------
Checks if seed embeddings loaded correctly into PRIME's state.
"""

import sys
import os

# Add project root to path
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

from src.neural.neural_bridge import NeuralBridge

print("=== SEED EMBEDDING DIAGNOSTIC ===")

bridge = NeuralBridge()

seeds = bridge.state.seed_embeddings

print("\nNumber of loaded seed embeddings:", len(seeds))

for i, s in enumerate(seeds):
    print(f"\nSeed {i+1}:")
    print("  text:", s.get("text", "NO TEXT"))
    print("  type:", s.get("type", "NO TYPE"))
    print("  embedding type:", type(s.get("embedding")))
    emb = s.get("embedding")
    if emb is not None:
        if hasattr(emb, "shape"):
            print("  embedding shape:", emb.shape)
        elif hasattr(emb, "__len__"):
            print("  embedding length:", len(emb))
        else:
            print("  embedding: ", emb)
    else:
        print("  embedding: None")

print("\nMemory store seed count:", 
      len([s for s in bridge.memory_store.data.get("thought_events", []) 
           if s.get("timestamp") == "seed"]))

print("\nTotal thought_events in memory_store:", 
      len(bridge.memory_store.data.get("thought_events", [])))

print("\n=== END SEED CHECK ===")

