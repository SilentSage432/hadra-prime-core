import json
import os
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def make_json_serializable(obj):
    """
    Recursively convert PyTorch tensors and other non-serializable objects
    to JSON-serializable formats.
    """
    if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except:
            return None


class MemoryStore:

    def __init__(self, filename="prime_memory.json"):
        self.filename = filename
        self.data = {
            "reflections": [],
            "identity_updates": [],
            "thought_events": [],
            "drift_history": [],
            "memory_recall_events": []
        }
        # Load existing memory if present
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    self.data = json.load(f)
            except:
                pass  # Start fresh if corrupted
        
        # Import seed memories
        try:
            from .memory_seed import SEED_MEMORIES
            
            # If memory is empty, inject seed concepts
            if len(self.data["thought_events"]) == 0 and len(self.data["reflections"]) == 0:
                for item in SEED_MEMORIES:
                    self.data["thought_events"].append({
                        "timestamp": "seed",
                        "data": item
                    })
                self.save()
        except ImportError:
            # If seed file doesn't exist, continue without seeds
            pass

    def save(self):
        # Convert tensors to JSON-serializable format before saving
        serializable_data = make_json_serializable(self.data)
        with open(self.filename, "w") as f:
            json.dump(serializable_data, f, indent=2)

    def add_event(self, category, payload):
        # Convert payload to JSON-serializable format
        serializable_payload = make_json_serializable(payload)
        timestamped = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": serializable_payload
        }
        self.data[category].append(timestamped)
        self.save()

    # Convenience helpers
    def log_reflection(self, reflection):
        self.add_event("reflections", reflection)

    def log_identity_update(self, identity):
        self.add_event("identity_updates", identity)

    def log_thought_event(self, dbg):
        self.add_event("thought_events", dbg)

    def log_drift(self, drift_state):
        self.add_event("drift_history", drift_state)

    def log_memory_recall(self, recalled):
        self.add_event("memory_recall_events", recalled)

    def log_perception(self, perception):
        self.add_event("thought_events", {
            "type": "perception",
            "content": perception
        })

