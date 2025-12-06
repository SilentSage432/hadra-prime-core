import json
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


class LogWriter:

    def __init__(self, filename="prime_runtime.log"):
        self.filename = filename

    def write(self, entry: dict):
        entry["timestamp"] = datetime.utcnow().isoformat()
        # Convert tensors to JSON-serializable format
        serializable_entry = make_json_serializable(entry)
        with open(self.filename, "a") as f:
            f.write(json.dumps(serializable_entry) + "\n")

