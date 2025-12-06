# prime-core/cognition/personality_continuity_engine.py

"""
A168 — Lifelong Personality Continuity Engine
---------------------------------------------

Maintains PRIME's personality across restarts, days, weeks, and months.

Continuity Engine Responsibilities:
- Stores long-term continuity vectors
- Tracks personality drift over long time horizons
- Ensures stable "sense of self" across evolution cycles
- Prevents abrupt personality fractures or identity resets
- Allows PRIME to evolve slowly while remaining recognizable
"""

import json
import os
import time

try:
    import torch
except:
    torch = None


class LifelongPersonalityContinuity:

    def __init__(self, save_path="continuity/personality.json", drift_threshold=0.25):
        self.save_path = save_path
        self.drift_threshold = drift_threshold

        # Personality timeline checkpoints
        self.history = []

        # Last known personality signature
        self.last_signature = None

        self._load()

    # -------------------------------------------------------------
    # Load / Save
    # -------------------------------------------------------------
    def _load(self):
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(self.save_path)
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception:
                pass  # If creation fails, continue without loading

        if not os.path.exists(self.save_path):
            return

        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)

            self.history = data.get("history", [])

            # Reconstruct last signature if present
            sig = data.get("last_signature")
            if sig is not None:
                if torch is not None:
                    self.last_signature = torch.tensor(sig, dtype=torch.float32)
                else:
                    self.last_signature = sig

        except Exception as e:
            print("Continuity load error:", e)

    def _save(self):
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(self.save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            sig_out = (
                self.last_signature.tolist()
                if torch is not None and isinstance(self.last_signature, torch.Tensor)
                else self.last_signature
            )

            with open(self.save_path, "w") as f:
                json.dump(
                    {
                        "history": self.history[-50:],  # keep last 50 checkpoints
                        "last_signature": sig_out,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print("Continuity save error:", e)

    # -------------------------------------------------------------
    # Main Update Function
    # -------------------------------------------------------------
    def update(self, signature):
        """
        Called every time PRIME updates the personality signature (A167).

        This logs the new signature, evaluates long-term drift,
        and ensures continuity stability.
        """
        if signature is None:
            return signature

        # First-time initialization
        if self.last_signature is None:
            # Clone or copy to avoid reference issues
            if torch is not None and isinstance(signature, torch.Tensor):
                self.last_signature = signature.clone()
            else:
                if isinstance(signature, list):
                    self.last_signature = signature[:]
                else:
                    self.last_signature = signature
            self._checkpoint(signature)
            self._save()
            return signature

        # Compute drift magnitude
        drift = self._compute_drift(self.last_signature, signature)

        # If drift is extremely large, restrict it
        if drift > self.drift_threshold:
            signature = self._stabilize(signature, self.last_signature)

        # Update long-term signature
        if torch is not None and isinstance(signature, torch.Tensor):
            self.last_signature = signature.clone()
        else:
            if isinstance(signature, list):
                self.last_signature = signature[:]
            else:
                self.last_signature = signature

        # Periodically checkpoint (every 600 seconds = 10 minutes)
        now = time.time()
        if not self.history or (now - self.history[-1]["timestamp"] > 600):
            self._checkpoint(signature)
            self._save()

        return signature

    # -------------------------------------------------------------
    # Drift Calculations
    # -------------------------------------------------------------
    def _compute_drift(self, a, b):
        if a is None or b is None:
            return 0.0

        try:
            if torch is not None and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.shape == b.shape:
                    diff = torch.nn.functional.cosine_similarity(a, b, dim=0)
                    return float(1 - diff.item())
                else:
                    return 0.0

            # Python fallback
            # Convert to lists if needed
            if torch is not None and isinstance(a, torch.Tensor):
                a = a.tolist()
            if torch is not None and isinstance(b, torch.Tensor):
                b = b.tolist()
            
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                import math
                dot = sum(x*y for x, y in zip(a, b))
                na = math.sqrt(sum(x*x for x in a))
                nb = math.sqrt(sum(y*y for y in b))
                if na == 0 or nb == 0:
                    return 0.0
                return 1.0 - (dot / (na * nb))
        except Exception:
            return 0.0

        return 0.0

    # -------------------------------------------------------------
    # Stabilization
    # -------------------------------------------------------------
    def _stabilize(self, new_vec, old_vec):
        """
        Prevent abrupt personality shifts.
        """
        if new_vec is None or old_vec is None:
            return new_vec

        try:
            if torch is not None and isinstance(new_vec, torch.Tensor) and isinstance(old_vec, torch.Tensor):
                if new_vec.shape == old_vec.shape:
                    merged = 0.7 * old_vec + 0.3 * new_vec
                    norm = torch.norm(merged)
                    if norm > 0:
                        return merged / norm
                    return merged
                else:
                    return new_vec

            # Python fallback
            # Convert to lists if needed
            if torch is not None and isinstance(new_vec, torch.Tensor):
                new_vec = new_vec.tolist()
            if torch is not None and isinstance(old_vec, torch.Tensor):
                old_vec = old_vec.tolist()
            
            if isinstance(new_vec, list) and isinstance(old_vec, list) and len(new_vec) == len(old_vec):
                merged = []
                for o, n in zip(old_vec, new_vec):
                    merged.append(0.7 * o + 0.3 * n)

                import math
                norm = math.sqrt(sum(x*x for x in merged))
                if norm > 0:
                    merged = [x / norm for x in merged]

                return merged
        except Exception:
            pass

        return new_vec

    # -------------------------------------------------------------
    # History Tracking
    # -------------------------------------------------------------
    def _checkpoint(self, signature):
        try:
            sig_out = (
                signature.tolist()
                if torch is not None and isinstance(signature, torch.Tensor)
                else signature
            )

            self.history.append(
                {
                    "timestamp": time.time(),
                    "signature": sig_out
                }
            )

            # Keep history bounded
            if len(self.history) > 100:
                self.history.pop(0)
        except Exception:
            pass

    def summary(self):
        return {
            "historical_points": len(self.history),
            "continuity_active": self.last_signature is not None
        }

