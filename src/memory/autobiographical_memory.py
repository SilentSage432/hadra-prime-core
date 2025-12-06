# prime-core/memory/autobiographical_memory.py

"""
Autobiographical Memory Matrix (A170)
-------------------------------------
Stores PRIME's evolving self-history across time.

Tracks:
- identity snapshots
- long-horizon identity deltas
- reflective insights
- cognitive events marked as meaningful
- stability/drift trends
"""

import time


class AutobiographicalMemory:

    def __init__(self):
        self.timeline = []  # ordered sequence of autobiographical entries

    def record_event(self, event_type, identity_vec, long_horizon_vec, reflection=None, drift=None):
        """
        Record an autobiographical event with identity snapshots.
        
        Args:
            event_type: Type of cognitive event (e.g., "generate_reflection", "update_identity")
            identity_vec: Current identity vector snapshot
            long_horizon_vec: Long-horizon identity vector snapshot
            reflection: Reflection embedding if this was a reflection event
            drift: Drift state information
        """
        # Convert vectors to lists for JSON serialization
        identity_snapshot = None
        if identity_vec is not None:
            if hasattr(identity_vec, "tolist"):
                identity_snapshot = identity_vec.tolist()
            elif isinstance(identity_vec, list):
                identity_snapshot = identity_vec[:]
            else:
                identity_snapshot = list(identity_vec) if hasattr(identity_vec, '__iter__') else None
        
        long_horizon_snapshot = None
        if long_horizon_vec is not None:
            if hasattr(long_horizon_vec, "tolist"):
                long_horizon_snapshot = long_horizon_vec.tolist()
            elif isinstance(long_horizon_vec, list):
                long_horizon_snapshot = long_horizon_vec[:]
            else:
                long_horizon_snapshot = list(long_horizon_vec) if hasattr(long_horizon_vec, '__iter__') else None
        
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "identity_snapshot": identity_snapshot,
            "long_horizon_snapshot": long_horizon_snapshot,
            "reflection": reflection,
            "drift": drift,
        }
        
        self.timeline.append(entry)
        
        # Keep timeline bounded (last 1000 entries)
        if len(self.timeline) > 1000:
            self.timeline.pop(0)
        
        return entry

    def get_recent(self, n=10):
        """Get the most recent N autobiographical entries."""
        return self.timeline[-n:] if self.timeline else []

    def summarize(self):
        """Generate a summary of autobiographical memory state."""
        if not self.timeline:
            return {"summary": "No autobiographical data yet."}

        latest = self.timeline[-1]
        return {
            "entries_total": len(self.timeline),
            "latest_event": latest.get("type"),
            "latest_reflection": latest.get("reflection"),
            "oldest_timestamp": self.timeline[0].get("timestamp") if self.timeline else None,
            "newest_timestamp": latest.get("timestamp"),
        }

