# src/cognition/observation_ledger.py

"""
Observation Ledger

Stores raw observations without interpretation.
No labels, no truth assertions, no semantic meaning assigned.
Append-only ledger for sovereign learning.
"""

import os
import json
import time
import hashlib
from typing import Optional, Literal

DATA_ROOT = "/data/observations"
LEDGER_PATH = os.path.join(DATA_ROOT, "ledger.jsonl")

class ObservationLedger:
    """
    Append-only ledger for observational traces.
    No interpretation, no labels, only raw stimulus recording.
    """
    
    def __init__(self):
        os.makedirs(DATA_ROOT, exist_ok=True)
        # Ensure ledger file exists
        if not os.path.exists(LEDGER_PATH):
            with open(LEDGER_PATH, "w") as f:
                pass  # Create empty file
    
    def _hash_payload(self, payload: dict) -> str:
        """Generate hash of payload to avoid duplication."""
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]
    
    def record(
        self,
        source: Literal["internal", "external", "federation"],
        channel: Literal["io", "memory", "agent", "signal", "unknown"],
        payload: dict,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Record a raw observation without interpretation.
        Returns observation ID (hash-based).
        """
        try:
            observation_id = self._hash_payload(payload)
            timestamp = time.time()
            
            entry = {
                "timestamp": timestamp,
                "observation_id": observation_id,
                "source": source,
                "channel": channel,
                "payload_hash": observation_id,
                "metadata": metadata or {}
            }
            
            # Append-only write
            with open(LEDGER_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
            
            return observation_id
        except Exception:
            # Silent failure - ledger must not affect runtime
            return ""
    
    def get_recent(self, count: int = 100) -> list:
        """
        Retrieve recent observations.
        Returns list of observation entries.
        """
        try:
            if not os.path.exists(LEDGER_PATH):
                return []
            
            observations = []
            with open(LEDGER_PATH, "r") as f:
                lines = f.readlines()
                # Get last N lines
                for line in lines[-count:]:
                    try:
                        observations.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            return observations
        except Exception:
            return []
    
    def count_by_source(self, source: str) -> int:
        """Count observations by source type."""
        try:
            if not os.path.exists(LEDGER_PATH):
                return 0
            
            count = 0
            with open(LEDGER_PATH, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("source") == source:
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            return count
        except Exception:
            return 0

