# integration/intent_emitter.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict

from integration.sovereign_client import SovereignClient


@dataclass
class EmitPolicy:
    min_interval_seconds: int = 60
    max_per_hour: int = 30
    min_confidence: float = 0.60
    allowed_actions: tuple[str, ...] = ("sync_with_sage",)


class IntentEmitter:
    """
    Organic, rate-limited intent emitter.
    Observational only.
    """

    def __init__(self, client: SovereignClient, policy: EmitPolicy):
        self.client = client
        self.policy = policy
        self._last_emit_ts = 0.0
        self._hour_bucket_start = time.time()
        self._hour_count = 0

    def _roll_hour(self) -> None:
        now = time.time()
        if now - self._hour_bucket_start >= 3600:
            self._hour_bucket_start = now
            self._hour_count = 0

    def should_emit(self, action: str, confidence: float) -> bool:
        self._roll_hour()
        now = time.time()

        if action not in self.policy.allowed_actions:
            return False
        if confidence < self.policy.min_confidence:
            return False
        if now - self._last_emit_ts < self.policy.min_interval_seconds:
            return False
        if self._hour_count >= self.policy.max_per_hour:
            return False

        return True

    def emit(
        self,
        *,
        action: str,
        goal: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        intent = {
            "intent_id": str(uuid.uuid4()),
            "goal": goal,
            "confidence": float(confidence),
            "requested_scopes": ["sage.sync"],  # SAFE, non-executing
            "context": {
                "action": action,
                **context,
            },
        }

        outcome = self.client.submit_intent(intent)

        self._last_emit_ts = time.time()
        self._hour_count += 1

        return outcome
