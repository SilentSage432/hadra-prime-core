# integration/sovereign_client.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SovereignClientConfig:
    sovereign_core_path: str
    enabled: bool = True


class SovereignClient:
    """
    Read-only client for the Sovereign Core (sage-sovereign-core).

    HARD GUARANTEES:
    - No token minting
    - No job routing
    - No execution triggers
    """

    def __init__(self, cfg: SovereignClientConfig):
        self.cfg = cfg
        self._ready = False

    def _ensure_import_path(self) -> None:
        if self._ready:
            return

        path = self.cfg.sovereign_core_path
        if not path or not os.path.isdir(path):
            raise RuntimeError(
                "Invalid SAGE_SOVEREIGN_CORE_PATH. "
                "Set it to the absolute path of sage-sovereign-core."
            )

        if path not in sys.path:
            sys.path.insert(0, path)

        self._ready = True

    def submit_intent(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {
                "intent_id": intent.get("intent_id", "unknown"),
                "status": "DENIED",
                "summary": "Sovereign client disabled",
                "costs": {"time": 0.0, "money": 0.0, "risk": 0.0},
                "policy_feedback": ["client_disabled"],
            }

        self._ensure_import_path()

        # Import after path injection
        from services.adrae_bridge.bridge import submit_intent as _submit  # type: ignore

        return _submit(intent)


def build_client_from_env() -> SovereignClient:
    core_path = os.environ.get("SAGE_SOVEREIGN_CORE_PATH", "").strip()
    enabled = os.environ.get("SAGE_SOVEREIGN_ENABLED", "1").lower() not in ("0", "false")
    return SovereignClient(
        SovereignClientConfig(
            sovereign_core_path=core_path,
            enabled=enabled,
        )
    )
