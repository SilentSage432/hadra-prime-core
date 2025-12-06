# prime-core/cognition/cognitive_growth_scheduler.py

"""
A164 — Self-Directed Cognitive Growth Scheduler
------------------------------------------------

This module gives PRIME:
 - Autonomous control over when to evolve
 - Safety-based stabilization windows
 - Drift-governed evolution dampening
 - Scheduled reflection + memory consolidation cycles
 - Executive control over cognitive growth pacing

This acts like PRIME's proto-prefrontal-executive system.
"""

import time


class CognitiveGrowthScheduler:

    def __init__(self):
        self.last_growth_time = time.time()
        self.last_stability_check = time.time()
        self.cooldown_seconds = 2.0

        # thresholds
        self.max_drift_for_growth = 0.002
        self.min_coherence_for_growth = 0.5

        # history logs
        self.history = []

    def should_trigger_growth(self, drift, coherence, trend):
        """
        Main decision function:
        Returns True when PRIME should run an evolution cycle.
        """

        now = time.time()

        # cooldown
        if now - self.last_growth_time < self.cooldown_seconds:
            return False

        # drift too high → no growth
        if drift is not None and drift > self.max_drift_for_growth:
            return False

        # coherence too low → stabilize first
        if coherence is not None and coherence < self.min_coherence_for_growth:
            return False

        # trajectory analysis
        if trend == "upward":
            return True

        if trend == "stable" or trend == "neutral":
            # allow occasional steady growth
            return (now - self.last_growth_time) > (self.cooldown_seconds * 2)

        # downward trajectory → introspection needed
        return False

    def should_trigger_stability(self, drift, trend):
        """
        Returns True when PRIME should prioritize stabilization.
        """

        # Drift rising or unstable trajectory → stabilize
        if drift is not None and drift > self.max_drift_for_growth:
            return True

        if trend == "unstable" or trend == "downward":
            return True

        return False

    def update(self, drift, coherence, trend):
        """
        Returns a recommended cognitive mode:

            - "evolve"
            - "stabilize"
            - "reflect"
            - "idle"
        """

        mode = "idle"

        if self.should_trigger_stability(drift, trend):
            mode = "stabilize"
        elif self.should_trigger_growth(drift, coherence, trend):
            mode = "evolve"
        elif trend == "unstable" or trend == "downward":
            mode = "reflect"

        # save history
        self.history.append({
            "mode": mode,
            "drift": drift,
            "coherence": coherence,
            "trend": trend,
            "ts": time.time(),
        })

        # Keep history bounded
        if len(self.history) > 100:
            self.history.pop(0)

        if mode == "evolve":
            self.last_growth_time = time.time()

        return mode

