# prime-core/neural/adaptive_evolution_engine.py

"""
Adaptive Evolution Engine (A156)
--------------------------------
Handles PRIME's transition into Adaptive Evolution Mode.

This engine activates ONLY when:
- Self-Stability Engine (A155) indicates convergence
- PRIME has remained stable for N cycles
- Cognitive action system selects "enter_adaptive_evolution"

Once activated:
- Evolution mode remains on until explicitly exited
- PRIME begins preparing for A160+ PyTorch transition phases
"""


class AdaptiveEvolutionEngine:

    def __init__(self):
        self.active = False
        self.activation_cycle = None
        self.reason = None

    def try_activate(self, stability_report, cycle_count):
        """
        Check if PRIME should enter adaptive evolution.
        """

        if self.active:
            return {
                "activated": True,
                "reason": self.reason,
                "cycle": self.activation_cycle
            }

        if not stability_report or not stability_report.get("ready_for_adaptive_evolution"):
            return {"activated": False, "reason": "Not stable"}

        # If stability is sufficient, activate evolution mode
        self.active = True
        self.activation_cycle = cycle_count
        self.reason = "Self-stability threshold achieved"

        return {
            "activated": True,
            "cycle": self.activation_cycle,
            "reason": self.reason
        }

    def status(self):
        return {
            "evolution_active": self.active,
            "activation_cycle": self.activation_cycle,
            "reason": self.reason
        }

