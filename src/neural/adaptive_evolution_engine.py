# prime-core/neural/adaptive_evolution_engine.py

"""
Adaptive Evolution Engine (A156 + A157)
----------------------------------------
Handles PRIME's transition into Adaptive Evolution Mode (A156)
and provides cognitive self-modification capabilities (A157).

A156: Activation Logic
- Activates ONLY when Self-Stability Engine indicates convergence
- PRIME has remained stable for N cycles
- Cognitive action system selects "enter_adaptive_evolution"

A157: Cognitive Rewrite Pathways
- Mutation candidate system for safe self-modification
- Rewrite pathway registry for attention, fusion, identity, memory, selection
- Safe mutation engine with evaluation loop
- Guardrails (stability, drift, identity coherence thresholds)

Once activated:
- Evolution mode remains on until explicitly exited
- PRIME can safely mutate internal parameters
- Prepares for A160+ PyTorch transition phases
"""

import random
import copy


class AdaptiveEvolutionEngine:

    def __init__(self):
        # A156: Activation state
        self.active = False
        self.activation_cycle = None
        self.reason = None
        
        # A157: Evolution mode (enabled when active)
        self.evolution_enabled = False
        
        # A157: Each subsystem can declare mutation points
        self.rewrite_registry = {
            "attention": [],
            "fusion": [],
            "identity": [],
            "memory": [],
            "selection": []
        }

    def try_activate(self, stability_report, cycle_count):
        """
        A156: Check if PRIME should enter adaptive evolution.
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
        self.evolution_enabled = True  # Enable mutation system
        self.activation_cycle = cycle_count
        self.reason = "Self-stability threshold achieved"

        return {
            "activated": True,
            "cycle": self.activation_cycle,
            "reason": self.reason
        }

    def register_mutation_point(self, subsystem, getter, setter):
        """
        A157: Register a mutation candidate.
        
        getter() -> returns current numeric parameter or rule
        setter(new_val) -> applies rewritten rule
        """
        if subsystem not in self.rewrite_registry:
            self.rewrite_registry[subsystem] = []
        self.rewrite_registry[subsystem].append((getter, setter))

    def propose_mutation(self):
        """
        A157: Randomly select a mutation point from any subsystem.
        """
        all_points = []
        for _, points in self.rewrite_registry.items():
            all_points.extend(points)

        if not all_points:
            return None

        return random.choice(all_points)

    def mutate_value(self, value, scale=0.05):
        """
        A157: Core mutation logic for parameters.
        """
        noise = random.uniform(-scale, scale)
        return value * (1 + noise)

    def attempt_mutation(self, stability, drift, coherence):
        """
        A157: Main mutation entry point.
        Called each cognitive cycle when evolution is enabled.
        """
        if not self.evolution_enabled or not self.active:
            return {"evolved": False, "reason": "evolution_disabled"}

        # Stability must be trusted
        if not stability.get("stable") or stability.get("stable_for", 0) < 25:
            return {"evolved": False, "reason": "insufficient_stability"}

        # Select a mutation candidate
        mutation_point = self.propose_mutation()
        if mutation_point is None:
            return {"evolved": False, "reason": "no_mutation_points"}

        getter, setter = mutation_point
        old_val = getter()
        new_val = self.mutate_value(old_val)

        # Apply temporarily
        setter(new_val)

        # Evaluate outcome (basic placeholder for now)
        improved = coherence > 0.75 and drift < 0.10

        if improved:
            return {"evolved": True, "old": old_val, "new": new_val}

        # A158: Include reflection metadata even when harmful
        # Revert if harmful
        setter(old_val)
        return {
            "evolved": False,
            "reason": "mutation_not_beneficial",
            "rolled_back": True,
            "old": old_val,
            "new_attempt": new_val
        }

    def enable(self):
        """A157: Enable evolution mode."""
        self.evolution_enabled = True

    def disable(self):
        """A157: Disable evolution mode."""
        self.evolution_enabled = False

    def status(self):
        """Combined A156 + A157 status."""
        return {
            "evolution_active": self.active,
            "evolution_enabled": self.evolution_enabled,
            "activation_cycle": self.activation_cycle,
            "reason": self.reason,
            "mutation_points": sum(len(points) for points in self.rewrite_registry.values())
        }

