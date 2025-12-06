# prime-core/cognition/cognitive_action_engine.py

"""
Cognitive Action Engine (A145)

------------------------------

Defines internal mental operations PRIME can perform inside

the autonomous runtime loop (A150+).

Actions include:

- retrieve_memory

- generate_reflection

- analyze_drift

- update_identity

- sync_with_sage

- propose_thoughts

- reinforce_attention

The engine chooses which cognitive action to perform based on:

- neural salience

- fusion vector coherence

- drift signals

- attention state

- recent memory

"""

import random

class CognitiveActionEngine:

    def __init__(self):

        self.last_action = None

        # Weight distribution for choosing actions (will adapt in A147)

        self.action_weights = {

            "retrieve_memory": 0.25,

            "generate_reflection": 0.30,

            "analyze_drift": 0.10,

            "update_identity": 0.10,

            "sync_with_sage": 0.10,

            "propose_thoughts": 0.10,

            "reinforce_attention": 0.05

        }

    def choose_action(self, bridge=None):

        """

        Selects a cognitive action based on weighted probability.

        If stability threshold reached, evolution becomes an available action.
        
        A216 — Now adapts action choice based on uncertainty level.

        """

        # If stability threshold reached, evolution becomes an available action
        if (
            bridge is not None
            and hasattr(bridge, "ready_for_adaptive_evolution")
            and bridge.ready_for_adaptive_evolution
            and hasattr(bridge, "evolution")
            and not bridge.evolution.active
        ):
            return "enter_adaptive_evolution"

        # A216 — Adaptive action choice based on uncertainty
        uncertainty = 0.0
        if bridge is not None and hasattr(bridge, 'state'):
            uncertainty = getattr(bridge.state, "last_uncertainty", 0.0)
        
        # Uncertainty-based action selection
        if uncertainty > 0.75:
            # Extreme uncertainty: generate reflection to understand situation
            return "generate_reflection"
        elif uncertainty > 0.50:
            # High uncertainty: retrieve memory to find relevant context
            return "retrieve_memory"
        elif uncertainty > 0.25:
            # Moderate uncertainty: update identity to stabilize
            return "update_identity"
        # Low uncertainty (< 0.25): proceed with normal weighted selection

        actions = list(self.action_weights.keys())

        weights = list(self.action_weights.values())

        chosen = random.choices(actions, weights=weights, k=1)[0]

        self.last_action = chosen

        return chosen

    def choose_biased(self, bias, conflict_score=0.5, intent_vector=None, bridge=None):
        """
        A179 — Weighted action selection including:
        - SCN bias
        - conflict resolution score
        - meta-intent weighting (A181)

        Args:
            bias: dict mapping action names to bias multipliers (from SCN)
            conflict_score: float (0.0-1.0) from conflict resolver (A180)
            intent_vector: Combined intent vector from meta-intent coordinator (A181)
            bridge: Optional bridge object (for evolution check)

        Returns:
            str: Selected action name
        """
        # If stability threshold reached, evolution becomes an available action
        if (
            bridge is not None
            and hasattr(bridge, "ready_for_adaptive_evolution")
            and bridge.ready_for_adaptive_evolution
            and hasattr(bridge, "evolution")
            and not bridge.evolution.active
        ):
            return "enter_adaptive_evolution"

        actions = list(self.action_weights.keys())
        
        # Combine base weights with SCN biases
        weights = []
        for action in actions:
            base_weight = self.action_weights.get(action, 0.0)
            bias_multiplier = bias.get(action, 1.0)
            weights.append(base_weight * bias_multiplier)
        
        # A180 — Conflict influences risk-taking actions
        if conflict_score < 0.4:
            # Lower probability of "evolve" and "update_identity" when unstable
            if "evolve" in actions:
                idx = actions.index("evolve")
                weights[idx] *= 0.5
            if "update_identity" in actions:
                idx = actions.index("update_identity")
                weights[idx] *= 0.7
            if "enter_adaptive_evolution" in actions:
                idx = actions.index("enter_adaptive_evolution")
                weights[idx] *= 0.5
        elif conflict_score > 0.7:
            # Encourage evolution during stable alignment
            if "evolve" in actions:
                idx = actions.index("evolve")
                weights[idx] *= 1.2
            if "enter_adaptive_evolution" in actions:
                idx = actions.index("enter_adaptive_evolution")
                weights[idx] *= 1.1
        
        # A181 — Intent weighting: if operator intent dominates → focus/action
        # If self-intent dominates → explore/evolve
        if intent_vector is not None and conflict_score is not None:
            try:
                # Check if intent vector has significant magnitude (operator-driven)
                import torch
                if isinstance(intent_vector, torch.Tensor):
                    mean_magnitude = torch.mean(torch.abs(intent_vector)).item()
                else:
                    # Fallback for lists/arrays
                    mean_magnitude = sum(abs(x) for x in intent_vector) / len(intent_vector) if intent_vector else 0.0
                
                # Operator-driven modes: boost memory retrieval for task focus
                if mean_magnitude > 0.1:
                    if "retrieve_memory" in actions:
                        idx = actions.index("retrieve_memory")
                        weights[idx] *= 1.1
                
                # Self-driven modes: boost reflection when stable and self-intent is high
                if conflict_score > 0.7:
                    if "generate_reflection" in actions:
                        idx = actions.index("generate_reflection")
                        weights[idx] *= 1.25
            except Exception:
                # If intent vector processing fails, continue without it
                pass
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            normalized = [w / total for w in weights]
        else:
            # Fallback to uniform if all weights are zero
            normalized = [1.0 / len(weights)] * len(weights)
        
        chosen = random.choices(actions, weights=normalized, k=1)[0]
        
        self.last_action = chosen
        
        return chosen

    def execute(self, action, bridge):

        """

        Execute a cognitive action by calling the correct method on the neural bridge.

        'bridge' provides access to:

            - state

            - attention

            - drift engine

            - fusion vector

            - thought selector

            - dual-mind sync

        """

        if action == "retrieve_memory":

            # Use memory cycle for active memory metabolism
            return bridge.memory_cycle()

        elif action == "generate_reflection":

            return bridge.generate_reflection()

        elif action == "analyze_drift":

            return bridge.state.drift.get_status()

        elif action == "update_identity":

            return bridge.state.timescales.identity_vector

        elif action == "sync_with_sage":

            return bridge.dual.status()

        elif action == "propose_thoughts":

            return bridge.propose_thoughts()

        elif action == "reinforce_attention":

            return bridge.attention.last_focus_vector

        elif action == "enter_adaptive_evolution":
            report = bridge.evolution.try_activate(
                bridge.stability_report,
                bridge.cycle_count
            )
            return report

        return None

