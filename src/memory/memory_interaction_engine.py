# prime-core/memory/memory_interaction_engine.py

"""
Memory Interaction Engine (A147)
--------------------------------
Manages PRIME's live memory dynamics:

- context-aware recall
- reinforcement of important memories
- decay of unused memories
- memory compression hooks (A160)
- integration with neural state and attention

This engine is used inside the A150 runtime loop.
"""

from math import exp
import random
import time
from .neural.neural_memory_manager import NeuralMemoryManager


class MemoryInteractionEngine:

    def __init__(self, decay_rate=0.002, reinforcement_rate=0.05):
        self.decay_rate = decay_rate
        self.reinforcement_rate = reinforcement_rate
        # Track access counts and strengths for each memory
        self.memory_strengths = {}  # memory_id -> strength
        self.access_counts = {}  # memory_id -> access_count
        self.last_access = {}  # memory_id -> timestamp

    def context_recall(self, fusion_vec, attention_vec, memory_manager):
        """
        Retrieve memories most relevant to the current cognitive state.
        Uses similarity scoring, attention weighting, and recency.
        """

        if fusion_vec is None or memory_manager is None:
            return []

        # Use semantic memory search
        semantic_results = memory_manager.semantic.find_similar(fusion_vec, top_k=3)
        similar = [(score, {"type": "semantic", "name": name, "vector": vec}) 
                   for score, name, vec in semantic_results]

        # Use episodic memory
        episodic_results = memory_manager.episodic.retrieve_similar(fusion_vec, top_k=2)
        episodic = [(score, {"type": "episodic", "entry": entry}) 
                    for score, entry in episodic_results]

        # Combine and track access
        recalled = similar + episodic
        for score, mem_data in recalled:
            mem_id = self._get_memory_id(mem_data)
            self._track_access(mem_id)

        return recalled

    def store(self, embedding, context, memory_manager):
        """
        Store new memory with semantic + episodic tagging.
        """
        if embedding is None or memory_manager is None:
            return

        # Store as episodic memory (experiences)
        entry = memory_manager.store_episode(embedding, meta=context)
        
        # Initialize strength for new memory
        mem_id = self._get_memory_id({"type": "episodic", "entry": entry})
        self.memory_strengths[mem_id] = 1.0
        self.access_counts[mem_id] = 0
        self.last_access[mem_id] = time.time()

    def reinforce(self, memories, memory_manager):
        """
        Increase strength of frequently recalled memories.
        """
        for score, mem_data in memories:
            mem_id = self._get_memory_id(mem_data)
            if mem_id in self.memory_strengths:
                # Reinforce based on access frequency
                current_strength = self.memory_strengths.get(mem_id, 1.0)
                access_count = self.access_counts.get(mem_id, 0)
                
                # More frequent access = more reinforcement
                reinforcement = self.reinforcement_rate * (1 + access_count * 0.1)
                self.memory_strengths[mem_id] = min(1.0, current_strength + reinforcement)
            else:
                # New memory, initialize
                self.memory_strengths[mem_id] = 1.0
                self.access_counts[mem_id] = 1

    def decay(self, memory_manager):
        """
        Gradually weaken memories that haven't been accessed recently.
        """
        current_time = time.time()
        
        # Decay all memories
        for mem_id in list(self.memory_strengths.keys()):
            last_access_time = self.last_access.get(mem_id, current_time)
            time_since_access = current_time - last_access_time
            
            # Decay based on time since last access
            decay_factor = exp(-self.decay_rate * time_since_access)
            self.memory_strengths[mem_id] *= decay_factor
            
            # Remove very weak memories
            if self.memory_strengths[mem_id] < 0.01:
                del self.memory_strengths[mem_id]
                if mem_id in self.access_counts:
                    del self.access_counts[mem_id]
                if mem_id in self.last_access:
                    del self.last_access[mem_id]

    def maintenance_cycle(self, fusion_vec, attention_vec, memory_manager):
        """
        Full memory metabolism step.
        """
        recalled = self.context_recall(fusion_vec, attention_vec, memory_manager)

        # Reinforce what was recalled
        self.reinforce(recalled, memory_manager)

        # Decay everything else
        self.decay(memory_manager)

        return recalled

    def _get_memory_id(self, mem_data):
        """
        Generate a unique ID for a memory entry.
        """
        if mem_data.get("type") == "semantic":
            return f"semantic:{mem_data.get('name', 'unknown')}"
        elif mem_data.get("type") == "episodic":
            entry = mem_data.get("entry", {})
            timestamp = entry.get("timestamp", 0)
            return f"episodic:{timestamp}"
        return f"unknown:{id(mem_data)}"

    def _track_access(self, mem_id):
        """
        Track when a memory is accessed.
        """
        self.access_counts[mem_id] = self.access_counts.get(mem_id, 0) + 1
        self.last_access[mem_id] = time.time()
        
        # Initialize strength if new
        if mem_id not in self.memory_strengths:
            self.memory_strengths[mem_id] = 1.0

