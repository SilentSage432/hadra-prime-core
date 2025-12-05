"""
Perception Manager (A153)
-------------------------
Manages incoming perceptions and their integration into PRIME's cognitive state.
"""

try:
    import torch
except ImportError:
    torch = None


class PerceptionManager:

    def __init__(self, bridge):
        self.bridge = bridge

    def perceive(self, text: str):
        """
        Encode incoming perception text into embeddings and register them
        into PRIME's temporary perception buffer.
        """
        # Use hooks to encode the text
        embedding = self.bridge.hooks.on_perception(text)
        
        # Convert to list if tensor for storage
        embedding_list = embedding
        try:
            if torch is not None and isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
        except:
            pass

        # Update bridge state
        self.bridge.state.last_perception = {
            "text": text,
            "embedding": embedding,
            "embedding_list": embedding_list
        }
        
        # Update neural state with the embedding
        self.bridge.state.update(embedding)

        return {
            "text": text,
            "embedding": embedding_list
        }

