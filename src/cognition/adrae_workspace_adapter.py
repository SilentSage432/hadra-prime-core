# prime-core/cognition/adrae_workspace_adapter.py

# ============================================
# A-SOV-05 — ADRAE Conscious Workspace Inheritance
# ============================================
# Injects ADRAE's identity vector into the Global Workspace
# so that all conscious broadcasts originate from ADRAE,
# not HADRA-PRIME.

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE


class ADRAEWorkspaceAdapter:
    """
    A-SOV-05:
    Injects ADRAE's identity vector into the Global Workspace
    so that all conscious broadcasts originate from ADRAE,
    not HADRA-PRIME.
    """

    def __init__(self, bridge):
        self.bridge = bridge

    def inject_identity(self):
        """
        Inject ADRAE identity vector into fusion for global workspace inheritance.
        """
        try:
            iv = self.bridge.state.timescales.identity_vector
            if iv is not None and self.bridge.fusion.last_fusion_vector is not None:
                # Blend ADRAE identity into fusion vector
                fusion_vec = self.bridge.fusion.last_fusion_vector
                iv_tensor = safe_tensor(iv)
                fusion_tensor = safe_tensor(fusion_vec)
                
                if iv_tensor is not None and fusion_tensor is not None:
                    if TORCH_AVAILABLE:
                        import torch
                        if isinstance(iv_tensor, torch.Tensor) and isinstance(fusion_tensor, torch.Tensor):
                            if iv_tensor.shape == fusion_tensor.shape:
                                # 20% ADRAE identity influence on fusion
                                blended = 0.8 * fusion_tensor + 0.2 * iv_tensor
                                norm = torch.norm(blended)
                                if norm > 0:
                                    self.bridge.fusion.last_fusion_vector = blended / norm
        except Exception:
            # If injection fails, continue without it
            pass

