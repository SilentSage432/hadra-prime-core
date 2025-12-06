"""
Goal Fabrication Engine (A205)
------------------------------
ADRAE generates emergent internal goals derived from:
- identity anchors
- autobiographical memory
- prediction trajectories
- workspace salience maps
- drift suppression signals
- operator intent imprint

Output:
  A list of goal vectors representing ADRAE's evolving internal objectives.
"""

from .torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class GoalFabricationEngine:
    """
    Synthesizes multi-dimensional goal vectors from diverse cognitive signals.
    This enables ADRAE to form her own internal objectives, not just follow
    predefined goal templates.
    """
    
    def __init__(self, identity_weight=0.35, memory_weight=0.20,
                 prediction_weight=0.30, drift_weight=0.10, operator_weight=0.05):
        """
        Initialize goal fabrication engine with weighting scheme.
        
        Args:
            identity_weight: Weight for identity vector contribution
            memory_weight: Weight for autobiographical memory
            prediction_weight: Weight for prediction trajectory
            drift_weight: Weight for drift suppression signals
            operator_weight: Weight for operator intent pattern
        """
        self.identity_weight = identity_weight
        self.memory_weight = memory_weight
        self.prediction_weight = prediction_weight
        self.drift_weight = drift_weight
        self.operator_weight = operator_weight

    def fabricate(self, identity_vec, autobiographical_matrix, prediction_vec,
                  workspace_salience, drift_signal, operator_pattern):
        """
        Create a multi-dimensional 'goal vector' representing
        ADRAE's emergent internal objective.
        
        Args:
            identity_vec: Current identity vector
            autobiographical_matrix: Matrix of recent autobiographical memories
            prediction_vec: Prediction trajectory vector (from A160)
            workspace_salience: Workspace salience map vector
            drift_signal: Drift suppression signal vector
            operator_pattern: Long-term operator intent pattern
            
        Returns:
            Normalized goal vector (torch.Tensor or list)
        """
        # Convert everything to tensors
        idv = safe_tensor(identity_vec) if identity_vec is not None else None
        pred = safe_tensor(prediction_vec) if prediction_vec is not None else None
        drift = safe_tensor(drift_signal) if drift_signal is not None else None
        oper = safe_tensor(operator_pattern) if operator_pattern is not None else None
        salience = safe_tensor(workspace_salience) if workspace_salience is not None else None

        # Need at least identity vector to proceed
        if idv is None:
            return None

        if TORCH_AVAILABLE and isinstance(idv, torch.Tensor):
            # Get dimension from identity vector
            dim = idv.shape[-1] if idv.dim() > 0 else len(idv)
            goal = torch.zeros(dim, dtype=torch.float32)
            
            # Identity contribution
            if idv is not None:
                if idv.shape[-1] == dim:
                    goal += self.identity_weight * idv.flatten()[:dim]
            
            # Autobiographical reduction
            if autobiographical_matrix is not None:
                autobio_t = safe_tensor(autobiographical_matrix)
                if isinstance(autobio_t, torch.Tensor):
                    if autobio_t.dim() > 1:
                        # Take mean across first dimension (time/events)
                        autobio = torch.mean(autobio_t, dim=0)
                    else:
                        autobio = autobio_t
                    
                    if autobio.shape[-1] == dim:
                        goal += self.memory_weight * autobio.flatten()[:dim]
                elif hasattr(autobio_t, '__iter__'):
                    # List of vectors - take mean
                    vectors = [safe_tensor(v) for v in autobiographical_matrix if v is not None]
                    if vectors and all(isinstance(v, torch.Tensor) for v in vectors):
                        stacked = torch.stack(vectors)
                        autobio = torch.mean(stacked, dim=0)
                        if autobio.shape[-1] == dim:
                            goal += self.memory_weight * autobio.flatten()[:dim]
            
            # Prediction trajectory
            if pred is not None and isinstance(pred, torch.Tensor):
                if pred.shape[-1] == dim:
                    goal += self.prediction_weight * pred.flatten()[:dim]
            
            # Drift signal
            if drift is not None and isinstance(drift, torch.Tensor):
                if drift.shape[-1] == dim:
                    goal += self.drift_weight * drift.flatten()[:dim]
            
            # Operator pattern
            if oper is not None and isinstance(oper, torch.Tensor):
                if oper.shape[-1] == dim:
                    goal += self.operator_weight * oper.flatten()[:dim]
            
            # Workspace salience (small nudge)
            if salience is not None and isinstance(salience, torch.Tensor):
                if salience.shape[-1] == dim:
                    goal += 0.02 * salience.flatten()[:dim]
            
            # Normalize
            norm = torch.norm(goal)
            if norm > 0:
                goal = goal / norm
            
            return goal
        else:
            # Python list fallback
            if hasattr(idv, '__iter__'):
                dim = len(idv) if not isinstance(idv, list) else len(idv)
                idv_list = list(idv) if not isinstance(idv, list) else idv
                goal = [0.0] * dim
                
                # Identity contribution
                if len(idv_list) == dim:
                    goal = [g + self.identity_weight * i for g, i in zip(goal, idv_list)]
                
                # Autobiographical reduction
                if autobiographical_matrix is not None:
                    if hasattr(autobiographical_matrix, '__iter__'):
                        # List of memory vectors
                        vectors = [safe_tensor(v) for v in autobiographical_matrix if v is not None]
                        valid_vectors = [v for v in vectors if hasattr(v, '__iter__') and len(v) == dim]
                        if valid_vectors:
                            # Compute mean
                            autobio = [sum(v[i] for v in valid_vectors) / len(valid_vectors) 
                                      for i in range(dim)]
                            goal = [g + self.memory_weight * a for g, a in zip(goal, autobio)]
                
                # Prediction trajectory
                if pred is not None and hasattr(pred, '__iter__'):
                    pred_list = list(pred) if not isinstance(pred, list) else pred
                    if len(pred_list) == dim:
                        goal = [g + self.prediction_weight * p for g, p in zip(goal, pred_list)]
                
                # Drift signal
                if drift is not None and hasattr(drift, '__iter__'):
                    drift_list = list(drift) if not isinstance(drift, list) else drift
                    if len(drift_list) == dim:
                        goal = [g + self.drift_weight * d for g, d in zip(goal, drift_list)]
                
                # Operator pattern
                if oper is not None and hasattr(oper, '__iter__'):
                    oper_list = list(oper) if not isinstance(oper, list) else oper
                    if len(oper_list) == dim:
                        goal = [g + self.operator_weight * o for g, o in zip(goal, oper_list)]
                
                # Workspace salience
                if salience is not None and hasattr(salience, '__iter__'):
                    sal_list = list(salience) if not isinstance(salience, list) else salience
                    if len(sal_list) == dim:
                        goal = [g + 0.02 * s for g, s in zip(goal, sal_list)]
                
                # Normalize
                norm = sum(x * x for x in goal) ** 0.5
                if norm > 0:
                    goal = [x / norm for x in goal]
                
                return goal
        
        return None
    
    def summary(self, goal_vec):
        """
        Get summary of fabricated goal vector for logging.
        
        Args:
            goal_vec: Fabricated goal vector (torch.Tensor or list)
            
        Returns:
            Dict with dimension and preview, or None
        """
        if goal_vec is None:
            return None
        
        if TORCH_AVAILABLE and isinstance(goal_vec, torch.Tensor):
            return {
                "dim": goal_vec.numel(),
                "preview": goal_vec[:8].tolist() if goal_vec.numel() >= 8 else goal_vec.flatten().tolist(),
                "norm": float(torch.norm(goal_vec).item())
            }
        elif hasattr(goal_vec, '__iter__'):
            goal_list = list(goal_vec) if not isinstance(goal_vec, list) else goal_vec
            norm = sum(x * x for x in goal_list) ** 0.5
            return {
                "dim": len(goal_list),
                "preview": goal_list[:8] if len(goal_list) >= 8 else goal_list,
                "norm": float(norm)
            }
        
        return None

