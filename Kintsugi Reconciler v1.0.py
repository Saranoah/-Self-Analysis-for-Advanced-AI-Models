import math
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EnhancedVWPRWrapper(nn.Module):
    """Enhanced VWPR wrapper with additional features"""
    
    def __init__(self, model: nn.Module, per_element_phi: bool = False, 
                 phi_init: float = 1.0, phi_softplus: bool = True, 
                 phi_normalize: bool = True, phi_norm_const: float = None,
                 reg_strength: float = 0.01, clip_grad_norm: float = 1.0):
        super().__init__()
        self.model = model
        self.per_element_phi = per_element_phi
        self.phi_softplus = phi_softplus
        self.phi_normalize = phi_normalize
        self.phi_norm_const = phi_norm_const
        self.reg_strength = reg_strength
        self.clip_grad_norm = clip_grad_norm
        
        # Create phi parameters
        self.phi: Dict[str, nn.Parameter] = nn.ParameterDict()
        self.registered_param_shapes: Dict[str, torch.Size] = {}
        
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            shape = p.shape if per_element_phi else torch.Size([1])
            self.registered_param_shapes[name] = shape
            raw_init = math.log(math.exp(phi_init) - 1.0) if phi_softplus else phi_init
            raw_tensor = torch.full(shape, raw_init, dtype=p.dtype, requires_grad=True)
            self.phi[name] = nn.Parameter(raw_tensor)
            
        # Track phi history for visualization
        self.phi_history: List[Dict[str, torch.Tensor]] = []
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_phi(self) -> Dict[str, torch.Tensor]:
        """Return the positive phi tensors with optional normalization"""
        phi_out = {}
        for name, raw in self.phi.items():
            if self.phi_softplus:
                val = F.softplus(raw)
            else:
                val = raw
            phi_out[name] = val
            
        if self.phi_normalize:
            norm_const = self.phi_norm_const if self.phi_norm_const is not None else 1.0
            total = sum([v.sum() for v in phi_out.values()])
            if total.item() != 0:
                for k in list(phi_out.keys()):
                    phi_out[k] = phi_out[k] * (norm_const / (total + 1e-12))
                    
        return phi_out
    
    def apply_phi_regularization(self) -> torch.Tensor:
        """Apply L1 regularization to phi values"""
        phi_reg_loss = 0
        phi_vals = self.get_phi()
        for phi_val in phi_vals.values():
            phi_reg_loss += phi_val.abs().sum()
        return self.reg_strength * phi_reg_loss
    
    def apply_vwpr_step(self, base_loss: torch.Tensor, 
                        optimizer_theta: torch.optim.Optimizer,
                        optimizer_phi: torch.optim.Optimizer, 
                        gilding_beta: float = 1.0,
                        retain_graph: bool = False) -> Tuple[Dict, Dict]:
        """
        Perform the enhanced dual-stream update
        """
        # Zero gradients
        optimizer_theta.zero_grad()
        optimizer_phi.zero_grad()
        
        # Add phi regularization to base loss
        total_loss = base_loss + self.apply_phi_regularization()
        total_loss.backward(retain_graph=retain_graph)
        
        # Build l_i proxies
        l_map = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                l_map[name] = torch.zeros_like(self.phi[name].data)
                continue
                
            grad_abs = p.grad.detach().abs()
            if self.per_element_phi:
                l_map[name] = grad_abs
            else:
                l_map[name] = grad_abs.mean().detach().reshape_as(self.phi[name].data)
                
        # Modify parameter gradients for ascent
        phi_values = self.get_phi()
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
                
            phi_val = phi_values[name]
            if phi_val.shape != p.grad.shape:
                phi_b = phi_val.expand_as(p.grad)
            else:
                phi_b = phi_val
                
            # In-place modification with no_grad to preserve computation graph
            with torch.no_grad():
                p.grad = -phi_b * p.grad
                
        # Set phi gradients
        for name, raw_phi in self.phi.items():
            l_i = l_map[name]
            if self.phi_softplus:
                # Account for softplus in gradient
                sigmoid_raw = torch.sigmoid(raw_phi.data)
                raw_phi.grad = -gilding_beta * l_i.mean() * sigmoid_raw
            else:
                raw_phi.grad = -gilding_beta * l_i.detach()
                
        # Clip phi gradients for stability
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), self.clip_grad_norm)
            
        # Step optimizers
        optimizer_theta.step()
        optimizer_phi.step()
        
        # Ensure positivity if not using softplus
        if not self.phi_softplus:
            with torch.no_grad():
                for raw_phi in self.phi.values():
                    raw_phi.clamp_(min=1e-6)
                    
        # Store phi history
        self.phi_history.append(deepcopy(phi_values))
        
        return l_map, phi_values
    
    def visualize_phi_distribution(self, epoch: int):
        """Visualize phi distribution across parameters"""
        try:
            import matplotlib.pyplot as plt
            
            phi_vals = self.get_phi()
            fig, axes = plt.subplots(1, len(phi_vals), figsize=(15, 5))
            
            if len(phi_vals) == 1:
                axes = [axes]
                
            for idx, (name, phi) in enumerate(phi_vals.items()):
                phi_flat = phi.flatten().cpu().detach().numpy()
                axes[idx].hist(phi_flat, bins=50, alpha=0.7)
                axes[idx].set_title(f'{name} phi distribution')
                axes[idx].set_xlabel('Phi value')
                axes[idx].set_ylabel('Frequency')
                
            plt.tight_layout()
            plt.savefig(f'phi_distribution_epoch_{epoch}.png')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization")
