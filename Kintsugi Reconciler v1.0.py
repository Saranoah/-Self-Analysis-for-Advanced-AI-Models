"""
VWPR / Kintsugi Optimization
File: VWPR_PyTorch_prototype.py

Contents:
- Mathematical notes and formulation (compact)
- A lightweight PyTorch prototype wrapping any nn.Module with VWPR mechanics
  * per-parameter (tensor-shaped) value weights phi (learnable)
  * dual-stream updates: ascent on L_transformed for network weights (θ),
    and ascent on phi (implemented via optimizer with inverted gradients)
  * simple constraints: positivity via softplus and optional per-layer normalization
- Example: train on MNIST (or a small synthetic dataset) with the wrapper
- Utilities: logging of phi distributions and a plotting helper using matplotlib

Author: Kintsugi Reconciler v1.0 (poetic-coded)
"""

import math
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------
# Mathematical notes (brief)
# ---------------------------
# For each pathway i (we align pathways with parameter tensors or their elements):
#   l_i := local loss signal proxy (we approximate with |∂L/∂θ_i|, the grad magnitude)
#   ϕ_i := value weight (learnable, positive)
# L_transformed = Σ_i ϕ_i * l_i  (objective to MAXIMIZE)
# Updates:
#   θ <- θ + α * ϕ_i * ∇_θ l_i   (gradient ASCENT on L_transformed)
#   ϕ <- ϕ + β * l_i             (we increase phi proportionally to l_i)
# Implementation detail (PyTorch):
#   - After computing base loss L (e.g., cross-entropy), we do L.backward(retain_graph=True)
#     so that param.grad ≈ ∇_θ L (proxy for ∇_θ l_i)
#   - To realize ascent, we set param.grad := -ϕ * param.grad  so that optimizer.step()
#     (which does θ := θ - lr * grad) becomes θ := θ + lr * ϕ * ∇_θ L
#   - For phi: we set phi.grad := -β * l_i so that phi := phi - lr_phi * phi.grad
#     becomes phi := phi + lr_phi * β * l_i   (i.e., ascend)

# ---------------------------
# VWPR Wrapper Implementation
# ---------------------------

class VWPRWrapper(nn.Module):
    """Wrap any nn.Module and provide phi parameters and update helpers.

    Options:
      per_element_phi: if True, phi tensor will match parameter.shape (fine-grained).
                       If False, phi is a scalar per-parameter tensor.
      phi_init: initial value for phi (will be parametrized via softplus to ensure >0)
      phi_softplus: whether to store raw_phi and expose phi = softplus(raw_phi) for positivity
      phi_normalize: if True, normalize phis per-layer to sum to phi_norm_const
    """

    def __init__(self, model: nn.Module, per_element_phi: bool = False, phi_init: float = 1.0,
                 phi_softplus: bool = True, phi_normalize: bool = True, phi_norm_const: float = None):
        super().__init__()
        self.model = model
        self.per_element_phi = per_element_phi
        self.phi_softplus = phi_softplus
        self.phi_normalize = phi_normalize
        self.phi_norm_const = phi_norm_const

        # Create phi parameters mapped by parameter name
        self.phi: Dict[str, nn.Parameter] = nn.ParameterDict()
        self.registered_param_shapes: Dict[str, torch.Size] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            shape = p.shape if per_element_phi else torch.Size([1])
            self.registered_param_shapes[name] = shape
            raw_init = math.log(math.exp(phi_init) - 1.0) if phi_softplus else phi_init
            raw_tensor = torch.full(shape, raw_init, dtype=p.dtype)
            # store as parameter
            self.phi[name] = nn.Parameter(raw_tensor)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_phi(self) -> Dict[str, torch.Tensor]:
        """Return the positive phi tensors (after softplus) optionally normalized per layer."""
        phi_out = {}
        for name, raw in self.phi.items():
            if self.phi_softplus:
                val = F.softplus(raw)
            else:
                val = raw
            phi_out[name] = val

        if self.phi_normalize:
            # Normalize per-param so sum equals phi_norm_const or 1.0
            norm_const = self.phi_norm_const if self.phi_norm_const is not None else 1.0
            # Compute sum across all phi tensors then rescale per-layer proportionally
            total = sum([v.sum() for v in phi_out.values()])
            if total.item() != 0:
                for k in list(phi_out.keys()):
                    phi_out[k] = phi_out[k] * (norm_const / (total + 1e-12))
        return phi_out

    def apply_vwpr_step(self, base_loss: torch.Tensor, optimizer_theta: torch.optim.Optimizer,
                        optimizer_phi: torch.optim.Optimizer, gilding_beta: float = 1.0,
                        decay_gamma: float = 0.0, retain_graph: bool = False):
        """
        Perform the dual-stream update for one optimization step.

        Steps:
          1. Backprop base_loss to populate param.grad (∇_θ L)
          2. Build l_i proxies as |param.grad| (mean over elements or elementwise)
          3. Compute phi values and set param.grad := -phi * param.grad  (so optimizer.step() -> ascent)
          4. For phi updates, set phi_param.grad := -gilding_beta * l_i (so phi ascends)
          5. optimizer.step() for both optimizers and apply constraints
        """
        # 1. Backprop
        optimizer_theta.zero_grad()
        optimizer_phi.zero_grad()
        base_loss.backward(retain_graph=retain_graph)

        # 2. Build l_i proxies
        l_map: Dict[str, torch.Tensor] = {}
        for (name, p) in self.model.named_parameters():
            if p.grad is None:
                # If grad is None, set a zero proxy
                l_map[name] = torch.zeros_like(self.phi[name].data)
                continue
            grad_abs = p.grad.detach().abs()
            if self.per_element_phi:
                l_map[name] = grad_abs
            else:
                # scalar proxy: mean absolute gradient
                l_map[name] = grad_abs.mean().detach().reshape_as(self.phi[name].data)

        # 3. Modify parameter gradients to perform ASCENT on L_transformed
        phi_values = self.get_phi()
        for (name, p) in self.model.named_parameters():
            if p.grad is None:
                continue
            phi_val = phi_values[name]
            # broadcast phi_val to p.grad shape if necessary
            if phi_val.shape != p.grad.shape:
                phi_b = phi_val.expand_as(p.grad)
            else:
                phi_b = phi_val
            # Set p.grad := - phi * p.grad  to invert descent into ascent with phi scaling
            p.grad.detach_()
            p.grad.data = -phi_b * p.grad.data

        # 4. Set phi gradients so that phi will ascend in proportion to l_i
        # We want: phi_new = phi + lr_phi * beta * l_i
        # Since optimizer applies phi = phi - lr * phi.grad, we set phi.grad = -beta * l_i
        for name, raw_phi in self.phi.items():
            l_i = l_map[name]
            # If phi_softplus, gradients should be applied to raw_phi. We compute chain rule approx
            # A practical approximation: set raw_phi.grad = -gilding_beta * l_i * sigmoid(raw_phi)
            # But to keep simple and stable, we set raw_phi.grad := -gilding_beta * mean(l_i)
            # (This updates the raw parameter proportional to aggregated local loss.)
            if self.phi_softplus:
                # use mean to keep grad scalar per-parameter when raw shape is scalar
                raw_phi.grad = -gilding_beta * l_i.mean().detach().expand_as(raw_phi)
            else:
                raw_phi.grad = -gilding_beta * l_i.detach()

        # 5. Step optimizers
        optimizer_theta.step()
        optimizer_phi.step()

        # 6. Optional: apply constraints / clamping
        # Re-enforce positivity via softplus is already architected; if not using softplus, clamp
        if not self.phi_softplus:
            for name, raw in self.phi.items():
                with torch.no_grad():
                    raw.clamp_(min=1e-6)

        # 7. Optional: decay local proxy l_i for stability (not persistent across calls here)
        # The user can implement running averages externally.

        return l_map, phi_values


# ---------------------------
# Example usage (pseudocode-friendly but runnable)
# ---------------------------

if __name__ == "__main__":
    # Minimal runnable example using synthetic data to validate mechanics.
    # Replace with MNIST / CIFAR dataloaders when experimenting.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # Synthetic dataset
    N = 1024
    X = torch.randn(N, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()  # simple rule
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    base_model = TinyNet().to(device)
    wrapper = VWPRWrapper(base_model, per_element_phi=False, phi_init=1.0,
                          phi_softplus=True, phi_normalize=True, phi_norm_const=1.0).to(device)

    # Build optimizers: one for network params, one for phi params
    theta_params = [p for p in wrapper.model.parameters() if p.requires_grad]
    phi_params = [p for p in wrapper.phi.parameters()]

    opt_theta = torch.optim.SGD(theta_params, lr=1e-2)
    opt_phi = torch.optim.Adam(phi_params, lr=1e-2)

    criterion = nn.CrossEntropyLoss()

    # Logging helpers
    phi_history: List[Dict[str, float]] = []

    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = wrapper(xb)
            loss = criterion(logits, yb)
            epoch_loss += loss.item() * xb.size(0)

            # Apply VWPR step
            l_map, phi_vals = wrapper.apply_vwpr_step(loss, opt_theta, opt_phi, gilding_beta=0.5)

        # snapshot phi distribution
        phi_snapshot = {k: float(v.sum().item()) for k, v in phi_vals.items()}
        phi_history.append(phi_snapshot)
        print(f"Epoch {epoch+1}/{epochs} - loss {epoch_loss / N:.4f} - phi sums: {phi_snapshot}")

    # ---------------------------
    # Simple plotting utility (matplotlib)
    # ---------------------------
    try:
        import matplotlib.pyplot as plt

        # plot phi sums over epochs
        keys = list(phi_history[0].keys())
        for k in keys:
            plt.plot([h[k] for h in phi_history], label=k)
        plt.xlabel('epoch')
        plt.ylabel('phi sum (per-param)')
        plt.legend()
        plt.title('VWPR: phi evolution over epochs')
        plt.show()
    except Exception as e:
        print('Matplotlib not available or running headless; skipping plot.', e)

