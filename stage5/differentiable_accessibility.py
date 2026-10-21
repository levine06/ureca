"""Differentiable accessibility from C-alpha coordinates (smooth neighbour density)."""
import torch

# Calibrated on the development split by validate_accessibility.py (see README).
PARAMS = dict(r0=8.65, tau=1.0, b=9.022, s=3.379)


def neighbor_density(coords, r0=PARAMS["r0"], tau=PARAMS["tau"]):
    """rho_i = sum_{j!=i} sigmoid((r0 - d_ij) / tau). coords (N, 3) -> (N,)."""
    d = torch.cdist(coords, coords)
    contact = torch.sigmoid((r0 - d) / tau)
    return contact.sum(-1) - contact.diagonal()  # drop the self term (d = 0)


def accessibility(coords, r0=PARAMS["r0"], tau=PARAMS["tau"], b=PARAMS["b"], s=PARAMS["s"]):
    """a_i = sigmoid((b - rho_i) / s) in (0, 1). coords (N, 3) -> (N,)."""
    return torch.sigmoid((b - neighbor_density(coords, r0, tau)) / s)


def accessibility_loss(coords, ref, **kw):
    """Mean squared error between estimated and reference accessibility."""
    return ((accessibility(coords, **kw) - ref) ** 2).mean()
