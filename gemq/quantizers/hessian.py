"""Stable FP32 inverse-Hessian factor without clearing weight columns."""

import torch


def inverse_hessian_factor(hessian, percdamp):
    hessian = hessian.float()
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or not torch.isfinite(hessian).all():
        raise ValueError("Hessian must be a finite square matrix")
    if percdamp <= 0:
        raise ValueError("Hessian damping must be positive")
    hessian = (hessian + hessian.T) * 0.5
    mean = hessian.diagonal().mean()
    if mean <= 0:
        return torch.eye(hessian.shape[0], device=hessian.device)
    identity = torch.eye(hessian.shape[0], device=hessian.device)
    def diagonal_inverse():
        return torch.diag(1.0 / hessian.diagonal().clamp_min(mean * 0.01))
    cholesky = None
    for ratio in (percdamp, 0.1, 1.0, 10.0, 100.0):
        try:
            cholesky = torch.linalg.cholesky(hessian + ratio * mean * identity)
            break
        except torch.linalg.LinAlgError:
            pass
    inverse = diagonal_inverse() if cholesky is None else torch.cholesky_inverse(cholesky)
    inverse = (inverse + inverse.T) * 0.5
    if not torch.isfinite(inverse).all():
        inverse = diagonal_inverse()
    try:
        factor = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError:
        factor = torch.linalg.cholesky(diagonal_inverse(), upper=True)
    if not torch.isfinite(factor).all():
        raise RuntimeError("Hessian inverse factor is non-finite")
    return factor
