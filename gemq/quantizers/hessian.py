"""Stable FP32 inverse-Hessian factor without clearing weight columns."""

import math

import torch


def _diagonal_inverse(hessian, diagonal_mean):
    diagonal = hessian.diagonal().clamp_min(diagonal_mean * 0.01)
    return torch.diag(1.0 / diagonal)


def inverse_hessian_factor(hessian, percdamp):
    hessian = hessian.float()
    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or not torch.isfinite(hessian).all()
    ):
        raise ValueError("Hessian must be a finite square matrix")
    if not math.isfinite(percdamp) or percdamp <= 0:
        raise ValueError("Hessian damping must be finite and positive")

    hessian = (hessian + hessian.T) * 0.5
    diagonal_mean = hessian.diagonal().mean()
    if diagonal_mean <= 0:
        return torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)

    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    cholesky = None
    for attempt in range(5):
        damping = percdamp * 10 ** attempt
        try:
            cholesky = torch.linalg.cholesky(hessian + damping * diagonal_mean * identity)
            break
        except torch.linalg.LinAlgError:
            pass

    if cholesky is None:
        inverse = _diagonal_inverse(hessian, diagonal_mean)
    else:
        inverse = torch.cholesky_inverse(cholesky)
        inverse = (inverse + inverse.T) * 0.5
        if not torch.isfinite(inverse).all():
            inverse = _diagonal_inverse(hessian, diagonal_mean)
    try:
        factor = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError:
        factor = torch.linalg.cholesky(_diagonal_inverse(hessian, diagonal_mean), upper=True)
    if not torch.isfinite(factor).all():
        raise RuntimeError("Hessian inverse factor is non-finite")
    return factor
