"""Scale-invariant SDR (SI-SDR) in dB."""

from __future__ import annotations

import numpy as np
import torch


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SDR between 1-D estimate and reference (Le Roux et al., 2019)."""
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    n = min(est.size, ref.size)
    if n == 0:
        return float("nan")
    est = est[:n]
    ref = ref[:n]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    dot = float(np.dot(est, ref))
    ref_energy = float(np.dot(ref, ref)) + eps
    proj = (dot / ref_energy) * ref
    noise = est - proj
    return 10.0 * float(np.log10((np.dot(proj, proj) + eps) / (np.dot(noise, noise) + eps)))


def si_sdr_loss_torch(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean (−SI-SDR) over batch for training (minimize).

    Accepts ``(B, T)``, ``(B, C, T)``, or ``(B, S, C, T)``; non-time dims are
    flattened into the batch before the SI-SDR reduction.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: estimate {tuple(estimate.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    # Flatten all but time into batch: (N, T)
    est = estimate.reshape(-1, estimate.shape[-1]).float()
    ref = reference.reshape(-1, reference.shape[-1]).float()
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True).clamp_min(eps)
    proj = (dot / ref_energy) * ref
    noise = est - proj
    proj_energy = (proj * proj).sum(dim=-1).clamp_min(eps)
    noise_energy = (noise * noise).sum(dim=-1).clamp_min(eps)
    si_sdr_db = 10.0 * torch.log10(proj_energy / noise_energy)
    return -si_sdr_db.mean()
