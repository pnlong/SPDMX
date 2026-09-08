"""Scale-invariant SDR (SI-SDR) in dB."""

from __future__ import annotations

import numpy as np


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
