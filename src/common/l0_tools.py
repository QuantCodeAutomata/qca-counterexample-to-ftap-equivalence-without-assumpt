"""
Common L^0 tools for small-cone verification.

L^0(Omega, F, P) is the space of all real-valued random variables on the
probability space, with convergence in probability.  Boundedness in L^0
means: for the family H,  sup_{h in H} P(|h| >= n) -> 0 as n -> infinity.
This is strictly weaker than L^1 or L^2 boundedness.

Custom — Context7 found no library equivalent (analytic definitions from
paper Section 2 and Examples 5.1–5.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, List, Sequence


def tail_probability(
    samples: np.ndarray,
    threshold: float,
) -> float:
    """
    Estimate P(|h| >= threshold) from Monte Carlo samples of h.

    Parameters
    ----------
    samples : np.ndarray, shape (N,)
        Realizations of a random variable h.
    threshold : float
        Threshold n >= 0.

    Returns
    -------
    float
        Empirical tail probability P(|h| >= threshold).
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return float(np.mean(np.abs(samples) >= threshold))


def sup_tail_probability(
    sample_matrix: np.ndarray,
    threshold: float,
) -> float:
    """
    Estimate sup_{h in H} P(|h| >= threshold) over a family H.

    The family is represented as rows of sample_matrix, each row being
    N Monte Carlo samples of one generator h_k.

    Parameters
    ----------
    sample_matrix : np.ndarray, shape (K, N)
        K generators, each observed at N sample points.
    threshold : float
        Threshold n.

    Returns
    -------
    float
        Estimated supremum of tail probabilities.
    """
    if sample_matrix.ndim != 2:
        raise ValueError("sample_matrix must be 2-D with shape (K, N)")
    tails = np.mean(np.abs(sample_matrix) >= threshold, axis=1)  # shape (K,)
    return float(np.max(tails))


def l0_boundedness_table(
    sample_matrix: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Build a table of sup_H P(|h| >= n) over increasing thresholds n.

    Used to diagnose whether a generator family H is bounded in L^0.

    Parameters
    ----------
    sample_matrix : np.ndarray, shape (K, N)
        K generators, N samples each.
    thresholds : sequence of float, optional
        Values of n to probe; defaults to [1, 2, 5, 10, 20, 50, 100].

    Returns
    -------
    pd.DataFrame with columns ['threshold', 'sup_tail_prob'].
    """
    if thresholds is None:
        thresholds = [1, 2, 5, 10, 20, 50, 100, 200]
    records = []
    for n in thresholds:
        sup_p = sup_tail_probability(sample_matrix, float(n))
        records.append({"threshold_n": n, "sup_P_abs_h_ge_n": sup_p})
    return pd.DataFrame(records)


def convergence_in_prob_check(
    sequence: np.ndarray,
    limit: np.ndarray,
    eps: float = 0.1,
) -> np.ndarray:
    """
    Estimate P(|h_k - h| >= eps) for each element of a sequence.

    Parameters
    ----------
    sequence : np.ndarray, shape (K, N)
        K random variables, each with N sample observations.
    limit : np.ndarray, shape (N,)
        N samples of the limit random variable.
    eps : float
        Convergence tolerance.

    Returns
    -------
    np.ndarray, shape (K,)
        Tail probabilities P(|h_k - limit| >= eps) for k=1,...,K.
    """
    diffs = np.abs(sequence - limit[np.newaxis, :])  # (K, N)
    return np.mean(diffs >= eps, axis=1)


def is_zero_excluded(
    sample_matrix: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """
    Heuristic check that zero is not essentially in the generator family H.

    For each row (generator), compute P(|h| <= tol) as a proxy; if any
    generator is almost-surely zero under the sample, flag it.

    Parameters
    ----------
    sample_matrix : np.ndarray, shape (K, N)
        K generators, N samples each.
    tol : float
        Tolerance for near-zero detection.

    Returns
    -------
    bool
        True if no generator appears to be zero a.s.
    """
    near_zero_fracs = np.mean(np.abs(sample_matrix) <= tol, axis=1)
    return bool(np.all(near_zero_fracs < 0.999))
