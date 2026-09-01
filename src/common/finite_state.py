"""
Finite-state probability space utilities for exact conic and linear feasibility
computations.

Payoffs are represented as vectors in R^S where S is the number of states.
Probability weights p in R^S_{>0} with sum 1 define the measure P.
A measure Q ~ P corresponds to q in R^S_{>0} with sum 1.
Martingale condition: sum_i q_i * X_1(i) = X_0.

Custom — Context7 found no library equivalent for this specific finite-state
FTAP helper structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class FiniteStateModel:
    """
    Encapsulates a one-period finite-state market model.

    Parameters
    ----------
    p : np.ndarray, shape (S,)
        Strictly positive physical probabilities.
    x0 : float
        Initial discounted asset price.
    x1 : np.ndarray, shape (S,)
        Terminal discounted asset price in each state.
    """

    def __init__(self, p: np.ndarray, x0: float, x1: np.ndarray) -> None:
        p = np.asarray(p, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        assert p.ndim == 1, "p must be 1-D"
        assert x1.shape == p.shape, "x1 and p must have the same shape"
        assert np.all(p > 0), "All state probabilities must be strictly positive"
        assert np.isclose(p.sum(), 1.0, atol=1e-10), "p must sum to 1"
        self.p = p
        self.x0 = float(x0)
        self.x1 = x1
        self.S = len(p)

    @property
    def n_states(self) -> int:
        return self.S

    def e_p(self, payoff: np.ndarray) -> float:
        """Expectation of payoff under physical measure P."""
        return float(np.dot(self.p, payoff))

    def e_q(self, payoff: np.ndarray, q: np.ndarray) -> float:
        """Expectation of payoff under measure Q with weights q."""
        return float(np.dot(q, payoff))

    def dynamic_gain(self, phi: float) -> np.ndarray:
        """Terminal gain of predictable strategy phi_1: phi_1*(X_1 - X_0)."""
        return phi * (self.x1 - self.x0)

    def is_martingale_measure(self, q: np.ndarray, tol: float = 1e-8) -> bool:
        """Check whether q defines an equivalent martingale measure."""
        q = np.asarray(q, dtype=float)
        sums_to_one = np.isclose(q.sum(), 1.0, atol=tol)
        all_positive = np.all(q > tol)
        martingale = np.isclose(np.dot(q, self.x1), self.x0, atol=tol)
        return bool(sums_to_one and all_positive and martingale)

    def find_martingale_measure(self) -> Optional[np.ndarray]:
        """
        Find an equivalent martingale measure Q ~ P via linear programming.

        Uses scipy.optimize.linprog to find q in R^S_{>0} such that
        sum q_i = 1 and E_q[X_1] = X_0.

        Returns
        -------
        np.ndarray or None
            Probability vector q if feasible, else None.
        """
        from scipy.optimize import linprog

        S = self.S
        # Minimise 0 subject to:
        #   sum q_i = 1
        #   sum q_i * x1_i = x0
        #   q_i >= eps  (strict positivity approximated)
        eps = 1e-9
        A_eq = np.vstack([np.ones((1, S)), self.x1.reshape(1, S)])
        b_eq = np.array([1.0, self.x0])
        bounds = [(eps, None)] * S
        c = np.zeros(S)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success:
            q = res.x
            if self.is_martingale_measure(q, tol=1e-6):
                return q
        return None


def check_no_arbitrage(
    model: FiniteStateModel,
    static_generators: Optional[np.ndarray] = None,
) -> dict:
    """
    Check (semi-)static no-arbitrage in a finite-state model.

    A payoff w = sum_k lambda_k * g_k + phi * (X_1 - X_0) is arbitrage-free
    if w >= 0 a.s. implies w = 0 a.s.

    This is tested by checking whether the conic combination of generators
    and dynamic gains intersects L^0_+ only at zero.

    Parameters
    ----------
    model : FiniteStateModel
    static_generators : np.ndarray, shape (K, S), optional
        Rows are payoff vectors of static generators g_k.

    Returns
    -------
    dict with keys 'dynamic_na', 'semi_static_na', 'arbitrage_payoff'.
    """
    from scipy.optimize import linprog

    S = model.S
    # Dynamic gains span: phi*(x1 - x0) for phi in R
    gain_vec = model.x1 - model.x0  # shape (S,)

    if static_generators is None or len(static_generators) == 0:
        K = 0
        generators = gain_vec.reshape(1, S)
        # Variables: phi (unrestricted), slack s>=0
        # We parameterize phi = phi_plus - phi_minus, phi_plus,phi_minus>=0
        # payoff = phi_plus * gain - phi_minus * gain >= 0 and != 0
        # This is equivalent to checking gain_vec is not a.s. >= 0 and a.s. <= 0
        gain_non_neg = np.all(gain_vec >= -1e-12)
        gain_non_pos = np.all(gain_vec <= 1e-12)
        dyn_na = not (gain_non_neg and not np.allclose(gain_vec, 0))
        # More carefully: dynamic NA <=> no phi s.t. phi*gain >= 0 a.s. and !=0
        # This holds iff gain is not a.s. >= 0 and not a.s. <= 0 simultaneously
        # i.e., gain changes sign across states
        has_pos = np.any(gain_vec > 1e-12)
        has_neg = np.any(gain_vec < -1e-12)
        dyn_na = has_pos and has_neg
        return {
            "dynamic_na": dyn_na,
            "semi_static_na": dyn_na,
            "arbitrage_payoff": None,
        }

    K = len(static_generators)
    # Semi-static payoff: phi*(x1-x0) + sum_k lambda_k * g_k
    # Variables: [phi_plus, phi_minus, lambda_1,...,lambda_K]
    # phi = phi_plus - phi_minus, lambda_k >= 0
    # payoff_i = (phi_plus - phi_minus)*gain_i + sum_k lambda_k * g_k_i >= 0
    # for all i, with strict inequality for some i (not all zero)
    # Check feasibility of: payoff >= 0, sum payoff > 0, using LP
    # We look for a nonneg nonzero payoff in the cone.
    # min -sum(w) s.t. w = (phi+  - phi-)*gain + G^T lambda >= 0
    # phi+, phi- >= 0, lambda >= 0, w >= 0

    # Variables z = [phi_plus, phi_minus, lambda_1,...,lambda_K, w_1,...,w_S]
    n_vars = 2 + K + S
    # Constraints:
    # w_i = (phi_plus - phi_minus)*gain_i + sum_k lambda_k g_k_i
    # => (phi_plus - phi_minus)*gain_i + sum_k lambda_k g_k_i - w_i = 0
    A_eq = np.zeros((S, n_vars))
    for i in range(S):
        A_eq[i, 0] = gain_vec[i]      # phi_plus
        A_eq[i, 1] = -gain_vec[i]     # phi_minus
        for k in range(K):
            A_eq[i, 2 + k] = static_generators[k, i]  # lambda_k
        A_eq[i, 2 + K + i] = -1.0     # -w_i
    b_eq = np.zeros(S)

    # Objective: minimize -sum(w) to find maximum nonneg payoff
    c = np.zeros(n_vars)
    c[2 + K: 2 + K + S] = -1.0  # -w_i

    bounds = [(0, None)] * n_vars  # all nonneg
    # phi is split into phi_plus - phi_minus, both >= 0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        obj = -res.fun
        w = res.x[2 + K:]
        if obj > 1e-8:
            return {
                "dynamic_na": False,
                "semi_static_na": False,
                "arbitrage_payoff": w,
            }

    return {
        "dynamic_na": True,
        "semi_static_na": True,
        "arbitrage_payoff": None,
    }
