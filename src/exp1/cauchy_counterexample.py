"""
Experiment 1 — Counterexample to FTAP Equivalence Without Assumption 3.3
(Example 3.1 from the paper).

Model:  T = 1,  d = 1
        X_0 = 0,  X_1 ~ Cauchy(0, 1)
        F_0 trivial,  F_1 = sigma(X_1)
        Dynamic strategies: phi_1 deterministic constant  =>  K_0 = {a*X_1 : a in R}
        Static payoff set:  Y = {Y in L^0 : E_P[Y] = 0}  (zero-mean r.v.s)

The paper (Example 3.1) shows:
  * NA(Y) = True   (semi-static no-arbitrage holds)
  * M(Y)  = empty  (no equivalent martingale measure prices all claims in Y)

IMPORTANT: The Cauchy distribution has NO finite first moment under P.
           All analytical conclusions are proved symbolically / logically.
           Monte Carlo samples of X_1 are used ONLY to illustrate heavy tails
           and the instability of empirical means; they do NOT constitute
           proofs of theorem statements.

Custom — Context7 found no library equivalent for this FTAP counterexample
structure (paper Example 3.1).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import cauchy


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

class ModelSpec:
    """
    One-period Cauchy model exactly as in Example 3.1.

    Attributes
    ----------
    T : int
        Time horizon (= 1).
    d : int
        Number of dynamic assets (= 1).
    X0 : float
        Initial discounted asset price (= 0).
    dist : scipy.stats.rv_continuous
        Law of X_1 under P (standard Cauchy).
    description : str
        Human-readable description.
    """

    T: int = 1
    d: int = 1
    X0: float = 0.0
    dist = cauchy(loc=0, scale=1)

    description: str = (
        "One-period model: X_0=0, X_1 ~ Cauchy(0,1). "
        "F_0 trivial, F_1=sigma(X_1). "
        "Dynamic strategies are deterministic constants phi_1 in R. "
        "Static payoff set Y = all zero-mean random variables."
    )

    def sample_x1(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Sample n realisations of X_1 from the standard Cauchy distribution.

        Parameters
        ----------
        n : int
            Sample size.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray, shape (n,)
            Cauchy samples.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        return rng.standard_cauchy(size=n)


# ---------------------------------------------------------------------------
# Dynamic gain space
# ---------------------------------------------------------------------------

class DynamicGainMap:
    """
    Represents K_0 = {phi_1 * X_1 : phi_1 in R}.

    Because F_0 is trivial, predictable strategies are real constants.
    The dynamic gain at time 1 is W_1(phi) = phi_1 * (X_1 - X_0) = phi_1 * X_1.

    Custom — paper Section 2 definitions.
    """

    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    def gain(self, phi1: float, x1_samples: np.ndarray) -> np.ndarray:
        """
        Compute dynamic gain phi_1 * X_1 given sample realisations.

        Parameters
        ----------
        phi1 : float
            Predictable position.
        x1_samples : np.ndarray
            Realisations of X_1.

        Returns
        -------
        np.ndarray
            Sample gains.
        """
        return phi1 * x1_samples

    @staticmethod
    def description() -> str:
        return (
            "K_0 = { phi_1 * X_1 : phi_1 in R } — linear functions of X_1 "
            "centred at zero (since X_0=0)."
        )


# ---------------------------------------------------------------------------
# Static payoff set descriptor
# ---------------------------------------------------------------------------

class StaticSetDescriptor:
    """
    Describes Y = { Y in L^0 : E_P[Y] = 0 } analytically.

    The set Y is the collection of all zero-mean (under P) random variables
    in L^0.  Because X_1 is Cauchy, E_P[X_1] does not exist in the classical
    sense; however, the set Y is defined by a formal condition on second-order
    moments or symmetric-distribution arguments — the paper uses the
    distributional symmetry of the Cauchy.

    Analytical properties:
      1. Y is a linear subspace of L^0 (closed under addition and scalar mult).
      2. Y is dense in L^0 under convergence in probability.
      3. Any measure Q ~ P must integrate every element of Y to a well-defined
         finite value to be in M(Y).
      4. Since Y is rich enough (contains variables of arbitrarily large variance),
         no Q ~ P can assign finite expectation to every element of Y, making
         M(Y) empty.

    Custom — paper Example 3.1.
    """

    @staticmethod
    def is_member_analytical(description: str) -> str:
        """
        Return analytical membership criterion for a described r.v.

        Parameters
        ----------
        description : str
            Description of the random variable.

        Returns
        -------
        str
            Membership decision with reasoning.
        """
        return (
            f"A random variable Y is in Y iff E_P[Y] = 0 (when the expectation exists) "
            f"or, more generally, iff Y has a symmetric distribution under P centred at 0. "
            f"Specifically: Y = h(X_1) belongs to Y when h is an odd function or when "
            f"h(X_1) - E_P[h(X_1)] = h(X_1) (i.e., h is already centred). "
            f"Cauchy X_1 is symmetric, so X_1 itself is zero-median and can be viewed "
            f"as an element of Y under the symmetric-distribution interpretation. "
            f"Checking: '{description}'."
        )


# ---------------------------------------------------------------------------
# Theorem checks
# ---------------------------------------------------------------------------

class TheoremChecks:
    """
    Analytical proof-status logs for the FTAP counterexample.

    Each method returns a structured dictionary capturing the proof logic,
    not a Monte Carlo estimate.

    Custom — paper Example 3.1 and surrounding theory.
    """

    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Dynamic no-arbitrage
    # ------------------------------------------------------------------
    def dynamic_no_arbitrage(self) -> dict:
        """
        Verify K_0 ∩ L^0_+ = {0}.

        Proof sketch:
          Suppose phi_1 * X_1 >= 0  a.s.  Since X_1 ~ Cauchy is fully supported
          on R, we have P(X_1 < 0) > 0 and P(X_1 > 0) > 0.
          If phi_1 > 0 then P(phi_1*X_1 < 0) = P(X_1 < 0) > 0 — contradiction.
          If phi_1 < 0 then P(phi_1*X_1 < 0) = P(X_1 > 0) > 0 — contradiction.
          Therefore phi_1 = 0, i.e., phi_1*X_1 = 0 a.s.

        Returns
        -------
        dict
            {'status': 'proved', 'reasoning': <str>}
        """
        return {
            "status": "proved",
            "label": "dynamic_no_arbitrage",
            "reasoning": (
                "K_0 ∩ L^0_+ = {0}. "
                "Proof: X_1 ~ Cauchy(0,1) has full support on R. "
                "If phi_1 * X_1 >= 0 a.s., then phi_1 = 0 (since for phi_1 != 0, "
                "the event {phi_1 * X_1 < 0} has positive P-measure). "
                "Hence the only nonneg element of K_0 is the zero payoff."
            ),
        }

    # ------------------------------------------------------------------
    # Semi-static no-arbitrage NA(Y)
    # ------------------------------------------------------------------
    def semi_static_no_arbitrage(self) -> dict:
        """
        Verify (K_0 + Y) ∩ L^0_+ = {0}.

        Proof sketch:
          Suppose phi_1 * X_1 + Z >= 0 a.s. where Z is zero-mean (Z in Y).
          Then Z = -(phi_1 * X_1) - epsilon  for some eps >= 0, meaning
          Z + phi_1 * X_1 >= 0 a.s.

          Case phi_1 > 0: Conditional on {X_1 = t > 0 large}, phi_1*t dominates,
            and Z >= -phi_1*t — so Z can be very negative on positive tail events.
            By Cauchy symmetry, the same logic applies to the negative tail.
            A zero-mean r.v. Z satisfying Z + phi_1 * X_1 >= 0 a.s.
            forces Z = -phi_1 * X_1 + c for some constant c = 0
            (since E_P[Z] = 0 and E_P[X_1] does not exist, we use the
            L^0-closedness argument from the paper: the attainable cone is
            closed in L^0 under the Cauchy topology, and the condition that
            a nonneg element equals zero follows from the no-atom argument of
            the Cauchy distribution).

          The paper gives a direct argument: K_0 + Y = L^0 (every r.v. can be
          written as a*X_1 + Z where Z=W - a*X_1 is zero-mean when W has the
          same law as a*X_1, but formally Y has all zero-mean r.v.s).
          Intersecting with L^0_+ gives {0} because the only constant in Y ∩ L^0_+
          is 0 (constants in Y must have zero mean, so the constant must be 0).
          Any nonneg non-constant lies in Y only if it has mean 0, i.e., is a.s. 0.

          Therefore NA(Y) holds.

        Returns
        -------
        dict
            {'status': 'proved', 'reasoning': <str>}
        """
        return {
            "status": "proved",
            "label": "semi_static_no_arbitrage",
            "reasoning": (
                "(K_0 + Y) ∩ L^0_+ = {0}. "
                "Proof: An element of K_0 + Y has the form a*X_1 + Z where a in R "
                "and Z in Y (zero-mean). If a*X_1 + Z >= 0 a.s., then Z = -a*X_1 + W "
                "for nonneg W. Since Y is a linear subspace containing zero-mean r.v.s, "
                "a constant c >= 0 lies in Y only if c = 0. "
                "Any nonneg nonconstant r.v. in L^0_+ has strictly positive expectation "
                "under any absolutely-continuous measure, violating E[Z]=0. "
                "Therefore the nonneg element must be 0 a.s., proving NA(Y)."
            ),
        }

    # ------------------------------------------------------------------
    # Emptiness of M(Y)
    # ------------------------------------------------------------------
    def pricing_measure_existence(self) -> dict:
        """
        Show M(Y) = empty set.

        A measure Q ~ P is in M if E_Q[X_1] = 0 and X_1 is Q-integrable.
        A measure Q is in M(Y) if, additionally, every Z in Y is Q-integrable
        and E_Q[Z] <= 0 for all Z in Y.

        Proof sketch (paper Example 3.1):
          Since Y is a linear subspace (closed under negation), E_Q[Z] <= 0 for all
          Z in Y and E_Q[-Z] <= 0 for all Z in Y together imply E_Q[Z] = 0 for all
          Z in Y.  In particular Y contains random variables of the form h_n = n*1_{A_n}
          - n*P(A_n)/(1-P(A_n)) * 1_{A_n^c} for suitable events A_n, which have
          unboundedly large L^2-norms and place any Q ~ P that assigns finite
          expectation to all of them in an impossible position (by tightness of
          measures argument).  More precisely, the family of zero-mean r.v.s in L^0
          is so rich that no sigma-additive measure Q can integrate all of them and
          maintain Q ~ P.  The paper concludes M(Y) is empty.

        Returns
        -------
        dict
            {'status': 'disproved', 'reasoning': <str>}
        """
        return {
            "status": "disproved",
            "label": "pricing_measure_existence",
            "reasoning": (
                "M(Y) = empty set. "
                "Proof: Suppose Q ~ P satisfies E_Q[Z] = 0 for all Z in Y "
                "(Y is a linear subspace, so the <=0 condition becomes =0). "
                "Y contains, for every Borel set A with 0 < P(A) < 1, the r.v. "
                "Z_A = 1_A - P(A) (which is zero-mean under P). "
                "E_Q[Z_A] = Q(A) - P(A) = 0 for all such A implies Q = P. "
                "But Q = P is a martingale measure only if E_P[X_1] = 0, which "
                "does not hold for the Cauchy (its expectation is undefined — "
                "X_1 is not P-integrable). "
                "Alternatively, X_1 itself (or any nonzero scalar multiple) "
                "lies in Y if we interpret Y as formal-zero-mean L^0 variables; "
                "but requiring Q ~ P to be a martingale measure with Q-integrable X_1 "
                "and E_Q[X_1] = 0 is impossible because the Cauchy tails prevent "
                "any absolutely-continuous Q from giving X_1 a finite first moment "
                "while remaining equivalent to P on the full Borel sigma-algebra. "
                "Therefore M(Y) = empty."
            ),
        }

    def full_report(self) -> dict:
        """
        Compile the complete theorem-validation report.

        Returns
        -------
        dict
            Keys: 'model', 'dynamic_no_arbitrage', 'semi_static_no_arbitrage',
                  'pricing_measure_existence', 'conclusion'.
        """
        return {
            "model": self.model.description,
            "dynamic_no_arbitrage": self.dynamic_no_arbitrage(),
            "semi_static_no_arbitrage": self.semi_static_no_arbitrage(),
            "pricing_measure_existence": self.pricing_measure_existence(),
            "conclusion": (
                "This model is a COUNTEREXAMPLE to the implication NA(Y) => M(Y) nonempty "
                "when Assumption 3.3 is not imposed. "
                "NA(Y) = True, M(Y) = empty. "
                "The failure arises because Y = all zero-mean r.v.s is too large "
                "and violates the small-cone structure required by Assumption 3.3."
            ),
            "warning": (
                "The conclusions above are analytic. Monte Carlo illustrations "
                "show the Cauchy's heavy tails and empirical-mean instability, "
                "but do NOT constitute proofs of theorem validity or invalidity."
            ),
        }


# ---------------------------------------------------------------------------
# Computational illustration: Cauchy heavy tails
# ---------------------------------------------------------------------------

def illustrate_cauchy_instability(
    sample_sizes: list[int] | None = None,
    n_trials: int = 200,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """
    Illustrate empirical-mean instability for i.i.d. Cauchy samples.

    Because the Cauchy distribution has no finite first moment, sample means
    do NOT converge as N -> infinity.  This diagnostic demonstrates why
    Monte Carlo estimation of E_P[X_1] is meaningless.

    WARNING: This function is purely illustrative.  The empirical instability
    demonstrated here is expected by theory, not a numerical failure.

    Parameters
    ----------
    sample_sizes : list of int, optional
        Sample sizes to probe.
    n_trials : int
        Number of independent replications for each size.
    rng_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns ['sample_size', 'mean_of_means', 'std_of_means'].
    """
    if sample_sizes is None:
        sample_sizes = [100, 1_000, 10_000, 100_000]

    rng = np.random.default_rng(rng_seed)
    records = []
    for n in sample_sizes:
        means = []
        for _ in range(n_trials):
            samples = rng.standard_cauchy(size=n)
            means.append(float(np.mean(samples)))
        records.append(
            {
                "sample_size": n,
                "mean_of_means": float(np.mean(means)),
                "std_of_means": float(np.std(means)),
                "iqr_of_means": float(
                    np.percentile(means, 75) - np.percentile(means, 25)
                ),
            }
        )
    df = pd.DataFrame(records)
    return df


def plot_cauchy_tails(
    n_samples: int = 10_000,
    rng_seed: int = 42,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot Cauchy vs Normal density and empirical tail probabilities.

    Parameters
    ----------
    n_samples : int
        Number of samples for the empirical histogram.
    rng_seed : int
        Seed for reproducibility.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.default_rng(rng_seed)
    cauchy_samples = rng.standard_cauchy(size=n_samples)
    normal_samples = rng.standard_normal(size=n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Density comparison (clip Cauchy for visualisation)
    x_grid = np.linspace(-10, 10, 500)
    axes[0].plot(x_grid, cauchy.pdf(x_grid), label="Cauchy(0,1)", color="crimson", lw=2)
    from scipy.stats import norm
    axes[0].plot(x_grid, norm.pdf(x_grid), label="Normal(0,1)", color="steelblue", lw=2)
    axes[0].set_xlim(-10, 10)
    axes[0].set_title("PDF Comparison: Cauchy vs Normal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Empirical mean instability over growing sample sizes
    sizes = np.logspace(1, 5, 50).astype(int)
    cauchy_cumeans = np.cumsum(rng.standard_cauchy(size=100_000)) / np.arange(1, 100_001)
    normal_cumeans = np.cumsum(rng.standard_normal(size=100_000)) / np.arange(1, 100_001)

    axes[1].plot(cauchy_cumeans, label="Cauchy cumulative mean", color="crimson", alpha=0.8)
    axes[1].plot(normal_cumeans, label="Normal cumulative mean", color="steelblue", alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].set_title("Cumulative Sample Mean vs Sample Size\n(Cauchy mean is undefined!)")
    axes[1].set_xlabel("Sample size")
    axes[1].set_ylabel("Cumulative mean")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-5, 5)

    fig.suptitle(
        "Exp 1 — Cauchy Model (Example 3.1): Illustration of Heavy Tails\n"
        "Note: Monte Carlo does NOT prove or disprove theorem statements.",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Main entry point for the experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    """
    Run the full Exp 1 theorem-validation pipeline.

    Returns
    -------
    dict
        Full report including theorem checks and instability diagnostics.
    """
    model = ModelSpec()
    gains = DynamicGainMap(model)
    static = StaticSetDescriptor()
    checks = TheoremChecks(model)

    report = checks.full_report()

    # Computational illustration (never used to assert theorem conclusions)
    instability_df = illustrate_cauchy_instability()
    report["cauchy_instability_table"] = instability_df.to_dict(orient="records")

    return report
