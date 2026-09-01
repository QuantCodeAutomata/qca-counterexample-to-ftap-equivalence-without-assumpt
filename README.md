# FTAP Counterexample — Semi-Static No-Arbitrage with Small Cones

A Python repository implementing and validating the theoretical results from
the paper on the Fundamental Theorem of Asset Pricing (FTAP) for semi-static
trading strategies with structured static payoff sets ("small cones").

## Overview

The project reproduces five experiments:

| Exp | Title | Key Result |
|-----|-------|------------|
| 1 | Cauchy Counterexample (Example 3.1) | NA(Y)=✓ but M(Y)=∅ without Assumption 3.3 |
| 2 | Small-Cone Constructions (Examples 5.1–5.3) | Three generator families all qualify as valid small cones |
| 3 | Call-Option Cone (Examples 5.4–5.6, Theorem 3.5) | Centered calls → valid small cone, FTAP holds |
| 4 | Put-Option Cone (Examples 5.7–5.9, Theorem 3.5) | Centered puts → valid small cone, FTAP holds |
| 5 | Utility Maximization (Theorem 4.1) | Optimizer exists under bounded-above concave utility |

## Repository Structure

```
├── data/                 # Synthetic data generators (no empirical data)
├── src/
│   ├── common/           # Shared L⁰-boundedness tools
│   ├── exp1/             # Cauchy counterexample
│   ├── exp2/             # Small-cone constructions
│   ├── exp3/             # Call-option small cone
│   ├── exp4/             # Put-option small cone
│   └── exp5/             # Utility maximization
├── exp/
│   └── run_all_experiments.py
├── tests/                # pytest test suite (174 tests)
├── results/
│   ├── RESULTS.md        # Summary of all findings
│   ├── *.json            # Per-experiment structured reports
│   └── figures/          # Diagnostic plots
└── README.md
```

## Usage

### Run all experiments

```bash
python exp/run_all_experiments.py
```

Outputs JSON reports and figures into `results/`.

### Run tests

```bash
pytest tests/ -q
```

All 174 tests should pass.

## Dependencies

- numpy, scipy, pandas
- cvxpy (finite-state optimization)
- matplotlib, seaborn (plots)
- sympy (optional, symbolic documentation)
- pytest (testing)

No empirical market data or external API access is required.
All models use synthetic probability spaces with standard distributions
(Cauchy, lognormal, normal, gamma, finite-state discrete).

## Key Concepts

### Small Cone
A generator set H ⊂ L⁰ \ {0} that is convex, closed in L⁰ (convergence in
probability), and bounded in L⁰ (uniform tail decay). The resulting cone
L = C(H) = {λh : λ ≥ 0, h ∈ H} is the "small cone".

### FTAP Equivalence
Under Assumption 3.3 (small-cone structure on static payoffs Y) and
semi-static no-arbitrage NA(Y), Theorem 3.5 gives:
  NA(Y) ⟺ M(Y) ≠ ∅

Experiment 1 shows this implication fails without Assumption 3.3.
Experiments 3–4 confirm it holds with structured call/put families.

## Results Summary

See `results/RESULTS.md` for the full findings table.
