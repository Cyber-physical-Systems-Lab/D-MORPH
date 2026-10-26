# R4 metaplasticity: bounded/unbounded x R4 on/off experiment — hypotheses and measures

Written 2026-10-22, before any run of `experiments/r4.py`. The published D-MORPH configuration is unchanged; the
earlier R4a/R4b null results (docs/ablation_hypotheses.md) stand and are not replaced.

## Factors

- **R4**: on (published) or off. Off removes both mechanisms together: BCM extra decay (bcm_gain = 0) and
  neuromodulation (neuromod_gain = 0, neuromod_explore = 0). R1-R3, allocation, thresholds and all other parameters
  are unchanged.
- **Weight bound**: bounded (published: w clipped to [0, 1]) or unbounded (w clipped below at 0 only). Response
  probability is unchanged, 0.25 + 0.75 clip(w_j[i], 0, 1); partner ranking uses the unclipped w.

## Settings (Medium and Large, the 20 paper seeds; each seed runs all four configurations)

- **S1 reference**: published parameters, package location shift (T = 2400).
- **S2 sustained reinforcement + change in useful partners**: retention gamma = 0.999 (instead of 0.95; see Amendment 1), T = 800;
  half of the Pickers move at 1/3 speed (hidden) and slow and fast Pickers swap at step 400.
  gamma = 0.99 is set a priori: the steady state of R1 is about alpha J / (1 - gamma) = 11.5 J, so unbounded
  weights exceed 1 for pairs with co-assignment share J > 0.09. No pilot run was used to choose it.
- **S3 sustained reinforcement + delivery disruption**: retention gamma = 0.999 (Amendment 1), package location shift (T = 2400).

## Primary measures

- **Reversal time (S2)**: g(t) = mean over AGVs of [mean w_a[j] over currently fast Pickers - mean w_a[j] over
  currently slow Pickers]. After the swap at step 400 the labels swap with the speeds. Reversal time = steps after
  400 until the 25-step moving average of g first becomes > 0; 400 if it never does (censored at T = 800).
- **Recovery time r-bar (S3)**: Table 6 definition (mean of r_3..r_6).

## Primary hypotheses and tests (8 tests; Bonferroni alpha = 0.05 / 8 = 0.00625)

Paired two-sided Wilcoxon signed-rank tests over seeds; "effect of R4" = value with R4 off minus value with R4 on.

| # | Setting | Hypothesis | Test | Scales |
|---|---|---|---|---|
| H1 | S2 | With unbounded weights, R4 shortens reversal time | R4 effect on reversal time, unbounded | Medium, Large |
| H2 | S2 | The R4 effect on reversal time is larger with unbounded than with bounded weights | interaction (effect unbounded - effect bounded) | Medium, Large |
| H3 | S3 | With unbounded weights, R4 shortens recovery time | R4 effect on r-bar, unbounded | Medium, Large |
| H4 | S3 | The R4 effect on recovery is larger with unbounded than with bounded weights | interaction on r-bar | Medium, Large |

## Secondary measures (reported, not tested for the claims)

Pre-swap gap g over steps 300-399 (manipulation check: preferences must favour fast Pickers before the swap);
share of weights > 1 and maximum w (unbounded); share of partner requests from AGVs that go to a currently fast
Picker after the swap; new links formed in the 50 steps after each shift; per-robot eta; deliveries; S1 recovery
and degradation with R4 on/off.

The combined R4 ablation does not separate BCM from neuromodulation; a BCM-only / neuromodulation-only follow-up and
the R1-R3 x R4 interaction design are not part of this run.

## Amendment 1 (2026-10-22, before any outcome was computed)

The steady-state formula above omitted R1's Hebbian scale cov = 0.05: the per-step increment is alpha cov J =
0.0058 J, so at gamma = 0.99 the steady state is 0.58 J and unbounded weights cannot exceed 1. A one-episode smoke
test (Medium, seed 42, unbounded, gamma = 0.99) confirmed this: maximum w = 0.30. No outcome measure was computed.
gamma is changed to 0.999 for S2 and S3, chosen from the corrected formula: after 400 steps a pair with J = 1 reaches
0.0058 (1 - 0.999^400) / 0.001 = 1.9 from R1 alone, before completion credit. Hypotheses, measures, tests and the
correction are unchanged.

## Amendment: multiplicity correction removed (2026-10-22, after the results were known)

At the author's decision, the paper reports all tests without correction for multiple comparisons: two-sided
Wilcoxon signed-rank tests, significant at p < 0.05, with exact p-values reported. This replaces the Bonferroni
correction specified above. The change was made after the results of this experiment were known.
