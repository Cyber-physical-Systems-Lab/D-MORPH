# R4 regulation and R1-R3 x R4 interactions: hypotheses and measures

Written 2026-10-22, before any run of `experiments/r4reg.py`. Earlier R4 results (docs/ablation_hypotheses.md,
docs/r4_hypotheses.md) stand. This experiment tests (1) whether R4 performs its intended regulation and (2) whether
R4 changes the effect of R1-R3 on their own primary measures. It does not test throughput or recovery.

Paired two-sided Wilcoxon signed-rank tests over the 20 paper seeds; Bonferroni over all 22 primary tests
(alpha = 0.05 / 22 = 0.00227). Published parameters unless stated.

## Part 1: does R4 perform its intended regulation?

| # | Claim | Setting | Comparison | Primary measure | Scales |
|---|---|---|---|---|---|
| E | BCM suppresses excessive potentiation | scale study, T = 800, retention 0.999 and unbounded weights (stress condition of docs/r4_hypotheses.md) | BCM on vs BCM off (bcm_gain = 0; neuromodulation on in both) | 95th percentile of positive w_i[j], pooled over robots and steps 200-799 (every 10th step) | 4 |
| F | Neuromodulation changes structural exploration | scale study, T = 800, published parameters, full D-MORPH | actual link formations vs the formations an unmodulated threshold (theta_form) would make from the identical candidate states | link formations per episode that occur only because of the arousal-lowered threshold (formed with score below theta_form) | 4 |

Test F compares the actual count with the counterfactual count in the same episode; the difference is >= 0 by
construction, so the test establishes whether arousal changes any formation decisions, and its size is reported
as the share of all formations. Secondary for E: share of positive weights > 1. Secondary for F: induced
formations in the 100 steps after each location shift vs the 100 steps before it (Medium and Large).

## Part 2: does R4 change the effect of R1-R3?

For rule X with primary measure Y (signed so that a positive Delta means X improves Y):
Delta_X(+M) = Y(with X) - Y(without X), both with modulator M on;
Delta_X(-M) = Y(with X, without M) - Y(without X, without M);
interaction I_X = Delta_X(+M) - Delta_X(-M), tested per seed. I_X > 0: X is more effective with M.

| # | Rule X | Modulator M | Y and sign | Setting | Scales |
|---|---|---|---|---|---|
| I1 | R1 completion credit (noreward) | BCM (nobcm) | credit gap (with - without) | scale study | 4 |
| I2 | R2 homeostasis (nohomeo) | all of R4 (bcm_gain = neuromod_gain = neuromod_explore = 0) | degree-target deviation (without - with) | scale study | 4 |
| I3 | R3 predictive formation (nopred) | neuromodulation (noneuro) | links at step 50 (with - without) | scale study | 4 |
| I4 | R3 structural plasticity (nostruct) | neuromodulation (noneuro) | post-shift linked-pair coverage (with - without) | location shift, T = 2400 | Medium, Large |

Count of primary tests: E 4 + F 4 + I1 4 + I2 4 + I3 4 + I4 2 = 22.

Measures Y are those of docs/ablation_hypotheses.md (same definitions and code).

## Amendment: multiplicity correction removed (2026-10-22, after the results were known)

At the author's decision, the paper reports all tests without correction for multiple comparisons: two-sided
Wilcoxon signed-rank tests, significant at p < 0.05, with exact p-values reported. This replaces the Bonferroni
correction specified above. The change was made after the results of this experiment were known.
