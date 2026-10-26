# Rule ablations: hypotheses and primary measures

Written 2026-10-22, before the runs in `results/rules.json`. Each hypothesis restates the function the manuscript's
Method section assigns to the rule; each rule has one primary measure, chosen from that statement. Deliveries are
reported for every ablation as a secondary measure. All tests are paired Wilcoxon signed-rank tests over the 20
paper seeds (full D-MORPH vs. the ablation, same seed), two-sided, Bonferroni-corrected over the 18 primary tests
below (alpha = 0.05 / 18 = 0.0028). Measures are taken after step 200 unless stated.

| Rule | Method-section claim | Ablation | Setting | Primary measure | Predicted direction (full vs. ablation) |
|---|---|---|---|---|---|
| R1 completion credit | only preferences contributing to completed deliveries are strengthened | `noreward` (reward_alpha = 0) | scale study, 4 scales | mean w_i[j] of AGV-Picker pairs that completed a delivery together in the last 50 steps minus mean w_i[j] of pairs that were co-assigned in the last 50 steps without completing one | larger with credit |
| R2 homeostasis | prevents isolation and hub collapse; keeps degree near target | `nohomeo` (beta = 0) | scale study, 4 scales | mean over robots and steps of abs(links_i - target) | smaller with R2 |
| R3 predictive formation | closes the cold-start gap before co-assignment has occurred | `nopred` (pred_boost = 0) | scale study, 4 scales | active links at step 50 | more with the hint |
| R3 structural plasticity | without it, links are fixed and new coordination relationships cannot be discovered | `nostruct` (no formation or pruning after step 400) | location shift, Medium and Large | share of co-assigned AGV-Picker pairs that are linked, phases 2-6 | larger with R3 |
| R4a BCM threshold | without it, preferences saturate and resist pruning after their relationship dissolves | `nobcm` (bcm_gain = 0) | location shift, Medium and Large | share of links active 25 steps before a shift that are still active 50 steps after it (shifts 1-5) | smaller with BCM |
| R4b neuromodulation | primary mechanism for accelerating recovery after shifts | `noneuro` (neuromod_gain = neuromod_explore = 0) | location shift, Medium and Large | mean recovery time r-bar (Table 6 definition) | shorter with neuromodulation |

Count of primary tests: 4 + 4 + 4 + 2 + 2 + 2 = 18.

## Amendment: multiplicity correction removed (2026-10-22, after the results were known)

At the author's decision, the paper reports all tests without correction for multiple comparisons: two-sided
Wilcoxon signed-rank tests, significant at p < 0.05, with exact p-values reported. This replaces the Bonferroni
correction specified above. The change was made after the results of this experiment were known.

## Note (2026-10-26): files and reporting

- `experiments/rules.py` is now `experiments/ablation.py` and `results/rules.json` is `results/ablation.json`
  (`python -m experiments.run ablation`); the earlier six-variant ablation (v1, no BCM threshold, no reward credit, no
  neuromodulation, no predictive formation, no homeostasis on deliveries and links) was removed from the repository.
- The six hypotheses above were run as specified. The combined R4 ablation (`nor4`: BCM decay and neuromodulation both
  off; measure: mean active links) was added after the results were known and is labelled as additional.
- The manuscript reports the five ablations No completion credit (R1), No homeostasis (R2), No predictive formation
  (R3), Frozen structure (R3) and No R4. The two pre-specified R4 component tests (`nobcm`, `noneuro`) are not reported
  in the manuscript; they remain in `results/ablation.json` and `docs/results.md` (p = 0.19 and 0.25 for `nobcm`,
  0.074 and 0.23 for `noneuro`, Medium and Large).
