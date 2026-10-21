"""
params.py — D-MORPH's parameters as used in the paper (identical at every fleet size).

Selected with MORPH's procedure (Optuna TPE, 200 configurations on development seeds 101-110, the
best 10 re-evaluated on seeds 111-120) and then frozen; the reported seeds were never used for tuning.
"""

TUNED_PARAMS = dict(
    # R1 synaptic
    alpha=0.11536284398087643,          # Hebbian learning rate
    decay=0.9499857043247211,           # gamma, per-step retention of w_i[j]
    cov=0.05,                           # Hebbian scale (MORPH's constant)
    window=10,                          # steps of target history behind J_ij
    reward_alpha=0.17615321885588503,   # completion-credit rate
    burst_thresh=0.25,                  # credit only if the pair's task overlap J exceeded this
    # R2 homeostatic
    beta=0.08065240597281449,
    target_deg_frac=0.1606030221516968, # target share of the other robots linked
    # R3 structural
    theta_form_start=0.8459777343599962,
    theta_form_end=0.5547689966966961,
    theta_form_anneal=100,              # steps over which theta_form anneals from start to end
    theta_prune=0.012846351512931578,
    grace_steps=20,                     # a new link cannot be pruned before this age
    k_slow=1,                           # structural update every k_slow steps
    mi_window=20,                       # steps of co-assignment history behind C_ij
    max_new_local=1,                    # links formed per robot per structural step
    pred_boost=0.4,                     # weight of the proposal hint P_ij in formation
    # R4 metaplastic
    bcm_tau=0.9619711638744608,         # BCM threshold smoothing
    bcm_gain=0.5,
    neuromod_gain=0.4,                  # alpha scaling when performance is above expectation
    neuromod_explore=0.1,               # theta_form reduction when below expectation
    neuromod_ema=0.03,                  # fast average of own outcomes
    slow_ema=0.006,                     # slow average of own outcomes (the expectation)
)
