"""
config.py — fixed settings of every experiment in the paper (seeds, fleet sizes, horizons, baselines).
"""

SEEDS = [13, 42, 87, 123, 256, 512, 777, 1024, 1337, 2021,
         4096, 5555, 6789, 8192, 9999, 12345, 22222, 31415, 42424, 65536]   # the 20 reported seeds

SCALES = [                     # (name, TA-RWARE environment id, robots N)
    ('tiny',   'tarware-tiny-5agvs-3pickers-partialobs-v1',     8),
    ('small',  'tarware-small-8agvs-4pickers-partialobs-v1',   12),
    ('medium', 'tarware-medium-12agvs-6pickers-partialobs-v1', 18),
    ('large',  'tarware-large-16agvs-8pickers-partialobs-v1',  24),
]
N_AGV = {'tiny': 5, 'small': 8, 'medium': 12, 'large': 16}

T_SCALE = 800                  # steps per episode in the scale, frontier and ablation experiments
T_SHIFT = 2400                 # package location shift: six 400-step phases
SHIFT_PHASE = 400
SHIFT_SCALES = ('medium', 'large')
T_EMERGENT, EMERGENT_SCALE, EMERGENT_SEED = 1500, 'medium', 42
EMERGENT_SNAPSHOTS = (99, 199, 399, 799, 1499)

# ── methods ──────────────────────────────────────────────────────────────────
# Full Graph, MORPH, Proximity-r and TSG run in MORPH's allocation loop (nearest free AGV to each request,
# nearest free Picker proposed for each AGV task, response gated by the link or preference).
# D-MORPH and both Random-Sparse variants use pair allocation (oldest request -> nearest available robot i
# -> i requests a partner by its own preferences -> the partner accepts with 0.25 + 0.75 w_j[i]).
METHODS = ['full', 'dmorph', 'morph', 'proximity', 'random_fixed', 'random_step', 'tsg']
LABELS = {'full': 'Full Graph', 'dmorph': 'D-MORPH', 'morph': 'MORPH (centralised)', 'proximity': 'Proximity-r',
          'random_fixed': 'Random-Sparse (fixed)', 'random_step': 'Random-Sparse (per-step)', 'tsg': 'TSG'}
PROX_R = 5                                 # Proximity radius in the scale and shift experiments
PROX_GRID = list(range(1, 13))             # radii of the Proximity-r sweep (Figure 4)
# Random-Sparse link budget: D-MORPH's mean number of links on development seeds, rounded
RANDOM_LINKS = {'tiny': 6, 'small': 12, 'medium': 18, 'large': 24}

# D-MORPH ablations: one or more mechanisms switched off, everything else unchanged
ABLATIONS = {
    'v1':       dict(bcm_gain=0.0, reward_alpha=0.0, neuromod_gain=0.0, neuromod_explore=0.0, pred_boost=0.0),
    'nobcm':    dict(bcm_gain=0.0),
    'noreward': dict(reward_alpha=0.0),
    'noneuro':  dict(neuromod_gain=0.0, neuromod_explore=0.0),
    'nopred':   dict(pred_boost=0.0),
}
ABLATION_LABELS = {'v1': 'v1 (R1-R3)', 'nobcm': 'no BCM threshold', 'noreward': 'no reward credit',
                   'noneuro': 'no neuromodulation', 'nopred': 'no predictive formation'}

# Centralised MORPH: the published MORPH parameters (Optuna TPE search on TA-RWARE)
MORPH_PARAMS = dict(
    alpha=0.14606852756899355, beta=0.08486507862257056, decay=0.9550542005105067,
    theta_form_start=0.5778923366768645, theta_form_end=0.3821591539998435, theta_form_anneal=100,
    theta_prune=0.014446945935515619, target_deg_frac=0.16294004175784596, grace_steps=20, k_slow=5,
    bcm_tau=0.9658382457133111, bcm_gain=0.5, reward_alpha=0.14380425940210126,
    neuromod_gain=0.4, neuromod_explore=0.10, neuromod_ema=0.03, pred_boost=0.4,
)
MORPH_EXPECTED_RATE = {'tiny': 0.07, 'small': 0.09, 'medium': 0.12, 'large': 0.14}


def morph_params(N, scale):
    return dict(MORPH_PARAMS, max_new=min(6, max(3, N // 6)), expected_delivery_rate=MORPH_EXPECTED_RATE[scale])
