"""
run_dmorph.py — run D-MORPH (or a baseline) for one TA-RWARE episode and print what happened.

    python scripts/run_dmorph.py                              # D-MORPH, Medium (18 robots), seed 42, 800 steps
    python scripts/run_dmorph.py --scale large --steps 1500
    python scripts/run_dmorph.py --method full --seed 13      # a baseline from experiments/config.py
    python scripts/run_dmorph.py --shift --steps 2400         # package location shift (upper/lower half)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import config as C                     # noqa: E402
from experiments.harness import run_episode             # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--method', default='dmorph', choices=C.METHODS)
ap.add_argument('--scale', default='medium', choices=[s for s, _, _ in C.SCALES])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--steps', type=int, default=C.T_SCALE)
ap.add_argument('--shift', action='store_true', help='package location shift every 400 steps')
ap.add_argument('--reward', default='pair', choices=['pair', 'task'], help="D-MORPH credit: 'pair' (paper) or 'task' settlement")
a = ap.parse_args()

out = run_episode(a.method, a.scale, a.seed, a.steps, shift=a.shift, reward=a.reward)
N = len(out['agents']); d = out['deliveries']
print(f'{C.LABELS[a.method]} on TA-RWARE {a.scale} (N = {N}), seed {a.seed}, {a.steps} steps')
print(f'  deliveries: {d[-1]}  (every 100 steps: {", ".join(str(x) for x in d[99::100])})')
print(f'  active links: mean {out["links"].mean():.1f} of {N * (N - 1) // 2} possible pairs, {out["links"][-1]} at the end')
coord = out['coordinator']
if a.method == 'dmorph':
    n_agv = C.N_AGV[a.scale]
    U = np.triu((coord.A > 0) | (coord.A.T > 0), 1)
    print(f'  links joining an AGV and a Picker: {int(U[:n_agv, n_agv:].sum())} of {int(U.sum())}')
    print(f'  mean preference on links: {coord.W[coord.A > 0].mean():.3f}' if coord.A.any() else '  no links at the end')
    st = out['stats']
    print(f'  partner differs from the nearest robot in {100 * st["non_nearest"] / max(st["requests"], 1):.1f}% of requests; '
          f'{100 * st["refusals"] / max(st["requests"], 1):.0f}% of requests refused and retried')
