"""
r4reg.py — R4 regulation and R1-R3 x R4 interactions (docs/r4_regulation_hypotheses.md).

    python -m experiments.r4reg [jobs]     # -> results/r4reg.json, prints the 22 primary tests
"""
import json
import os
import sys
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
from scipy.stats import wilcoxon

from experiments import config as C
from experiments.ablation import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = C.ABLATIONS
NOR4 = dict(bcm_gain=0.0, neuromod_gain=0.0, neuromod_explore=0.0)
STRESS = dict(decay=0.999, bounded=False)
CONFIGS = {   # name: (kind, parameters)
    'full': ('scale', {}), 'noreward': ('scale', A['noreward']), 'nobcm': ('scale', A['nobcm']),
    'noreward+nobcm': ('scale', {**A['noreward'], **A['nobcm']}),
    'nohomeo': ('scale', A['nohomeo']), 'nor4': ('scale', NOR4), 'nohomeo+nor4': ('scale', {**A['nohomeo'], **NOR4}),
    'nopred': ('scale', A['nopred']), 'noneuro': ('scale', A['noneuro']), 'nopred+noneuro': ('scale', {**A['nopred'], **A['noneuro']}),
    'stress': ('scale', STRESS), 'stress+nobcm': ('scale', {**STRESS, **A['nobcm']}),
    'shift:full': ('shift', {}), 'shift:nostruct': ('shift', C.NOSTRUCT), 'shift:noneuro': ('shift', A['noneuro']),
    'shift:nostruct+noneuro': ('shift', {**C.NOSTRUCT, **A['noneuro']}),
}
ALPHA = 0.05 / 22
S4 = [s for s, _, _ in C.SCALES]


def episode(job):
    name, sc, seed = job
    kind, params = CONFIGS[name]
    return measure(kind, name, params or None, sc, seed)


def jobs():
    return [(n, sc, s) for n, (kind, _) in CONFIGS.items()
            for sc in (S4 if kind == 'scale' else C.SHIFT_SCALES) for s in C.SEEDS]


def vec(R, name, sc, k):
    rows = sorted([x for x in R if x['ablation'] == name and x['scale'] == sc], key=lambda x: x['seed'])
    return np.array([np.nan if x[k] is None else x[k] for x in rows], float)


def test(x):
    x = x[~np.isnan(x)]
    return float(x.mean()), (float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0)


def analyse(R):
    out = {}
    for sc in S4:   # E: BCM on vs off under stress; effect = off - on (positive: BCM lowers the 95th percentile)
        on, off = vec(R, 'stress', sc, 'w95'), vec(R, 'stress+nobcm', sc, 'w95')
        d, p = test(off - on)
        out[f'E_{sc}'] = dict(on=float(on.mean()), off=float(off.mean()), diff=d, p=p,
                              above1_on=float(vec(R, 'stress', sc, 'above1').mean()), above1_off=float(vec(R, 'stress+nobcm', sc, 'above1').mean()))
    for sc in S4:   # F: induced formations (actual minus unmodulated counterfactual) in full D-MORPH
        ind, frm = vec(R, 'full', sc, 'induced'), vec(R, 'full', sc, 'formed')
        d, p = test(ind)
        out[f'F_{sc}'] = dict(induced=float(ind.mean()), formed=float(frm.mean()), share=float(ind.sum() / frm.sum()), diff=d, p=p)
    for sc in C.SHIFT_SCALES:
        a, b = vec(R, 'shift:full', sc, 'induced_after'), vec(R, 'shift:full', sc, 'induced_before')
        d, p = test(a - b)
        out[f'Fshift_{sc}'] = dict(after=float(a.mean()), before=float(b.mean()), diff=d, p=p)
    specs = [  # (id, Y, sign, X on/off with M on, X on/off with M off, scales)
        ('I1', 'credit_gap', +1, ('full', 'noreward'), ('nobcm', 'noreward+nobcm'), S4),
        ('I2', 'deg_dev', -1, ('full', 'nohomeo'), ('nor4', 'nohomeo+nor4'), S4),
        ('I3', 'links50', +1, ('full', 'nopred'), ('noneuro', 'nopred+noneuro'), S4),
        ('I4', 'coassigned_linked', +1, ('shift:full', 'shift:nostruct'), ('shift:noneuro', 'shift:nostruct+noneuro'), C.SHIFT_SCALES),
    ]
    for tid, Y, sgn, (xon, xoff), (xon_m, xoff_m), scales in specs:
        for sc in scales:
            dp = sgn * (vec(R, xon, sc, Y) - vec(R, xoff, sc, Y))
            dm = sgn * (vec(R, xon_m, sc, Y) - vec(R, xoff_m, sc, Y))
            d, p = test(dp - dm)
            out[f'{tid}_{sc}'] = dict(delta_with=float(np.nanmean(dp)), delta_without=float(np.nanmean(dm)), interaction=d, p=p)
    return out


def report(out):
    print(f'22 primary tests, Bonferroni alpha = {ALPHA:.5f}')
    for k, v in out.items():
        sig = ' *' if 'p' in v and not k.startswith('Fshift') and v['p'] < ALPHA else ''
        print(f'  {k:16s} ' + '  '.join(f'{a}={b:.4g}' for a, b in v.items()) + sig)


if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() // 2) as pool:
        R = pool.map(episode, jobs(), chunksize=1)
    json.dump(dict(experiment='r4reg', seeds=C.SEEDS, records=R), open(os.path.join(ROOT, 'results', 'r4reg.json'), 'w'), separators=(',', ':'))
    report(analyse(R))
