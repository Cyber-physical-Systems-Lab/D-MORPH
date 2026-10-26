"""
ablation.py — rule ablations with one pre-specified primary measure per rule (docs/ablation_hypotheses.md).

    python -m experiments.ablation         # or: python -m experiments.run ablation -> results/ablation.json

Scale study (T = 800, four scales): full D-MORPH and the ablations noreward (R1 credit), nohomeo (R2), nopred
(R3 predictive formation) and nor4 (R4, both parts). Location shift (T = 2400, Medium and Large): full D-MORPH and
nostruct (R3 frozen after step 400), plus the pre-specified R4 component tests nobcm (R4a) and noneuro (R4b).
The recorder only reads the state; the episodes are those of the other experiments.
"""
import json
import os
import sys
from collections import deque
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
from scipy.stats import wilcoxon
from tarware.heuristic import AgentType

from experiments import config as C
from experiments.analysis import endpoints
from experiments.harness import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN, T0 = 50, 200                     # window of the R1 measure; measures start after step T0
SCALE_ABL = ('noreward', 'nohomeo', 'nopred', 'nor4')
SHIFT_ABL = ('nostruct', 'nobcm', 'noneuro')
PRIMARY = {'noreward': 'credit_gap', 'nohomeo': 'deg_dev', 'nopred': 'links50',
           'nostruct': 'coassigned_linked', 'nobcm': 'stale_survival', 'noneuro': 'rbar',
           'nor4': 'links'}                  # nor4: combined R4 ablation, added after the pre-specified tests
ALPHA = 0.05                      # no multiplicity correction (paper)


def _params(ab):
    return C.ABLATIONS[ab] if ab else None


def episode(job):
    kind, ab, sc, seed = job
    return measure(kind, ab or 'full', _params(ab), sc, seed)


def measure(kind, label, params, sc, seed):
    """One D-MORPH episode with the primary measures of docs/ablation_hypotheses.md (and of
    docs/r4_regulation_hypotheses.md: weight percentiles, arousal-induced formations)."""
    shift = kind == 'shift'
    T = C.T_SHIFT if shift else C.T_SCALE
    st = dict(prev_pair={}, delivered=deque(), coassigned=deque(), gap=[], dev=[], linked=[], snap={})

    def rec(t, agents, coordinator, assigned_agvs, assigned_pickers, **_):
        idx = {a: i for i, a in enumerate(agents)}
        W, A = coordinator.W, coordinator.A > 0
        # pairs that completed a delivery this step: the AGV's recorded partner before the step, AGV rewarded
        for a, p in st['prev_pair'].items():
            if coordinator.rewards[a] > 0.5:
                st['delivered'].append((t, a, p))
        st['prev_pair'] = dict(coordinator.pair)
        # co-assigned AGV-Picker pairs: the Picker's mission is at the AGV's mission location
        loc = {(m.location_x, m.location_y): idx[agv] for agv, m in assigned_agvs.items()}
        co = {(loc[(m.location_x, m.location_y)], idx[p]) for p, m in assigned_pickers.items()
              if (m.location_x, m.location_y) in loc}
        for a, p in co:
            st['coassigned'].append((t, a, p))
        for q in ('delivered', 'coassigned'):
            while st[q] and st[q][0][0] <= t - WIN:
                st[q].popleft()
        if t >= T0:
            D = {(a, p) for _, a, p in st['delivered']}
            Cn = {(a, p) for _, a, p in st['coassigned']} - D
            if D and Cn:
                w = lambda S: np.mean([(W[a, p] + W[p, a]) / 2 for a, p in S])
                st['gap'].append(w(D) - w(Cn))
            st['dev'].append(float(np.mean(np.abs(A.sum(axis=1) - coordinator.target_deg))))
            if co:
                st['linked'].append((t, np.mean([A[a, p] or A[p, a] for a, p in co])))
            if t % 10 == 0:
                st.setdefault('wpos', []).extend(W[W > 0].tolist())
        if t == 49:
            st['links50'] = int(A.sum())
        if shift:
            for k in range(1, 6):
                if t == k * C.SHIFT_PHASE - 25:
                    st['snap'][k] = A.copy()
                if t == k * C.SHIFT_PHASE + 50 and k in st['snap']:
                    before = st['snap'][k]
                    st.setdefault('surv', []).append(float((before & A).sum() / max(before.sum(), 1)))

    out = run_episode('dmorph', sc, seed, T, params=params, shift=shift, recorder=rec)
    d = out['deliveries']; c = out['coordinator']
    wp = np.array(st.get('wpos', [0.0]))
    ind = np.array(c.induced_h)
    r = dict(kind=kind, ablation=label, scale=sc, seed=seed, deliveries=int(d[-1]),
             credit_gap=float(np.mean(st['gap'])) if st['gap'] else None, deg_dev=float(np.mean(st['dev'])),
             links50=st.get('links50'), deliveries100=int(d[99]), links=float(out['links'].mean()), w95=float(np.percentile(wp, 95)),
             above1=float((wp > 1).mean()), formed=int(sum(c.form_h)), induced=int(ind.sum()))
    if shift:
        r['induced_after'] = float(np.mean([ind[k * C.SHIFT_PHASE:k * C.SHIFT_PHASE + 100].sum() for k in range(1, 6)]))
        r['induced_before'] = float(np.mean([ind[k * C.SHIFT_PHASE - 100:k * C.SHIFT_PHASE].sum() for k in range(1, 6)]))
    if shift:
        sd = np.diff(np.concatenate(([0], d))).astype(int).tolist()
        e = endpoints(sd)
        lk = [v for t_, v in st['linked'] if t_ >= C.SHIFT_PHASE]          # phases 2-6
        r.update(coassigned_linked=float(np.mean(lk)) if lk else None,
                 stale_survival=float(np.mean(st['surv'])), rbar=e['rbar'], dbar=e['dbar'], P=e['P'])
    return r


def jobs():
    J = [('scale', ab, sc, s) for ab in (None,) + SCALE_ABL for sc, _, _ in C.SCALES for s in C.SEEDS]
    J += [('shift', ab, sc, s) for ab in (None,) + SHIFT_ABL for sc in C.SHIFT_SCALES for s in C.SEEDS]
    return J


def report(R):
    def v(kind, ab, sc, k):
        rows = sorted([x for x in R if x['kind'] == kind and x['ablation'] == ab and x['scale'] == sc], key=lambda x: x['seed'])
        return np.array([x[k] for x in rows], float)
    print(f'Primary measures: paired two-sided Wilcoxon, full vs ablation, p < {ALPHA} (no multiplicity correction)')
    for ab in SCALE_ABL + SHIFT_ABL:
        kind = 'scale' if ab in SCALE_ABL else 'shift'
        k = PRIMARY[ab]
        scales = [s for s, _, _ in C.SCALES] if kind == 'scale' else C.SHIFT_SCALES
        for sc in scales:
            f, a = v(kind, 'full', sc, k), v(kind, ab, sc, k)
            ok = ~np.isnan(f) & ~np.isnan(a)
            p = wilcoxon(f[ok] - a[ok]).pvalue if np.any(f[ok] != a[ok]) else 1.0
            df, da = v(kind, 'full', sc, 'deliveries'), v(kind, ab, sc, 'deliveries')
            pd = wilcoxon(df - da).pvalue if np.any(df != da) else 1.0
            print(f'  {ab:9s} {sc:7s} {k:18s} full {np.nanmean(f):8.4f}  ablation {np.nanmean(a):8.4f}  p={p:.2g}{" *" if p < ALPHA else "  "}'
                  f' | deliveries full {df.mean():6.1f} ablation {da.mean():6.1f} (p={pd:.3f})')


if __name__ == '__main__':
    out = os.path.join(ROOT, 'results', 'ablation.json')
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() // 2) as pool:
        R = pool.map(episode, jobs(), chunksize=1)
    json.dump(dict(experiment='ablation', seeds=C.SEEDS, records=R), open(out, 'w'), separators=(',', ':'))
    report(R)
