"""
r4.py — R4 metaplasticity: bounded/unbounded weights x R4 on/off (docs/r4_hypotheses.md).

    python -m experiments.r4 [jobs]        # -> results/r4.json, prints the 8 primary tests and secondary measures

S1: published parameters, package location shift. S2: retention 0.999, hidden slow Pickers swapped at step 400.
S3: retention 0.999, package location shift. Medium and Large, the 20 paper seeds, all four configurations per seed.
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
from experiments.analysis import endpoints
from experiments.harness import run_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R4_OFF = dict(bcm_gain=0.0, neuromod_gain=0.0, neuromod_explore=0.0)
STRESS = dict(decay=0.999)
SETTINGS = {'S1': (dict(), True), 'S2': (STRESS, False), 'S3': (STRESS, True)}   # (parameters, location shift)
CONFIGS = [(b, r) for b in (True, False) for r in (True, False)]                  # (bounded, R4 on)
SCALES = ('medium', 'large')
ALPHA = 0.05 / 8
MA = 25


def episode(job):
    setting, bounded, r4, sc, seed = job
    base, shift = SETTINGS[setting]
    params = dict(base, bounded=bounded, **({} if r4 else R4_OFF))
    T = C.T_SHIFT if shift else C.T_SCALE
    st = dict(g=[], prev=None, new=[], eta=[], wmax=0.0, above1=[])

    def rec(t, agents, coordinator, slow_mask, **_):
        c = coordinator; W = c.W; A = c.A > 0
        n_agv = c.n_agv
        if slow_mask is not None:
            sm = slow_mask(t)
            fast = [j for j in range(n_agv, c.N) if not sm[j]]; slow = [j for j in range(n_agv, c.N) if sm[j]]
            st['g'].append(float(np.mean(W[:n_agv][:, fast].mean(axis=1) - W[:n_agv][:, slow].mean(axis=1))))
        st['new'].append(int((A & ~st['prev']).sum()) if st['prev'] is not None else 0)
        st['prev'] = A.copy()
        if t >= 200:
            st['eta'].append(float(np.mean(np.clip((c.d_fast - c.d_slow) / (c.d_slow + 1e-9), -1, 1))))
            nz = W[W > 0]
            st['above1'].append(float((nz > 1).mean()) if nz.size else 0.0)
        st['wmax'] = max(st['wmax'], float(W.max()))

    out = run_episode('dmorph', sc, seed, T, params=params, shift=shift, slow=None if shift else 'swap', recorder=rec)
    d = out['deliveries']
    r = dict(setting=setting, bounded=bounded, r4=r4, scale=sc, seed=seed, deliveries=int(d[-1]), wmax=st['wmax'],
             above1=float(np.mean(st['above1'])), eta=float(np.mean(st['eta'])))
    if shift:
        e = endpoints(np.diff(np.concatenate(([0], d))).astype(int).tolist())
        r.update(rbar=e['rbar'], dbar=e['dbar'], tr=e['tr'],
                 new50=float(np.mean([sum(st['new'][k * C.SHIFT_PHASE:k * C.SHIFT_PHASE + 50]) for k in range(1, 6)])))
    else:
        g = np.array(st['g']); sw = C.SWAP_AT
        ma = np.convolve(g, np.ones(MA) / MA, mode='full')[:len(g)]          # trailing 25-step moving average
        hit = np.where(ma[sw + MA:] > 0)[0]                                  # window entirely after the swap
        r.update(pre_gap=float(g[sw - 100:sw].mean()), reversal=float(hit[0] + MA) if len(hit) else float(T - sw),
                 fast_after=float(np.mean([b for t_, b in out['selection'] if t_ >= sw])) if any(t_ >= sw for t_, _ in out['selection']) else None)
    return r


def jobs():
    return [(s, b, r, sc, seed) for s in SETTINGS for b, r in CONFIGS for sc in SCALES for seed in C.SEEDS]


def report(R):
    def v(s, b, r, sc, k):
        rows = sorted([x for x in R if x['setting'] == s and x['bounded'] == b and x['r4'] == r and x['scale'] == sc], key=lambda x: x['seed'])
        return np.array([np.nan if x.get(k) is None else x[k] for x in rows], float)
    pv = lambda x: wilcoxon(x[~np.isnan(x)]).pvalue if np.any(x[~np.isnan(x)] != 0) else 1.0
    print(f'Primary tests (effect of R4 = R4 off minus R4 on; Bonferroni alpha = {ALPHA:.5f})')
    for h, s, k in (('H1/H2', 'S2', 'reversal'), ('H3/H4', 'S3', 'rbar')):
        for sc in SCALES:
            eu = v(s, False, False, sc, k) - v(s, False, True, sc, k)
            eb = v(s, True, False, sc, k) - v(s, True, True, sc, k)
            it = eu - eb
            print(f'  {h} {s} {k:8s} {sc:6s} unbounded: on {np.nanmean(v(s, False, True, sc, k)):6.1f} off {np.nanmean(v(s, False, False, sc, k)):6.1f} '
                  f'effect {np.nanmean(eu):+6.1f} p={pv(eu):.3g}{" *" if pv(eu) < ALPHA else "  "} | bounded: on {np.nanmean(v(s, True, True, sc, k)):6.1f} '
                  f'off {np.nanmean(v(s, True, False, sc, k)):6.1f} effect {np.nanmean(eb):+6.1f} | interaction {np.nanmean(it):+6.1f} p={pv(it):.3g}{" *" if pv(it) < ALPHA else ""}')
    print('Secondary (means): setting scale bounded r4 -> ...')
    for s in SETTINGS:
        for sc in SCALES:
            for b, r in CONFIGS:
                m = lambda k: np.nanmean(v(s, b, r, sc, k))
                extra = (f"pre_gap {m('pre_gap'):+.4f} reversal {m('reversal'):6.1f} fast_after {m('fast_after'):.2f}" if s == 'S2'
                         else f"rbar {m('rbar'):5.1f} dbar {m('dbar'):+5.2f} new50 {m('new50'):5.1f}")
                print(f"  {s} {sc:6s} {'bounded  ' if b else 'unbounded'} R4 {'on ' if r else 'off'} | deliveries {m('deliveries'):6.1f} "
                      f"wmax {m('wmax'):.2f} w>1 {m('above1'):.3f} eta {m('eta'):+.2f} | {extra}")


if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() // 2) as pool:
        R = pool.map(episode, jobs(), chunksize=1)
    json.dump(dict(experiment='r4', seeds=C.SEEDS, records=R), open(os.path.join(ROOT, 'results', 'r4.json'), 'w'), separators=(',', ':'))
    report(R)
