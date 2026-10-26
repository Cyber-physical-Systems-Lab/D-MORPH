"""
run.py — runs the paper's experiments and writes one results file per experiment.

    python -m experiments.run scale      # Table 4, Figure 3: seven methods, four fleet sizes, 20 seeds, T = 800
    python -m experiments.run frontier   # Figure 4: Proximity-r for r = 1..12
    python -m experiments.run ablation   # Figure 5, Supplementary Table S1: rule ablations (experiments/ablation.py)
    python -m experiments.run shift      # Table 5, Figure 6: package location shift, Medium and Large, T = 2400
    python -m experiments.run emergent   # Figure 7: one long D-MORPH episode (Medium, seed 42, T = 1500)
    python -m experiments.run mechanism  # Table 4 refused / idle / task-time columns (also queue wait, clashes, stucks)
    python -m experiments.run all --jobs 16        # the six experiments above

Additional analyses that are not reported in the manuscript (run separately):
    python -m experiments.run control    # pair allocation with every pair linked and frozen uniform preferences
    python -m experiments.run partner    # hidden slow Pickers: partner-selection quality and adaptation
    python -m experiments.run r4         # R4 on/off x bounded/unbounded preference weights (docs/r4_hypotheses.md)
    python -m experiments.run r4reg      # R4 regulation and R1-R3 x R4 interactions (docs/r4_regulation_hypotheses.md)

Outputs: results/{scale,frontier,ablation,shift,emergent,mechanism}.json; control, partner, r4, r4reg
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per worker process

import numpy as np

from experiments import ablation, config as C, r4, r4reg
from experiments.harness import run_episode

MODULES = {'ablation': ablation, 'r4': r4, 'r4reg': r4reg}   # experiments with their own episode and report code
PAPER = ['scale', 'frontier', 'ablation', 'shift', 'emergent', 'mechanism']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
NSIZE = {name: n for name, _, n in C.SCALES}


def _cross_role(A, n_agv):
    """Share of a coordinator's links that join an AGV and a Picker."""
    U = np.triu((A > 0) | (A.T > 0), 1)
    return float(U[:n_agv, n_agv:].sum() / U.sum()) if U.sum() else None


def _episode(job):
    kind, method, scale, seed, extra = job
    T = C.T_SHIFT if kind == 'shift' else C.T_SCALE
    kw = dict(shift=kind == 'shift')
    if kind == 'frontier':
        kw['r'] = extra
    if kind in ('control', 'mechanism') and method == 'uniform':
        kw['w_uniform'] = extra
    if kind == 'partner':
        kw['slow'], w = extra
        if w is not None:
            kw['w_uniform'] = w
    out = run_episode(method, scale, seed, T, **kw)
    d = out['deliveries']
    rec = dict(method=method, scale=scale, seed=seed, deliveries=int(d[-1]), links=float(out['links'].mean()))
    if kind == 'frontier':
        rec['r'] = extra
    if kind == 'control':
        st = out['stats']
        rec['w_uniform'] = extra
        rec['refusals'] = st['refusals'] / max(st['requests'], 1)
    if kind == 'mechanism':
        cg = out['congestion']
        rec['w_uniform'] = extra
        rec.update({k: cg[k] for k in ('proposals', 'refusals', 'clashes', 'stucks', 'auctioned', 'resolved')})
        rec.update({k: cg[k] / T for k in ('agv_idle', 'active_missions', 'picking_missions', 'wait_for_picker')})
        rec['queue_wait'] = float(np.mean(cg['queue_wait'])) if cg['queue_wait'] else None
        rec['cycle'] = float(np.mean(cg['cycle'])) if cg['cycle'] else None
        rec['agv_idle_rate'] = cg['agv_idle_steps'] / (T * C.N_AGV[scale])
        rec['picker_idle_rate'] = cg['picker_idle_steps'] / (T * (NSIZE[scale] - C.N_AGV[scale]))
        rec['tasks_done'] = len(cg['task_return'])
        for k in ('task_delivery', 'task_return', 'task_agv_wait', 'task_picker_wait'):
            rec[k] = float(np.mean(cg[k])) if cg[k] else None
    if kind == 'partner':
        rec['slow'], rec['w_uniform'] = extra
        rec['selection'] = [[int(a), int(b)] for a, b in out['selection']]
        rec['pref_gap'] = out['pref_gap']
        rec['deliveries_ts'] = [int(x) for x in d[24::25]]
        cg = out['congestion']
        rec['task_return'] = float(np.mean(cg['task_return'])) if cg['task_return'] else None
    if kind == 'scale':
        rec['deliveries_ts'] = [int(x) for x in d[24::25]]                 # cumulative deliveries every 25 steps
        rec['w_mean'] = float(out['w_mean'].mean())
        if out['coordinator'] is not None:
            rec['cross_role'] = _cross_role(out['coordinator'].A, C.N_AGV[scale])
        if method in ('dmorph', 'random_fixed', 'random_step'):
            st = out['stats']
            rec['non_nearest'] = st['non_nearest'] / max(st['requests'], 1)  # partner differs from the nearest robot
            rec['refusals'] = st['refusals'] / max(st['requests'], 1)
            rec['failed'] = {k: st[k] for k in ('no_link', 'same_role', 'busy')}
    if kind == 'shift':
        rec['step_deliveries'] = np.diff(np.concatenate(([0], d))).astype(int).tolist()
        rec['links_ts'] = [float(x) for x in out['links'][24::25]]
        rec['zone_share'] = {str(k): v for k, v in out['zone_share'].items()}
    return rec


def _jobs(kind):
    S = [s for s, _, _ in C.SCALES]
    if kind == 'scale':
        return [('scale', m, sc, s, None) for m in C.METHODS for sc in S for s in C.SEEDS]
    if kind == 'frontier':
        return [('frontier', 'proximity', sc, s, r) for r in C.PROX_GRID for sc in S for s in C.SEEDS]
    if kind == 'partner':
        return [('partner', m, sc, s, (k, w)) for k in ('static', 'swap') for m, w in C.PARTNER_METHODS
                for sc in S for s in C.SEEDS]
    if kind == 'mechanism':
        ms = [(m, None) for m in C.METHODS]                    # the seven methods of Table 4
        return [('mechanism', m, sc, s, w) for m, w in ms for sc in S for s in C.SEEDS]
    if kind == 'control':
        return [('control', 'uniform', sc, s, w) for w in C.UNIFORM_W for sc in S for s in C.SEEDS]
    if kind == 'shift':
        return [('shift', m, sc, s, None) for m in C.METHODS for sc in C.SHIFT_SCALES for s in C.SEEDS]
    raise ValueError(kind)


def emergent():
    """Each robot's own links and preferences at snapshots, links by pair type, mean preference on links and the
    lifetime of every link, in one long D-MORPH episode."""
    sc, seed, T = C.EMERGENT_SCALE, C.EMERGENT_SEED, C.T_EMERGENT
    N, na = NSIZE[sc], C.N_AGV[sc]; is_agv = np.arange(N) < na
    snaps, ap, aa, pp, wl = {}, [], [], [], []
    born, life, prev = {}, [], np.zeros((N, N), bool)

    def rec(t, coordinator, **_):
        nonlocal prev
        Ad = coordinator.A > 0; Wd = coordinator.W
        ii, jj = np.nonzero(np.triu(Ad | Ad.T, 1))
        a_ = int(np.sum(is_agv[ii] != is_agv[jj])); b_ = int(np.sum(is_agv[ii] & is_agv[jj]))
        ap.append(a_); aa.append(b_); pp.append(len(ii) - a_ - b_)
        wl.append(float(Wd[Ad].mean()) if Ad.any() else 0.0)
        for i, j in zip(*np.nonzero(Ad & ~prev)):
            born[(i, j)] = t
        for i, j in zip(*np.nonzero(prev & ~Ad)):
            life.append(t - born.pop((i, j)))
        prev = Ad.copy()
        if t in C.EMERGENT_SNAPSHOTS:
            snaps[t] = dict(A=Ad.astype(int).tolist(), W=np.round(Wd, 5).tolist())

    out = run_episode('dmorph', sc, seed, T, recorder=rec)
    st = out['stats']
    return dict(scale=sc, seed=seed, T=T, N=N, n_agv=na, deliveries=int(out['deliveries'][-1]),
                snapshots={str(k): v for k, v in snaps.items()}, links_agv_picker=ap, links_agv_agv=aa,
                links_picker_picker=pp, w_mean_links=wl, lifetimes=[int(x) for x in life],
                lifetimes_open=[int(T - 1 - t0) for t0 in born.values()],
                non_nearest=st['non_nearest'] / max(st['requests'], 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment', choices=PAPER + ['control', 'partner', 'r4', 'r4reg', 'all'])
    ap.add_argument('--jobs', type=int, default=os.cpu_count() // 2)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    for kind in (PAPER if args.experiment == 'all' else [args.experiment]):
        t0 = time.time()
        if kind == 'emergent':
            data = emergent()
        elif kind in MODULES:
            mod = MODULES[kind]; jobs = mod.jobs()
            print(f'{kind}: {len(jobs)} episodes on {args.jobs} processes')
            with Pool(args.jobs) as pool:
                records = pool.map(mod.episode, jobs, chunksize=1)
            data = dict(experiment=kind, seeds=C.SEEDS, records=records)
        else:
            jobs = _jobs(kind)
            print(f'{kind}: {len(jobs)} episodes on {args.jobs} processes')
            with Pool(args.jobs) as pool:                         # processes: each episode seeds numpy's global RNG
                records = pool.map(_episode, jobs, chunksize=1)
            data = dict(experiment=kind, seeds=C.SEEDS, T=C.T_SHIFT if kind == 'shift' else C.T_SCALE, records=records)
        with open(os.path.join(RES, f'{kind}.json'), 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        print(f'{kind}: results/{kind}.json written ({time.time() - t0:.0f} s)')
        if kind in MODULES:
            MODULES[kind].report(MODULES[kind].analyse(records) if kind == 'r4reg' else records)


if __name__ == '__main__':
    main()
