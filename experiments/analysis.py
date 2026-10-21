"""
analysis.py — statistics, tables and figures of the paper, from results/*.json.

    python -m experiments.analysis

Outputs
  docs/figures/fig2_environment ... fig7_emergent (.pdf and .png)
  docs/tables/table4_scale.tex, docs/tables/table5_shift.tex
  docs/results.md            all reported numbers as Markdown tables

Statistics: paired over the 20 seeds; mean difference with a 95% t confidence interval and a two-sided
Wilcoxon signed-rank test; Bonferroni correction over the comparisons of each experiment.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import t as tdist, wilcoxon

from experiments import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, DOCS = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'docs')
FIG, TAB = os.path.join(DOCS, 'figures'), os.path.join(DOCS, 'tables')
S = [s for s, _, _ in C.SCALES]
NSZ = {s: n for s, _, n in C.SCALES}
BASE = [m for m in C.METHODS if m != 'dmorph']

# ── plot style ───────────────────────────────────────────────────────────────
MM = 1 / 25.4; W2 = 180 * MM
INK, MUTED, GRIDC, GREY, DARK = '#3A3936', '#7D7B75', '#DDDCD5', '#9C9A94', '#222222'
RED, BLUE, GREEN = '#971F30', '#2a78d6', '#1E8C45'
SHELF, AGV_C, PICK_C, GOAL_C = '#E4DCCB', '#1F5696', '#E2781E', '#6B6B6B'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 2.0, 'axes.linewidth': 0.8,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
ST = {'dmorph': (RED, '*', '-', 10), 'full': (DARK, 'D', '--', 5), 'morph': ('#6B5B95', 'P', '-.', 6),
      'proximity': (BLUE, 'o', '-', 5), 'random_fixed': ('#B8B5AE', 's', '-', 5),
      'random_step': ('#8C6D3F', 'p', '--', 6), 'tsg': ('#7F7C75', 'v', ':', 5)}
SHORT = {'full': 'Full', 'dmorph': 'D-MORPH', 'morph': 'MORPH', 'proximity': 'Prox.', 'random_fixed': 'Rand. fixed',
         'random_step': 'Rand. step', 'tsg': 'TSG'}
LW = lambda m: 2.6 if m == 'dmorph' else 1.1
ZO = lambda m: 10 if m == 'dmorph' else 3


def style(ax, grid_axis='x'):
    ax.grid(axis=grid_axis, color=GRIDC, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)


def panel_letter(fig, ax, letter):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.045, bb.y1 + 0.012, f'({letter})', fontsize=9, fontweight='bold', color=INK, ha='left', va='bottom')


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f'{name}.pdf'), facecolor='white')
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=200, facecolor='white')
    plt.close(fig); print('saved docs/figures/' + name)


# ── data ─────────────────────────────────────────────────────────────────────
def load(kind):
    return json.load(open(os.path.join(RES, f'{kind}.json')))


def by_seed(records, **match):
    return {r['seed']: r for r in records if all(r.get(k) == v for k, v in match.items())}


def mean(cur, k='deliveries'):
    return float(np.mean([v[k] for v in cur.values()]))


def sd(cur, k='deliveries'):
    return float(np.std([v[k] for v in cur.values()], ddof=1))


def ci(x):
    x = np.asarray(x, float)
    return float(tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def paired(a, b):
    """a - b over the shared seeds: mean, 95% CI half-width, Wilcoxon p."""
    s = sorted(set(a) & set(b)); x = np.array([a[q] - b[q] for q in s], float)
    return float(x.mean()), ci(x), (float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0)


# ── scale study: Table 4, Figure 3 ───────────────────────────────────────────
def scale_stats(R):
    alpha = 0.05 / (len(S) * len(BASE))
    out = dict(alpha=alpha, rows={}, tests={})
    for sc in S:
        E = NSZ[sc] * (NSZ[sc] - 1) / 2; fg = mean(by_seed(R, method='full', scale=sc))
        dm = by_seed(R, method='dmorph', scale=sc)
        for m in C.METHODS:
            cur = by_seed(R, method=m, scale=sc)
            row = dict(deliveries=mean(cur), sd=sd(cur), pct_full=100 * mean(cur) / fg, links=mean(cur, 'links'),
                       links_pct=100 * mean(cur, 'links') / E)
            if 'non_nearest' in next(iter(cur.values())):
                row['non_nearest_pct'] = 100 * mean(cur, 'non_nearest')
            if m.startswith('random'):
                row['failed'] = {k: float(np.mean([v['failed'][k] for v in cur.values()])) for k in ('no_link', 'same_role', 'busy')}
                row['stalled'] = int(sum(v['deliveries_ts'][-1] == v['deliveries_ts'][-9] for v in cur.values()))
            out['rows'][f'{m}_{sc}'] = row
            if m != 'dmorph':
                d, h, p = paired({s: v['deliveries'] for s, v in dm.items()}, {s: v['deliveries'] for s, v in cur.items()})
                out['tests'][f'{m}_{sc}'] = dict(diff=d, ci=h, p=p, significant=p < alpha)
    return out


def table4(st):
    rows = [r'\begin{table}[!ht]', r'\centering',
            r"\caption{Scale study ($T=800$, 20 seeds, mean $\pm$ SD). \%FG: deliveries as \% of Full Graph. Links: mean active "
            r"links (\% of maximum possible). Full Graph, MORPH, Proximity-$r$ ($r=5$) and TSG use MORPH's original allocation "
            r"loop (the nearest free Picker is proposed and the response is gated by the link or preference); D-MORPH and both "
            r"Random-Sparse variants use preference-based partner selection. Bold: most deliveries and fewest active links at each scale.}",
            r'\label{tab:main}', r'\footnotesize', r'\begin{tabular}{llrrr}', r'\hline',
            r'Scale ($N$) & Method & Deliveries & \%FG & Links (\%max) \\', r'\hline']
    B = lambda txt, on: (r'\textbf{' + txt + '}') if on else txt
    for sc in S:
        V = {m: st['rows'][f'{m}_{sc}'] for m in C.METHODS}
        bJ = max(round(v['deliveries'], 1) for v in V.values()); bL = min(round(v['links'], 1) for v in V.values())
        for i, m in enumerate(C.METHODS):
            v = V[m]; isJ = round(v['deliveries'], 1) == bJ
            lab = r'\textbf{D-MORPH}' if m == 'dmorph' else C.LABELS[m].replace('Proximity-r', 'Proximity-$r$')
            rows.append((f'{sc.capitalize()} ({NSZ[sc]})' if i == 0 else '') + f' & {lab} & '
                        + B(f"{v['deliveries']:.1f} $\\pm$ {v['sd']:.1f}", isJ) + ' & ' + B(f"{v['pct_full']:.0f}\\%", isJ) + ' & '
                        + B(f"{v['links']:.1f} ({v['links_pct']:.0f}\\%)", round(v['links'], 1) == bL) + r' \\')
        rows.append(r'\hline')
    rows += [r'\end{tabular}', r'\end{table}', '']
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(TAB, 'table4_scale.tex'), 'w').write('\n'.join(rows))


def fig3_scale(R):
    fig = plt.figure(figsize=(W2, 215 * MM))
    gA = fig.add_gridspec(1, 4, left=0.07, right=0.975, top=0.86, bottom=0.69, wspace=0.25)
    gB = fig.add_gridspec(1, 2, left=0.1, right=0.99, top=0.60, bottom=0.42, wspace=0.22)
    gC = fig.add_gridspec(1, 4, left=0.07, right=0.99, top=0.32, bottom=0.07, wspace=0.25)
    t = np.concatenate(([0], np.arange(24, C.T_SCALE, 25) + 1))
    for k, sc in enumerate(S):
        a = fig.add_subplot(gA[0, k]); style(a, 'y')
        for m in C.METHODS:
            X = np.array([v['deliveries_ts'] for v in by_seed(R, method=m, scale=sc).values()], float)
            mu, sd_ = np.concatenate(([0], X.mean(0))), np.concatenate(([0], X.std(0)))
            col, mk, ls, ms = ST[m]
            a.fill_between(t, mu - sd_, mu + sd_, color=col, alpha=0.15 if m == 'dmorph' else 0.05, lw=0, zorder=1)
            a.plot(t, mu, color=col, ls=ls, lw=LW(m), zorder=ZO(m))
        dm, fg = mean(by_seed(R, method='dmorph', scale=sc)), mean(by_seed(R, method='full', scale=sc))
        a.set_title(f'{sc.capitalize()} N={NSZ[sc]}\nD-MORPH {dm:.0f}, Full {fg:.0f}\n({100 * round(dm / fg, 3):.0f}% of Full Graph)',
                    color=INK, fontsize=7, loc='left', fontweight='bold')
        a.set_xlabel('Step'); a.set_xlim(0, C.T_SCALE); a.set_xticks([0, 200, 400, 600, 800])
        if k == 0:
            a.set_ylabel('Cumulative deliveries'); panel_letter(fig, a, 'A')
    xs = [NSZ[s] for s in S]
    rel = {m: [100 * mean(by_seed(R, method=m, scale=s)) / mean(by_seed(R, method='full', scale=s)) for s in S] for m in C.METHODS}
    # (B) throughput with a broken y-axis: methods near Full Graph above, the Random-Sparse variants below
    gb = gB[0, 0].subgridspec(2, 1, height_ratios=[1.6, 1], hspace=0.08)
    b, b2 = fig.add_subplot(gb[0]), fig.add_subplot(gb[1])
    hi = [v for m in C.METHODS if not m.startswith('random') for v in rel[m]]
    lo = [v for m in C.METHODS if m.startswith('random') for v in rel[m]]
    for ax in (b, b2):
        style(ax, 'y')
        for m in C.METHODS:
            col, mk, ls, ms = ST[m]
            ax.plot(xs, rel[m], color=col, marker=mk, ls=ls, ms=ms, lw=LW(m), mec=DARK, mew=0.4, zorder=ZO(m))
        ax.set_xticks(xs)
    b.set_ylim(np.floor(min(hi)) - 1.5, np.ceil(max(hi)) + 1.5)
    b2.set_ylim(max(0, np.floor(min(lo) / 10) * 10 - 5), np.ceil(max(lo) / 10) * 10 + 5)
    b.spines['bottom'].set_visible(False); b2.spines['top'].set_visible(False)
    b.tick_params(axis='x', bottom=False, labelbottom=False)
    d_ = 0.012
    for ax, y0 in ((b, 0), (b2, 1)):
        kw = dict(transform=ax.transAxes, color=DARK, clip_on=False, lw=0.8)
        ax.plot((-d_, d_), (y0 - d_ * 2, y0 + d_ * 2), **kw); ax.plot((1 - d_, 1 + d_), (y0 - d_ * 2, y0 + d_ * 2), **kw)
    b2.set_xlabel('Agents (N)'); b.set_ylabel('% Full-Graph deliveries'); b.yaxis.set_label_coords(-0.11, 0.2)
    b.set_title('Throughput', color=INK, fontsize=8, fontweight='bold'); panel_letter(fig, b, 'B')
    c = fig.add_subplot(gB[0, 1]); style(c, 'y')
    for m in C.METHODS:
        col, mk, ls, ms = ST[m]
        c.plot(xs, [100 * mean(by_seed(R, method=m, scale=s), 'links') / (NSZ[s] * (NSZ[s] - 1) / 2) for s in S],
               color=col, marker=mk, ls=ls, ms=ms, lw=LW(m), mec=DARK, mew=0.4, zorder=ZO(m))
    c.set_xticks(xs); c.set_xlabel('Agents (N)'); c.set_ylabel('Links (% full graph)'); c.set_ylim(0, 105)
    c.set_title('Active links', color=INK, fontsize=8, fontweight='bold'); panel_letter(fig, c, 'C')
    for k, sc in enumerate(S):
        a = fig.add_subplot(gC[0, k]); style(a, 'y')
        fg = mean(by_seed(R, method='full', scale=sc)); top = 0
        for i_, m in enumerate(C.METHODS):
            cur = by_seed(R, method=m, scale=sc); y = mean(cur); sdv = np.std([v['deliveries'] for v in cur.values()])
            a.bar(i_, y, width=0.72, color=ST[m][0], edgecolor=DARK if m == 'dmorph' else 'none', lw=1.2, zorder=3,
                  yerr=sdv, error_kw=dict(lw=0.8, capsize=2))
            a.text(i_, y + sdv + 1.5, f'{y:.0f} ({100 * round(y / fg, 3):.0f}%)', ha='center', va='bottom', fontsize=6, rotation=90,
                   color=ST[m][0] if not m.startswith('random') else MUTED, fontweight='bold' if m == 'dmorph' else 'normal')
            top = max(top, y + sdv)
        a.set_xticks(range(len(C.METHODS))); a.set_xticklabels([SHORT[m] for m in C.METHODS], rotation=35, ha='right', fontsize=6.5)
        a.set_ylim(0, top * 1.42); a.set_title(f'{sc.capitalize()} N={NSZ[sc]}', color=INK, fontsize=7.5, fontweight='bold')
        if k == 0:
            a.set_ylabel('Deliveries'); panel_letter(fig, a, 'D')
    h = [Line2D([], [], color=ST[m][0], ls=ST[m][2], lw=LW(m), marker=ST[m][1], ms=ST[m][3] - 2, mec=DARK, label=C.LABELS[m])
         for m in C.METHODS]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=7, columnspacing=1.0, handlelength=2.4)
    save(fig, 'fig3_scale')


# ── Figure 4: Proximity-r sweep ──────────────────────────────────────────────
def fig4_frontier(R, F):
    from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
    fig, axes = plt.subplots(1, 4, figsize=(W2, 78 * MM)); fig.subplots_adjust(left=0.07, right=0.99, top=0.7, bottom=0.17, wspace=0.32)
    best = {}
    for k, sc in enumerate(S):
        a = axes[k]; style(a, 'both'); E = NSZ[sc] * (NSZ[sc] - 1) / 2; fg = mean(by_seed(R, method='full', scale=sc))
        pts = []
        for q in C.PROX_GRID:
            cur = by_seed(F, scale=sc, r=q)
            pts.append((mean(cur, 'links'), mean(cur) / fg, q, mean(cur)))
        best[sc] = max(pts, key=lambda p: p[3])
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, marker='o', ms=4, lw=2.0, mec='white', mew=0.5, zorder=1, alpha=0.85)
        a.axhline(1.0, color=DARK, ls='--', lw=0.8); a.plot(E, 1.0, 'D', color=DARK, ms=5)
        for m in ('random_fixed', 'random_step', 'tsg', 'morph', 'dmorph'):
            col, mk, ls, ms = ST[m]; cur = by_seed(R, method=m, scale=sc)
            a.plot(mean(cur, 'links'), mean(cur) / fg, mk, color=col, ms=ms + (6 if m == 'dmorph' else 1), mec=DARK,
                   mew=0.8 if m == 'dmorph' else 0.5, zorder=ZO(m))
            if m == 'dmorph':
                a.annotate('D-MORPH', (mean(cur, 'links'), mean(cur) / fg), xytext=(6, 6), textcoords='offset points',
                           fontsize=7, color=RED, fontweight='bold')
        a.set_xscale('log'); a.set_xlim(1.5, E * 1.6)
        tk = [t for t in (2, 5, 10, 20, 50, 200) if t < E * 1.6]
        a.xaxis.set_major_locator(FixedLocator(tk)); a.set_xticklabels([str(t) for t in tk])
        a.xaxis.set_minor_locator(NullLocator()); a.xaxis.set_minor_formatter(NullFormatter())
        a.set_title(f'{sc.capitalize()} (N={NSZ[sc]})', color=INK); a.set_xlabel('active links')
        if k == 0:
            a.set_ylabel('deliveries / Full Graph')
        panel_letter(fig, a, 'ABCD'[k])
    h = [Line2D([], [], color=ST[m][0], marker=ST[m][1], ls='', ms=ST[m][3] + 1, mec=DARK, label=C.LABELS[m])
         for m in ('dmorph', 'morph', 'random_fixed', 'random_step', 'tsg')]
    h += [Line2D([], [], color=BLUE, marker='o', ms=4, lw=2.0, label='Proximity-r, r = 1 ... 12'),
          Line2D([], [], color=DARK, marker='D', ls='--', ms=5, lw=0.8, label='Full Graph')]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=7.5)
    save(fig, 'fig4_frontier')
    return {sc: dict(r=b[2], deliveries=b[3], links=b[0]) for sc, b in best.items()}


# ── Figure 5: ablations ──────────────────────────────────────────────────────
def fig5_ablation(R, Ab):
    rows = [('dmorph', 'D-MORPH (all rules)')] + list(C.ABLATION_LABELS.items())
    alpha = 0.05 / (len(S) * len(C.ABLATIONS)); tests = {}
    fig, axes = plt.subplots(2, 4, figsize=(W2, 105 * MM), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.11, hspace=0.45, wspace=0.1)
    for k, sc in enumerate(S):
        dm = by_seed(R, method='dmorph', scale=sc)
        for a_ in C.ABLATIONS:
            cur = by_seed(Ab, scale=sc, ablation=a_)
            d, h, p = paired({s: v['deliveries'] for s, v in cur.items()}, {s: v['deliveries'] for s, v in dm.items()})
            tests[f'{a_}_{sc}'] = dict(deliveries=mean(cur), links=mean(cur, 'links'), diff=d, ci=h, p=p,
                                       rel=mean(cur) / mean(dm), links_rel=mean(cur, 'links') / mean(dm, 'links'), significant=p < alpha)
        for r_, key in enumerate(('deliveries', 'links')):
            a = axes[r_, k]; style(a, 'x')
            for i, (m, _) in enumerate(rows):
                cur = dm if m == 'dmorph' else by_seed(Ab, scale=sc, ablation=m)
                x = [v[key] for v in cur.values()]
                a.errorbar(np.mean(x), i, xerr=ci(x), fmt='o', color=RED if m == 'dmorph' else MUTED, ms=4.5, mec=DARK, mew=0.4, capsize=1.5)
            if key == 'deliveries':
                a.axvline(mean(by_seed(R, method='full', scale=sc)), color=DARK, ls='--', lw=1)
            a.set_yticks(range(len(rows))); a.set_yticklabels([l for _, l in rows])
            if k == 0:
                a.invert_yaxis(); panel_letter(fig, a, 'AB'[r_])
            if r_ == 0:
                a.set_title(f'{sc.capitalize()} (N={NSZ[sc]})', color=INK)
            a.set_xlabel('deliveries per episode' if key == 'deliveries' else 'active links')
    save(fig, 'fig5_ablation')
    return dict(alpha=alpha, tests=tests)


# ── package location shift: Table 5, Figure 6 ───────────────────────────────
PH, W_REC = C.SHIFT_PHASE, 50


def _fwd(sd, t0, t1):
    seg = np.asarray(sd[t0:t1], float); cs = np.concatenate(([0.0], np.cumsum(seg)))
    return np.array([(cs[d] - cs[max(0, d - W_REC)]) / (d - max(0, d - W_REC)) for d in range(1, len(seg) + 1)])


def rec_time(sd, k):
    """MORPH's recovery time after the shift into phase k (k >= 3): the first step at which the delivery rate over the
    following steps (at most 50) reaches 90% of the rate of phase k - 2 over its last 50 steps; 400 if never."""
    base = _fwd(sd, PH * (k - 3), PH * (k - 2))[-1]; post = _fwd(sd, PH * (k - 1), PH * k)
    h = np.where(post >= 0.9 * base)[0]
    return float(h[0] + 1) if len(h) and base > 1e-9 else float(PH)


def endpoints(sd):
    P = [sum(sd[k * PH:(k + 1) * PH]) for k in range(6)]
    tr = np.mean([sum(sd[(k - 1) * PH:(k - 1) * PH + 100]) / max(sum(sd[(k - 2) * PH - 100:(k - 2) * PH]), 1) for k in range(3, 7)])
    return dict(P=P, dbar=((P[4] - P[2]) + (P[5] - P[3])) / 2, tr=float(tr), r=[rec_time(sd, k) for k in range(3, 7)],
                rbar=float(np.mean([rec_time(sd, k) for k in range(3, 7)])), stalled=int(sum(sd[-200:]) == 0))


def shift_stats(Sh):
    alpha = 0.05 / (len(C.SHIFT_SCALES) * len(BASE) * 2)
    E = {(r['method'], r['scale'], r['seed']): endpoints(r['step_deliveries']) for r in Sh}
    out = dict(alpha=alpha, methods={}, tests={},
               zone_share=float(np.mean([np.mean(list(r['zone_share'].values())) for r in Sh if r['zone_share']])))
    for sc in C.SHIFT_SCALES:
        for m in C.METHODS:
            v = [E[(m, sc, s)] for s in C.SEEDS]
            out['methods'][f'{m}_{sc}'] = dict(P=list(np.mean([x['P'] for x in v], axis=0)), r=list(np.mean([x['r'] for x in v], axis=0)),
                                                **{k: float(np.mean([x[k] for x in v])) for k in ('dbar', 'tr', 'rbar')},
                                                stalled=int(sum(x['stalled'] for x in v)))
        for m in BASE:
            for k in ('dbar', 'tr', 'rbar'):
                d, h, p = paired({s: E[('dmorph', sc, s)][k] for s in C.SEEDS}, {s: E[(m, sc, s)][k] for s in C.SEEDS})
                out['tests'][f'{k}_{m}_{sc}'] = dict(diff=d, lo=d - h, hi=d + h, p=p,
                                                     significant=bool(k != 'rbar' and p < alpha and (d - h > 0 or d + h < 0)))
    return out


def table5(st):
    rows = [r'\begin{table}[!ht]', r'\centering',
            r"\caption{Package location shift (20 seeds, mean). New requests only from the upper (U) or lower (L) half of the "
            r"shelves, alternating every 400 steps; the delivery stations are on the lower edge. P1--P6: deliveries in each phase. "
            r"$r_3$--$r_6$: recovery time after the shift into phase $k$, the steps until the delivery rate over the following "
            r"steps (at most 50) reaches 90\% of the rate of the previous phase with demand in the same half ($k-2$) over its last "
            r"50 steps (MORPH's definition; 400 if never); $\bar r$: mean of $r_3$--$r_6$. Bold: best value in each column and "
            r"scale (most deliveries, shortest recovery).}",
            r'\label{tab:resilience}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llrrrrrrrrrrr}', r'\hline',
            r' & & \multicolumn{6}{c}{Deliveries per phase} & \multicolumn{5}{c}{Recovery time (steps)} \\',
            r'Scale & Condition & P1 U & P2 L & P3 U & P4 L & P5 U & P6 L & $r_3$ U & $r_4$ L & $r_5$ U & $r_6$ L & $\bar r$ \\', r'\hline']
    order = ['dmorph'] + BASE
    for sc in C.SHIFT_SCALES:
        vals = {m: st['methods'][f'{m}_{sc}']['P'] + st['methods'][f'{m}_{sc}']['r'] + [st['methods'][f'{m}_{sc}']['rbar']] for m in order}
        best = [max(vals[m][c] for m in order) if c < 6 else min(vals[m][c] for m in order) for c in range(11)]
        for i_, m in enumerate(order):
            cells = [(r'\textbf{' + f'{x:.1f}' + '}') if f'{x:.1f}' == f'{best[c]:.1f}' else f'{x:.1f}' for c, x in enumerate(vals[m])]
            lab = r'\textbf{D-MORPH}' if m == 'dmorph' else C.LABELS[m].replace('Proximity-r', 'Proximity-$r$')
            rows.append((f'{sc.capitalize()} ({NSZ[sc]})' if i_ == 0 else '') + f' & {lab} & ' + ' & '.join(cells) + r' \\')
        rows.append(r'\hline')
    rows += [r'\end{tabular}', r'\end{table}', '']
    open(os.path.join(TAB, 'table5_shift.tex'), 'w').write('\n'.join(rows))


def fig6_shift(Sh, st):
    order = ['dmorph'] + BASE
    fig = plt.figure(figsize=(W2, 205 * MM))
    gA = fig.add_gridspec(1, 2, left=0.1, right=0.98, top=0.87, bottom=0.68, wspace=0.18)
    gB = fig.add_gridspec(1, 2, left=0.1, right=0.98, top=0.58, bottom=0.40, wspace=0.18)
    gC = fig.add_gridspec(1, 2, left=0.25, right=0.98, top=0.29, bottom=0.06, wspace=0.4)
    H_, T_ = 100, C.T_SHIFT
    for c_, sc in enumerate(C.SHIFT_SCALES):
        a = fig.add_subplot(gA[0, c_]); style(a, 'y')
        for m in order:
            col, mk, ls, ms = ST[m]
            D_ = np.array([by_seed(Sh, method=m, scale=sc)[s]['step_deliveries'] for s in C.SEEDS], float)
            cs = np.concatenate((np.zeros((len(D_), 1)), np.cumsum(D_, axis=1)), axis=1)
            lo = np.maximum(np.arange(1, T_ + 1) - H_, 0); win = np.arange(1, T_ + 1) - lo
            Rr = 100 * (cs[:, 1:] - cs[:, lo]) / win                     # deliveries per 100 steps, window from step 0
            mu = Rr.mean(0); tt = np.arange(T_)
            if m == 'dmorph':
                hw = tdist.ppf(0.975, len(Rr) - 1) * Rr.std(0, ddof=1) / np.sqrt(len(Rr))
                a.fill_between(tt, mu - hw, mu + hw, color=col, alpha=0.2, lw=0, zorder=1)
            a.plot(tt, mu, color=col, ls=ls, lw=2.4 if m == 'dmorph' else 0.9, alpha=1.0 if m == 'dmorph' else 0.8, zorder=ZO(m))
        for k in range(6):
            if k % 2:
                a.axvspan(k * PH, (k + 1) * PH, color=GREY, alpha=0.14, lw=0, zorder=0)
            a.text(PH * k + PH / 2, 1.01, f'P{k + 1} ' + ('U' if k % 2 == 0 else 'L'), transform=a.get_xaxis_transform(), ha='center', va='bottom', fontsize=7)
        a.set_xlim(0, T_); a.set_xlabel('simulation step')
        a.set_title(f'{sc.capitalize()} (N={NSZ[sc]})', color=INK, fontsize=8, pad=13)
        if c_ == 0:
            a.set_ylabel('delivery rate\n(deliveries per 100 steps)'); panel_letter(fig, a, 'A')
    for c_, sc in enumerate(C.SHIFT_SCALES):
        b = fig.add_subplot(gB[0, c_]); style(b, 'y')
        for m in order:
            col, mk, ls, ms = ST[m]
            P = np.array([endpoints(by_seed(Sh, method=m, scale=sc)[s]['step_deliveries'])['P'] for s in C.SEEDS], float); mu = P.mean(0)
            if m == 'dmorph':
                hw = tdist.ppf(0.975, len(P) - 1) * P.std(0, ddof=1) / np.sqrt(len(P))
                b.fill_between(range(1, 7), mu - hw, mu + hw, color=col, alpha=0.2, lw=0)
            b.plot(range(1, 7), mu, color=col, ls=ls, marker=mk, ms=ms - 3 if m == 'dmorph' else ms - 1, mec=DARK, mew=0.4,
                   lw=2.6 if m == 'dmorph' else 0.9, alpha=1.0 if m == 'dmorph' else 0.8, zorder=ZO(m))
        b.set_xticks(range(1, 7)); b.set_xticklabels([f'P{k}\n' + ('U' if k % 2 else 'L') for k in range(1, 7)], fontsize=7)
        if c_ == 0:
            b.set_ylabel('deliveries per phase'); panel_letter(fig, b, 'B')
    for c_, (key, ttl, unit) in enumerate((('dbar', 'Degradation $\\bar\\Delta$: D-MORPH minus baseline\n(positive: D-MORPH degrades less)', 'deliveries per phase'),
                                           ('tr', 'Transient retention: D-MORPH minus baseline\n(positive: D-MORPH retains more)', 'share of the earlier rate'))):
        d = fig.add_subplot(gC[0, c_]); style(d, 'x')
        for i_, m in enumerate(BASE):
            for o, (sc, mk) in enumerate(zip(C.SHIFT_SCALES, ('o', 's'))):
                y = i_ + (o - 0.5) * 0.3; r = st['tests'][f'{key}_{m}_{sc}']
                d.errorbar(r['diff'], y, xerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]], fmt=mk, color='#6E6E6E', mec='#6E6E6E',
                           mfc='white' if o else '#6E6E6E', ms=4.5, elinewidth=1, capsize=2, label=(sc.capitalize() if i_ == 0 else None))
        d.axvline(0, color=GREY, lw=0.8, ls='--'); d.set_yticks(range(len(BASE)))
        d.set_yticklabels([C.LABELS[m] for m in BASE] if c_ == 0 else [''] * len(BASE), fontsize=6.5); d.invert_yaxis()
        d.set_xlabel(unit, fontsize=7); d.set_title(ttl, fontsize=7, color=INK); panel_letter(fig, d, 'CD'[c_])
        if c_ == 0:
            d.legend(frameon=False, fontsize=6.5, loc='lower right')
    h = [Line2D([], [], color=ST[m][0], ls=ST[m][2], lw=2.4 if m == 'dmorph' else 0.9, label=C.LABELS[m]) for m in order]
    fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, fontsize=6.5, columnspacing=0.9)
    save(fig, 'fig6_shift')


# ── Figure 7: emergent allocation structure ─────────────────────────────────
def fig7_emergent(e):
    from matplotlib.colors import ListedColormap
    ts = sorted(int(t) for t in e['snapshots']); na, N, T_ = e['n_agv'], e['N'], e['T']
    snap = {t: {k: np.array(v) for k, v in e['snapshots'][str(t)].items()} for t in ts}
    vmax = max(float(snap[t]['W'].max()) for t in ts) or 1.0
    fig = plt.figure(figsize=(W2, 140 * MM))
    gA = fig.add_gridspec(2, len(ts), wspace=0.12, hspace=0.32, left=0.08, right=0.9, top=0.95, bottom=0.44)
    for row, (key, cmap, vm, lab) in enumerate((('A', ListedColormap(['white', RED]), 1.0, None), ('W', 'Blues', vmax, 'w_i[j]'))):
        for k, t in enumerate(ts):
            ax = fig.add_subplot(gA[row, k])
            im = ax.imshow(snap[t][key], cmap=cmap, vmin=0, vmax=vm, interpolation='nearest')
            ax.axhline(na - 0.5, color=INK, lw=0.8); ax.axvline(na - 0.5, color=INK, lw=0.8)
            ax.set_xticks([0, na, N - 1]); ax.set_yticks([0, na, N - 1] if k == 0 else []); ax.tick_params(labelsize=6.5)
            if row == 0:
                ax.set_title(f't = {t + 1}', color=INK, fontsize=8)
            if k == 0:
                ax.set_ylabel('robot i', fontsize=7); panel_letter(fig, ax, 'AB'[row])
            if row == 1:
                ax.set_xlabel('robot j', fontsize=7)
        if lab:
            cax = fig.add_axes([0.915, 0.44, 0.012, 0.2]); cb = fig.colorbar(im, cax=cax); cb.set_label(lab, fontsize=7); cb.ax.tick_params(labelsize=6)
    gb = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.32, bottom=0.09, wspace=0.45)
    sm = lambda x: np.convolve(x, np.ones(25) / 25, mode='valid'); tt = np.arange(T_)[12:T_ - 12]
    ax = fig.add_subplot(gb[0, 0]); style(ax, 'y')
    ax.plot(tt, sm(e['links_agv_picker']), color=RED, lw=2, label='AGV-Picker')
    ax.plot(tt, sm(e['links_agv_agv']), color=BLUE, lw=1.5, ls='--', label='AGV-AGV')
    ax.plot(tt, sm(e['links_picker_picker']), color=GREY, lw=1.5, ls=':', label='Picker-Picker')
    ax.set_xlabel('step'); ax.set_ylabel('active links (25-step mean)'); ax.set_ylim(0, 40)
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc='upper right', handlelength=2.2, borderaxespad=0.2); panel_letter(fig, ax, 'C')
    ax = fig.add_subplot(gb[0, 1]); style(ax, 'y')
    ax.plot(tt, sm(e['w_mean_links']), color=BLUE, lw=2)
    ax.set_xlabel('step'); ax.set_ylabel('mean w_i[j] on links\n(25-step mean)'); ax.set_ylim(0, None); panel_letter(fig, ax, 'D')
    ax = fig.add_subplot(gb[0, 2]); style(ax, 'y')
    L = np.array(e['lifetimes']); bins = np.arange(0, 380, 10)
    ax.axvspan(0, 20, color=GREY, alpha=0.25, lw=0, zorder=0)
    ax.text(45, 0.35, 'shaded: grace period\n(first 20 steps, no pruning)', transform=ax.get_xaxis_transform(), ha='left', va='center', fontsize=6, color=INK)
    n_, _, _ = ax.hist(L, bins=bins, color=RED, edgecolor='white', lw=0.4, zorder=3)
    share = np.mean((L >= 21) & (L <= 22))
    ax.annotate(f'{100 * share:.0f}% pruned right\nafter the grace period', (25, n_[2]), xytext=(70, n_[2] * 0.9), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=0.6, color=INK), color=INK, va='top')
    ax.set_xlim(0, 380); ax.set_xlabel('link lifetime (steps)'); ax.set_ylabel('pruned links'); panel_letter(fig, ax, 'E')
    save(fig, 'fig7_emergent')
    return dict(deliveries=e['deliveries'], agv_picker=float(np.mean(e['links_agv_picker'][200:])),
                agv_agv=float(np.mean(e['links_agv_agv'][200:])), picker_picker=float(np.mean(e['links_picker_picker'][200:])),
                w_mean=float(np.mean(e['w_mean_links'][200:])), pruned=len(L), lifetime_median=float(np.median(L)),
                share_after_grace=float(share), share_over_100=float(np.mean(L > 100)), open_at_end=len(e['lifetimes_open']),
                non_nearest=e['non_nearest'])


# ── Figure 2: the warehouse with D-MORPH's links ────────────────────────────
def fig2_environment():
    """TA-RWARE at t = 0 (before any allocation) and t = 300 with D-MORPH's links, seed 42, four fleet sizes."""
    import gymnasium as gym
    from tarware.definitions import AgentType
    from experiments.harness import run_episode, ENV_ID
    snap_t = 300
    fig, axes = plt.subplots(2, 4, figsize=(W2, 125 * MM))
    for c, (sc, env_id, N) in enumerate(C.SCALES):
        static, frames = {}, {}

        def rec(t, env, agents, coordinator, deliveries, **_):
            if not static:
                static.update(grid=env.grid_size, shelves=np.array([(s.x, s.y) for s in env.shelfs]), goals=np.array(env.goals),
                              is_agv=np.array([a.type == AgentType.AGV for a in agents]))
            if t == snap_t:
                L = coordinator.A > 0
                est = L & (coordinator.age > coordinator.grace); est = est | est.T
                frames[t] = dict(xy=np.array([(a.x, a.y) for a in agents], float), A=L | L.T, E=est,
                                 req=np.array([(q.x, q.y) for q in env.request_queue]), deliveries=int(deliveries))
        run_episode('dmorph', sc, 42, snap_t + 1, recorder=rec)
        u = gym.make(env_id).unwrapped; u.reset(seed=42)
        frames[0] = dict(xy=np.array([(a.x, a.y) for a in u.agents], float), A=np.zeros((N, N), bool), E=np.zeros((N, N), bool),
                         req=np.array([(q.x, q.y) for q in u.request_queue]), deliveries=0)
        for r, t in enumerate((0, snap_t)):
            fr = frames[t]; ax = axes[r, c]
            ax.set_xlim(-1, static['grid'][1]); ax.set_ylim(35, -1); ax.set_aspect('equal'); ax.axis('off')
            ax.scatter(*static['shelves'].T, marker='s', s=9, c=SHELF, edgecolors='none', zorder=1)
            ax.scatter(*static['goals'].T, marker='s', s=9, c=GOAL_C, edgecolors='none', zorder=1)
            ax.scatter(*fr['req'].T, marker='s', s=9, facecolors='none', edgecolors=PICK_C, linewidths=0.7, zorder=2)
            xy, is_agv = fr['xy'], static['is_agv']
            U = np.triu(fr['A'], 1); Es = np.triu(fr['E'], 1)
            seg_e = [[xy[i], xy[j]] for i, j in zip(*np.nonzero(U & Es))]
            seg_c = [[xy[i], xy[j]] for i, j in zip(*np.nonzero(U & ~Es))]
            if seg_e:
                ax.add_collection(LineCollection(seg_e, colors=GREEN, linewidths=1.8, zorder=3))
            if seg_c:
                ax.add_collection(LineCollection(seg_c, colors=RED, linewidths=1.2, linestyles=(0, (3, 2)), zorder=3))
            ax.scatter(xy[is_agv, 0], xy[is_agv, 1], marker='h', s=26, c=AGV_C, edgecolors='white', linewidths=0.5, zorder=4)
            ax.scatter(xy[~is_agv, 0], xy[~is_agv, 1], marker='D', s=17, c=PICK_C, edgecolors='white', linewidths=0.5, zorder=4)
            ax.set_title(f'{sc.capitalize()} (N = {N}), t = {t}\nlinks = {len(seg_e)} established + {len(seg_c)} candidate\n'
                         f'delivered = {fr["deliveries"]}', color=INK, fontsize=6.5)
    h = [Line2D([], [], marker='h', ls='', color=AGV_C, ms=6, label='AGV'), Line2D([], [], marker='D', ls='', color=PICK_C, ms=5, label='Picker'),
         Line2D([], [], marker='s', ls='', color=SHELF, ms=5, label='shelf'),
         Line2D([], [], marker='s', ls='', mfc='none', mec=PICK_C, ms=5, label='requested shelf'),
         Line2D([], [], marker='s', ls='', color=GOAL_C, ms=5, label='delivery station'),
         Line2D([], [], color=GREEN, lw=1.8, label='established link (older than grace period)'),
         Line2D([], [], color=RED, lw=1.2, ls=(0, (3, 2)), label='candidate link (in grace period)')]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.1, hspace=0.3, wspace=0.04)
    for r, letter in enumerate('AB'):
        bb = axes[r, 0].get_position()
        fig.text(0.005, bb.y1 + 0.03, f'({letter})', fontsize=9, fontweight='bold', color=INK, va='top')
    save(fig, 'fig2_environment')


# ── Markdown summary ─────────────────────────────────────────────────────────
def results_md(sc_st, best_prox, ab, sh, em):
    L = ['# Results', '', 'Generated by `python -m experiments.analysis` from `results/*.json` (20 seeds per condition).', '',
         '## Scale study (Table 4, Figure 3)', '', f'T = 800 steps. Paired Wilcoxon against D-MORPH, Bonferroni alpha = {sc_st["alpha"]:.4f} '
         f'({len(S) * len(BASE)} comparisons).', '',
         '| Scale | Method | Deliveries (mean ± SD) | % of Full Graph | Active links (% of pairs) | D-MORPH minus method [95% CI], p |',
         '|---|---|---|---|---|---|']
    for sc in S:
        for m in C.METHODS:
            v = sc_st['rows'][f'{m}_{sc}']
            tst = '' if m == 'dmorph' else (lambda t: f"{t['diff']:+.2f} [{t['diff'] - t['ci']:+.2f}, {t['diff'] + t['ci']:+.2f}], "
                                                     f"p = {t['p']:.4f}{' *' if t['significant'] else ''}")(sc_st['tests'][f'{m}_{sc}'])
            L.append(f"| {sc.capitalize()} ({NSZ[sc]}) | {C.LABELS[m]} | {v['deliveries']:.1f} ± {v['sd']:.1f} | {v['pct_full']:.0f}% | "
                     f"{v['links']:.1f} ({v['links_pct']:.0f}%) | {tst} |")
    L += ['', '\\* significant after correction.', '', 'Partner chosen differs from the nearest available robot (share of requests): '
          + ', '.join(f"{sc} {sc_st['rows'][f'dmorph_{sc}']['non_nearest_pct']:.2f}%" for sc in S) + ' (D-MORPH).', '',
          'Random-Sparse failed requests per episode (no link / same-role partner / busy partner) and stalled episodes '
          '(no delivery in the last 200 steps):', '', '| Scale | Variant | No link | Same role | Busy | Stalled |', '|---|---|---|---|---|---|']
    for sc in S:
        for m in ('random_fixed', 'random_step'):
            v = sc_st['rows'][f'{m}_{sc}']; f = v['failed']
            L.append(f"| {sc.capitalize()} | {C.LABELS[m]} | {f['no_link']:.0f} | {f['same_role']:.0f} | {f['busy']:.0f} | {v['stalled']}/20 |")
    L += ['', '## Proximity-r sweep (Figure 4)', '', 'Highest-throughput radius per scale: '
          + ', '.join(f"{sc} r = {b['r']} ({b['deliveries']:.1f} deliveries, {b['links']:.1f} links)" for sc, b in best_prox.items()) + '.', '',
          '## Ablations (Figure 5)', '', f'Paired Wilcoxon against full D-MORPH, Bonferroni alpha = {ab["alpha"]:.4f}.', '',
          '| Scale | Variant | Deliveries | vs D-MORPH | Links | Links vs D-MORPH | p |', '|---|---|---|---|---|---|---|']
    for sc in S:
        for a_, lab in C.ABLATION_LABELS.items():
            t = ab['tests'][f'{a_}_{sc}']
            L.append(f"| {sc.capitalize()} | {lab} | {t['deliveries']:.1f} | {100 * (t['rel'] - 1):+.1f}% | {t['links']:.1f} | "
                     f"{100 * (t['links_rel'] - 1):+.0f}% | {t['p']:.4f}{' *' if t['significant'] else ''} |")
    L += ['', '## Package location shift (Table 5, Figure 6)', '', f"Demand alternates between the upper and lower half every 400 steps "
          f"(share of new requests in the active half: {sh['zone_share']:.3f}). Δ̄ = ((P5 − P3) + (P6 − P4)) / 2; retention: first 100 "
          f"steps after a shift relative to the last 100 of the previous same-half phase; r̄: mean recovery time (MORPH's definition). "
          f"Bonferroni alpha = {sh['alpha']:.4f} over Δ̄ and retention.", '',
          '| Scale | Method | P1–P6 | Δ̄ | Retention | r̄ (steps) | Stalled | D-MORPH minus method: Δ̄ [95% CI], p |', '|---|---|---|---|---|---|---|---|']
    for sc in C.SHIFT_SCALES:
        for m in ['dmorph'] + BASE:
            v = sh['methods'][f'{m}_{sc}']
            tst = '' if m == 'dmorph' else (lambda t: f"{t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}], p = {t['p']:.4f}"
                                                     f"{' *' if t['significant'] else ''}")(sh['tests'][f'dbar_{m}_{sc}'])
            L.append(f"| {sc.capitalize()} | {C.LABELS[m]} | {' '.join(f'{x:.1f}' for x in v['P'])} | {v['dbar']:+.2f} | {v['tr']:.3f} | "
                     f"{v['rbar']:.1f} | {v['stalled']}/20 | {tst} |")
    L += ['', '## Emergent allocation structure (Figure 7)', '', f"Medium, seed 42, T = 1500: {em['deliveries']} deliveries. After step 200, "
          f"{em['agv_picker']:.1f} AGV–Picker, {em['agv_agv']:.1f} AGV–AGV and {em['picker_picker']:.1f} Picker–Picker links on average; "
          f"mean preference on links {em['w_mean']:.3f}. {em['pruned']} links pruned (median lifetime {em['lifetime_median']:.0f} steps; "
          f"{100 * em['share_after_grace']:.0f}% pruned at 21–22 steps, {100 * em['share_over_100']:.0f}% lived longer than 100 steps); "
          f"{em['open_at_end']} active at the end. Partner differs from the nearest robot in {100 * em['non_nearest']:.1f}% of requests.", '']
    open(os.path.join(DOCS, 'results.md'), 'w').write('\n'.join(L))
    print('saved docs/results.md')


def main():
    R, F, Ab, Sh, em = (load(k) for k in ('scale', 'frontier', 'ablation', 'shift', 'emergent'))
    R, F, Ab, Sh = R['records'], F['records'], Ab['records'], Sh['records']
    sc_st = scale_stats(R); table4(sc_st); fig3_scale(R)
    best_prox = fig4_frontier(R, F)
    ab = fig5_ablation(R, Ab)
    sh = shift_stats(Sh); table5(sh); fig6_shift(Sh, sh)
    em_st = fig7_emergent(em)
    fig2_environment()
    results_md(sc_st, best_prox, ab, sh, em_st)


if __name__ == '__main__':
    main()
