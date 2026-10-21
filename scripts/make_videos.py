"""
make_videos.py — Supplementary Videos 1-4: D-MORPH in the TA-RWARE warehouse at the four fleet sizes over one
episode (T = 800, seed 42, the paper's parameters), in the style of Figure 2.

Every frame comes from experiments/harness.run_episode through its read-only `recorder` hook. Each frame shows
shelves, requested shelves, delivery stations, AGVs and Pickers, and the links held by the robots: green solid,
established (older than the grace period); red dashed, candidate (still in the grace period). Below the warehouse:
cumulative deliveries and active links up to the current step.

    python scripts/make_videos.py  ->  docs/videos/supp_video_{1..4}_{scale}.mp4 and .gif   (needs ffmpeg)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import subprocess
import imageio.v2 as imageio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tarware.definitions import AgentType
from experiments import config as C
from experiments.harness import run_episode

OUT = os.path.join(ROOT, 'docs', 'videos')
INK, RED, GREEN, SHELF, AGV_C, PICK_C, GOAL_C = '#3A3936', '#971F30', '#1E8C45', '#E4DCCB', '#1F5696', '#E2781E', '#6B6B6B'
T = 800
FPS_MP4, GIF_STRIDE, FPS_GIF = 20, 4, 15          # MP4: every step at 20 fps (40 s); GIF: every 4th step at 15 fps


def record(scale, seed=42):
    static, frames = {}, []

    def rec(t, env, agents, coordinator, deliveries, **_):
        if not static:
            static.update(grid=env.grid_size, shelves=np.array([(s.x, s.y) for s in env.shelfs]),
                          goals=np.array(env.goals), is_agv=np.array([a.type == AgentType.AGV for a in agents]))
        L = np.asarray(coordinator.A) > 0
        est = L & (np.asarray(coordinator.age, float) > coordinator.grace); est = est | est.T
        frames.append(dict(t=t + 1, xy=np.array([(a.x, a.y) for a in agents], float), A=L | L.T, E=est,
                           req=np.array([(q.x, q.y) for q in env.request_queue]), deliveries=int(deliveries)))

    run_episode('dmorph', scale, seed, T, recorder=rec)
    return static, frames


def render(scale, N, static, frames, path_mp4, path_gif):
    H, Wd = static['grid']
    fig = plt.figure(figsize=(7.2, 7.2 * (H + 8) / (Wd + 2) * 0.62 + 1.6), dpi=110)
    ax = fig.add_axes([0.02, 0.27, 0.96, 0.62]); st = fig.add_axes([0.1, 0.07, 0.8, 0.14])
    ax.set_xlim(-1, Wd); ax.set_ylim(H, -1); ax.set_aspect('equal'); ax.axis('off')
    sh, g = static['shelves'], static['goals']; is_agv = static['is_agv']
    ax.scatter(sh[:, 0], sh[:, 1], marker='s', s=14, c=SHELF, edgecolors='none', zorder=1)
    ax.scatter(g[:, 0], g[:, 1], marker='s', s=14, c=GOAL_C, edgecolors='none', zorder=1)
    req = ax.scatter([], [], marker='s', s=14, facecolors='none', edgecolors=PICK_C, linewidths=0.9, zorder=2)
    est_lc = LineCollection([], colors=GREEN, linewidths=2.2, zorder=3); ax.add_collection(est_lc)
    cand_lc = LineCollection([], colors=RED, linewidths=1.4, linestyles=(0, (3, 2)), zorder=3); ax.add_collection(cand_lc)
    agv = ax.scatter([], [], marker='h', s=60, c=AGV_C, edgecolors='white', linewidths=0.6, zorder=4)
    pk = ax.scatter([], [], marker='D', s=40, c=PICK_C, edgecolors='white', linewidths=0.6, zorder=4)
    title = fig.text(0.5, 0.965, '', ha='center', va='top', color=INK, fontsize=10, fontweight='bold')
    sub = fig.text(0.5, 0.925, '', ha='center', va='top', color=INK, fontsize=8.5)
    h = [Line2D([], [], marker='h', ls='', color=AGV_C, ms=7, label='AGV'),
         Line2D([], [], marker='D', ls='', color=PICK_C, ms=6, label='Picker'),
         Line2D([], [], marker='s', ls='', mfc='none', mec=PICK_C, ms=6, label='requested shelf'),
         Line2D([], [], marker='s', ls='', color=GOAL_C, ms=6, label='delivery station'),
         Line2D([], [], color=GREEN, lw=2.2, label='established link'),
         Line2D([], [], color=RED, lw=1.4, ls=(0, (3, 2)), label='candidate link (grace period)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.215))
    tt = np.array([f['t'] for f in frames]); dd = np.array([f['deliveries'] for f in frames])
    ll = np.array([int(np.triu(f['A'], 1).sum()) for f in frames])
    st.set_xlim(0, T); st.set_ylim(0, max(dd.max(), 1) * 1.1); st.set_xlabel('step', fontsize=8); st.tick_params(labelsize=7)
    st.set_ylabel('deliveries', fontsize=8, color=INK)
    st2 = st.twinx(); st2.set_ylim(0, max(ll.max(), 1) * 1.2); st2.set_ylabel('active links', fontsize=8, color=GREEN); st2.tick_params(labelsize=7, colors=GREEN)
    for s_ in ('top',):
        st.spines[s_].set_visible(False); st2.spines[s_].set_visible(False)
    dl, = st.plot([], [], color=INK, lw=1.4); lk, = st2.plot([], [], color=GREEN, lw=1.0)
    fig.canvas.draw(); w0, h0 = fig.canvas.get_width_height()
    w0, h0 = w0 - w0 % 2, h0 - h0 % 2                       # libx264 with yuv420p needs even dimensions
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w0}x{h0}',
                           '-r', str(FPS_MP4), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', path_mp4],
                          stdin=subprocess.PIPE)
    gif = []
    for k, f in enumerate(frames):
        xy = f['xy']; U = np.triu(f['A'], 1); Es = np.triu(f['E'], 1)
        est_lc.set_segments([[xy[i], xy[j]] for i, j in zip(*np.nonzero(U & Es))])
        cand_lc.set_segments([[xy[i], xy[j]] for i, j in zip(*np.nonzero(U & ~Es))])
        req.set_offsets(f['req'] if len(f['req']) else np.empty((0, 2)))
        agv.set_offsets(xy[is_agv]); pk.set_offsets(xy[~is_agv])
        ne, nc = int((U & Es).sum()), int((U & ~Es).sum())
        title.set_text(f'D-MORPH in TA-RWARE: {scale.capitalize()} (N = {N}, {int(is_agv.sum())} AGVs + {int((~is_agv).sum())} Pickers)')
        sub.set_text(f'step {f["t"]} / {T}    delivered {f["deliveries"]}    links: {ne} established + {nc} candidate')
        dl.set_data(tt[:k + 1], dd[:k + 1]); lk.set_data(tt[:k + 1], ll[:k + 1])
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        ff.stdin.write(np.ascontiguousarray(img[:h0, :w0]).tobytes())
        if k % GIF_STRIDE == 0 or k == len(frames) - 1:
            gif.append(img[::2, ::2])
    ff.stdin.close(); ff.wait(); plt.close(fig)
    imageio.mimsave(path_gif, gif, duration=1000 / FPS_GIF, loop=0)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for n, (sc, _, N) in enumerate(C.SCALES, start=1):
        static, frames = record(sc)
        base = os.path.join(OUT, f'supp_video_{n}_{sc}')
        render(sc, N, static, frames, base + '.mp4', base + '.gif')
        print(f'saved {base}.mp4/.gif  ({len(frames)} frames, {frames[-1]["deliveries"]} deliveries)')
