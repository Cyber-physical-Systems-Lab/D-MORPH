"""
core.py — D-MORPH: each robot keeps its own directed partner preferences w_i[j] and links A_i[j] and
updates them from its own observations with MORPH's four plasticity rules (R1-R4), written as per-robot
dictionaries (row i of W and A belongs to robot i and is changed only by robot i).

What robot i observes at each step (nothing is sensed within a radius and nothing is sent between robots):
  its own current target and the robots co-assigned to it (the robots with the same target),
  its own reward from the environment,
  the partner it was proposed or that it proposed (note_proposal),
  the completion of a mission it served (note_pair + its own reward, or task settlement).

Rules, as applied by robot i to its own row:
  R1  synaptic      w_i[j] <- gamma_ij w_i[j] + alpha_i J_ij cov A_ij  (+ completion credit)
                    J_ij = share of i's recent targets (last `window` steps) shared with j
  R2  homeostatic   H_i <- clip(H_i + beta (target_deg - deg_i) / (N - 1), 0, 1)
  R3  structural    every k_slow steps: prune links with w_i[j] < theta_prune older than the grace period;
                    form the best candidate if (C_ij + pred_boost P_ij)(1 + 2 H_i) >= theta_form
                    (C_ij: share of the last `mi_window` steps co-assigned with j; P_ij = 1 if j was proposed)
  R4  metaplastic   BCM sliding threshold theta_ij (extra decay of over-potentiated mature links) and
                    neuromodulation eta_i from robot i's own outcomes (scales alpha_i up when eta_i > 0,
                    lowers theta_form for i when eta_i < 0)

Completion credit (reward modes):
  'pair'  symbiotic credit used in the paper: when an AGV delivers a mission, the AGV and the Picker that
          loaded it each add reward_alpha * J (if J > burst_thresh) to their preference for the other.
  'task'  end-of-task settlement (extension, not in the paper): an AGV credits every Picker that served
          its shelf cycle when the cycle closes; a Picker credits the AGV when its own load or unload ends.
          Partners stay partners until the task ends, and co-occurrence is by task identity, not by target
          cell. The credit is directed: each robot credits only what it observed itself.

Allocation is outside this class (dmorph.allocation): the primary robot requests the available partner
with the highest w_i[j] (nearest on a tie) and the partner accepts with probability 0.25 + 0.75 w_j[i].
"""
from collections import deque

import numpy as np

from dmorph.params import TUNED_PARAMS


class DMORPH:
    def __init__(self, N, n_agv, reward='pair', **params):
        p = dict(TUNED_PARAMS); p.update(params)
        unknown = set(p) - set(TUNED_PARAMS)
        if unknown:
            raise TypeError(f'unknown D-MORPH parameters: {sorted(unknown)}')
        if reward not in ('pair', 'task'):
            raise ValueError("reward must be 'pair' or 'task'")
        self.N, self.n_agv, self.reward, self.params = N, n_agv, reward, p
        self.alpha, self.beta, self.decay = p['alpha'], p['beta'], p['decay']
        self.tf_start, self.tf_end, self.tf_anneal = p['theta_form_start'], p['theta_form_end'], p['theta_form_anneal']
        self.tp, self.grace, self.k_slow = p['theta_prune'], p['grace_steps'], p['k_slow']
        self.target_deg = p['target_deg_frac'] * (N - 1)
        self.bcm_tau, self.bcm_gain, self.reward_alpha = p['bcm_tau'], p['bcm_gain'], p['reward_alpha']
        self.ng, self.ne_explore, self.fast, self.slow = p['neuromod_gain'], p['neuromod_explore'], p['neuromod_ema'], p['slow_ema']
        self.pred_boost, self.cov, self.burst_thresh = p['pred_boost'], p['cov'], p['burst_thresh']
        self.window, self.mi_window, self.max_new_local = p['window'], p['mi_window'], p['max_new_local']
        self.structure_until, self.bounded = p['structure_until'], p['bounded']

        self.t = 0
        self.W = np.zeros((N, N)); self.A = np.zeros((N, N))          # row i: robot i's preferences and links
        self.age = np.zeros((N, N)); self.theta = np.zeros((N, N))
        self.H = np.ones(N) * p['target_deg_frac']
        self.d_fast = np.zeros(N); self.d_slow = np.full(N, 1e-3)
        self.hist = [deque(maxlen=max(self.window, self.mi_window)) for _ in range(N)]   # (target, partners)
        self.proposals = []
        self.rewards = np.zeros(N)
        self.pair, self.pairJ, self._jt = {}, {}, np.zeros((N, N))      # 'pair' credit bookkeeping
        self.cur_tok, self.cum = [None] * N, [set() for _ in range(N)]  # 'task' settlement bookkeeping
        # histories (measurement only)
        self.ne_h, self.W_mean_h, self.eta_h, self.form_h, self.prune_h = [], [], [], [], []
        self.induced_h = []                                            # formations below theta_form (arousal-induced)
        self.life = []                                                 # lifetimes of pruned links (steps)

    # ── observations ─────────────────────────────────────────────────────────
    def set_rewards(self, r):
        """Each robot's own reward this step (TA-RWARE: 1 for a delivery, 0.1 for a load or unload)."""
        self.rewards = np.asarray(r, dtype=float)

    def note_proposal(self, i, j):
        """Robot i requested robot j (both observe the request)."""
        self.proposals.append((i, j))

    def note_pair(self, agv, picker):
        """An accepted partnership for a new mission ('pair' credit)."""
        self.pair[agv] = picker
        self.pairJ[(agv, picker)] = 0.0; self.pairJ[(picker, agv)] = 0.0

    def reset_plasticity(self):
        self.W[:] = 0; self.A[:] = 0; self.age[:] = 0; self.theta[:] = 0
        self.H[:] = self.target_deg / max(self.N - 1, 1)
        for h in self.hist:
            h.clear()

    @property
    def theta_form(self):
        p = min(self.t / self.tf_anneal, 1.0)
        return self.tf_start - (self.tf_start - self.tf_end) * p

    # ── one step of R1-R4 ───────────────────────────────────────────────────
    def step(self, targets, task_obs=None):
        """targets: each robot's current target (hashable, None if idle); robots with the same target are
        co-assigned. task_obs ('task' mode): (tokens, partners) per robot, its own task identifier and the
        partner(s) it directly observed this step."""
        N = self.N
        self.t += 1
        Rtask = np.zeros((N, N))
        if task_obs is not None:
            tokens, direct = task_obs
            for i in range(N):
                if tokens[i] != self.cur_tok[i]:
                    if self.cur_tok[i] is not None:                   # own task ended: settle with its partners
                        for j in self.cum[i]:
                            Rtask[i, j] = 1.0
                    self.cur_tok[i] = tokens[i]; self.cum[i] = set()
                if tokens[i] is not None:
                    self.cum[i] |= set(direct[i])
        by_target = {}
        for i, x in enumerate(targets):
            if x is not None:
                by_target.setdefault(x, []).append(i)
        J = np.zeros((N, N)); C = np.zeros((N, N))
        for i in range(N):
            x = targets[i]
            if task_obs is not None:
                x = task_obs[0][i]; partners = frozenset(self.cum[i])
            else:
                partners = frozenset(j for j in by_target.get(x, ()) if j != i) if x is not None else frozenset()
            self.hist[i].append((x, partners))
            recent = list(self.hist[i])[-self.window:]
            Ti = {y for y, _ in recent if y is not None}
            if Ti:
                shared = {}
                for y, ps in recent:
                    for j in ps:
                        shared.setdefault(j, set()).add(y)
                for j, s in shared.items():
                    J[i, j] = len(s) / len(Ti)
            long = list(self.hist[i])[-self.mi_window:]
            for _, ps in long:
                for j in ps:
                    C[i, j] += 1.0 / len(long)
        P = np.zeros((N, N))
        for a, p in self.proposals:
            P[a, p] = P[p, a] = 1.0
        self.proposals = []

        # completion credit
        for (a, b) in list(self.pairJ):
            self.pairJ[(a, b)] = max(self.pairJ[(a, b)], J[a, b])
        R = np.zeros((N, N))
        if self.reward == 'task':
            R = Rtask
        else:
            for a, p in list(self.pair.items()):
                if self.rewards[a] > 0.5:                              # the AGV delivered this mission
                    for (x, y) in ((a, p), (p, a)):
                        R[x, y] = 1.0
                        self._jt[x, y] = self.pairJ.pop((x, y), 0.0)
                    del self.pair[a]
        d = (R.sum(axis=1) > 0).astype(float)

        # R4 neuromodulation from robot i's own outcomes
        self.d_fast += self.fast * (d - self.d_fast)
        self.d_slow += self.slow * (d - self.d_slow)
        eta = np.clip((self.d_fast - self.d_slow) / (self.d_slow + 1e-9), -1.0, 1.0)
        alpha_i = self.alpha * (1.0 + self.ng * np.maximum(eta, 0.0))
        tf_i = np.maximum(self.theta_form - self.ne_explore * np.maximum(-eta, 0.0), self.tf_end * 0.8)

        # R4 BCM threshold (before R1) and BCM-modulated decay
        self.theta = self.bcm_tau * self.theta + (1 - self.bcm_tau) * self.W
        mature = (self.age > self.grace).astype(float)
        extra = self.bcm_gain * np.clip(self.W / (self.theta + 1e-9) - 1.0, 0.0, 3.0) * (1 - self.decay) * mature
        self.W *= np.clip(self.decay - extra, 0.5, 1.0)
        # R1 Hebbian term and completion credit on robot i's own links
        self.W += alpha_i[:, None] * J * self.cov * self.A
        if self.reward == 'task':
            self.W += self.reward_alpha * R * self.A
        else:
            credit = np.where(self._jt > self.burst_thresh, self._jt, 0.0)
            self.W += self.reward_alpha * R * credit * self.A
            self._jt[:] = 0.0
        np.clip(self.W, 0.0, 1.0 if self.bounded else None, out=self.W); np.fill_diagonal(self.W, 0.0)
        # R2 homeostasis
        self.H += self.beta * (self.target_deg - self.A.sum(axis=1)) / (N - 1)
        np.clip(self.H, 0.0, 1.0, out=self.H)

        # R3 structural plasticity
        formed = pruned = induced = 0
        if self.t % self.k_slow == 0 and (self.structure_until is None or self.t <= self.structure_until):
            pm = (self.A > 0) & (self.W < self.tp) & (self.age > self.grace)
            pruned = int(pm.sum())
            self.life.extend(self.age[pm].tolist())
            self.A[pm] = 0; self.W[pm] = 0; self.theta[pm] = 0; self.age[pm] = 0
            score = (C + self.pred_boost * P) * (1.0 + 2.0 * self.H[:, None])
            score[self.A > 0] = -1.0
            np.fill_diagonal(score, -1.0)
            for i in range(N):
                for _ in range(self.max_new_local):
                    j = int(np.argmax(score[i]))
                    if score[i, j] < tf_i[i] or score[i, j] <= 0:
                        break
                    induced += int(score[i, j] < self.theta_form)        # formed only because arousal lowered the threshold
                    self.A[i, j] = 1.0; self.age[i, j] = 0.0; score[i, j] = -1.0
                    formed += 1
        self.age = np.where(self.A > 0, self.age + 1, 0.0)
        und = (self.A > 0) | (self.A.T > 0)
        self.ne_h.append(int(np.triu(und, 1).sum()))
        self.W_mean_h.append(float(self.W[self.A > 0].mean()) if self.A.any() else 0.0)
        self.eta_h.append(float(eta.mean())); self.form_h.append(formed); self.prune_h.append(pruned)
        self.induced_h.append(induced)
