"""
Shared TA-RWARE observation/signal utilities.

get_current_targets      — instantaneous per-agent assignment location.
compute_coassign_jaccard — instantaneous, symmetric co-assignment indicator
                            (memoryless; backs the TSG baseline).
TargetHistory             — per-agent trailing window of assignment locations.
compute_directed_overlap  — directed containment coefficient built from
                            TargetHistory windows; feeds MORPH's R1 update.
compute_pred_jac          — anticipatory AGV -> nearest-picker hint matrix.
compute_proximity_matrix  — Manhattan-distance adjacency (Proximity baseline).

compute_directed_overlap replaces a symmetric Jaccard overlap with a directed
containment coefficient, J_hat[i, j] = |T_i ∩ T_j| / |T_i|, so that
J_hat[i, j] != J_hat[j, i] whenever agents i and j have covered different
amounts of recent workload. Feeding this into MORPH's Hebbian rule (which is
otherwise symmetric in its arguments) is what makes the learned preference
matrix W genuinely directed rather than a relabelled undirected affinity.
"""
from collections import deque
import numpy as np
from tarware.heuristic import AgentType, MissionType

DEFAULT_HISTORY_WINDOW = 10


def get_current_targets(agents, assigned_agvs, assigned_pickers):
    """Each agent's current assignment location, or None if unassigned."""
    targets = {}
    for a in agents:
        if a in assigned_agvs:
            m = assigned_agvs[a]
            targets[a.id] = (m.location_x, m.location_y)
        elif a in assigned_pickers:
            m = assigned_pickers[a]
            targets[a.id] = (m.location_x, m.location_y)
        else:
            targets[a.id] = None
    return targets


def compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N):
    """Instantaneous, symmetric, memoryless co-assignment indicator.

    Jaccard overlap of two singleton current-target sets: 1 if agents i and j
    share a target this step, 0 otherwise. This is the TSG baseline signal:
    MORPH's co-assignment evidence stripped of memory and plasticity.
    """
    targets = get_current_targets(agents, assigned_agvs, assigned_pickers)
    agent_ids = [a.id for a in agents]
    jac = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ti, tj = targets[agent_ids[i]], targets[agent_ids[j]]
            if ti and tj and ti == tj:
                jac[i, j] = jac[j, i] = 1.0
    return jac


class TargetHistory:
    """Per-agent trailing window of assignment locations, keyed by agent id."""

    def __init__(self, agent_ids, window=DEFAULT_HISTORY_WINDOW):
        self.window = window
        self._hist = {aid: deque(maxlen=window) for aid in agent_ids}

    def update(self, current_targets):
        for aid, loc in current_targets.items():
            if loc is not None:
                self._hist[aid].append(loc)

    def sets(self, agent_ids_ordered):
        return [set(self._hist[aid]) for aid in agent_ids_ordered]


def compute_directed_overlap(target_sets, N):
    """Directed containment coefficient: J_hat[i,j] = |T_i ∩ T_j| / |T_i|.

    Asymmetric whenever |T_i| != |T_j|: measures how much of agent i's own
    recent workload agent j also covered, rather than a shared, symmetric
    overlap between the two.
    """
    J = np.zeros((N, N))
    for i in range(N):
        Ti = target_sets[i]
        if not Ti:
            continue
        ni = len(Ti)
        for j in range(N):
            if i == j:
                continue
            Tj = target_sets[j]
            if not Tj:
                continue
            inter = len(Ti & Tj)
            if inter:
                J[i, j] = inter / ni
    return J


def compute_pred_jac(agents, assigned_agvs, agent_idx, N):
    """Anticipatory hint: newly assigned AGV -> nearest available picker."""
    pred = np.zeros((N, N))
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    for agv, m in assigned_agvs.items():
        if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
            continue
        dists = [abs(p.x - m.location_x) + abs(p.y - m.location_y) for p in pickers]
        if not dists:
            continue
        nearest = pickers[int(np.argmin(dists))]
        i, j = agent_idx[agv.id], agent_idx[nearest.id]
        pred[i, j] = pred[j, i] = 1.0
    return pred


def _proximity_from_coords(coords, r):
    """
    Manhattan-radius adjacency from an (N, 2) coordinate array.

    Split out of compute_proximity_matrix so that the Proximity baseline and
    the decentralized communication model (dmorph.comm.CommModel) share one
    definition of "within radius r" — otherwise the two are not comparable.
    """
    coords = np.asarray(coords, dtype=float)
    N = len(coords)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]) <= r:
                A[i, j] = A[j, i] = 1.0
    return A


def compute_proximity_matrix(agents, r):
    return _proximity_from_coords([(a.x, a.y) for a in agents], r)
