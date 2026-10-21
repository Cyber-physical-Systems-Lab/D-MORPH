"""
harness.py — one TA-RWARE episode with D-MORPH or a baseline.

TA-RWARE (Krnjaic et al., 2024) supplies the warehouse, the request queue, A* path planning and the
resolution of movement conflicts. This file supplies task allocation and the AGV mission controller,
carried over from MORPH's experiment loop:

  MORPH's loop (Full Graph, MORPH, Proximity-r, TSG): each queued request goes to the nearest free AGV; for
  each AGV task the nearest free Picker is proposed and responds with probability 0.25 + 0.75 x, where x is
  the link (A) or preference (W) between them (always 1 for Full Graph).

  Pair allocation (D-MORPH, Random-Sparse): the oldest pending request goes to the nearest available robot i
  (AGV or Picker); i requests a partner j of the complementary role by its own preferences (dmorph.allocation)
  and j accepts with probability 0.25 + 0.75 w_j[i]; a refused request is retried at the next step. An AGV
  that needs a Picker again (returning the shelf) is the primary robot and requests one the same way.

The optional `recorder` callback is called after every step with read-only state (videos, Figure 2,
emergent-structure measurements); it cannot change the episode.
"""
import warnings
from collections import OrderedDict, deque

import numpy as np
import gymnasium as gym
import tarware                                                     # noqa: F401  (registers the environments)
from tarware.heuristic import AgentType, Mission, MissionType

from dmorph import DMORPH, accepts, select_partner
from experiments import config as C
from experiments.morph import MORPH
from experiments.signals import (get_current_targets, compute_coassign_jaccard, TargetHistory,
                                 compute_directed_overlap, compute_pred_jac, compute_proximity_matrix)

warnings.filterwarnings('ignore')
ENV_ID = {name: env for name, env, _ in C.SCALES}
LOOP_METHODS = ('full', 'morph', 'proximity', 'tsg')               # MORPH's allocation loop
PAIR_METHODS = ('dmorph', 'random_fixed', 'random_step')           # pair allocation


def _install_shift(u):
    """Package location shift: every new request comes from the upper (U) half of the shelves in phases 1, 3, 5
    and from the lower (L) half in phases 2, 4, 6 (SHIFT_PHASE steps each); the number of requests is unchanged.
    TA-RWARE replaces each delivered shelf by one drawn uniformly from the shelves neither requested nor
    carried; only that draw is replaced. The delivery stations are on the lower edge."""
    from tarware.warehouse import CollisionLayers, RewardType
    ys = sorted(s.y for s in u.shelfs)
    mid_y = ys[len(ys) // 2]
    top = {s.id for s in u.shelfs if s.y < mid_y}
    allid = {s.id for s in u.shelfs}
    q = 1.0                                                        # strict: every new request from the active half
    state = {'t': 0, 'zone_share': {}}

    def zone(t):
        return top if (t // C.SHIFT_PHASE) % 2 == 0 else allid - top

    def draw(cands, z):
        inz = [s for s in cands if s.id in z]
        out = [s for s in cands if s.id not in z]
        part = inz if (np.random.random() < q) else out
        if not part:
            part = out if part is inz else inz
        new = np.random.choice(part)
        return new, new.id in z

    def patched(rewards):
        z = zone(state['t'])
        shelf_deliveries = 0
        for y, x in u.goals:
            shelf_id = u.grid[CollisionLayers.CARRIED_SHELVES, x, y]
            if not shelf_id or u.shelfs[shelf_id - 1] not in u.request_queue:
                continue
            carried = [a.carrying_shelf for a in u.agents if a.carrying_shelf]
            cands = sorted(set(u.shelfs) - set(u.request_queue) - set(carried), key=lambda s: s.id)
            new_request, in_zone = draw(cands, z)
            u.request_queue[u.request_queue.index(u.shelfs[shelf_id - 1])] = new_request
            zs = state['zone_share'].setdefault(state['t'] // C.SHIFT_PHASE, [0, 0])
            zs[0] += int(in_zone); zs[1] += 1
            agent = u.agents[u.grid[CollisionLayers.AGVS, x, y] - 1]
            if not agent.has_delivered:
                agent.has_delivered = True
                if u.reward_type == RewardType.GLOBAL:
                    rewards += 1
                elif u.reward_type == RewardType.INDIVIDUAL:
                    rewards[agent.id - 1] += 1
            shelf_deliveries += 1
        if shelf_deliveries:
            u._cur_inactive_steps = 0
        else:
            u._cur_inactive_steps += 1
        return rewards, shelf_deliveries

    u.process_shelf_deliveries = patched
    queue = []                                                     # initial queue under the phase-1 rule
    for _ in range(u.request_queue_size):
        cands = sorted(set(u.shelfs) - set(queue), key=lambda s: s.id)
        queue.append(draw(cands, zone(0))[0])
    u.request_queue = queue
    return state


def _random_graph(N, k, rs_links, rs_w):
    """k random undirected links among all robot pairs (same-role pairs included), with independent directed
    preferences w_i[j] ~ U(0, 1) on them."""
    A = np.zeros((N, N))
    iu = np.array(np.triu_indices(N, 1)).T
    for e in rs_links.permutation(len(iu))[:k]:
        i, j = iu[e]; A[i, j] = A[j, i] = 1.0
    return A, A * rs_w.uniform(0.0, 1.0, size=(N, N))


def _task_obs(agents, assigned_agvs, assigned_pickers, agent_idx, agv_cycle):
    """D-MORPH 'task' settlement: each robot's own task identifier and directly observed partner.
    AGV: its shelf cycle (new mission to the shelf's return); partner: the Picker serving its pickup or
    return. Picker: one load or unload for one AGV; partner: that AGV."""
    tokens = [None] * len(agents); direct = [() for _ in agents]
    loc_agv = {}
    for agv, m in assigned_agvs.items():
        ia = agent_idx[agv.id]
        if m.mission_type == MissionType.PICKING:
            agv_cycle[ia] = m.assigned_time                        # a new pickup mission starts a new cycle
        cyc = agv_cycle.get(ia)
        if cyc is None:
            cyc = agv_cycle.setdefault(ia, ('late', id(m)))
        tokens[ia] = ('c', ia, cyc)
        if m.mission_type in (MissionType.PICKING, MissionType.RETURNING):
            loc_agv[(m.location_x, m.location_y)] = (ia, m.mission_type)
    for ia in [k for k in agv_cycle if tokens[k] is None]:
        del agv_cycle[ia]                                          # cycle closed (shelf returned)
    for p, mp in assigned_pickers.items():
        ip = agent_idx[p.id]
        hit = loc_agv.get((mp.location_x, mp.location_y))
        if hit is not None:
            ia, mt = hit
            tokens[ip] = ('p', tokens[ia], mt); direct[ip] = (ia,)
            direct[ia] = direct[ia] + (ip,)
        else:
            tokens[ip] = ('p', 'loc', mp.location_x, mp.location_y)
    return tokens, direct


def run_episode(method, scale, seed, T, params=None, reward='pair', r=C.PROX_R, shift=False, recorder=None):
    """method: one of config.METHODS; params: D-MORPH parameter overrides (ablations); reward: D-MORPH credit
    mode ('pair' as in the paper, 'task' settlement); r: Proximity radius; shift: package location shift.
    Returns a dict with per-step cumulative deliveries, active links and mean preference on links, the
    coordinator object (D-MORPH or MORPH), the agents and allocation counts."""
    env = gym.make(ENV_ID[scale])
    u = env.unwrapped
    env.reset(seed=seed)
    np.random.seed(seed)

    agents = u.agents
    agvs = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]
    N = len(agents)
    agent_idx = {a.id: i for i, a in enumerate(agents)}
    coords_map = {v: k for k, v in u.action_id_to_coords_map.items()}
    non_goal_ids = np.array([i for i, c in u.action_id_to_coords_map.items() if (c[1], c[0]) not in u.goals])

    coord = None; Wb = None
    A = np.zeros((N, N)); W = np.zeros((N, N))
    if method == 'full':
        A = np.ones((N, N)) - np.eye(N); W = np.ones((N, N)) - np.eye(N)
    elif method == 'morph':
        coord = MORPH(N, **C.morph_params(N, scale))
    elif method == 'dmorph':
        coord = DMORPH(N, C.N_AGV[scale], reward=reward, **(params or {}))
    elif method in ('random_fixed', 'random_step'):
        A, Wb = _random_graph(N, C.RANDOM_LINKS[scale], np.random.RandomState(12345 + seed), np.random.RandomState(777 + seed))
        redraw_rng = np.random.RandomState(4242 + seed)
    elif method not in ('proximity', 'tsg'):
        raise ValueError(f'unknown method {method!r}; expected one of {C.METHODS}')
    pair = method in PAIR_METHODS

    shift_state = _install_shift(u) if shift else None
    target_hist = TargetHistory([a.id for a in agents]) if coord is not None else None
    assigned_agvs, assigned_pickers, assigned_items = OrderedDict(), OrderedDict(), OrderedDict()
    deliveries = 0
    deliv_h, link_h, w_mean_h = [], [], []
    stats = dict(requests=0, refusals=0, assign=0, non_nearest=0, no_link=0, same_role=0, busy=0)
    agv_cycle = {}

    def respond(j, i):
        if method == 'full':
            return True
        return accepts(W[j, i] if method in ('dmorph', 'morph') else (Wb[j, i] if Wb is not None else A[j, i]))

    def choose(i_, prim, cand, dist):
        """Pair allocation: the partner robot i_ requests, or None if the request fails this step."""
        if method == 'dmorph':
            j = select_partner(W[i_], [agent_idx[c.id] for c in cand], lambda q: dist(agents[q]))
            return agents[j]
        # Random-Sparse: the linked robot with the highest w_i[j], of either role (nearest on a tie). The request
        # fails if robot i has no link, or the chosen robot has the same role (it cannot serve the task) or is
        # busy; there is no fallback to the nearest robot.
        others = [c for c in agents if c is not prim]
        part = min(others, key=lambda c: (-Wb[i_, agent_idx[c.id]], dist(c)))
        if Wb[i_, agent_idx[part.id]] <= 0:
            stats['no_link'] += 1; return None
        if part.type == prim.type:
            stats['same_role'] += 1; return None
        if part not in cand:
            stats['busy'] += 1; return None
        return part

    for t in range(T):
        if shift_state is not None:
            shift_state['t'] = t
        rq, gls = u.request_queue, u.goals
        actions = {a: 0 for a in agents}

        if method == 'random_step':
            # Random-Sparse (per-step): new random links (same budget) and preferences at every step
            A, Wb_new = _random_graph(N, C.RANDOM_LINKS[scale], redraw_rng, redraw_rng)
            Wb[:] = Wb_new

        if pair:
            # new missions: oldest pending request -> nearest available robot -> its preferred partner
            for item in rq:
                if item.id in assigned_items.values() or (item.y, item.x) not in coords_map:
                    continue
                av = [a for a in agvs if not a.busy and not a.carrying_shelf and a not in assigned_agvs]
                pk = [p for p in pickers if p not in assigned_pickers]
                if not av or not pk:
                    continue
                dist = lambda a: abs(a.x - item.x) + abs(a.y - item.y)
                prim = min(av + pk, key=dist)
                cand = pk if prim.type == AgentType.AGV else av
                i_ = agent_idx[prim.id]
                part = choose(i_, prim, cand, dist)
                if part is None:
                    continue
                j_ = agent_idx[part.id]
                if coord is not None:
                    coord.note_proposal(i_, j_)
                stats['requests'] += 1
                stats['non_nearest'] += int(part is not min(cand, key=dist))
                if not respond(j_, i_):
                    stats['refusals'] += 1
                    continue
                stats['assign'] += 1
                agv, pkr = (prim, part) if prim.type == AgentType.AGV else (part, prim)
                if coord is not None:
                    coord.note_pair(agent_idx[agv.id], agent_idx[pkr.id])
                loc = coords_map[(item.y, item.x)]
                assigned_agvs[agv] = Mission(MissionType.PICKING, loc, item.x, item.y, t)
                assigned_items[agv] = item.id
                assigned_pickers[pkr] = Mission(MissionType.PICKING, loc, item.x, item.y, t)
        else:
            # MORPH's loop: each queued request goes to the nearest free AGV (A* path length)
            for item in rq:
                if item.id in assigned_items.values():
                    continue
                avail = [a for a in agvs if not a.busy and not a.carrying_shelf and a not in assigned_agvs]
                if not avail:
                    continue
                dists = [len(u.find_path((a.y, a.x), (item.y, item.x), a, care_for_agents=False)) for a in avail]
                best = avail[np.argmin(dists)]
                if (item.y, item.x) not in coords_map:
                    continue                                       # shelf in transit, not at an addressable slot
                assigned_agvs[best] = Mission(MissionType.PICKING, coords_map[(item.y, item.x)], item.x, item.y, t)
                assigned_items[best] = item.id

        # AGV mission controller: pick up -> deliver -> return the shelf
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                assigned_agvs[agv].at_location = (agv.x == m.location_x and agv.y == m.location_y)
            if agv not in assigned_agvs or agv.busy:
                continue
            m = assigned_agvs[agv]
            if m.mission_type == MissionType.PICKING and m.at_location and agv.carrying_shelf:
                paths = [u.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (x, y) in gls]
                bg = gls[np.argmin([len(p) for p in paths])]
                assigned_agvs[agv] = Mission(MissionType.DELIVERING, coords_map[(bg[1], bg[0])], bg[0], bg[1], t)
            elif m.mission_type == MissionType.DELIVERING and m.at_location and agv.carrying_shelf:
                empty = u.get_empty_shelf_information()
                eids = [i for i, e in zip(non_goal_ids, empty) if e > 0]
                taken = {m2.location_id for a2, m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids = [e for e in eids if e not in taken] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (y, x) in elocs]
                    be = eids[np.argmin([len(p) for p in paths])]
                    byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be, byx[1], byx[0], t)
            elif m.mission_type == MissionType.RETURNING and m.at_location and not agv.carrying_shelf:
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)
            elif m.mission_type == MissionType.RETURNING and m.at_location and agv.carrying_shelf:
                empty = u.get_empty_shelf_information()
                eids = [i for i, e in zip(non_goal_ids, empty) if e > 0]
                taken = {m2.location_id for a2, m2 in assigned_agvs.items()
                         if a2 is not agv and m2.mission_type == MissionType.RETURNING}
                eids = [e for e in eids if e not in taken and u.action_id_to_coords_map[e] != (agv.y, agv.x)] or eids
                if eids:
                    elocs = [u.action_id_to_coords_map[i] for i in eids]
                    paths = [u.find_path((agv.y, agv.x), (y, x), agv, care_for_agents=False) for (y, x) in elocs]
                    be = eids[np.argmin([len(p) for p in paths])]
                    byx = u.action_id_to_coords_map[be]
                    assigned_agvs[agv] = Mission(MissionType.RETURNING, be, byx[1], byx[0], t)

        # links of the instantaneous baselines
        if method == 'tsg':
            A = compute_coassign_jaccard(agents, assigned_agvs, assigned_pickers, N)
        elif method == 'proximity':
            A = compute_proximity_matrix(agents, r)

        # Pickers for AGV tasks that need one (one Picker per location)
        picker_locs = {(mp.location_x, mp.location_y) for mp in assigned_pickers.values()}
        for agv, m in assigned_agvs.items():
            if m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            if (m.location_x, m.location_y) in picker_locs:
                continue
            pk = [p for p in pickers if p not in assigned_pickers]
            if not pk:
                continue
            i_ = agent_idx[agv.id]
            dloc = lambda p: abs(p.x - m.location_x) + abs(p.y - m.location_y)
            if pair:
                part = choose(i_, agv, pk, dloc)
                if part is None:
                    continue
            else:
                part = pk[int(np.argmin([dloc(p) for p in pk]))]  # MORPH's loop: nearest free Picker proposed
            j_ = agent_idx[part.id]
            if coord is not None and pair:
                coord.note_proposal(i_, j_)
            ok = respond(j_, i_) if pair else respond(i_, j_)      # MORPH's loop gates by W[agv, picker]
            if ok:
                assigned_pickers[part] = Mission(MissionType.PICKING, m.location_id, m.location_x, m.location_y, t)
                picker_locs.add((m.location_x, m.location_y))

        # a Picker is released once no AGV task at its location needs it
        for picker in list(assigned_pickers.keys()):
            mp = assigned_pickers[picker]
            if not any(m.location_x == mp.location_x and m.location_y == mp.location_y
                       and m.mission_type in (MissionType.PICKING, MissionType.RETURNING) for m in assigned_agvs.values()):
                assigned_pickers.pop(picker)

        for agv, m in assigned_agvs.items():
            actions[agv] = m.location_id if not agv.busy else 0
        for p, m in assigned_pickers.items():
            actions[p] = m.location_id
        rewards = env.step(list(actions[a] for a in agents))[1]
        deliveries += sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)

        # learning step of the coordinator
        if coord is not None:
            current_targets = get_current_targets(agents, assigned_agvs, assigned_pickers)
            target_hist.update(current_targets)
            if method == 'dmorph':
                coord.set_rewards(np.asarray(rewards, dtype=float))
                task_obs = _task_obs(agents, assigned_agvs, assigned_pickers, agent_idx, agv_cycle) if reward == 'task' else None
                coord.step([current_targets[a.id] for a in agents], task_obs=task_obs)
            else:
                jac = compute_directed_overlap(target_hist.sets([a.id for a in agents]), N)
                obs = np.array([[a.x / u.grid_size[1], a.y / u.grid_size[0]] for a in agents], dtype=float)
                pred = compute_pred_jac(agents, assigned_agvs, agent_idx, N) if coord.pred_boost else None
                coord.step(obs, jac, 0.05, delivery=sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5), pred_jac=pred)
            A = coord.A.copy(); W = coord.W.copy()
            n_links, w_m = coord.ne_h[-1], coord.W_mean_h[-1]
        elif method == 'full':
            n_links, w_m = N * (N - 1) // 2, 1.0
        else:
            n_links, w_m = int(A.sum() // 2), 0.0
        deliv_h.append(deliveries); link_h.append(n_links); w_mean_h.append(w_m)

        if recorder is not None:
            recorder(t=t, env=u, agents=agents, coordinator=coord, A=A, W=W, deliveries=deliveries,
                     assigned_agvs=assigned_agvs, assigned_pickers=assigned_pickers)

    env.close()
    zone = {k: v[0] / v[1] for k, v in shift_state['zone_share'].items()} if shift_state else None
    return dict(deliveries=np.array(deliv_h), links=np.array(link_h), w_mean=np.array(w_mean_h),
                coordinator=coord, agents=agents, stats=stats, zone_share=zone)
