"""
harness.py — one TA-RWARE episode with D-MORPH or a baseline.

TA-RWARE (Krnjaic et al., 2024) supplies the warehouse, the request queue, A* path planning and the
resolution of movement conflicts. This file supplies task allocation and the AGV mission controller,
carried over from MORPH's experiment loop:

  MORPH's loop (Full Graph, MORPH, Proximity-r, TSG): each queued request goes to the nearest free AGV; for
  each AGV task the nearest free Picker is proposed and responds with probability 0.25 + 0.75 x, where x is
  the link (A) or preference (W) between them (always 1 for Full Graph).

  CBAA-Partner (Choi et al., 2009, converged single-assignment form): for each pending request the nearest
  available robot is the primary robot; every available robot of the complementary role bids -distance to the
  request, and claims are resolved greedily on the highest bid; the winner is assigned without a response
  step. AGVs that need a Picker again (returning the shelf) are served by the same greedy auction.

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
AUCTION_METHODS = ('cbaa',)                                        # greedy auction (CBAA-Partner)
PAIR_METHODS = ('dmorph', 'random_fixed', 'random_step', 'uniform', 'random_partner', 'oracle_fast')  # pair allocation


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


def _install_slow_pickers(u, agents, seed, kind):
    """Hidden partner effectiveness: half of the Pickers move at 1/SLOW_FACTOR speed (on two of every three
    steps a movement micro-action is replaced by a no-op after TA-RWARE has resolved the macro-actions);
    nothing a robot observes reveals which. kind 'static': the same Pickers are slow all episode; 'swap': slow
    and fast Pickers swap at step SWAP_AT. Returns t -> boolean mask over agents of the Pickers slow at step t."""
    from tarware.definitions import Action
    pk = [i for i, a in enumerate(agents) if a.type == AgentType.PICKER]
    rng = np.random.default_rng(seed + 7777)
    slow = set(rng.choice(pk, size=len(pk) // 2, replace=False).tolist()); fast = set(pk) - slow
    N = len(agents)

    def slow_mask(t):
        m = np.zeros(N, dtype=bool)
        m[list(slow if (kind == 'static' or t < C.SWAP_AT) else fast)] = True
        return m

    orig, state = u.attribute_macro_actions, {'t': 0}

    def patched(macro_actions):
        out = orig(macro_actions)
        t = state['t']
        if t % C.SLOW_FACTOR != 0:
            for i in np.flatnonzero(slow_mask(t)):
                if agents[i].req_action in (Action.FORWARD, Action.LEFT, Action.RIGHT):
                    agents[i].req_action = Action.NOOP
        state['t'] += 1
        return out

    u.attribute_macro_actions = patched
    return slow_mask


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


def run_episode(method, scale, seed, T, params=None, reward='pair', r=C.PROX_R, shift=False, recorder=None,
                w_uniform=0.0, slow=None, response='paper'):
    """method: one of config.METHODS; params: D-MORPH parameter overrides (ablations); reward: D-MORPH credit
    mode ('pair' as in the paper, 'task' settlement); r: Proximity radius; shift: package location shift;
    w_uniform: the frozen preference of every pair in the Uniform control (also the response of
    'random_partner' and 'oracle_fast'); slow: None, 'static' or 'swap' (hidden slow Pickers);
    response (D-MORPH only): 'paper' 0.25 + 0.75 w_j[i]; 'normalized' 0.25 + 0.75 w_j[i] / max_k w_j[k]
    (robot j always accepts its most-preferred partner); 'always' every request is accepted.
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
    if method in ('full', 'cbaa'):
        A = np.ones((N, N)) - np.eye(N); W = np.ones((N, N)) - np.eye(N)
    elif method in ('uniform', 'random_partner', 'oracle_fast'):
        # control: pair allocation with every pair linked and every w_i[j] frozen at w_uniform (no learning)
        A = np.ones((N, N)) - np.eye(N); W = w_uniform * A
    elif method == 'morph':
        coord = MORPH(N, **C.morph_params(N, scale))
    elif method == 'dmorph':
        coord = DMORPH(N, C.N_AGV[scale], reward=reward, **(params or {}))
    elif method in ('random_fixed', 'random_step'):
        A, Wb = _random_graph(N, C.RANDOM_LINKS[scale], np.random.RandomState(12345 + seed), np.random.RandomState(777 + seed))
        redraw_rng = np.random.RandomState(4242 + seed)
    elif method not in ('proximity', 'tsg', 'cbaa'):
        raise ValueError(f'unknown method {method!r}; expected one of {C.METHODS}')
    pair = method in PAIR_METHODS

    shift_state = _install_shift(u) if shift else None
    slow_mask = _install_slow_pickers(u, agents, seed, slow) if slow else None
    sel = []                                                       # (t, chose a fast Picker) when both kinds were free
    gap_h = []                                                     # AGVs' mean w to fast minus to slow Pickers
    target_hist = TargetHistory([a.id for a in agents]) if coord is not None else None
    assigned_agvs, assigned_pickers, assigned_items = OrderedDict(), OrderedDict(), OrderedDict()
    deliveries = 0
    deliv_h, link_h, w_mean_h = [], [], []
    stats = dict(requests=0, refusals=0, assign=0, non_nearest=0, no_link=0, same_role=0, busy=0)
    agv_cycle = {}
    # congestion and waiting measurements (read only; no random draws)
    cg = dict(proposals=0, refusals=0, clashes=0, stucks=0, agv_idle=0, active_missions=0, picking_missions=0,
              wait_for_picker=0, queue_wait=[], cycle=[], agv_idle_steps=0, picker_idle_steps=0,
              task_delivery=[], task_return=[], task_agv_wait=[], task_picker_wait=[], auctioned=0, resolved=0,
              ts_proposals=[], ts_refusals=[], ts_agv_idle=[], ts_picker_idle=[], task_end_t=[])
    queued, pick_start = {}, {}
    task = {}                                                      # AGV -> its current task (one shelf cycle)

    def respond(j, i):
        cg['proposals'] += 1
        ok = _respond(j, i)
        cg['refusals'] += int(not ok)
        return ok

    def _respond(j, i):
        if method == 'full' or (method == 'dmorph' and response == 'always'):
            return True
        if method == 'dmorph' and response == 'normalized':
            top = W[j].max()
            return accepts(W[j, i] / top if top > 0 else 0.0)
        return accepts(W[j, i] if method in ('dmorph', 'morph', 'uniform', 'random_partner', 'oracle_fast') else (Wb[j, i] if Wb is not None else A[j, i]))

    def choose(i_, prim, cand, dist):
        """Pair allocation: the partner robot i_ requests, or None if the request fails this step."""
        if method in ('dmorph', 'uniform', 'random_partner', 'oracle_fast'):
            ids = [agent_idx[c.id] for c in cand]
            if method == 'random_partner':
                j = ids[np.random.randint(len(ids))]
            elif method == 'oracle_fast':                              # reference: knows the hidden speeds
                sm = slow_mask(t)
                j = min(ids, key=lambda q: (bool(sm[q]), dist(agents[q])))
            else:
                j = select_partner(W[i_], ids, lambda q: dist(agents[q]))
            if slow_mask is not None and cand[0].type == AgentType.PICKER:
                sm = slow_mask(t)
                if any(sm[q] for q in ids) and not all(sm[q] for q in ids):
                    sel.append((t, not bool(sm[j])))
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
        taken_items = set(assigned_items.values())
        for item in rq:
            if item.id not in taken_items:
                queued.setdefault(item.id, t)                      # request waiting for an AGV since step t

        if method == 'random_step':
            # Random-Sparse (per-step): new random links (same budget) and preferences at every step
            A, Wb_new = _random_graph(N, C.RANDOM_LINKS[scale], redraw_rng, redraw_rng)
            Wb[:] = Wb_new

        if method == 'cbaa':
            # new missions: nearest available robot is primary; complementary robots bid -distance; greedy
            used, reqs = set(), []
            for item in rq:
                if item.id in assigned_items.values() or (item.y, item.x) not in coords_map:
                    continue
                av = [a for a in agvs if not a.busy and not a.carrying_shelf and a not in assigned_agvs and a not in used]
                pk = [p for p in pickers if p not in assigned_pickers and p not in used]
                if not av or not pk:
                    continue
                dist = (lambda it: (lambda a: abs(a.x - it.x) + abs(a.y - it.y)))(item)
                prim = min(av + pk, key=dist)
                used.add(prim)
                reqs.append((item, prim, pk if prim.type == AgentType.AGV else av, dist))
            bids = sorted(((-dist(c), k, c) for k, (item, prim, cand, dist) in enumerate(reqs) for c in cand
                           if c not in used), key=lambda b: (-b[0], b[1], b[2].id))
            cg['auctioned'] += len({k for _, k, _ in bids})            # requests with at least one bid
            won, taken = set(), set()
            for _, k, c in bids:
                if k in won or c in taken:
                    continue
                won.add(k); taken.add(c)
                item, prim, cand, dist = reqs[k]
                cg['proposals'] += 1; cg['resolved'] += 1
                stats['requests'] += 1; stats['assign'] += 1
                stats['non_nearest'] += int(c is not min(cand, key=dist))
                agv, pkr = (prim, c) if prim.type == AgentType.AGV else (c, prim)
                loc = coords_map[(item.y, item.x)]
                assigned_agvs[agv] = Mission(MissionType.PICKING, loc, item.x, item.y, t)
                assigned_items[agv] = item.id
                assigned_pickers[pkr] = Mission(MissionType.PICKING, loc, item.x, item.y, t)
        elif pair:
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

        taken_items = set(assigned_items.values())
        for sid in [s_ for s_ in queued if s_ in taken_items]:
            cg['queue_wait'].append(t - queued.pop(sid))
        for agv, m in assigned_agvs.items():
            if m.mission_type == MissionType.PICKING:
                pick_start.setdefault(agv, m.assigned_time)

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
                tk = task.pop(agv, None)
                if tk is not None and tk['deliv'] is not None:     # task assigned -> shelf delivered -> shelf returned
                    cg['task_delivery'].append(tk['deliv'] - tk['start']); cg['task_return'].append(t - tk['start'])
                    cg['task_end_t'].append(t)
                    cg['task_agv_wait'].append(tk['agv_wait']); cg['task_picker_wait'].append(tk['picker_wait'])
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
        if method == 'cbaa':
            # free Pickers bid -distance to every AGV task that needs one; greedy on the highest bid
            reqs = [(agv, m) for agv, m in assigned_agvs.items()
                    if m.mission_type in (MissionType.PICKING, MissionType.RETURNING)
                    and (m.location_x, m.location_y) not in picker_locs]
            pk = [p for p in pickers if p not in assigned_pickers]
            bids = sorted(((-(abs(p.x - m.location_x) + abs(p.y - m.location_y)), k, p) for k, (agv, m) in enumerate(reqs)
                           for p in pk), key=lambda b: (-b[0], b[1], b[2].id))
            cg['auctioned'] += len({k for _, k, _ in bids})
            won, taken = set(), set()
            for _, k, p in bids:
                if k in won or p in taken:
                    continue
                agv, m = reqs[k]
                if (m.location_x, m.location_y) in picker_locs:
                    continue
                won.add(k); taken.add(p); cg['proposals'] += 1; cg['resolved'] += 1
                assigned_pickers[p] = Mission(MissionType.PICKING, m.location_id, m.location_x, m.location_y, t)
                picker_locs.add((m.location_x, m.location_y))
        for agv, m in (assigned_agvs.items() if method != 'cbaa' else ()):
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
        cg['active_missions'] += len(assigned_agvs)
        ia, ip = sum(a not in assigned_agvs for a in agvs), sum(p not in assigned_pickers for p in pickers)
        cg['agv_idle_steps'] += ia; cg['picker_idle_steps'] += ip                 # robot without a task
        cg['ts_agv_idle'].append(ia); cg['ts_picker_idle'].append(ip)
        cg['ts_proposals'].append(cg['proposals']); cg['ts_refusals'].append(cg['refusals'])  # cumulative
        for agv, m in assigned_agvs.items():
            if m.mission_type == MissionType.PICKING and agv not in task:
                task[agv] = dict(start=m.assigned_time, deliv=None, agv_wait=0, picker_wait=0)
            if agv not in task or m.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            needs = (m.mission_type == MissionType.PICKING and not agv.carrying_shelf) or \
                    (m.mission_type == MissionType.RETURNING and agv.carrying_shelf)
            if not needs:
                continue
            here = [p for p, mp in assigned_pickers.items()
                    if (mp.location_x, mp.location_y) == (m.location_x, m.location_y) and (p.x, p.y) == (m.location_x, m.location_y)]
            if m.at_location and not here:
                task[agv]['agv_wait'] += 1                         # AGV at the shelf, no Picker there
            elif here and not m.at_location:
                task[agv]['picker_wait'] += 1                      # Picker at the shelf, AGV not there
        cg['picking_missions'] += sum(m.mission_type == MissionType.PICKING for m in assigned_agvs.values())
        cg['wait_for_picker'] += sum(                              # AGV at its shelf, waiting to be loaded/unloaded
            1 for agv, m in assigned_agvs.items() if m.at_location and (
                (m.mission_type == MissionType.PICKING and not agv.carrying_shelf) or
                (m.mission_type == MissionType.RETURNING and agv.carrying_shelf)))
        _, rewards, _, _, info = env.step(list(actions[a] for a in agents))
        deliveries += sum(1 for rv in rewards[:u.num_agvs] if rv > 0.5)
        cg['clashes'] += info['clashes']; cg['stucks'] += info['stucks']; cg['agv_idle'] += info['agvs_idle_time']
        for k, rv in enumerate(rewards[:u.num_agvs]):
            if rv > 0.5 and agents[k] in pick_start:
                cg['cycle'].append(t + 1 - pick_start.pop(agents[k]))  # pickup mission assigned -> shelf delivered
            if rv > 0.5 and agents[k] in task:
                task[agents[k]]['deliv'] = t + 1

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
        elif method in ('full', 'uniform', 'cbaa'):
            n_links, w_m = N * (N - 1) // 2, (w_uniform if method == 'uniform' else 1.0)
        else:
            n_links, w_m = int(A.sum() // 2), 0.0
        deliv_h.append(deliveries); link_h.append(n_links); w_mean_h.append(w_m)

        if slow_mask is not None and method == 'dmorph' and t % 25 == 24:
            sm = slow_mask(t); pk_idx = [agent_idx[p.id] for p in pickers]
            fast_i = [q for q in pk_idx if not sm[q]]; slow_i = [q for q in pk_idx if sm[q]]
            gap_h.append(float(np.mean([W[agent_idx[a.id], fast_i].mean() - W[agent_idx[a.id], slow_i].mean() for a in agvs])))

        if recorder is not None:
            recorder(t=t, env=u, agents=agents, coordinator=coord, A=A, W=W, deliveries=deliveries,
                     assigned_agvs=assigned_agvs, assigned_pickers=assigned_pickers, slow_mask=slow_mask)

    env.close()
    zone = {k: v[0] / v[1] for k, v in shift_state['zone_share'].items()} if shift_state else None
    return dict(deliveries=np.array(deliv_h), links=np.array(link_h), w_mean=np.array(w_mean_h),
                coordinator=coord, agents=agents, stats=stats, zone_share=zone, congestion=cg,
                selection=sel, pref_gap=gap_h)
