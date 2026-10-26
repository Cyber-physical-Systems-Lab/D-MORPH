"""
allocation.py — how D-MORPH's preferences enter task allocation (MORPH's response rule).

  select_partner  the primary robot i requests the candidate j with the highest w_i[j]; distance breaks
                  ties, so with no positive preference the nearest candidate is requested
  accepts         the requested robot j accepts with probability 0.25 + 0.75 w_j[i]; a refused request is
                  retried at the next step
"""
import numpy as np

RESPONSE_FLOOR = 0.25


def select_partner(prefs_i, candidates, distance):
    """prefs_i: robot i's preferences, indexable by robot index; candidates: robot indices available now;
    distance: robot index -> distance to the task. Returns the requested robot index."""
    return min(candidates, key=lambda j: (-prefs_i[j], distance(j)))


def accepts(w_ji, rng=np.random):
    """Robot j's response to robot i, given its preference w_j[i] in [0, 1]."""
    return rng.random() < RESPONSE_FLOOR + (1.0 - RESPONSE_FLOOR) * min(max(w_ji, 0.0), 1.0)
