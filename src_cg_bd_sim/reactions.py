# Handles binding/unbinding.
# Two modes:
# A. Pure force-based sticking
# B. Rule-based reaction


import numpy as np
from .types import ReactionRule
from .state import SimulationState


def _canonical(i : int, j: int ) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)

def attempt_binding(
        state: SimulationState,
        pairs: np.ndarray,
        deltas: np.ndarray,
        rules: list[ReactionRule],
        species_names: list[str],
        dt: float,
        rng: np.random.Generator,
        ) -> None: 
    """
    For each compatible A-B pair within r_bind that is NOT already bound, bind with probablity k_on * dt. 
    """

    if len(pairs) == 0 or not rules:
        return
    
    dist = np.linalg.norm(deltas, axis = 1)

    # - x - x - x - x - x - x - x - x - x - x
    # particle already bound (on either side) are locked - one-partner binding 
    # locked = set()
    # for (a, b) in state.bound_pairs:
    #     locked.add(a)
    #     locked.add(b)

    # for rule in rules:
    #     id_a = species_names.index(rule.species_a)
    #     id_b = species_names.index(rule.species_b)

    #     si = state.species_ids[pairs[:, 0]]
    #     sj = state.species_ids[pairs[:, 1]]

    #     mask = (
    #         ((si == id_a) & (sj == id_b)) |
    #         ((si == id_b) & (sj == id_a))
    #     ) & (dist < rule.r_bind)

    #     if not np.any(mask):
    #         continue

    #     candidates = pairs[mask]
    #     p_bind = rule.k_on * dt

    #     for (i, j ) in candidates:
    #         i, j = int(i), int(j)

    #         if i in locked or j in locked:
    #             continue

    #         if rng.random() < p_bind:
    #             state.bound_pairs.add(_canonical(i, j))
    #             locked.add(i)
    #             locked.add(j)

    # - x - x - x - x - x - x - x - x - x - x
    # count existing partners per particle 
    partner_count: dict[int, int] = {}
    for (a, b) in state.bound_pairs:
        partner_count[a] = partner_count.get(a, 0) + 1
        partner_count[b] = partner_count.get(b, 0) + 1

    for rule in rules:
        id_a = species_names.index(rule.species_a)
        id_b = species_names.index(rule.species_b)

        si = state.species_ids[pairs[:, 0]]
        sj = state.species_ids[pairs[:, 1]]

        mask = (
            ((si == id_a) & (sj == id_b)) |
            ((si == id_b) & (sj == id_a))
            ) & (dist < rule.r_bind)
        
        if not np.any(mask):
            continue

        candidates = pairs[mask]
        p_bind = rule.k_on * dt
        cap = rule.max_partners         

        for (i, j) in candidates:
            i, j = int(i), int(j)
            if cap > 0:
                if partner_count.get(i, 0) >= cap:
                    continue
                if partner_count.get(j, 0) >= cap:
                    continue
            
            if rng.random() < p_bind:
                state.bound_pairs.add(_canonical(i, j))
                partner_count[i] = partner_count.get(i, 0) + 1
                partner_count[j] = partner_count.get(j, 0) + 1


def attempt_unbinding(
        state: SimulationState,
        rules: list[ReactionRule],
        species_names: list[str],
        dt: float,
        rng: np.random.Generator,
        ) -> None:
    """
    For each bound pair, unbind with probablity k_off * dt (using the rule matching its species pair).
    """

    if not state.bound_pairs or not rules:
        return
    
    # build species-pair -> k_off lookup
    koff_lookup: dict[frozenset[int], float] = {}
    for rule in rules:
        id_a = species_names.index(rule.species_a)
        id_b = species_names.index(rule.species_b)
        koff_lookup[frozenset({id_a, id_b})] = rule.k_off

    to_remove = []
    for (i, j) in state.bound_pairs:
        key = frozenset({int(state.species_ids[i]), int(state.species_ids[j])})
        k_off = koff_lookup.get(key)
        if k_off is None: 
            continue
        if rng.random() < k_off * dt:
            to_remove.append((i, j))

    for pair in to_remove:
        state.bound_pairs.discard(pair)
