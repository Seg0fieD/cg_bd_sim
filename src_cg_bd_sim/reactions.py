# Rule-based binding and unbinding of particle pairs.
# Binding: compatible A-B pairs within r_bind bind with probability k_on * dt.
# Unbinding: each bound pair dissolves with probability k_off * dt.
# `max_partners` caps how many bonds a single particle can hold simultaneously.


import numpy as np
from .types import ReactionRule
from .state import SimulationState


def _canonical(i : int, j: int ) -> tuple[int, int]:
    """Return pair in sorted order so (i, j) and (j, i) map to the same key."""

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
    Attempt to bind compatible unbound pairs within r_bind(that is NOT already bound), respecting max_partners;
    bind probablity k_on * dt.
    """

    if len(pairs) == 0 or not rules:
        return
    
    dist = np.linalg.norm(deltas, axis = 1)

    # current bond count per particle, based on max_partners
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
    Attempt to unbind each bound pair using the probablity k_off * dt of its matching rule.
    """

    if not state.bound_pairs or not rules:
        return
    
    # species-pair -> k_off lookup (unordered pair as frozenset)
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
