# Defines interaction laws.
# Start with:
# Steric repulsion
# hard-sphere-like approximation
# or soft repulsion, e.g. harmonic overlap penalty
# Short-range attraction
# square well
# Morse-like
# simple patch attraction later
# Functions:
# compute_forces(state, pairs, params, box) -> forces
# compute_pair_energy(...)

import numpy as np
from .types import AttractionRule

# harmonic_repulsion — pushes overlapping particles apart (only when r < sigma)
# harmonic_bond_force — spring between bound pairs (attractive when stretched, repulsive when compressed)

def harmonic_repulsion(
        positions: np.ndarray,
        pairs: np.ndarray, 
        deltas: np.ndarray,
        sigma: float,
        k_rep: float,
        ) -> np.ndarray:
    """
    Soft harmonic repulsion between overlapping paris.
    F = k_rep * (sigma - r) * r_hat if r < sigma, else 0

    Parameters
    -----------
    positions : (N, 3)
    pairs     : (M, 2) index pairs from neighbors.py
    deltas    : (M, 3) displacement vector r_j - r_i (minimum image)
    sigma     : particle diameter (overlap threshold)
    k-rep     : repulsion spring constant

    Returns
    -------
    forces : (N, 3)
    """

    forces = np.zeros_like(positions)

    if len(pairs) == 0:
        return forces
    
    dist = np.linalg.norm(deltas, axis = 1)     #(M, 1)
    overlap_mask = dist < sigma

    if not np.any(overlap_mask):
        return forces
    
    d = dist[overlap_mask]
    dv = deltas[overlap_mask]
    p = pairs[overlap_mask]

    r_hat = dv / d[:, np.newaxis]           #unit vector
    magnitude = k_rep * (sigma - d)         #scaler force
    f = magnitude[:, np.newaxis] * r_hat    # (M, 3)

    # i pushed away from j, j pushed away from i
    np.add.at(forces, p[:, 0], -f)
    np.add.at(forces, p[:, 1],  f)

    return forces

def harmonic_bond_force(
        positions: np.ndarray,
        bound_pairs: set[tuple[int, int]],
        box_length: float,
        k_bond: float,
        r_bond: float,
        ) -> np.ndarray:
    """
    Harmonic spring between bound pairs.
    F = k_bond * (r - r_bond) * r_hat (attractive if r > r_bond, repulsive if r < r_bond)
    """
    forces = np.zeros_like(positions)

    if len(bound_pairs) == 0:
        return forces
    
    pairs = np.array(list(bound_pairs))                         # (M, 2)
    deltas = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    deltas -= box_length * np.round(deltas / box_length)        # minimum image

    dist = np.linalg.norm(deltas, axis = 1)
    valid = dist > 1e-8
    if not np.any(valid):
        return forces
    
    d  = dist[valid]
    dv = deltas[valid]
    p  = pairs[valid]

    r_hat = dv / d[:, np.newaxis]
    magnitude = k_bond * (d - r_bond)                           # positive when stretched
    f = magnitude[:, np.newaxis] * r_hat                        # pulls i toward j if stretched

    np.add.at(forces, p[:, 0], f)
    np.add.at(forces, p[:, 1], -f)

    return forces

def pair_energy_bond(
        positions: np.ndarray,
        bound_pairs: set[tuple[int, int]],
        box_length: float,
        k_bond: float,
        r_bond: float,     
        ) -> float:
    """ Total bond energy : U = 0.5 * k_bond * (r - r_bond)^2 summed over bound pairs. """
    if len(bound_pairs) == 0:
        return 0.0
    
    pairs = np.array(list(bound_pairs))
    deltas = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    deltas -= box_length * np.round(deltas / box_length)
    dist = np.linalg.norm(deltas, axis = 1) 

    # bond force computes its own deltas from positions (not the neighbor-list deltas), 
    # because bound pairs may be outside the neighbor cutoff if they get stretched.

    return 0.5 * k_bond * np.sum((dist - r_bond) ** 2)

def pair_energy_repulsion(
        pairs: np.ndarray,
        deltas: np.ndarray,
        sigma: float,
        k_rep: float,
        ) -> float: 
    """
    Total repulsive potential energy: U = 0.5 * k_rep * (sigma - r)^2
    """

    if len(pairs) == 0:
        return 0.0
    
    dist    = np.linalg.norm(deltas, axis = 1)
    overlap = dist[dist < sigma]

    return 0.5 * k_rep * np.sum((sigma - overlap) ** 2)

    
def square_well_attraction(
        positions: np.ndarray, 
        species_ids: np.ndarray,
        pairs: np.ndarray,
        deltas: np.ndarray,
        rules: list[AttractionRule],
        species_names: list[str],
        ) -> np.ndarray:
    """
    Square-well attractive force between compatible species.
    F = -epsilon * r_hat if sigma < r < cutoff, else 0

    Parameters
    ----------
    species_ids     : (N, ) int array - index into species_names per particle
    rules           : list of AttractionRule defining which pairs attract 
    species_names   : ordered list mapping id -> name
    """

    forces = np.zeros_like(positions)

    if len(pairs) == 0:
        return forces 
    
    dist = np.linalg.norm(deltas, axis = 1)

    for rule in rules: 
        id_a = species_names.index(rule.species_a)
        id_b = species_names.index(rule.species_b)

        si = species_ids[pairs[:, 0]]
        sj = species_ids[pairs[:, 1]]

        mask = (
            ((si == id_a) & (sj == id_b)) |
            ((si == id_b) & (sj == id_a))
            ) & (dist < rule.cutoff) & (dist > 1e-8)
        
        if not np.any(mask):
            continue


        d = dist[mask]
        dv = deltas[mask]
        p = pairs[mask]

        r_hat = dv / d[:, np.newaxis]
        f = rule.epsilon * r_hat            # pulls i toward j 

        np.add.at(forces, p[:, 0], f)
        np.add.at(forces, p[:, 1], -f)

    return forces


def pair_energy_attraction(
        species_ids: np.ndarray,
        pairs: np.ndarray,
        deltas: np.ndarray,
        rules: list[AttractionRule],
        species_names: list[str],
    ) -> float:
    """ Total attractive potential energy: U = -epsilon per attracted pair."""

    if len(pairs) == 0:
        return 0.0
    
    dist = np.linalg.norm(deltas, axis = 1)
    total = 0.0

    for rule in rules:
        id_a = species_names.index(rule.species_a)
        id_b = species_names.index(rule.species_b)

        si = species_ids[pairs[:, 0]]
        sj = species_ids[pairs[:, 1]]

        mask = (
            ((si == id_a) & (sj == id_b)) |
            ((si == id_b) & (sj == id_a))
            ) & (dist < rule.cutoff)
        
        total += -rule.epsilon * np.sum(mask)

    return total
        
        

