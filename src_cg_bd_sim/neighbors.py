# Neighbor list / near-pair detection.
# Version 1:
# rebuild neighbor structure every few steps
# use cKDTree
# generate close pairs within cutoff
# SciPy’s cKDTree is a good fit for fast nearest-neighbor and radius-based lookup.
# Interface idea:
# pairs = find_neighbor_pairs(positions, cutoff, box)

# Later:
# custom cell lists for full performance
# skin distance / Verlet-style updates
# I would prefer to use this skin distance and verlet- styled update 


import numpy as np 
from scipy.spatial import KDTree


def find_neighbor_pairs(
        positions: np.ndarray,
        cutoff: float,
        box_length: float,
        ) -> np.ndarray:
    """
    Return (N, 2) array of index pairs within cutoff distance. 
    Uses periodic boundary conditions via cKDTree boxsize.
    """

    tree = KDTree(positions, boxsize=box_length)
    pairs = tree.query_pairs(r = cutoff, output_type = "ndarray")

    return pairs


def compute_displacement_vectors(
        positions: np.ndarray,
        pairs: np.ndarray,
        box_length: float,
        ) -> np.ndarray:
    """
    Return displacement vectors r_ij = r_j - r_i with minimum image covertion.
    Shape: (n_pairs, 3)
    """
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]

    #minimum image
    delta -= box_length * np.round(delta / box_length)

    return delta