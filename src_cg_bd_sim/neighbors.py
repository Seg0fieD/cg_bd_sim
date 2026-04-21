# Neighbor-pair detection via KDTree with periodic boundaries.

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
    Return displacement vectors r_ij = r_j - r_i with minimum image convention.

    """
    delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]

    #minimum image
    delta -= box_length * np.round(delta / box_length)

    return delta