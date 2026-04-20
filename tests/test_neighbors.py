# We shall test the neighbours of the particle or protein here

import numpy as np
from src_cg_bd_sim.neighbors import find_neighbor_pairs, compute_displacement_vectors

def test_finds_close_pair():
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pairs  = find_neighbor_pairs(positions = positions, cutoff = 1.0, box_length = 10.0)
    assert len(pairs) == 1
    assert set(pairs[0]) == {0, 1}

def test_no_pairs_beyond_cutoff():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pairs     = find_neighbor_pairs(positions = positions, cutoff = 1.0, box_length = 10.0)
    assert len(pairs) == 0

def test_periodic_pair_detection():
    """Particle near opposite walls should be detected as neighbors."""
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    pairs = find_neighbor_pairs(positions = positions, cutoff = 1.0, box_length = 10.0)
    assert len(pairs) == 1


def test_minimum_image_displacement():
    positions = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
    pairs = find_neighbor_pairs(positions, cutoff=1.0, box_length=10.0)
    deltas = compute_displacement_vectors(positions, pairs, box_length=10.0)
    assert np.linalg.norm(deltas[0]) < 1.0