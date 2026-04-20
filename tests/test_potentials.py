import numpy as np
from src_cg_bd_sim.potentials import harmonic_repulsion, pair_energy_repulsion, square_well_attraction
from src_cg_bd_sim.types import AttractionRule


def test_repulsion_zero_when_no_overlap():
    positions   = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pairs       = np.array([[0, 1]])
    deltas      = np.array([[2.0, 0.0, 0.0]])
    forces      = harmonic_repulsion(positions  = positions,
                                     pairs      = pairs,
                                     deltas     = deltas,
                                     sigma      = 1.0,
                                     k_rep      = 100.0
                                     )
    assert np.allclose(forces, 0.0)

def test_repulsion_nonzero_on_overlap():
    positions   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pairs       = np.array([[0, 1]])
    deltas      = np.array([[0.5, 0.0, 0.0]])
    forces      = harmonic_repulsion(positions  = positions,
                                     pairs      = pairs,
                                     deltas     = deltas,
                                     sigma      = 1.0,
                                     k_rep      = 100.0
                                     )
    assert forces[0, 0] < 0.0   # particle 0 pushed left
    assert forces[1, 0] > 0.0   # particle 1 pushed right

def test_repulsion_energy_zero_no_overlap():
    pairs       = np.array([[0, 1]])
    deltas      = np.array([[2.0, 0.0, 0.0]])
    energy      = pair_energy_repulsion(pairs  = pairs,
                                        deltas = deltas,
                                        sigma  = 1.0,
                                        k_rep  = 100.0
                                        )
    assert energy == 0.0

def test_attraction_between_compatible_species():
    positions   = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    species_ids = np.array([0, 1])
    pairs       = np.array([[0, 1]])
    deltas      = np.array([[1.5, 0.0, 0.0]])
    rules       = [AttractionRule(species_a = "A", species_b = "B", epsilon = 2.0, cutoff = 2.0)]
    forces      = square_well_attraction(positions = positions,
                                         species_ids = species_ids,
                                         species_names= ["A", "B"],
                                         pairs = pairs,
                                         deltas = deltas, 
                                         rules= rules,
                                         )
    assert forces[0, 0] > 0.0       # A pulled toward B
    assert forces[1, 0] < 0.0       # B pulled toward A

def test_no_attraction_same_species():
    positions   = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    species_ids = np.array([0, 0])   # both A
    pairs       = np.array([[0, 1]])
    deltas      = np.array([[1.5, 0.0, 0.0]])
    rules       = [AttractionRule(species_a = "A", species_b = "B", epsilon = 2.0, cutoff = 2.0)]
    forces      = square_well_attraction(positions = positions,
                                         species_ids = species_ids,
                                         species_names= ["A", "B"],
                                         pairs = pairs,
                                         deltas = deltas, 
                                         rules= rules,
                                         )
    assert np.allclose(forces, 0.0)