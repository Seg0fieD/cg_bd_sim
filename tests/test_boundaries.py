import numpy as np
import pytest
from src_cg_bd_sim.config import load_config
from src_cg_bd_sim.init_system import create_initial_state
from src_cg_bd_sim.simulation import Simulation
from src_cg_bd_sim.neighbors import find_neighbor_pairs, compute_displacement_vectors

CONFIG = "configs/test_small.yaml"

def test_positions_stay_in_box():
    """All position must remain within [0, box_length] after excution"""

    cfg = load_config(CONFIG)
    state = create_initial_state(cfg = cfg)
    Simulation(cfg, state).run()

    assert np.all(state.positions >= 0.0)
    assert np.all(state.positions < cfg.box_length)

def test_no_severe_overlap():
    """Minimum pairwise distance should not be less than 0.5 * sigma"""
    cfg = load_config(CONFIG)
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()

    pairs = find_neighbor_pairs(state.positions, cfg.sigma, cfg.box_length)

    if len(pairs) == 0:
        return              # no overlapping pairs at all - pass 
    
    deltas = compute_displacement_vectors(state.positions, pairs, cfg.box_length)
    min_dist = np.linalg.norm(deltas, axis = 1).min()
    assert min_dist >= 0.5 * cfg.sigma


def test_no_nan_or_inf():
    """Positions must never contain NaN or Inf."""
    cfg = load_config(CONFIG)
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()

    assert np.all(np.isfinite(state.positions))

