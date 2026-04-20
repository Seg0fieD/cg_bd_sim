# Tests for total potential energy tracking

import numpy as np
from src_cg_bd_sim.config import load_config
from src_cg_bd_sim.init_system import create_initial_state
from src_cg_bd_sim.simulation import Simulation

CONFIG = "configs/test_small.yaml"

def test_energies_recorded_per_snapshot():
    """ One energy value per saved snapshot. """
    cfg   = load_config(CONFIG)
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()
    assert len(state.saved_energies) == len(state.saved_positions)
    assert len(state.saved_energies) > 0

def test_energies_finite():
    """ No NaN or Inf in energy series. """
    cfg  = load_config(CONFIG)
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()
    arr = np.array(state.saved_energies)
    assert np.all(np.isfinite(arr))

def test_energy_nonpositive_with_attraction_dominant():
    """ With strong attraction and weak repulsion, total PE should be <= 0
        on average across the run (attractive wells dominate).
    """
    cfg = load_config(CONFIG)
    cfg.attraction_rules = [{
        "species_a": "A", "species_b": "B",
        "epsilon": 5.0, "cutoff": 2.0
    }]
    cfg.reactions = []                      # disable bonds for a clean test
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()
    mean_E = float(np.mean(state.saved_energies))
    assert mean_E <= 0.0
    



