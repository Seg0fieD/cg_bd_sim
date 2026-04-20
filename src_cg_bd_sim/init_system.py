import numpy as np
from .config import SimulationConfig
from .state import SimulationState

def create_initial_state(cfg: SimulationConfig) -> SimulationState:
    rng = np.random.default_rng(cfg.seed)

    positions = rng.uniform(0.0, cfg.box_length, size=(cfg.n_particles, 3))

    # assign species by fraction
    species_ids = np.zeros(cfg.n_particles, dtype = int)
    if cfg.species:
        fractions   = [s["fraction"] for s in cfg.species]
        counts      = np.round(np.array(fractions) * cfg.n_particles).astype(int)
        counts[-1]  = cfg.n_particles - counts[: -1].sum()     # fix rounding
        species_ids = np.repeat(np.arange(len(cfg.species)), counts)
        rng.shuffle(species_ids)

    return SimulationState(
        positions          = positions.copy(),
        initial_positions  = positions.copy(),
        times              = [],
        saved_positions    = [],
        species_ids        = species_ids,
        bound_pairs        = set(),
        saved_bound_pairs  = [],
        saved_energies     = [],
        )


