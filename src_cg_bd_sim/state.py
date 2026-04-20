# Hold the simulation array | instead of per-particel objects , using an array 

from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationState: 
    positions: np.ndarray
    initial_positions: np.ndarray
    times: list[float]
    saved_positions: list[np.ndarray]
    species_ids: np.ndarray
    bound_pairs: set[tuple[int, int]] = field(default_factory = set) # convention pairs stored as (i, j) with i < j for uniqueness 
    saved_bound_pairs: list[set[tuple[int, int]]] = field(default_factory = list)
    saved_energies: list[float] = field(default_factory = list)
     
