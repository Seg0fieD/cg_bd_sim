# Periodic boundary wrap: keep positions inside [0, box_length).

import numpy as np

def apply_periodic_boundary(positions: np.ndarray, box_length: float) -> None:
    positions %= box_length