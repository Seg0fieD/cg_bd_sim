# Boundary handling.

# Functions:

# apply_periodic(positions, box)
# minimum_image(displacement, box)
# optional reflecting-wall logic later

# Keep this module clean because many other modules depend on it.

import numpy as np


def apply_periodic_boundary(positions: np.ndarray, box_length: float) -> None:
    positions %= box_length