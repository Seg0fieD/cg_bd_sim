# Real physics for free diffusion Δr = √(2DΔtη) with η ~ N(0,1)

import numpy as np 
from .boundaries import apply_periodic_boundary

def brownian_step(
    positions: np.ndarray,
    diffusion: float,
    dt: float,
    box_length: float,
    rng: np.random.Generator,
    apply_pbc: bool = True,
    forces: np.ndarray | None = None,
    kT: float = 1.0
    ) -> None:
    """
    Δr = (D/kT) * F * dt  +  √(2D dt) * η
    """

    sigma = np.sqrt(2.0 * diffusion * dt)
    noise = rng.normal(loc = 0.0, scale = sigma, size = positions.shape)

    drift = np.zeros_like(positions)
    if forces is not None:
        drift = (diffusion /kT) * forces * dt

    positions += drift + noise

    # displacement = rng.normal(loc=0.0, scale=sigma, size=positions.shape)
    # positions += displacement

    if apply_pbc:
        apply_periodic_boundary(positions, box_length)


