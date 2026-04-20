# correctness test 
# estime MSD and compare slope with 6D

from src_cg_bd_sim.integrator import brownian_step
import numpy as np

def test_free_diffusion_msd():
    n_particles = 5000
    n_steps = 200
    dt = 1e-3
    D = 1.0
    box_length = 1e6

    rng = np.random.default_rng(42)
    positions = np.zeros((n_particles, 3))
    initial = positions.copy()

    for _ in range(n_steps):
        brownian_step(positions, D, dt, box_length, rng, apply_pbc=False)

    disp = positions - initial
    msd = np.mean(np.sum(disp**2, axis = 1))
    expected = 6.0 * D * n_steps * dt

    assert np.isclose(msd, expected, rtol = 0.15)