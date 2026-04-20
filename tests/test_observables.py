# Tests for observables

import numpy as np
import pytest

from src_cg_bd_sim.observables import (
    compute_msd,
    compute_bound_fraction,
    compute_position_autocorrelation,
    compute_rdf,
    compute_structure_factor,
    compute_cluster_sizes,
    compute_cluster_lifetimes,
    compute_bond_lifetimes,
    compute_msd_split_bound_free,
)

# ================================= chunk A =================================
def test_msd_zero_displacement():
    """ Particles that don't move have MSD = 0. """
    init = np.zeros((10, 3))
    saved = [init.copy() for _ in range(5)]
    msd = compute_msd(saved, init)
    assert np.allclose(msd, 0.0)

def test_msd_linear_growth():
    """ Manual displacement r(t) = t * v should give MSD = t^2 * |v|^ 2. """

    n = 100
    init = np.zeros((n, 3))
    v = np.array([1.0, 0.0, 0.0])
    saved = [init + t * v for t in range(5)]
    msd = compute_msd(saved, init)
    expected = np.array([t **2 for t in range(5)], dtype = float)
    assert np.allclose(msd, expected)

def test_bound_fraction_empty():
    saved = [set(), set(), set()]
    frac = compute_bound_fraction(saved, n_particles = 10)
    assert np.allclose(frac, 0.0)

def test_bound_fraction_full():
    """ 5 Paris over 10 particles = 100% bound. """
    saved = [{(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)}]
    frac  = compute_bound_fraction(saved, n_particles = 10)
    assert np.isclose(frac[0], 1.0)

def test_position_autocorrelation_starts_at_one():
    rng = np.random.default_rng(0)
    saved = [rng.normal(size = (20, 3)) for _ in range(5)]
    C = compute_position_autocorrelation(saved)
    assert np.isclose(C[0], 1.0)


# ================================= chunk B =================================

def test_rdf_returns_correct_shape():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10.0, size = (50, 3))
    species = np.array([0] * 25 + [1] * 25)
    r, g = compute_rdf(pos, species, box_length = 10.0,
                       species_a = 0, species_b = 1, n_bins = 50)
    assert r.shape == (50,)
    assert g.shape == (50,)
    assert np.all(g >= 0)


def test_rdf_uniform_gas_approaches_one():
    """ Random uniform positions: g(r) should fluctuate around ~1 at large r."""
    rng = np.random.default_rng(0)
    n = 2000
    L = 20.0
    pos = rng.uniform(0, L, size = (n, 3))
    species = np.zeros(n, dtype = int)
    r, g = compute_rdf(pos, species, box_length = L,
                       species_a = 0, species_b = 0, n_bins = 40)
    # mid-range g(r) should average near 1 for ideal gas
    mid = g[len(g) // 4 : 3 * len(g) // 4]
    assert 0.7 < mid.mean() < 1.3

def test_structure_factor_shape():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10.0, size = (100, 3))
    q, S = compute_structure_factor(pos, box_length = 10.0, n_q = 10)
    assert q.shape == S.shape
    assert np.all(S >= 0)

# ================================= chunk C =================================

def test_cluster_sizes_all_monomers():
    hist = compute_cluster_sizes(set(), n_particles = 5)
    assert hist[1] == 5
    assert hist[2:].sum() == 0


def test_cluster_sizes_one_dimer():
    hist = compute_cluster_sizes({(0, 1)}, n_particles = 5)
    assert hist[1] == 3             # particles 2, 3, 4 are monomers
    assert hist[2] == 1             # one dimer

def  test_cluster_sizes_trimer_chain():
    """ 0-1, 1-2 -> one cluster of size 3."""
    hist = compute_cluster_sizes({(0, 1), (1, 2)}, n_particles = 5)
    assert hist[1] == 2
    assert hist[3] == 1

def test_cluster_lifetimes_stable():
    """ Same cluster across 5 frames -> lifetime = 5 * dt_snap."""
    snaps = [{(0, 1)} for _ in range(5)]
    lifetimes = compute_cluster_lifetimes(snaps, n_particles = 3, dt_snap = 0.1)
    assert len(lifetimes) == 1
    assert np.isclose(lifetimes[0], 0.5)

def test_cluster_lifetimes_breaks():
    """ Cluster alive frames 0-2, then gone."""
    snaps = [{(0, 1)}, {(0, 1)}, {(0,1)}, set(), set()]
    lifetimes = compute_cluster_lifetimes(snaps, n_particles = 3, dt_snap = 1.0)
    assert len(lifetimes) == 1
    assert np.isclose(lifetimes[0], 3.0)

def test_bond_lifetimes_persistent():
    snaps = [{(0, 1)} for _ in range(4)]
    lt = compute_bond_lifetimes(snaps, dt_snap = 0.5)
    assert len(lt) == 1
    assert np.isclose(lt[0], 2.0)

def test_bond_lifetimes_reform_counts_twice():
    """ Bond forms, breaks, re-forms -> two separate lifetimes."""
    snaps = [{(0, 1)}, {(0, 1)}, set(), {(0, 1)}]
    lt = compute_bond_lifetimes(snaps, dt_snap = 1.0)
    assert len(lt) == 2
    assert sorted(lt.tolist()) == [1.0, 2.0]

def test_msd_split_all_bound():
    """ All particles bound -> msd_free is NaN, mas_bound is finite. """
    init = np.zeros((4, 3))
    saved = [np.ones((4, 3))]                               # each particle moved by (1, 1, 1)
    saved_pairs = [{(0, 1), (2, 3)}]
    msd_b , msd_f = compute_msd_split_bound_free(saved, saved_pairs, init, n_particles = 4)
    assert np.isclose(msd_b[0], 3.0)                        # |(1, 1, 1)|^2  = 3
    assert np.isnan(msd_f[0])

def test_msd_split_mixed():
    init = np.zeros((4, 3))
    pos = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]], dtype = float)
    saved = [pos]
    saved_pairs = [{(0, 1)}]                                # 0, 1 bound; 2,3 free
    msd_b, msd_f = compute_msd_split_bound_free(saved, saved_pairs, init, n_particles = 4)
    assert np.isclose(msd_b[0], 1.0)                        # avg of 1, 1
    assert np.isclose(msd_f[0], 4.0)                        # avg of 4, 4