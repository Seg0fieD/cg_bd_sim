# =======================================================================================
# Observables — pure functions on saved trajectory data.
# Chunk A: time-series observables (MSD, bound fraction, position autocorrelation)
# =======================================================================================

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# unwrap positions in post-processing using minimum-image of frame-to-frame displacements. 
# As long as no particle moves more than L/2 per saved snapshot 
# (expected ~0.14 per step, L/2 = 5 — safe), unwrapping is exact.
def unwrap_positions(
        saved_positions: list[np.ndarray],
        box_length: float,
        ) -> list[np.ndarray]:
    """
    Undo PBC wrapping for MSD calculation. 
    Assumes frame-to-frame displacement < L/2 (no particle moves more 
    than half a box between save snapshots)
    """
    if len(saved_positions) == 0: 
        return []
    
    unwrapped = [saved_positions[0].copy()]
    for t in range(1, len(saved_positions)):
        delta = saved_positions[t] - saved_positions[t - 1]
        delta -= box_length * np.round(delta /box_length)     # minimum image
        unwrapped.append(unwrapped[-1] + delta)

    return unwrapped

def compute_msd(
        saved_position: list[np.ndarray],
        initial_position: np.ndarray,
        )-> np.ndarray:
    """ Mean Square Displacement over time.
    MSD(t) = <|r(t) - r(0)|^2> averaged over all particles.

    Note: assume positions are NOT wrapped by PBC (ot that displacement are small relative to box). For long runs with PBC, 
    use unwrapped positions.

    Parameters
    ----------
    saved_positions : list of (N, 3) arrays, one per snapshot
    initial_position : (N, 3) reference positions at t = 0


    Returns
    -------
    msd : (n_snapshots,) array
    """

    msd = np.zeros(len(saved_position))
    for i, pos in enumerate(saved_position):
        disp = pos - initial_position
        msd[i] = np.mean(np.sum(disp ** 2, axis = 1))
    return msd

def compute_bound_fraction(
        save_bound_pairs: list[set[tuple[int, int]]],
        n_particles: int,
        ) -> np.ndarray:
    """
    fraction of particles that are bound at each snapshot.
    Each bound pair locks 2 particles -> fraction = 2 * n_pairs / n_particles.

    Return
    ------
    bound_fraction : (n_snapshots,) array in [0, 1]
    """

    frac = np.zeros(len(save_bound_pairs))
    for i, pairs in enumerate(save_bound_pairs):
        frac[i] = 2.0 * len(pairs) / n_particles
    
    return frac


# uses centered positions (drift-subtracted). For overdamped BD with no forces, expect fast decay. 
# With attraction/bonds, expect slower decay → shows caging.
def compute_position_autocorrelation(
        saved_positions: list[np.ndarray],
        ) -> np.ndarray:                                        
    """
    Position autocorrelation: C(t) =<(r(t) - r(0)) . (r(0) - <r>)> style metric.

    Here we use the normalized displacement autocorrelation:
         C(t) = <dr(t) . dr(0)> / <dr(0) . dr(0)>
            where dr(t) = r(t) - <r>_t (per-frame mean subtracted).

    For overdamped Browian motion, this should decay rapidly (memoryless)

    Returns
    -------
    C : (n_snapshots,) array, C[0] = 1.0
    """

    n_snap = len(saved_positions)
    if n_snap == 0:
        return np.zeros(0)

    # subtract per-frame center of mass to remove drift 
    centered = [pos - pos.mean(axis = 0) for pos in saved_positions]
    ref = centered[0]
    norm = np.mean(np.sum(ref * ref, axis = 1))

    C = np.zeros(n_snap)
    if norm < 1e-12:
        return C

    for i, pos in enumerate(centered):
        C[i] = np.mean(np.sum(ref * pos, axis = 1)) / norm

    return C 

# =======================================================================================
# Chunk B: structure observables (g(r) per species pair, S(q))
# =======================================================================================
def compute_rdf(
        positions: np.ndarray,
        species_ids: np.ndarray,
        box_length: float,
        species_a: int,
        species_b: int,
        r_max: float | None = None,
        n_bins: int = 100,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial distribution function g(r) between species_a and species_b.
    g(r) = (local density at r) / (bulk density).

    Uses minimum image convention of PBC, r_max defaults to box_length / 2.

    Parameters
    ----------
    positions            : (N, 3)
    species_ids          : (N,) int
    species_a, species_b : integer species IDs (can be the same for A-A)
    r_max                : max distance to bin (default box_length / 2)
    n_bins               : number of radial bins

    Returns
    -------
    r_centers : (n_bins,) bin centers
    g         : (n_bins,) g(r) values  
    """

    if r_max is None: 
        r_max = box_length / 2.0

    idx_a = np.where(species_ids == species_a)[0]
    idx_b = np.where(species_ids == species_b)[0]
    n_a   = len(idx_a)
    n_b   = len(idx_b)

    if n_a == 0 or n_b == 0:
        r_centers = np.linspace(0, r_max, n_bins + 1)[:-1] + 0.5 * r_max / n_bins 
        return r_centers, np.zeros(n_bins)
    
    # pairwise distance with minimum image 
    pos_a   = positions[idx_a]                            # (n_a, 3)
    pos_b   = positions[idx_b]                            # (n_b, 3)
    delta   = pos_a[:, None, :] - pos_b[None, :, : ]      # (n_a, n_b, 3)  
    delta  -= box_length * np.round(delta / box_length)
    dist    = np.linalg.norm(delta, axis=2)             # (n_a, n_b)   


    # exclude self-pairs when a == b
    if species_a == species_b:
        np.fill_diagonal(dist, np.inf)

    # histogram
    mask = (dist > 0) & (dist < r_max)
    hist, edges = np.histogram(dist[mask], bins = n_bins, range = (0.0, r_max))
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # Normaliztion; shell volume * bulk density * count of reference particles
    dr = edges[1] - edges[0]
    shell_vol = 4.0 * np.pi * r_centers ** 2 * dr
    volume = box_length ** 3 
    bulk_density_b = n_b / volume


    if species_a == species_b:
        # each pair counted twice in the cress matrix
        norm = n_a * bulk_density_b * shell_vol

    else:
        norm = n_a * bulk_density_b * shell_vol

    g = hist / norm 

    return r_centers, g 

def compute_structure_factor(
        positions: np.ndarray,
        box_length: float,
        n_q: int = 30,
        q_max: float | None = None,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Static structure factor S(q) computed on a grid of q-vector compatible with 
    PBC: q = (2* pi/L) * (nx, ny, nz)
    
    S(q) = (1/N) * |sum_j exp(-i q.r_j)|^2

    Angular-averaged (binned by |q|).

    Parameter
    ---------
    positions   : (N, 3)
    box_length  : L
    n_q         : max integer index per axis (q_grid goes from -n_q to n_q)
    q_max       : optional cap on |q|


    Return
    ------
    q_centers  : (n_bins,) array of |q| bin centers
    S          : (n_bins,) angular_averaged S(q) 
    """

    n = positions.shape[0]
    two_pi_L = 2.0 * np.pi / box_length

    # integer grid of q-vectors (skip origin)
    ns          = np.arange(-n_q, n_q + 1)
    nx, ny, nz  = np.meshgrid(ns, ns, ns, indexing = "ij")      
    q_int       = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis = 1)              # (M, 3)
    q_int       = q_int[np.any(q_int != 0, axis = 1)]                                   # drop (0, 0, 0)
    q_vecs      = two_pi_L * q_int                                                      # (M, 3)
    q_mag       = np.linalg.norm(q_vecs, axis = 1)                                      # (M, )


    if q_max is None:
        q_max = float(q_mag.max())
        
    keep    = q_mag <= q_max
    q_vecs = q_vecs[keep]
    q_mag = q_mag[keep]

    # S(q) per q-vector 
    # phases: (M, N) = q_vecs @ positions.T 
    phase   = q_vecs @ positions.T                                                      # (M, N)
    rho_q   = np.sum(np.exp(-1j * phase), axis = 1)                                     # (M, )
    S_per_q = (rho_q.real**2 + rho_q.imag**2) / n                                    # (M, )

    # angular average into bins
    n_bins  = min(40, int(np.ceil(q_max / two_pi_L)))
    bins = np.linspace(0.0, q_max, n_bins + 1)
    idx = np.digitize(q_mag, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    S = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(S, idx, S_per_q)
    np.add.at(counts, idx, 1)
    counts[counts == 0] = 1
    S = S / counts

    q_centers = 0.5 * (bins[:-1] + bins[1:])
    return q_centers, S

# =======================================================================================
# Chunk C: cluster & bond observables
#   - compute_cluster_sizes
#   - compute_cluster_lifetimes
#   - compute_bond_lifetimes
#   - compute_msd_split_bound_free
# =======================================================================================

def _clusters_from_pairs(
        bound_pairs: set[tuple[int, int]],
        n_particles: int,
        ) -> np.ndarray:
    """
    Return (n_particles,) array of cluster labels via connected components.
    Isolated particles get their own unique label.
    """

    if len(bound_pairs) == 0:
        return np.arange(n_particles)
    
    pairs = np.array(list(bound_pairs))
    rows  = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols  = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data  = np.ones(len(rows), dtype = np.int8)
    adj   = csr_matrix((data, (rows, cols)), shape = (n_particles, n_particles))

    _, labels = connected_components(adj, directed = False)

    return labels

def compute_cluster_sizes(
    bound_pairs: set[tuple[int, int]],
    n_particles: int,
    ) -> np.ndarray:
    """
    Histogram of ccluster sizes at single snapshot.
    Monomers (size - 1) included.

    Returns
    -------
    sizes: (n_particles + 1,) array, sizes[k] = number of cluster of size k
    """

    labels = _clusters_from_pairs(bound_pairs, n_particles)
    _, counts = np.unique(labels, return_counts = True)
    hist = np.zeros(n_particles + 1, dtype = int)
    for c in counts:
        hist[c] += 1

    return hist

def compute_cluster_lifetimes(
        saved_bound_pairs: list[set[tuple[int, int]]],
        n_particles: int,
        dt_snap: float,
        ) -> np.ndarray:
    """
    Track cluster identity across frames. A cluster "persists" if its 
    exact membership set reappears in the next frame. Lifetime ends when
    membership changes (split, merge, or particle leaves/joins). 

    Clusters of size 1 (monomers) are excluded.

    Parameters
    ----------
    saved_bound_pairs: list of snapshots
    n_particles: N
    dt_shape: time between snapshots (cfg.dt * cfg.save_entry)

    Returns
    -------
    lifetimes: (n_clusters_observed,) array of lifetimes in simulation time units
    """
    if len(saved_bound_pairs) == 0:
        return np.zeros(0)
    
    # per-frame: set of frozensets of member indices (clusters of size >=2)
    frame_clusters: list[set[frozenset[int]]] = []
    for pairs in saved_bound_pairs:
        labels = _clusters_from_pairs(pairs, n_particles)
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)
        clusters ={frozenset(members) for members in groups.values() if len(members) >= 2}
        frame_clusters.append(clusters)

    # track: for each cluster currently alive, how many frames it has survived 
    alive: dict[frozenset[int], int] = {}               # cluster -> frames alive
    finished_lifetimes: list[int] = []    

    for t, clusters in enumerate(frame_clusters):
        # extend survivors 
        new_alive: dict[frozenset[int], int] = {}
        for c in clusters:
            if c in alive:
                new_alive[c] = alive[c] + 1
            else:
                new_alive[c] = 1
    
        # clusters in alive but not in clusters -> they died 
        for c, age in alive.items():
            if c not in clusters:
                finished_lifetimes.append(age)
        
        alive = new_alive

    # clusters sitll alive at end of simulation - record their current age too
    for age in alive.values():
        finished_lifetimes.append(age)

    return np.array(finished_lifetimes) * dt_snap

def compute_bond_lifetimes(
        saved_bound_pairs: list[set[tuple[int, int]]],
        dt_snap: float,
        ) -> np.ndarray:
    """
    Per-pair bond lifetime: for each (i, j) pair, count consecutive frames
    it stays bound. Broken and re-formed bonds count as sepatate lifetimes. 

    Parameters
    ----------
    saved_bound_pairs : list of snapshots (each a set of (i, j) with i < j)
    dt_snap           : time between snapshots 

    Returns
    -------
    lifetimes  : (n_bound_observed,) arrary of lifetimes in simulation time units
    """

    if len(saved_bound_pairs) == 0:
        return np.zeros(0)
    
    active: dict[tuple[int, int], int] = {}                 # pair -> frames bound so far 
    finished: list[int] = []
    
    for pairs in saved_bound_pairs:
        # extend / start
        next_active: dict[tuple[int, int], int] = {}
        for p in pairs: 
            next_active[p] = active.get(p, 0) + 1
        # pairs that were active but are gone now -> finished
        for p, age in active.items():
            if p not in pairs:
                finished.append(age)
        active = next_active

    for age in active.values():
        finished.append(age)
        
    return np.array(finished) * dt_snap

def compute_msd_split_bound_free(
        saved_positions: list[np.ndarray],
        saved_bound_pairs: list[set[tuple[int, int]]],
        initial_positions: np.ndarray,
        n_particles: int,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    MSD split by current bound/free status at each snapshot.
    A particle is "bound" at frame t if it appears in any pair in saved_bound_pairs[t].

    Returns
    -------
    msd_bound  : (n_snapshots,) - mean |r(t) - r(0)|^2 over particles bound at t
    msd_free   : (n_snapshots,) - same for particles free at t
    NaN where the relevant population in empty. 
    """
    n_snap    = len(saved_positions)
    msd_bound = np.full(n_snap, np.nan)
    msd_free  = np.full(n_snap, np.nan)

    for t, (pos, pairs) in enumerate(zip(saved_positions, saved_bound_pairs)):
        bound_set: set[int] = set()
        for (i, j) in pairs:
            bound_set.add(int(i))
            bound_set.add(int(j))

        is_bound = np.zeros(n_particles, dtype = bool)
        if bound_set:
            is_bound[list(bound_set)] = True

        disp = pos - initial_positions
        sq = np.sum(disp ** 2, axis = 1)

        if is_bound.any():
            msd_bound[t] = sq[is_bound].mean()

        if (~is_bound).any():
            msd_free[t]  = sq[~is_bound].mean()
        
    return msd_bound, msd_free

