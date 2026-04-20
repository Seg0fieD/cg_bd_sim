# cg_bd_sim

A coarse-grained Brownian dynamics simulator for protein-like particles in 3D. Particles diffuse in a periodic box, interact through soft repulsion and species-specific attraction, and can form transient bonds via rule-based binding/unbinding. Produces HDF5 trajectories and report-quality plots of dynamic, structural, and cluster observables.

## Physics

### Particles
Spherical coarse-grained particles of diameter σ in a cubic periodic box. Each particle belongs to a named species (e.g. A, B) with its own fraction of the population. All particles share the same diffusion coefficient D and thermal energy kT (configurable per-species support is in place but not yet used).

### Integrator
Overdamped Brownian dynamics:

    Δr = (D / kT) · F · Δt + √(2 D Δt) · η,     η ~ N(0, 1)

Periodic boundaries are applied after each step. Free-diffusion MSD scales as 6Dt, validated in `tests/test_diffusion.py`.

### Steric repulsion
Soft harmonic overlap penalty between any two particles with r < σ:

    F_rep = k_rep · (σ − r) · r̂

No force when particles don't overlap. Energy: `U = 0.5 · k_rep · (σ − r)²`.

### Short-range attraction
Species-pair-selective square well. Each `AttractionRule` names two species (possibly the same), a well depth ε, and a cutoff:

    F_att = ε · r̂   for r < cutoff, else 0

Only pairs whose species match the rule feel this force. Energy is a constant −ε per pair inside the well.

### Binding / unbinding (rule-based reactions)
Each `ReactionRule` defines how two species can form a transient bond:

- If two eligible particles come within `r_bind` of each other, they bind with probability `k_on · Δt` per step.
- A bound pair experiences a **harmonic spring** with stiffness `k_bond` and rest length `r_bond` (independent of the neighbor cutoff — stretched bonds still pull).
- Each bound pair unbinds with probability `k_off · Δt` per step.
- The `max_partners` field caps the number of simultaneous bonds per particle:
  - `1` → dimers only
  - `2` → linear chains
  - `3+` → branched networks
  - `0` → unlimited (gel-like connectivity)

Bond force: `F_bond = k_bond · (r − r_bond) · r̂` (attractive when stretched, repulsive when compressed).
Bond energy: `U_bond = 0.5 · k_bond · (r − r_bond)²`.

Neighbor detection uses `scipy.spatial.KDTree` with `boxsize` for minimum-image queries.

## Project structure

```
cg_bd_sim/
├── src_cg_bd_sim/             # core package
│   ├── config.py              # YAML → SimulationConfig
│   ├── types.py               # Species, AttractionRule, ReactionRule
│   ├── state.py               # SimulationState dataclass
│   ├── init_system.py         # random initial positions + species assignment
│   ├── boundaries.py          # PBC wrap
│   ├── neighbors.py           # KDTree pair finder + min-image displacements
│   ├── potentials.py          # repulsion, attraction, bond forces + energies
│   ├── reactions.py           # attempt_binding / attempt_unbinding
│   ├── integrator.py          # brownian_step
│   ├── simulation.py          # run loop
│   ├── observables.py         # MSD, g(r), S(q), cluster/bond lifetimes, …
│   └── io.py                  # HDF5 save/load (sparse or dense bond history)
├── scripts/
│   ├── run_sim.py             # run simulation → HDF5
│   ├── plot_dynamics.py       # MSD, MSD split, bound fraction, energy, autocorrelation
│   ├── plot_structure.py      # g(r) per species pair, S(q)
│   ├── plot_clusters.py       # cluster sizes, cluster lifetimes, bond lifetimes
│   └── animate_traj.py        # PyVista 3D interactive viewer
├── configs/                   # YAML experiment definitions
├── tests/                     # pytest suite
├── Snakefile                  # config → trajectory → 3 plots pipeline
└── pyproject.toml
```

## Installation

```bash
conda create -n cg_bd_sim python=3.12
conda activate cg_bd_sim
pip install -r requirements.txt
```

The package is used via `PYTHONPATH=.` rather than an editable install.

## Usage

### Snakemake pipeline (recommended)

Runs simulation + three plots in one command. The `config=<name>` key selects which YAML in `configs/` to use; outputs land in `outputs/<name>/` and `figures/<name>/`.

```bash
snakemake -j4 --config config=default
snakemake -j4 --config config=chains
snakemake -j4 --config config=gel
```

Re-running is cached — Snakemake skips steps whose inputs are unchanged.

### Manual execution

```bash
PYTHONPATH=. python scripts/run_sim.py \
    --config configs/default.yaml \
    --output outputs/run.h5 --no-stamp

PYTHONPATH=. python scripts/plot_dynamics.py  --input outputs/run.h5
PYTHONPATH=. python scripts/plot_structure.py --input outputs/run.h5
PYTHONPATH=. python scripts/plot_clusters.py  --input outputs/run.h5
```

Without `--no-stamp`, outputs get a `DDMMYY_HHMMSS` timestamp suffix.

### Interactive 3D viewer

```bash
PYTHONPATH=. python scripts/animate_traj.py --input outputs/default/trajectory.h5
```

Opens a PyVista window with a time slider, cluster-colored particles, bond tubes, and the PBC wireframe. `← / →` step through frames.

## Available configs

| Config                 | Description                                            |
|------------------------|--------------------------------------------------------|
| `default.yaml`         | Balanced reference: 1000 particles, dimers, ~80% bound |
| `test_small.yaml`      | 50 particles × 100 steps — CI-fast smoke test          |
| `chains.yaml`          | `max_partners=2` — linear chains                       |
| `networks.yaml`        | `max_partners=3` — branched networks                   |
| `gel.yaml`             | `max_partners=0` — unlimited bonds, gel-like           |
| `strong_attraction.yaml` | ε=5, no reactions — pure attractive clustering       |

## Observables

Every run exports to HDF5. The three plotting scripts compute:

**Dynamics** (`plot_dynamics.py`)
- Mean squared displacement vs 6Dt reference
- MSD split by current bound/free status
- Bound fraction over time
- Drift-subtracted position autocorrelation
- Total potential energy over time

**Structure** (`plot_structure.py`)
- Radial distribution function g(r) per species pair (A–A, A–B, B–B)
- Static structure factor S(q), angular-averaged

**Clusters & bonds** (`plot_clusters.py`)
- Cluster size histogram (final snapshot)
- Cluster lifetime distribution (membership-preserving)
- Per-bond lifetime distribution

Cluster identification uses connected components on the bond graph (`scipy.sparse.csgraph.connected_components`).

## Testing

```bash
PYTHONPATH=. pytest tests/ -q
```

41 tests covering diffusion, PBC, neighbors, repulsion/attraction, reactions, observables, and energy tracking.

## Implementation notes

- Bound pairs are stored canonically as `(i, j)` with `i < j`.
- HDF5 supports two storage formats for bond history: `sparse` (flat edge list + offsets, default) and `dense` (N×N adjacency per snapshot).
- Positions saved to HDF5 are PBC-wrapped. MSD post-processing calls `unwrap_positions` using frame-to-frame minimum-image displacements — valid as long as no particle moves more than L/2 between saved snapshots.
- The pipeline is deterministic: each config has a `seed`, and `Snakemake` caches on filename, so the same config → the same files.