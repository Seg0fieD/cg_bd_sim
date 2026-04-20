# Cluster & Bond  statistics: size histogram (final snapshot),
# cluster lifetime, bond lifetimes.

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src_cg_bd_sim.io import load_trajectory
from src_cg_bd_sim.observables import (
    compute_cluster_sizes,
    compute_cluster_lifetimes,
    compute_bond_lifetimes,
)
from src_cg_bd_sim.utils import stamped_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required = True, help = "Path to HDF5 trajectory")
    parser.add_argument("--output", default = "figures/clusters.png")
    parser.add_argument("--no-stamp", action = "store_true",
                        help = "disable timestamp suffix on output filename")
    args = parser.parse_args()

    traj               = load_trajectory(args.input)
    meta               = traj["metadata"]
    saved_bound_pairs  = traj["saved_bound_pairs"]
    n_particles        = int(meta["n_particles"])
    dt                 = float(meta["dt"])
    save_entry         = int(meta["save_entry"])
    dt_snap            = dt * save_entry

    # ----- Cluster size histogram on final snapshot -----
    final_pairs  = saved_bound_pairs[-1] if saved_bound_pairs else set ()
    size_hist    = compute_cluster_sizes(final_pairs, n_particles)
    # trim trailing zeros for cleaner plot
    max_k        = int(np.max(np.nonzero(size_hist)[0])) if size_hist.any() else 1
    ks           = np.arange(1, max_k + 1)
    counts       = size_hist[1:max_k + 1]

    # ------ Cluster lifetime (size >= 2) -------
    cluster_lt = compute_cluster_lifetimes(saved_bound_pairs, n_particles, dt_snap)

    # ------ bond lifetime (per pair) --------
    bond_lt = compute_bond_lifetimes(saved_bound_pairs, dt_snap)

    # ------ plot ------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Clusters & Bonds - {Path(args.input).name}", fontsize = 14)

    # (a) Cluster size histogram
    ax = axes[0]
    ax.bar(ks, counts, color = "C0", edgecolor = "k", alpha = 0.8)
    ax.set_xlabel("Cluster size k")
    ax.set_ylabel("number of clusters")
    ax.set_title("Cluster size distribution (final frame)")
    ax.set_xticks(ks)
    ax.grid(alpha = 0.3, axis = "y")

    # (b) Cluster lifetimes 
    ax = axes[1]
    if len(cluster_lt) > 0:
        ax.hist(cluster_lt, bins = min(10, len(cluster_lt)),
                color = "C3", edgecolor = "k", alpha = 0.8)
        ax.set_xlabel("lifetime")
        ax.set_ylabel("count")
        ax.set_title(f"Cluster lifetimes  (n={len(cluster_lt)}, mean={cluster_lt.mean():.3f})")

    else:
        ax.text(0.5, 0.5, "no cluster observed",
                ha = "center", va = "center", transform = ax.transAxes)
        ax.set_title("Cluster lifetimes")
    ax.grid(alpha = 0.3, axis = "y")

    # (c) bond lifetimes
    ax = axes[2]
    if len(bond_lt) > 0:
        ax.hist(bond_lt, bins = min(20, len(bond_lt)),
                color = "C2", edgecolor = "k", alpha = 0.8)
        ax.set_xlabel("lifetime")
        ax.set_ylabel("count")
        ax.set_title(f"Bond lifetimes (n = {len(bond_lt)}, mean={bond_lt.mean():.3f})")
    else:
        ax.text(0.5, 0.5, "no bonds observed", ha = "center", va = "center", 
                transform = ax.transAxes)
        ax.set_title("Bond lifetimes")
    ax.grid(alpha = 0.3, axis = "y")

    plt.tight_layout()

    # save figures
    # out_path = stamped_path(args.output)
    out_path = Path(args.output) if args.no_stamp else stamped_path(args.output)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 200)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()



