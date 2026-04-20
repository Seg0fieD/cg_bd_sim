# Spatial observables: radial distribution function g(r) per species pair 
# and static structures faccor S(q). Average over the last few snapshots
# to reduce single-frame noise. 

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src_cg_bd_sim.io import load_trajectory 
from src_cg_bd_sim.observables import compute_rdf, compute_structure_factor
from src_cg_bd_sim.utils import stamped_path

def main() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required = True, help= "Path to HDF5 trajectory" )
    parser.add_argument("--output", default="figures/structure.png")
    parser.add_argument("--n_avg", type = int, default = 5, help = "Average over the last N snapshots")
    parser.add_argument("--n_bins", type = int, default = 50)
    parser.add_argument("--n_q", type = int, default = 15)
    parser.add_argument("--no-stamp", action = "store_true",
                        help = "disable timestamp suffix on output filename")
    args = parser.parse_args()

    traj          = load_trajectory(args.input)
    meta          = traj["metadata"]
    species_ids   = traj["species_ids"]
    box_length    = float(meta["box_length"])
    species_names = list(meta.get("species_names", ["A"]))
    n_species     = len(species_names)

    positions_all = traj["positions"]                   # (n_snap, N, 3)
    n_snap        = positions_all.shape[0]
    n_avg         = min(args.n_avg, n_snap)
    frames        = positions_all[-n_avg:]              # average over last n_avg frames 

    # ------- g(r) for each unique species -------
    pairs  = [(i, j) for i in range(n_species) for j in range(i, n_species)]
    gr_results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (a, b) in pairs: 
        g_accum = None
        r_centers = None
        for pos in frames:
            r_centers, g = compute_rdf(
                pos, species_ids, box_length, species_a = a, 
                species_b = b, n_bins = args.n_bins,)
            g_accum = g if g_accum is None else g_accum + g

        assert g_accum is not None and r_centers is not None
        g_avg = g_accum / n_avg
        gr_results[(a, b)] = (r_centers, g_avg)

    # ------ S(q) averaged -------
    q_centers = None
    S_accum   = None
    for pos in frames:
        q_centers, S  = compute_structure_factor(pos, box_length, n_q = args.n_q)
        S_accum = S if S_accum is None else S_accum + S
    S_avg = S_accum / n_avg

    # --------- Plot --------------
    fig, axes = plt.subplots(1, 2, figsize = (14, 9))
    fig.suptitle(f"Structure - {Path(args.input).name} (avg over last {n_avg} frames)", 
                 fontsize = 14)
    
    # g(r) 
    ax = axes[0]
    for (a, b) , (r, g) in gr_results.items():
        label = f"{species_names[a]} - {species_names[b]}"
        ax.plot(r, g, "o-", label = label, lw = 1.5, markersize = 4)
    ax.axhline(1.0, color = "k", lw = 0.5, linestyle = "--")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    ax.legend()
    ax.grid(alpha = 0.3)

    # S(q)
    ax = axes[1]
    ax.plot(q_centers, S_avg, "o-", color = "C1", lw = 1.5, markersize = 4)
    ax.axhline(1.0, color = "k", lw = 0.5, alpha = 0.5, linestyle = "--")
    ax.set_xlabel("|q|")
    ax.set_ylabel("S(q)")
    ax.set_title("Static structure factor")
    ax.grid(alpha = 0.3)

    plt.tight_layout()

    # save plots
    # out_path = stamped_path(args.output)
    out_path = Path(args.output) if args.no_stamp else stamped_path(args.output)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 200)
    print(f"Saved figures of Structures to {out_path}")


if __name__ == "__main__":
    main()











