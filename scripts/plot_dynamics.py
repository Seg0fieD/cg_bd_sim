# Time-series observables : MSD, MSD split by bound/free, bound fraction,
# position autocorrelation. All share time on X-axis.

import argparse
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np

from src_cg_bd_sim.io import load_trajectory
from src_cg_bd_sim.observables import (
    compute_msd,
    compute_msd_split_bound_free,
    compute_bound_fraction,
    compute_position_autocorrelation,
    unwrap_positions,
)
from src_cg_bd_sim.utils import stamped_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required = True, help = "Path to HDF5 trajectory")
    parser.add_argument("--output", default = "figures/dynamics.png")
    parser.add_argument("--no-stamp", action = "store_true",
                        help = "disable timestamp suffix on output filename")
    args = parser.parse_args()

    traj            = load_trajectory(args.input)
    meta            = traj["metadata"]
    times           = traj["times"]
    saved_positions = [traj["positions"][t] for t in range(traj["positions"].shape[0])]
    initial         = traj["initial_positions"]
    energies        = traj.get("energies", np.zeros(0))
    box_length      = float(meta["box_length"])

    # unwrap for clean MSD
    saved_positions_unwrapped = unwrap_positions(saved_positions, box_length)
    saved_bound_pairs = traj["saved_bound_pairs"]
    n_particles = int(meta["n_particles"])
    D = float(meta["diffusion"])

    # ---- compute ----
    msd = compute_msd(saved_positions_unwrapped, initial)
    msd_b, msd_f = compute_msd_split_bound_free(saved_positions_unwrapped, saved_bound_pairs, initial, n_particles)
    frac = compute_bound_fraction(saved_bound_pairs, n_particles)
    C = compute_position_autocorrelation(saved_positions)

    # ---- plot -----
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f"Dynamics - {Path(args.input).name}", fontsize = 14)

    # MSD vs 6Dt reference
    ax = axes[0, 0]
    ax.plot(times, msd, "o-", label="simulation", lw=1.5)
    ax.plot(times, 6.0 * D * times, "--", color="k", label=r"$6Dt$ (free)")
    ax.set_xlabel("time")
    ax.set_ylabel("MSD")
    ax.set_title("Mean squared displacement")
    ax.legend()
    ax.grid(alpha=0.3)

    # MSD split bound / free
    ax = axes[0, 1]
    ax.plot(times, msd_b, "o-", label="bound", color="C3")
    ax.plot(times, msd_f, "s-", label="free", color="C0")
    ax.plot(times, 6.0 * D * times, "--", color="k", alpha=0.5, label=r"$6Dt$")
    ax.set_xlabel("time")
    ax.set_ylabel("MSD")
    ax.set_title("MSD split: bound vs free")
    ax.legend()
    ax.grid(alpha=0.3)

    # Bound fraction
    ax = axes[1, 0]
    ax.plot(times, frac, "o-", color="C2")
    ax.set_xlabel("time")
    ax.set_ylabel("fraction bound")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Bound fraction over time")
    ax.grid(alpha=0.3)

    # Position autocorrelation
    ax = axes[1,1]
    ax.plot(times, C, "o-", color = "C4")
    ax.axhline(0, color = "k", lw = 0.5)
    ax.set_xlabel("times")
    ax.set_ylabel("C(t)")
    ax.set_title("Position autocorrelation (drift-subtracted)")
    ax.grid(alpha = 0.3)

    # Total potential energy 
    ax = axes[0, 2]
    if energies.size > 0 and energies.size == len(times):
        ax.plot(times, energies, "o-", color = "C5")
        ax.set_xlabel("time")
        ax.set_ylabel("U (total potential)")
        ax.set_title("Total potential energy")
        ax.grid(alpha = 0.3)
    else:
        ax.text(0.5, 0.5, "no energy data", ha = "center", va = "center", 
                transform = ax.transAxes)
        ax.set_title("Total potential energy")
    # hide the empty slot
    axes[1, 2].axis("off")

    plt.tight_layout()

    # save figure
    # out_path = stamped_path(args.output)
    out_path = Path(args.output) if args.no_stamp else stamped_path(args.output)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 200)
    print(f"saved figure to {out_path}")



if __name__ == "__main__":
    main()







