

import argparse
from pathlib import Path

from src_cg_bd_sim.init_system import create_initial_state
from src_cg_bd_sim.config import load_config
from src_cg_bd_sim.simulation import Simulation
from src_cg_bd_sim.io import save_trajectory
from src_cg_bd_sim.utils import stamped_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True)
    parser.add_argument("--output", default = "outputs/trajectory.h5", help = "Path to HDF5 output file")
    parser.add_argument("--format", choices = ["sparse", "dense"], default = "sparse",
                        help = "bound_pairs storage: sparse (flat edges) or dense (adjacency)")
    parser.add_argument("--no-stamp", action = "store_true",
                        help = "disable timestamp suffic on output filename")
    args = parser.parse_args()

    cfg = load_config(args.config)
    state = create_initial_state(cfg = cfg)
    sim = Simulation(cfg = cfg, state = state)
    sim.run()

    # save output to the path
    #out_path = stamped_path(args.output)
    out_path = Path(args.output) if args.no_stamp else stamped_path(args.output)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    save_trajectory(state, cfg, out_path, fmt = args.format)

    print(f"Finished run with {cfg.n_particles} particles.")
    print(f"Saved {len(state.saved_positions)} snapshots to {out_path} (format = {args.format}).")


if __name__ == "__main__":
    main()