# HDF5 trajectory I/O.
# Bound-pair history has two storage formats:
#   "sparse": flat edge list + per-frame offsets (compact, default)
#   "dense":  (n_snapshots, N, N) boolean adjacency (simple but large)

# pyright: reportGeneralTypeIssues=false

from pathlib import Path
import h5py 
import numpy as np
from .state import SimulationState
from .config import SimulationConfig




def save_trajectory(
        state: SimulationState,
        cfg: SimulationConfig,
        path: str | Path,
        fmt: str = "sparse",
        ) -> None: 
    """ 
    Write trajectory + metadata to HDF5 . 
    
    Parameters 
    ----------
    fmt : "sparse" (flat efges + offsets) or "dense" (NxN adjacency per snapshot)
    """

    if fmt not in ("sparse" , "dense"): 
        raise ValueError(f"fmt must be 'sparse' or 'dense' , got {fmt!r}")
    
    path  = Path(path)
    n_snap = len(state.saved_positions)
    n_particles = cfg.n_particles

    with h5py.File(path, "w") as f: 
        #metadata 
        m = f.create_group("metadata")
        m.attrs["n_particles"]  = cfg.n_particles
        m.attrs["box_length"]   = cfg.box_length
        m.attrs["dt"]           = cfg.dt
        m.attrs["n_steps"]      = cfg.n_steps    
        m.attrs["save_entry"]   = cfg.save_entry        
        m.attrs["diffusion"]    = cfg.diffusion
        m.attrs["sigma"]        = cfg.sigma
        m.attrs["k_rep"]        = cfg.k_rep
        m.attrs["kT"]           = cfg.kT
        m.attrs["seed"]         = cfg.seed
        m.attrs["bound_format"] = fmt
        species_names = [s["name"] for s in cfg.species] if cfg.species else["A"]
        m.attrs["species_names"] = np.array(species_names, dtype="S")
        
        f.create_dataset("times", data = np.array(state.times, dtype = float))        
        f.create_dataset("species_ids", data = state.species_ids.astype(np.int32) )
        f.create_dataset("initial_positions", data = state.initial_positions.astype(np.float32))
        f.create_dataset("energies", data = np.array(state.saved_energies, dtype = float))


        if n_snap > 0:
            stacked = np.stack(state.saved_positions, axis = 0).astype(np.float32)
            f.create_dataset( "positions", data = stacked, compression = "gzip", compression_opts = 4)
        else:
            f.create_dataset("positions", shape = (0, n_particles, 3), dtype = np.float32)


        # bound_pairs history 
        # sparse
        if fmt == "sparse":
            offsets = [0]
            edges_list = []
            for pairs in state.saved_bound_pairs:
                if pairs:
                    edges_list.append(np.array(list(pairs), dtype = np.int32))
                offsets.append(offsets[-1] + len(pairs))
            
            if edges_list:
                edges = np.concatenate(edges_list, axis = 0)
            else:
                edges = np.zeros((0, 2), dtype = np.int32)

            f.create_dataset("bound_edges", data = edges,
                             compression = "gzip", compression_opts = 4)
            f.create_dataset("bound_offsets", 
                             data = np.array(offsets, dtype = np.int64))
        # dense
        else:  
            adj = np.zeros((n_snap, n_particles, n_particles), dtype = bool)
            for t, pairs in enumerate(state.saved_bound_pairs):
                for (i, j) in pairs:
                    adj[t, i, j] = True
                    adj[t, j, i] = True
            
            f.create_dataset("bound_adjacency", data = adj,
                             compression = "gzip", compression_opts = 4)
            

def load_trajectory(path : str | Path) -> dict:
    """
    Read trajectory back into a dict. Returns dict with keys:
        metadata (dict) , times species_ids, initial_position, positions,
        save_bound_pairs (list of set[tuple[int, int]])
    """
    path = Path(path)
    out: dict = {}
    with h5py.File(path, "r") as f:
        m = f["metadata"]
        meta = {k: m.attrs[k] for k in m.attrs.keys()}
        if "species_names" in meta:
            meta["species_names"] = [s.decode() for s in meta["species_names"]]
        out["metadata"]          = meta
        out["times"]             = f["times"][...]        
        out["species_ids"]       = f["species_ids"][...]
        out["initial_positions"] = f["initial_positions"][...]  
        out["positions"]         = f["positions"][...]
        # energies guard keeps older trajectory files (saved before this field) loadable
        out["energies"]          = f["energies"][...] if "energies" in f else np.zeros(0) 
                                                        
        fmt = meta.get("bound_format", "sparse")
        n_snap = out["positions"].shape[0]
        saved_pairs: list[set[tuple[int, int]]] = []

        # sparse
        if fmt == "sparse" or fmt == b"sparse":
            edges   = f["bound_edges"][...]
            offsets = f["bound_offsets"][...]
            for t in range(n_snap):
                block = edges[offsets[t]:offsets[t + 1]]
                saved_pairs.append({(int(i), int(j)) for (i, j) in block})
        # dense
        else:  
            adj = f["bound_adjacency"][...]
            for t in range(n_snap):
                ii, jj = np.where(np.triu(adj[t], k = 1))
                saved_pairs.append({(int(i), int(j)) for i, j in zip(ii, jj)})

        out["saved_bound_pairs"] = saved_pairs

    return out 











