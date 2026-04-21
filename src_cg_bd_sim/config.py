from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class SimulationConfig:
    n_particles: int
    dt: float         # timestep
    n_steps: int
    diffusion: float 
    box_length: float 
    save_entry: int 
    seed: int 
    sigma: float                                                    # particle diameter
    k_rep: float                                                    #repulsion spring constant
    kT:float                                                        #thermal energy 
    species: list[dict] = field(default_factory=list)               # list of species definitions from YAML
    attraction_rules: list[dict] = field(default_factory=list)      # list of attraction rules from YAML
    reactions: list[dict] = field(default_factory=list)
    


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding = "utf-8") as f:
        data = yaml.safe_load(f)

    return SimulationConfig(
        n_particles         = data["simulation"]["n_particles"],
        dt                  = data["simulation"]["dt"],
        n_steps             = data["simulation"]["n_steps"],
        diffusion           = data["simulation"]["diffusion"],
        box_length          = data["box"]["length"],
        save_entry          = data["output"]["save_entry"],
        seed                = data["simulation"]["seed"],
        sigma               = data["particles"]["sigma"],
        k_rep               = data["particles"]["k_rep"],
        kT                  = data["particles"].get("kT", 1.0),  # yml -> get  kT get value, No value -> 1.0 by default
        species             = data.get("species", []),
        attraction_rules    = data.get("attractions", []),
        reactions           = data.get("reactions", []),
        
    )
