# Small dataclasses for static metadata only.

from dataclasses import dataclass

# Species — static per-type physical properties
@dataclass(frozen = True)  # frozen True means these are immutable once created — correct for static metadata.
class Species:
    name: str
    sigma: float            # particle diameter
    diffusion: float        # diffusion coefficient
    mass: float = 1.0


# AttractionRule — which species attract and how strongly
@dataclass(frozen = True)
class AttractionRule:
    species_a: str
    species_b: str 
    epsilon: float          # attraction strength 
    cutoff: float           # range of attraction


# Reaction Rule - rule-based binding with harmonic bond while bound 
@dataclass(frozen = True)
class ReactionRule:
    species_a: str 
    species_b: str
    k_on: float             # binding rate (per unit time)
    k_off: float            # unbinding rate (per unit time)
    r_bind: float           # max distance for binding attempt
    k_bond: float           # harmonic spring constant while bound 
    r_bond: float           # equilibrium bond length 
    max_partners: int = 1   # 0 = unlimited, else cap on bonds per particle
