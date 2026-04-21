# Run loop: neighbor list -> forces -> reactions -> Brownian step -> snapshot.

import numpy as np 
from .config import SimulationConfig
from .state import SimulationState
from .integrator import brownian_step
from .neighbors import find_neighbor_pairs, compute_displacement_vectors
from .potentials import (
    harmonic_repulsion,
    square_well_attraction,
    harmonic_bond_force,
    pair_energy_repulsion,
    pair_energy_attraction,
    pair_energy_bond,
)
from .types import AttractionRule, ReactionRule
from .reactions import attempt_binding, attempt_unbinding


class Simulation:
    def __init__(self, cfg: SimulationConfig, state: SimulationState):
        self.cfg    = cfg
        self.state  = state
        self.rng    = np.random.default_rng(cfg.seed)

        # species name list and attractive rules 
        self.species_names    = [s["name"] for s in cfg.species] if cfg.species else ["A"]
        self.attraction_rules = [
            AttractionRule(
                species_a   = r["species_a"],
                species_b   = r["species_b"],
                epsilon     = r["epsilon"],
                cutoff      = r["cutoff"],
            )
            for r in cfg.attraction_rules
        ]

        #  species name list and list for reaction rules 
        self.reaction_rules = [
            ReactionRule(
                species_a     = r["species_a"],
                species_b     = r["species_b"],
                k_on          = r["k_on"],
                k_off         = r["k_off"],
                r_bind        = r["r_bind"],
                k_bond        = r["k_bond"],
                r_bond        = r["r_bond"],
                max_partners  = r.get("max_partners", 1),
            )
            for r in cfg.reactions
        ]

    def run(self) -> None:
        for step in range(self.cfg.n_steps):
            # neighbor cutoff must cover the longest interaction range
            cutoff = max(
                self.cfg.sigma * 2.0,
                *[r.cutoff for r in self.attraction_rules] or [0.0],
                *[r.r_bind for r in self.reaction_rules] or [0.0],
            )
            pairs = find_neighbor_pairs(
                positions  = self.state.positions,
                cutoff     = cutoff,           
                box_length = self.cfg.box_length,
            )
            
            forces = np.zeros_like(self.state.positions)
            deltas = None

            if len(pairs) > 0: 
                deltas = compute_displacement_vectors(self.state.positions, pairs, self.cfg.box_length)

                forces = harmonic_repulsion(positions = self.state.positions, 
                                            pairs     = pairs, deltas = deltas, 
                                            sigma     = self.cfg.sigma, 
                                            k_rep     = self.cfg.k_rep,)
                
                if self.attraction_rules:
                    forces += square_well_attraction(
                                positions     = self.state.positions,
                                species_ids   = self.state.species_ids,
                                pairs = pairs, deltas = deltas,
                                rules         = self.attraction_rules,
                                species_names = self.species_names,
                    )
            # bond forces run over previously bound pairs, independent of the
            # neighbor list — a stretched bond outside the cutoff still pulls.
            for rule in self.reaction_rules:
                forces += harmonic_bond_force(
                    positions    = self.state.positions,
                    bound_pairs  = self.state.bound_pairs,
                    box_length   = self.cfg.box_length,
                    k_bond       = rule.k_bond,
                    r_bond       = rule.r_bond,
                )   # if multiple reaction rules with different k_bond/r_bond exist,
                    # this applies them all to every bound pair

            # reaction updates (use current pairs + deltas for binding)
            if self.reaction_rules:
                if len(pairs) > 0 and deltas is not None:
                    attempt_binding(
                        state           = self.state,
                        pairs           = pairs,
                        deltas          = deltas,
                        rules           = self.reaction_rules,
                        species_names   = self.species_names,
                        dt              = self.cfg.dt,
                        rng             = self.rng
                        )
                    
                attempt_unbinding(
                    state           = self.state,
                    rules           = self.reaction_rules,
                    species_names    = self.species_names,
                    dt              = self.cfg.dt,
                    rng             = self.rng 
                    )

            brownian_step(
                positions  = self.state.positions,
                diffusion  = self.cfg.diffusion,
                dt         = self.cfg.dt,
                box_length = self.cfg.box_length,
                rng        = self.rng,
                forces     = forces,
                kT         = self.cfg.kT
                )
            
            if step % self.cfg.save_entry == 0:
                time = step * self.cfg.dt
                self.state.times.append(time)
                self.state.saved_positions.append(self.state.positions.copy())
                self.state.saved_bound_pairs.append(set(self.state.bound_pairs))
                
                # compute total potential energy 
                E_rep = (pair_energy_repulsion(pairs, deltas, self.cfg.sigma, self.cfg.k_rep)
                    if len(pairs) > 0 and deltas is not None
                    else 0.0)
                
                E_att = (pair_energy_attraction(self.state.species_ids, pairs, deltas,
                                               self.attraction_rules, self.species_names)
                            if len(pairs) > 0 and deltas is not None and self.attraction_rules else 0.0)
                E_bond = sum(
                    pair_energy_bond(self.state.positions, self.state.bound_pairs,
                                     self.cfg.box_length, rule.k_bond, rule.r_bond)
                                     for rule in self.reaction_rules
                            )
                self.state.saved_energies.append(float(E_rep + E_att + E_bond))
                
                    
