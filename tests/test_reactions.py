# we shall test the reactions here 
import numpy as np 
import pytest 
from src_cg_bd_sim.config import load_config
from src_cg_bd_sim.init_system import create_initial_state
from src_cg_bd_sim.simulation import Simulation
from src_cg_bd_sim.state import SimulationState
from src_cg_bd_sim.types import ReactionRule
from src_cg_bd_sim.reactions import attempt_binding, attempt_unbinding

CONFIG = "configs/test_small.yaml"

def test_binding_occurs_when_close():
    """ Two compatible particles within r_bind should bind with high k_on."""

    state = SimulationState(
        positions           = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        initial_positions   = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        times               = [],
        saved_positions     = [],
        species_ids         = np.array([0, 1]),
        bound_pairs         = set(),
    )

    rules   = [ReactionRule(species_a = "A", species_b = "B", k_on = 1e6, k_off = 0.0, r_bind = 1.5,
                          k_bond = 200.0, r_bond = 1.0)]
    pairs   = np.array([[0, 1]])
    deltas  = np.array([[1.0, 0.0, 0.0]])
    rng     = np.random.default_rng(0)

    attempt_binding(state, pairs, deltas, rules, ["A", "B"], dt = 1e-3, rng = rng)
    assert (0, 1) in state.bound_pairs


def test_no_binding_beyond_r_bind():
    """Pairs farther than r_bind must not bind even with huge k_on."""
    state = SimulationState(
        positions          = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        initial_positions  = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        times              = [],
        saved_positions    = [],
        species_ids        = np.array([0, 1]),
        bound_pairs        = set(),
    )
    rules = [
        ReactionRule(
            species_a = "A",
            species_b = "B",
            k_on      = 1e6,
            k_off     = 0.0,
            r_bind    = 1.5,
            k_bond    = 200.0,
            r_bond    = 1.0,
        )]

    pairs  = np.array([[0, 1]])
    deltas = np.array([[5.0, 0.0, 0.0]])
    rng    = np.random.default_rng(0)
    
    attempt_binding(state, pairs, deltas, rules, ["A", "B"], dt=1e-3, rng=rng)
    assert len(state.bound_pairs) == 0


def test_no_binding_same_species():
    """ Two A particle should not bind under an A-B rule."""
    state = SimulationState(
        positions         = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        initial_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        times             = [],
        saved_positions   = [],
        species_ids       = np.array([0, 0]),
        bound_pairs       = set(),
    )

    rules  = [ReactionRule("A", "B", k_on = 1e6, k_off = 0.0, r_bind = 1.5,
                          k_bond = 200.0, r_bond = 1.0)]
    pairs  = np.array([[0, 1]])
    deltas = np.array([[1.0, 0.0, 0.0]])
    rng    = np.random.default_rng(0)

    attempt_binding(state, pairs, deltas, rules, ["A", "B"], dt=1e-3, rng=rng)
    assert len(state.bound_pairs) == 0

def test_unbinding_occurs_with_high_k_off():
    """ Bound pair with huge k_off should unbind immediately."""
    state = SimulationState(
        positions         = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        initial_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        times             = [],
        saved_positions   = [],
        species_ids       = np.array([0, 1]),
        bound_pairs       = {(0, 1)},
    )

    rules = [ReactionRule("A", "B", k_on = 0.0, k_off = 1e6, r_bind = 1.5,
                          k_bond = 200.0, r_bond = 1.0)]
    rng   = np.random.default_rng(0)

    attempt_unbinding(state, rules, ["A", "B"], dt = 1e-3, rng = rng)
    assert len(state.bound_pairs) == 0

def test_one_partner_binding():
    """ A particle already bound cannot bind to a second partner in the same step."""
    state = SimulationState(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        initial_positions=np.zeros((3, 3)),
        times=[],
        saved_positions=[],
        species_ids=np.array([0, 1, 1]),  # A, B, B
        bound_pairs=set(),
    )
    rules  = [ReactionRule("A", "B", k_on=1e6, k_off=0.0, r_bind=1.5, k_bond=200.0, r_bond=1.0)]
    pairs  = np.array([[0, 1], [0, 2]])
    deltas = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rng    = np.random.default_rng(0)

    attempt_binding(state, pairs, deltas, rules, ["A", "B"], dt = 1e-3, rng = rng)
    # only the close pair (0, 1) can bind; (0, 2) is outside r_bind anyway,
    # but even if both were in range, particle 0 would lock after first bind.
    assert len(state.bound_pairs) == 1
    assert (0, 1) in state.bound_pairs

def test_bound_fraction_depends_on_rates():
    """ High k_on / low k_off => more bound pairs low k_on / high k_off. """
    cfg = load_config(CONFIG)

    # strong binding 
    cfg.reactions = [{
            "species_a": "A","species_b": "B",
            "k_on": 100.0, "k_off": 0.1,
            "r_bind": 1.5, "k_bond": 200.0, "r_bond": 1.0,}]
    state_hi = create_initial_state(cfg)
    Simulation(cfg, state = state_hi).run()

    # weak binding
    cfg.reactions = [{"species_a": "A", "species_b": "B",
                      "k_on": 0.1, "k_off": 100.0,
                      "r_bind": 1.5, "k_bond": 200.0, "r_bond": 1.0}]
    state_lo = create_initial_state(cfg)
    Simulation(cfg, state= state_lo).run()

    assert len(state_hi.bound_pairs) > len(state_lo.bound_pairs)


def test_full_simulation_runs_with_reactions():
    """ Smoke test: full sim with reactions block runs without error. """
    cfg = load_config(CONFIG)
    state = create_initial_state(cfg)
    Simulation(cfg, state).run()
    assert np.all(np.isfinite(state.positions))

def test_multiple_partners_allowed():
    """ With max_partners = 2, particle 0 can bind to both 1 and 2."""
    state = SimulationState(
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        initial_positions = np.zeros((3, 3)),
        times = [],
        saved_positions =  [],
        species_ids = np.array([0, 1, 1]),
        bound_pairs = set(), 
        )
    rules = [ReactionRule("A", "B", k_on = 1e6, k_off = 0.0, r_bind = 1.5,
                          k_bond = 200.0, r_bond = 1.0, max_partners = 2)]
    pairs = np.array([[0, 1], [0, 2]])
    deltas = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rng = np.random.default_rng(0)
    attempt_binding(state, pairs, deltas, rules, ["A", "B"], dt = 1e-3, rng = rng)
    assert len(state.bound_pairs) == 2