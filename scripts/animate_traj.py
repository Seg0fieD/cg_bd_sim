# Interactive 3D trajectory viewer (PyVista, GPU-accelerated).
# Opens a native desktop window with time slider, cluster coloring,
# bond overlay, and PBC box wireframe.

import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
from src_cg_bd_sim.io import load_trajectory
from src_cg_bd_sim.observables import _clusters_from_pairs


# ----- tunables ---------------------------------------------------------
PARTICLE_RADIUS = 0.4  # sphere radius (in simulation units)
BOND_RADIUS = 0.08  # tube radius for bond cylinders
MONOMER_COLOR = "#777777"
BG_COLOR = "#101014"
# ------------------------------------------------------------------------


def build_bond_lines(
    positions: np.ndarray,
    pairs: set[tuple[int, int]],
    box_length: float,
) -> pv.PolyData | None:
    """Build a PolyData line set from bound pairs; skip PBC-crossing bonds."""
    if not pairs:
        return None
    verts = []
    lines = []
    for i, j in pairs:
        delta = positions[j] - positions[i]
        if np.any(np.abs(delta) > box_length / 2):
            continue
        n = len(verts)
        verts.append(positions[i])
        verts.append(positions[j])
        lines.extend([2, n, n + 1])
    if not verts:
        return None
    poly = pv.PolyData(np.array(verts))
    poly.lines = np.array(lines)
    return poly


def build_pbc_wireframe(box_length: float) -> pv.PolyData:
    """12-edge wireframe cube of the periodic box."""
    L = box_length
    corners = np.array(
        [
            [0, 0, 0],
            [L, 0, 0],
            [L, L, 0],
            [0, L, 0],
            [0, 0, L],
            [L, 0, L],
            [L, L, L],
            [0, L, L],
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for e in edges:
        lines.extend([2, e[0], e[1]])
    cube = pv.PolyData(corners)
    cube.lines = np.array(lines)
    return cube


def cluster_color_values(
    labels: np.ndarray,
    counts_by_label: dict,
) -> np.ndarray:
    """
    Assign float color values: -1 for monomers (rendered gray),
    positive integers for each cluster (distinct color via colormap).
    """
    return np.array([-1.0 if counts_by_label[l] == 1 else float(l) for l in labels])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to HDF5 trajectory")
    parser.add_argument(
        "--stride", type=int, default=1, help="Use every Nth frame (smoother vs memory)"
    )
    args = parser.parse_args()

    traj = load_trajectory(args.input)
    meta = traj["metadata"]
    positions = traj["positions"][:: args.stride]
    saved_pairs = traj["saved_bound_pairs"][:: args.stride]
    times = traj["times"][:: args.stride]
    species_ids = traj["species_ids"]
    box_length = float(meta["box_length"])
    n_particles = int(meta["n_particles"])
    species_names = list(meta.get("species_names", ["A"]))
    n_frames = positions.shape[0]

    # ----- precompute colors per frame (fast slider) -------------------
    all_color_values = []
    for t in range(n_frames):
        labels = _clusters_from_pairs(saved_pairs[t], n_particles)
        unique, counts = np.unique(labels, return_counts=True)
        size_lookup = dict(zip(unique, counts))
        all_color_values.append(cluster_color_values(labels, size_lookup))

    # ----- plotter ----------------------------------------------------
    pl = pv.Plotter(window_size=(1400, 900))
    pl.set_background(BG_COLOR)
    pl.enable_anti_aliasing("ssaa")

    # PBC box wireframe (static)
    pbc = build_pbc_wireframe(box_length)
    pl.add_mesh(pbc, color="#3a3a44", line_width=1.2, opacity=0.7)

    # Glyph (sphere template for all particles)
    glyph = pv.Sphere(radius=PARTICLE_RADIUS, theta_resolution=18, phi_resolution=18)

    state = {
        "frame": 0,
        "particle_actor": None,
        "bond_actor": None,
    }

    def render_frame(t) -> None:
        t = int(t)
        state["frame"] = t

        # rebuild particle cloud
        cloud = pv.PolyData(positions[t])
        cloud["cluster_id"] = all_color_values[t]
        mesh = cloud.glyph(geom=glyph, scale=False, orient=False)

        if state["particle_actor"] is not None:
            pl.remove_actor(state["particle_actor"])
        state["particle_actor"] = pl.add_mesh(
            mesh,
            scalars="cluster_id",
            cmap="turbo",
            clim=[0, max(1, all_color_values[t].max())],
            nan_color=MONOMER_COLOR,
            below_color=MONOMER_COLOR,
            show_scalar_bar=False,
            smooth_shading=True,
            specular=0.4,
            specular_power=20,
        )

        # rebuild bonds
        if state["bond_actor"] is not None:
            pl.remove_actor(state["bond_actor"])
        bonds = build_bond_lines(positions[t], saved_pairs[t], box_length)
        if bonds is not None:
            tubes = bonds.tube(radius=BOND_RADIUS, n_sides=8)
            state["bond_actor"] = pl.add_mesh(
                tubes, color="#e8e8ee", smooth_shading=True, specular=0.6
            )

        pl.add_text(
            f"frame {t}/{n_frames - 1}   t = {times[t]:.3f}",
            name="hud",
            position="upper_left",
            font_size=11,
            color="#d0d0dc",
        )

    # initial render
    render_frame(0)

    # time slider
    pl.add_slider_widget(
        callback=render_frame,
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.25, 0.08),
        pointb=(0.75, 0.08),
        style="modern",
        color="#d0d0dc",
    )

    # keyboard: ← → = step
    def step(delta: int) -> None:
        nxt = max(0, min(n_frames - 1, state["frame"] + delta))
        render_frame(nxt)

    pl.add_key_event("Right", lambda: step(1))
    pl.add_key_event("Left", lambda: step(-1))

    # axes gizmo + camera
    pl.add_axes(line_width=2, color="#d0d0dc")
    pl.camera.focal_point = (box_length / 2,) * 3
    pl.camera.position = (box_length * 2.2, box_length * 2.2, box_length * 1.8)
    pl.camera.up = (0, 0, 1)

    pl.add_text(
        f"{Path(args.input).name}  |  N={n_particles}  |  species={','.join(species_names)}",
        position="lower_left",
        font_size=10,
        color="#888896",
    )
    pl.add_text(
        "← → step   drag to rotate   scroll to zoom",
        position="lower_right",
        font_size=9,
        color="#666676",
    )

    pl.show()


if __name__ == "__main__":
    main()
