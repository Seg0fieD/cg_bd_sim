# Snakefile - single-run pipeline: config -> trajectory -> 3 plots
# Usage: snakemake -j4 --config config=default
#        snakemake -j4 --config config=test_small

configfile_name = config.get("config", "default")

rule all:
    input:
        f"figures/{configfile_name}/dynamics.png",
        f"figures/{configfile_name}/structure.png",
        f"figures/{configfile_name}/clusters.png",

rule run_sim:
    input:
        cfg = "configs/{name}.yaml",
    output:
        traj = "outputs/{name}/trajectory.h5",
    shell:
        "PYTHONPATH=. python scripts/run_sim.py "
        "--config {input.cfg} --output {output.traj} "
        "--format sparse --no-stamp"

rule plot_dynamics:
    input:
        traj = "outputs/{name}/trajectory.h5",
    output:
        fig = "figures/{name}/dynamics.png",
    shell:
        "PYTHONPATH=. python scripts/plot_dynamics.py "
        "--input {input.traj} --output {output.fig} --no-stamp"

rule plot_structure:
    input:
        traj = "outputs/{name}/trajectory.h5",
    output:
        fig = "figures/{name}/structure.png",
    shell:
        "PYTHONPATH=. python scripts/plot_structure.py "
        "--input {input.traj} --output {output.fig} --no-stamp"

rule plot_clusters:
    input:
        traj = "outputs/{name}/trajectory.h5",
    output:
        fig = "figures/{name}/clusters.png",
    shell:
        "PYTHONPATH=. python scripts/plot_clusters.py "
        "--input {input.traj} --output {output.fig} --no-stamp"