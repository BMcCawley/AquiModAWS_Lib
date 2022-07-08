import os
from pathlib import Path
import pandas as pd


def read_aquimod_data(path):
    """Read tab-separated AquiMod data."""
    return pd.read_csv(path, sep="\t", index_col=False, engine="python")


def aquimod(model_dir, num_runs=None):
    """
    Execute AquiModAWS with chosen directory and number of runs. Returns calibration
    outputs.
    """
    # Delete contents of output folder
    for p in Path(model_dir, "Output").glob("*"):
        os.remove(p)

    # Set number of runs if specfied
    if num_runs is not None:
        input_path = Path(model_dir, "Input.txt")
        with open(input_path, "r") as f:
            lines = f.readlines()
            lines[10] = f"{num_runs}\n"
        with open(input_path, "w") as f:
            f.writelines(lines)

    # Run AquiModAWS
    os.system(f"AquiModAWS {model_dir}")

    # Read outputs
    output_list = [read_aquimod_data(p) for p in Path(model_dir, "Output").glob("*")]
    return pd.concat(output_list, axis=1)

