from pathlib import Path

import numpy as np
import pandas as pd

from .aquimod import AquiModAWS, read_parameter_lims, run_aquimod, write_eval_file


def cce(
    model: AquiModAWS,
    complx: pd.DataFrame,
    q: int,
    alpha=1,
    beta=None,
):
    """
    complx: complex
    q: number of points in simplex
    m: number of points in complex
    alpha: user-defined number of evolution iterations per simplex
    beta:

    1. Assign weights to each point in the complex
    2. Randomly select weighted simplex points
    3. Reorder simplex by best performing points
    4. Compute the centroid of the simplex
    5. Reflect the worst performing point through the centroid
    6. Check that the reflected point is within the parameter space:
        - If true, go to 8
        - If false, go to 7
    7. Generate random point within the parameter space (mutation)
    8. Run AquiMod for new point (either mutated or reflected) to get NSE
    9. Check that new point performs better than previous worst point:
        - If true, go to 15
        - If false, go to 10
    10. Contract the worst performing point towards the centroid
    11. Run AquiMod for contracted point to get NSE
    12. Check that contracted point performs better than previous worst point:
        - If true, go to 15
        - If false, go to 13
    13. Generate random point within the parameter space (mutation)
    14. Run AquiMod with the mutated point
    15. Replace worst performing point with new point
    16. Repeat steps 1 - 15 alpha times where alpha >= 1
    """
    m = len(complx)
    complx = complx.reset_index(drop=True)
    # Calculate triangular distribution probability
    complx["weight"] = (2 * (m + 1 + (-complx.index))) / (m * (m + 1))
    # Normalise weights so that their sum == 1
    complx["weight"] /= complx["weight"].sum()
    simplx = complx.loc[
        np.random.choice(complx.index, q, replace=False, p=complx["weight"])
    ]
    # Restore order back to simplx
    simplx = simplx.sort_by("ObjectiveFunction", ascending=False)

    # Compute the centroid of the simplex
    centroid = simplx.mean().to_frame().T

    # Drop the weight and objective function value of the centroid
    centroid = centroid.drop(["ObjectiveFunction", "weight"], axis=1)

    # Perform reflection step of the worst performing point through centroid
    reflected = (2 * centroid) - simplx.iloc[-1]

    # Check that the reflected point is still within parameter space
    # Get parameter bounds
    # TODO rewrite this section using updated AquiModAWS API
    parameters = {}
    for dct in module_config.values():
        parameters |= dct
    # Check that reflected parameter values are within bounds
    within_parameter_space = True
    for col in reflected.columns:
        min, max = parameters[col]
        if not min <= reflected[col].values[0] <= max:
            within_parameter_space = False

    # If reflected point outside parameter space, perform mutation instead
    if not within_parameter_space:
        # Mutation performed but labelled as reflection
        reflected = run_aquimod(model_dir, sim_mode="c", num_runs=1)
    else:
        # Run AquiMod using the reflected point
        # AquiMod must be run in evaluation mode for this
        # Evaluation files need to be written for each module
        df_dict = {
            module: reflected[params] for module, params in module_config.items()
        }
        write_eval_file(df_dict)
        reflected = run_aquimod(model_dir, sim_mode="e", num_runs=1)

    return simplx
