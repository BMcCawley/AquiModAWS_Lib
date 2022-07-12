from .aquimod import AquiModAWS
from .cce import cce


def sce(model: AquiModAWS, p, m):
    """
    n: number of dimensions in calibration
    p: number of complexes
    m: number of points in complex

    s: initial sample size
    s = p*m

    1. Run AquiMod for s random points in the parameter space.
    2. Sort s points in order from best objective function value.
    3. Partition points into p complexes of m points.
        - Points partitioned in repeating order [1:a, 2:b, 3:c, 4:a, 5:b, 6:c]
    4. Evolve each complex according to CCE algorithm.
    5. Shuffle complexes.
        - Combine poiints in evolved complexes back into a single population
        - Sort the population in rank order
        - Repartition into complexes as before
    6. Check for convergence criteria
        - Stop if maximum number of trials reached
        - Stop if objective function value not significantly increased
    7. Return to step 3/4.
    """

    # Total sample points
    s = p * m
    # Run AquiMod for all points - returned in order of ObjectiveFunction by default
    df_all_points = model.run(sim_mode="c", num_runs=s)
    # Partition into complexes
    complxes = []
    for i in range(p):
        bool_mask = [(j % p) == i for j in range(len(df_all_points.index))]
        complx = df_all_points.iloc[bool_mask]
        # Call CCE here
