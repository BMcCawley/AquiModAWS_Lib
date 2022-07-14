from .aquimod import AquiModAWS
from .cce import cce


def sce(model: AquiModAWS, p: int, m: int, q: int):
    """
    n: number of dimensions (parameters/degrees of freedom) to calibrate
    p: number of complexes [p >= 1]
    m: number of points in each complex [m >= n+1]
    q: number of points in simplex [2 <= q <= m]

    s: initial sample size [s = p * m]

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
    # 1. Run AquiMod for all points
    model.run(sim_mode="c", num_runs=s)
    # 2. Get results (returned in order of ObjectiveFunction by default)
    all_points = pd.concat(model.read_performance_output().values(), axis=1)
    # 3. Partition into complexes
    complxes = []
    for i in range(p):
        # Create a boolean mask to select rows of the i-th complex
        bool_mask = [(j % p) == i for j in range(len(all_points.index))]
        complx = all_points.iloc[bool_mask]
        # Call CCE here
        cce(model, complx, q)
