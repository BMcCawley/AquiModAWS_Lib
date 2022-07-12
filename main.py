#%%
from autocal.aquimod import run_aquimod
from autocal.sce import sce
from autocal.cce import cce

df1 = run_aquimod("model", num_runs=10)
df2 = cce(df1, 5)

module_config = {
    "FAO": {
        "FieldCapacity(-)": [0, 1],
        "WiltingPoint(-)": [0, 1],
        "MaxRootDepth(mm)": [0, 1],
        "DepletionFactor(-)": [0, 1],
        "BaseflowIndex(-)": [0, 1],
    },
    "Weibull": {"k(-)": [0, 1], "lambda(-)": [0, 1], "n(timesteps)": [0, 1],},
    "Q3K3S1": {
        "AquiferLength(m)": [0, 1],
        "SpecificYield(%)": [0, 1],
        "K3(m/day)" "K2(m/day)": [0, 1],
        "K1(m/day)": [0, 1],
        "z3(m)": [0, 1],
        "z2(m)": [0, 1],
        "z1(m)": [0, 1],
        "Alpha(-)": [0, 1],
    },
}
