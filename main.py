from autocal.aquimod_temp import AquiModAWS
from autocal.cce import cce
from autocal.sce import sce

model = AquiModAWS("model")
sce(model, p=2, m=50)
