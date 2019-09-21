# from metapopulationModel import latineHypercube_selectedParameters as latineHypercube
from metapopulationModel import latineHypercube as latineHypercube

lh = latineHypercube.LatineHypercubeHandler("Carabus depressus")
lh.launchSampling()

# lh = latineHypercube.LatineHypercubeHandler("Pterostichus flavofemoratus")
# lh.launchSampling()