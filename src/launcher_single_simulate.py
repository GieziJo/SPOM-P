# #%%
from importlib import reload
from metapopulationModel import setup as setup
from datetime import datetime
import numpy as np
from metapopulationModel import metapopulation_Pomp_optimized as metapop

species = "Pterostichus flavofemoratus"
species = "Carabus depressus"

gpnp_setup = setup.GPNP_Setup_Pomp(species)

##Pterostichus flavofemoratus

# vector = np.array([
# 	#c
# 	187.62337718,
# 	#e
# 	0.00153354,
# 	#D
# 	189.91084922,
# 	#eo
# 	-1.75970967,
# 	3.5788129,
# 	3.21953745,
# 	0.42051246,
# 	1.54163118,
# 	-2.4460815,
# 	-2.54605306,
# 	-2.68337365,
# 	3.81966495,
# 	#p_FP
# 	0.01,
# 	#p_TP
# 	0.78597096
# ])



##Carabus depressus

vector = np.array([
	#c
	7.33067033e+02,
	#e
	1.20556571e-02,
	#D
	1.73002180e+02,
	#eo
	3.44926054e+00,
	-5.25146380e-01,
	-2.82423197e-01,
	8.06332804e-01,
	-3.61304232e+00,
	-2.68104299e+00,
	-3.99929653e+00,
	1.43597809e+00,
	-8.13614035e-02,
	#p_FP
	1.00000000e-02,
	#p_TP
	6.31428998e-01
])

gpnp_setup.parameters.from_vector(vector)

timeStamp = str(datetime.now().strftime("%Y_%m_%d_%H_%M"))

# #%%
# reload(metapop)
Metapop = metapop.Metapopulation()
Metapop.setSaveProgressOption()
Metapop.setupFromSetup(gpnp_setup)
Metapop.setTimeStamp(timeStamp)
# Metapop.setPlotProgressOption()
Metapop.setDisplayPlots()
# Metapop.setPlotBurninTrajectory()
Metapop.setPlotPresenceMaps()
# Metapop.setPlotInitialMaps()
# Metapop.setPlotPairPlot()
# Metapop.setPlotParticleMovement()
Metapop.setParallel()

Metapop.setPresenceMapFromFile("/home/giezenda/Documents/metadata/data/precomputed/" + species.replace(" ", "_") + "_180/current/initial_presence.obj")

# #%%

Metapop.simluateCalibrated()


# import pprofile
# profiler = pprofile.Profile()
# with profiler:
# 	Metapop.performPomp()

# profiler.print_stats( )