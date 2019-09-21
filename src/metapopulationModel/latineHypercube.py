from metapopulationModel import metapopulation_Pomp_optimized as metapop

from pyDOE import *
import numpy as np
from metapopulationModel import setup as setup
from datetime import datetime

class LatineHypercubeHandler():
	def __init__(self, species):
		self.species = species
		#todo: save and load
		# self.nb_sample_points = 45
		self.nb_sample_points = 65
		self.sampleRatio = False
		nbSampleAxis = 14
		if self.sampleRatio:
			nbSampleAxis = 15
		
		# set seed to get same parameters to be able to split
		np.random.seed(19890905)
		self.samples = lhs(nbSampleAxis, samples = self.nb_sample_points, criterion = 'center')

		#* c
		#pterosticus
		#self.samples[:,0] = np.interp(self.samples[:,0], (0, 1), (250, 400))
		#carabus
		self.samples[:,0] = np.interp(self.samples[:,0], (0, 1), (600, 1200))
		#* e
		#pterosticus
		# self.samples[:,1] = np.interp(self.samples[:,1], (0, 1), (0.0015, 0.0025))
		#carabus
		self.samples[:,1] = np.interp(self.samples[:,1], (0, 1), (0.005, 0.012))
		#* D
		self.samples[:,2] = np.interp(self.samples[:,2], (0, 1), (170, 225))
		#* eo_0
		self.samples[:,3] = np.interp(self.samples[:,3], (0, 1), (-.5, 3.5))
		#* eo_1 - temperature
		self.samples[:,4] = np.interp(self.samples[:,4], (0, 1), (-2, 2))
		#* eo_2 - wetness
		self.samples[:,5] = np.interp(self.samples[:,5], (0, 1), (-.5, 4))
		#* eo_3 - brightness
		self.samples[:,6] = np.interp(self.samples[:,6], (0, 1), (-3.5, 3.5))
		#* eo_4 - greenness
		self.samples[:,7] = np.interp(self.samples[:,7], (0, 1), (-4, 3))
		#* eo_5 - eastness
		self.samples[:,8] = np.interp(self.samples[:,8], (0, 1), (-2.5, 4))
		#* eo_6 - northness
		self.samples[:,9] = np.interp(self.samples[:,9], (0, 1), (-4, 4))
		#* eo_7 - slope
		self.samples[:,10] = np.interp(self.samples[:,10], (0, 1), (-2, 2))
		#* eo_8 - forest_presence
		self.samples[:,11] = np.interp(self.samples[:,11], (0, 1), (-2, 2))
		#* p_FP
		self.samples[:,12] = np.interp(self.samples[:,12], (0, 1), (0.005, 0.01))
		#* p_TP
		self.samples[:,13] = np.interp(self.samples[:,12], (0, 1), (.6, 0.7))
		if self.sampleRatio:
			#* ratio
			self.samples[:,14] = np.interp(self.samples[:,14], (0, 1), (1, 100))

	def launchSampling(self):
		gpnp_setup = setup.GPNP_Setup_Pomp(self.species) 
		timeStamp = str(datetime.now().strftime("%Y_%m_%d_%H_%M"))
		for sample in range(4,self.nb_sample_points,5):#range(self.nb_sample_points):
			print("starting sampling: " + str(sample) + " of " + str(self.nb_sample_points))
			vector = np.copy(np.squeeze(self.samples[sample,:]))
			if self.sampleRatio:
				ratio = self.samples[sample,-1]
				vector = np.copy(np.squeeze(self.samples[sample,0:-1]))
				vector[0:2] /= ratio
				gpnp_setup.ratio = ratio
			

			print("with parameters: " + str(vector))
			
			gpnp_setup.parameters.from_vector(vector)
			Metapop = metapop.Metapopulation()
			Metapop.setupFromSetup(gpnp_setup)
			Metapop.setLH_num(sample)
			Metapop.setTimeStamp(timeStamp)
			Metapop.setPlotProgressOption()
			Metapop.setDisplayPlots()
			# Metapop.setPlotBurninTrajectory()
			Metapop.setPlotPresenceMaps()
			# Metapop.setPlotInitialMaps()
			# Metapop.setPlotPairPlot()
			# Metapop.setPlotParticleMovement()
			Metapop.setParallel()
			
			Metapop.performPomp()
			print("done with sample: " + str(sample))



