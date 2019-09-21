import numpy as np
import pandas as pd
from numba import jit, njit, prange
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import special
import time
import copy
from operator import itemgetter
from joblib import Parallel, delayed
import multiprocessing
import seaborn as sns

import os
import pickle


class Metapopulation:
	"""
	Create a new object of class Metapopulation
	"""

	def __init__(self):
		self.totalSteps = 4
		self.lh_num = 0
		self.displayPlots = True
		self.resampleParamBool = True
		self.postCalibrationPhase = False
		self.plotProgress = False
		self.saveProgress = False
		self.plotBurnin = False
		self.plotLikelihoodScatter = False
		self.plotPresenceMaps = False
		self.plotInitialMaps = False
		self.plotPairPlot = False
		self.plotParticleMovement = False
		self.setParallel(False, False)
		self.lastPassage = False
		self.sample_p_FP = True
		matplotlib.use('Agg')
		
		self.calibrate = True
		self.simulate = False

	def setupFromSetup(self, setupData):
		self.species = setupData.species

		self.setInsituMap(
			np.copy(setupData.metapopSpeciesData.observed),
			np.copy(setupData.metapopSpeciesData.sampled),
			np.copy(setupData.metapopSpeciesData.time)
		)

		self.setEOData(np.copy(setupData.metapopEOData.EOMatrix), dx=setupData.res, dy=setupData.res)
		self.res = setupData.res
		self.setSimulationConstants(ratio=setupData.ratio)
		self.setParams(setupData.parameters)
		self.setPresenceMapFromProbability(np.copy(setupData.initialPresence))

		self.__initialiseFitness()

		self.__computeDispersalDistance()

		self.__initialiseRandomVector()
	
	def setLH_num(self, lh_num):
		self.lh_num = lh_num
	
	def setTimeStamp(self, timeStamp):
		self.timeStamp = timeStamp


	def setInsituMap(self, Observed, Sampled, insituTimes):
		self.insituTimes = insituTimes - insituTimes[0]
		# self.Observed = Observed
		# self.Sampled = Sampled
		self.n = [None] * len(self.insituTimes)
		self.k = [None] * len(self.insituTimes)
		self.binom = [None] * len(self.insituTimes)
		self.samplingSites = [None] * len(self.insituTimes)
		for t_insitu in range(len(self.insituTimes)):
			self.samplingSites[t_insitu] = ~np.isnan(Sampled[:, :, t_insitu])
			self.n[t_insitu] = Sampled[self.samplingSites[t_insitu], t_insitu].flatten()
			self.k[t_insitu] = Observed[self.samplingSites[t_insitu], t_insitu].flatten()
			self.binom[t_insitu] = special.binom(self.n[t_insitu], self.k[t_insitu])

	def setEOData(self, EOMatrix, dx=90, dy=90):
		"""
		Set the EO matrix for the simulation
		:param EOMatrix: the Earth Observation Matrix of dimension 0:#Rows, 1:#Cols, 2:#TimeSteps, 3:#params.eo - 1
		"""
		self.dx = dx
		self.dy = dy
		self.mapDimensions = EOMatrix.shape[0:2]
		self.EOMatrix = [None] * EOMatrix.shape[2]
		for t in range(EOMatrix.shape[2]):
			self.EOMatrix[t] = EOMatrix[:,:,t,:].reshape(-1,EOMatrix.shape[3])
		self.EOShape = EOMatrix.shape[0:3]


	##Carabus depresus
	def setSimulationConstants(self, burn_in_iter = 120, burninLimitLikelihood = .9, burn_in_iter_children = 15, nb_particles=480, nb_iterations=30, nb_iterationsCalibrated=15, sigma = .02, alpha = 0.1, ratio = 1, i_steps = 5):
	## pterosticus flavemoratus
	# def setSimulationConstants(self, burn_in_iter = 400, burninLimitLikelihood = .9, burn_in_iter_children = 50, nb_particles=96, nb_iterations=30, nb_iterationsCalibrated=15, sigma = .02, alpha = 0.1, ratio = 1, i_steps = 5):
		"""
		! Set the simulation parameters
		:param nbReps: the number of repetitions that the simulation should be run
		"""
		multFactor = int(np.interp(ratio, (1, 100), (1, 20)))
		self.nb_burn_in_iter = burn_in_iter * multFactor
		self.burninLimitLikelihood = burninLimitLikelihood
		self.burn_in_iter_children = burn_in_iter_children * multFactor
		self.nb_particles = nb_particles
		self.nb_iterations = nb_iterations
		self.nb_iterationsCalibrated = nb_iterationsCalibrated
		self.sigma0 = sigma
		self.alpha = alpha
		self.i_steps = i_steps

		if self.saveProgress:
			self.saveProgress_likelihood = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_c = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_e = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_D = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_p_FP = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_p_TP = np.ones([self.nb_iterations * 4,3])
			self.saveProgress_eff_samp = np.ones([self.nb_iterations * 4])
			self.saveProgress_eo = np.ones([self.nb_iterations * 4,9,3])

	def setParams(self, params):
		"""
		Set a species for the metapop simulation using the species class
		:param params: the species parameters
		"""
		self.params0 = params
		self.params = [None] * self.nb_particles
		
		for k in range(self.nb_particles):
			self.params[k] = copy.copy(self.params0)
			self.params[k].eo = np.copy(self.params[0].eo)
	
	def setParallel(self, burninParallel = True, iterationParallel = True):
		self.burninParallel = burninParallel
		self.iterationParallel = iterationParallel

	def setPresenceMapFromProbability(self, P):
		self.W = np.ones(
			[self.mapDimensions[0], self.mapDimensions[1], self.nb_particles], dtype=bool)
		
		# R = np.ones([self.mapDimensions[0], self.mapDimensions[1]]) * 0.5
		for k in range(self.nb_particles):
			R = np.random.rand(P.shape[0], P.shape[1])
			self.W[:, :, k] = (R < P)
		
		self.initialPresence = np.copy(self.W)
	
	def setPresenceMapFromFile(self, filePath):
		filehandler = open(filePath,'rb')
		initialPresence = pickle.load(filehandler)
		filehandler.close()

		self.setPresenceMapFromProbability(initialPresence)
	
	def __SelectInitialMapFromMean(self):
		P = np.mean(self.initialPresence, axis=2)
		self.setPresenceMapFromProbability(P)

	def __resetPresenceMapToInitial(self):
		self.W = np.copy(self.initialPresence)

	def __initialiseRandAccessVector(self):
		self.randParticleOrder = [None] * self.nb_particles
		for k in range(self.nb_particles):
			self.randParticleOrder[k] = np.concatenate([np.random.choice(self.nb_randMap, self.nb_randMap, replace=False), np.random.choice(self.nb_randMap, self.nb_randMap, replace=False)])

	def __initialiseParticleAccessIndex(self):
		self.particleAccessIndex = np.zeros(self.nb_particles, dtype=int)
	
	def __resetParticleAccessIndex(self):
		self.particleAccessIndex *= 0

	def __scrambleRandAccessVector(self):
		self.randParticleOrder = itemgetter(*np.random.choice(self.nb_particles, self.nb_particles, replace=False))(self.randParticleOrder)
	
	def __initialiseRandomVector(self):
		self.nb_randMap = self.nb_burn_in_iter + len(range(self.insituTimes[0], self.insituTimes[-1]))
		self.nb_randMap = int(np.ceil(self.nb_randMap / 2))
		self.R = np.random.random([self.mapDimensions[0], self.mapDimensions[1], self.nb_randMap])

	def __initialiseFitness(self):
		self.C = [None] * self.nb_particles
		self.E = [None] * self.nb_particles
		self.F = [None] * self.nb_particles

		for p in range(self.nb_particles):
			self.F[p] = np.zeros(self.EOMatrix[0].shape[0])
			self.C[p] = np.zeros(self.mapDimensions)
			self.E[p] = np.zeros(self.mapDimensions)

	@jit
	def __computeDispersalDistance(self):
		"""
		Compute the dispersal map using the terrain properties
		"""
		self.d = np.square(
			((self.mapDimensions[0] / 2) - np.abs(np.cumsum(np.ones(self.mapDimensions), axis=0) - 1 - (self.mapDimensions[0]) / 2)) * self.dx)
		self.d = np.sqrt(self.d + np.square(((self.mapDimensions[1] / 2) - np.abs(
			np.cumsum(np.ones(self.mapDimensions), axis=1) - 1 - (self.mapDimensions[1]) / 2)) * self.dy))
		self.d[0, 0] = np.inf
		self.fftd	= [None] * self.nb_particles
		i_map = np.fft.fft2(np.zeros(self.d.shape))
		for p in range(self.nb_particles):
			self.fftd[p] = np.copy(i_map)
	
	# !####################### Parameter Resampling #########

	# def __reshuffleRandomVector(self):
	# 	self.R = self.R[:, :, np.random.choice(self.nb_randMap,self.nb_randMap,replace=False)]

	@jit
	def __resampleParams(self):
		"""
		resamples all the parametersets from their current parameter
		"""
		for k in range(self.nb_particles):
			self.__resampleParam(k)

	@jit
	def __resampleParam(self, k):
		self.params[k].c = self.__sampleParam_positive(self.params[k].c)
		self.params[k].e = self.__sampleParam_positive(self.params[k].e)
		self.params[k].D = self.__sampleParam_positive(self.params[k].D)
		if self.sample_p_FP:
			self.params[k].p_FP = self.__sampleParam_01(self.params[k].p_FP)
		self.params[k].p_TP = self.__sampleParam_01(self.params[k].p_TP)
		for j in range(len(self.params[k].eo)):
			self.params[k].eo[j] = self.__sampleParam(self.params[k].eo[j])
	
	def __sampleParam(self,mu):
		return np.random.normal(mu, self.__activeSigma())
	
	def __sampleParam_positive(self,mu):
		return np.exp(self.__sampleParam(np.log(mu)))
	
	def __sampleParam_01(self,mu):
		if mu == 1:
			mu -= 1e-10
		if mu == 0:
			mu += 1e-10
		x = np.log(mu / (1 - mu))
		y = self.__sampleParam(x)
		return 1 / (1 + np.exp(-y))

	def __activeSigma(self):
		# return self.sigma0 * self.alpha ** (self.iter / 50.0)
		return self.sigma0 * self.alpha ** ((self.step + self.iter * self.totalSteps) / (50.0 * self.totalSteps))
	
	@jit
	def __computeFitness_forParticle(self, p, t):
		np.dot(self.EOMatrix[t], self.params[p].eo[1:len(self.params[p].eo)], out = self.F[p])
		
		self.F[p] += self.params[p].eo[0]
		np.exp(-self.F[p], out = self.F[p])
		
		self.F[p] += 1
		
		self.C[p][:] = 0.0
		self.C[p] += self.params[p].c / self.F[p].reshape(self.mapDimensions)
		self.E[p][:] = 0.0
		self.E[p] -= np.expm1(- self.params[p].e * self.F[p].reshape(self.mapDimensions))
	
	@jit
	def __computeDispersal_forParticle(self, p, t):
		self.fftd[p] = np.fft.fft2(np.exp(-self.d / self.params[p].D) / self.params[p].D ** 2 / 2 / np.pi)
	
	# !####################### Simulation #########

	@jit
	def __takeNormalStepForTime(self, particle, t):

		# resample parameters
		if self.resampleParamBool:
			self.__resampleParam(particle)

		# recompute fitness and dispersal for parameters
		self.__computeFitness_forParticle(particle,t)
		self.__computeDispersal_forParticle(particle,t)

		# take simulation step with particle
		for i_step_i in range(self.i_steps):
			self.__simulationStep(particle)

	@jit
	def __takeNormalStepForTimes(self, particle, times):
		for t in times:
			self.__takeNormalStepForTime(particle, t)
		
		if not self.lastPassage:
			self.likelihood[particle] = self.computeLikelihood_particle(times[-1] + 1,particle)
	

	@jit
	def __burninSteps(self, particle):

		if self.resampleParamBool:
			self.__resampleParam(particle)

		self.__computeFitness_forParticle(particle, 0)
		self.__computeDispersal_forParticle(particle, 0)

		self.likelihood[particle] = self.__simulationSteps_burnin(particle)
	


	@jit
	def __simulationSteps_burnin(self, particle):
		"""
		Perform one step of the markov chain
		:param W: the presence matrix for the given step and repetition
		"""
		count = 0
		likelihood = 0.0
		for k in range(self.nb_burn_in_iter):
			self.__simulationStep(particle)

			if k >= np.floor(self.burninLimitLikelihood * self.nb_burn_in_iter):
				likelihood += self.computeLikelihood_particle(0,particle)
				count += 1
			
			if self.plotBurnin and self.iter%15 == 0:
				self.__storeBurnin(k, particle)

		return likelihood / count

	@jit
	def __simulationStep(self, particle):
		"""
		Perform one step of the markov chain
		:param W: the presence matrix for the given step and repetition
		"""
		fftc = np.fft.fft2(self.W[:,:,particle] * self.C[particle])

		C = 1 - np.exp(-np.real(np.fft.ifft2(fftc * self.fftd[particle])))
		
		R = np.random.random(self.mapDimensions)
		

		C = (R < C) & (~self.W[:,:,particle])
		E = (R < self.E[particle]) & (self.W[:,:,particle])

		self.W[C, particle] = True
		self.W[E, particle] = False

	def __performBurnIn(self, parallel = False):
		if parallel:
			num_cores = multiprocessing.cpu_count()
			Parallel(n_jobs=num_cores, prefer="threads")(delayed(self.__burninSteps)(i)for i in range(self.nb_particles))
		else:
			self.__performBurnIn_serialised()
	
	@jit
	def __performBurnIn_serialised(self):
		for p in range(self.nb_particles):
			self.__burninSteps(p)

	def __performSimulationSteps(self, times, parallel = False):
		if parallel:
			num_cores = multiprocessing.cpu_count()
			Parallel(n_jobs=num_cores, prefer="threads")(delayed(self.__takeNormalStepForTimes)(i, times)for i in range(self.nb_particles))
		else:
			self.__performSimulationSteps_serialised(times)
	
	@jit
	def __performSimulationSteps_serialised(self, times):
		for p in range(self.nb_particles):
				self.__takeNormalStepForTimes(p, times)

	
	def performSimulation(self):
		"""
		Perform all time steps
		"""

		if self.plotPresenceMaps and self.iter%15 == 0:
			self.__clearPresenceMapPlot()
		
		if self.plotInitialMaps and self.iter%15 == 0:
			self.__clearInitialMapPlot()

		self.step = 0
		
		self.__resetPresenceMapToInitial()

		self.__resetParticleAccessIndex()
		self.__scrambleRandAccessVector()

		if self.plotPresenceMaps and self.iter%15 == 0:
			self.__plotPresenceMap(-1)

		if self.plotInitialMaps and self.iter%15 == 0:
			self.__plotInitialMap(-2)

		# perform burn-in
		print('performing burn-in')
		

		t0 = time.time()
		self.likelihood = np.zeros(self.nb_particles)
		
		self.__performBurnIn(parallel=self.burninParallel)

		self.step += 1

		self.__saveNewInitial()

		if self.plotPresenceMaps and self.iter%15 == 0:
			self.__plotPresenceMap(0)

		if self.plotInitialMaps and self.iter%15 == 0:
			self.__plotInitialMap(-1)
		
		if not self.simulate:
			self.__filterFromWeights(0)

		if self.postCalibrationPhase:
			self.calibratedLikelihood_particle -= np.log(self.likelihood)
		
		print('done. performing burnin took ' + str(time.time() - t0))

		if self.plotInitialMaps and self.iter%15 == 0:
			self.__plotInitialMap(0)

		if self.plotBurnin and self.iter%15 == 0:
			self.__plotBurnin()
		
		if self.lastPassage:
			self.doLastStepStuff(0)



		for t_i in range(len(self.insituTimes[0:-1])):
			# * select times for computation
			times = range(self.insituTimes[t_i],self.insituTimes[t_i + 1])
			print('performing steps ' + str(times[0]) + " to " + str(times[-1]))
			# * compute on times
			if not self.lastPassage:
				self.__performSimulationSteps(times, parallel=self.iterationParallel)
			else:
				t_obj = [None]
				for timeStep in times:
					t_obj[0] = timeStep
					self.__performSimulationSteps(t_obj, parallel=self.iterationParallel)
					if not ((timeStep + 1) == self.insituTimes[t_i + 1]):
						self.doLastStepStuff(timeStep + 1)

			self.step += 1

			if self.plotPresenceMaps and self.iter%15 == 0:
				self.__plotPresenceMap(t_i + 1)

			# * resample
			if not self.simulate:
				self.__filterFromWeights(self.insituTimes[t_i + 1])

			if self.postCalibrationPhase:
				self.calibratedLikelihood_particle -= np.log(self.likelihood)

			if self.lastPassage:
				self.doLastStepStuff(self.insituTimes[t_i + 1])

			if self.plotInitialMaps and self.iter%15 == 0:
				self.__plotInitialMap(t_i + 1)
			
			print('done. performing steps ' + str(times[0]) + " to " + str(times[-1]) + " took " + str(time.time() - t0))

	
	
	def performPomp(self):
		self.simulationName = self.species.replace(" ", "_") + "/" + self.timeStamp + "/" + "res_" + str(self.res) + "_it_" + str(self.nb_iterations) + "_part_" + str(self.nb_particles)  + "_bur_" + str(self.nb_burn_in_iter) + "/"
			
		self.iter = 0
		self.__initialiseRandAccessVector()
		self.__initialiseRandomVector()
		self.__initialiseParticleAccessIndex()

		if self.calibrate:
			self.__setupFigures()

			#*
			#* Calibration
			#*

			for self.iter in range(self.nb_iterations):
				print('starting iteration ' + str(self.iter) + ' of ' + str(self.nb_iterations - 1))
			
				if self.plotPairPlot and self.iter%15 == 0:
					self.__initialisePairPlot()

				self.performSimulation()

				if self.plotPairPlot and self.iter%15 == 0:
					self.__showPairPlot()
				
				if (self.plotInitialMaps or self.plotPresenceMaps or self.plotPairPlot) and self.displayPlots and self.iter%15 == 0:
					plt.pause(.1)
				
				if self.plotPairPlot and self.iter%15 == 0:
					self.__savePairPlot(self.iter)

				self.nb_burn_in_iter = self.burn_in_iter_children

				# If progress plot needs to be saved
				# if self.iter%15 == 0:
				# 	self.__saveProgressPlot()
				
				if self.iter%15 == 0:
					self.__savePresenceMapPlot(self.iter)
					self.__saveInitialPresenceMapPlot(self.iter)
					self.__saveParticleMovementPlot(self.iter)

				if self.plotBurnin and self.iter%15 == 0:
					self.__initialiseBurninTraj()
				
				if np.sum(self.initialPresence.flatten()) == 0:
					self.__closeAllFigs()
					return
			
			self.__saveProgressPlot()
			self.__saveProgressToDisk()

			self.__closeAllFigs()


		#*
		#* Post Calibration Filtering
		#*
		self.nb_burn_in_iter = self.burn_in_iter_children
		self.saveProgress = False

		self.postCalibrationPhase = True
		self.simulationName += "calibrated_"
		self.__SelectParameterMedian()
		self.__SelectInitialMapFromMean()

		self.__saveCalibratedParams()
		
		self.__setupFigures()

		self.resampleParamBool = False
		self.plotPairPlot = False

		self.calibratedLikelihoodVector = np.ones(self.nb_iterationsCalibrated) * np.NaN

		for self.iter in range(self.nb_iterationsCalibrated):
			print('starting calibrated iteration ' + str(self.iter) + ' of ' + str(self.nb_iterationsCalibrated - 1))

			self.calibratedLikelihood_particle = np.zeros(self.nb_particles)

			self.performSimulation()

			self.calibratedLikelihoodVector[self.iter] = np.mean(self.calibratedLikelihood_particle)
			
			# if (self.plotInitialMaps or self.plotPresenceMaps or self.plotPairPlot) and self.displayPlots:
			# 	plt.pause(.1)
			
			# if self.iter%15 == 0:
			# 	self.__saveProgressPlot()
			
			if self.iter%15 == 0:
				self.__savePresenceMapPlot(self.iter)
				self.__saveInitialPresenceMapPlot(self.iter)
			
			if self.plotBurnin and self.iter%15 == 0:
				self.__initialiseBurninTraj()

		plt.pause(.1)
		self.__saveProgressPlot()
		self.__saveCalibratedLikelihood()
		self.__saveFinalSetup()

		self.lastPassage = True
		self.performSimulation()

		self.simulate = True
		self.simulationName += "simulated_"
		self.performSimulation()
		
		self.__closeAllFigs()

	
	
	def simluateCalibrated(self):
		self.simulationName = self.species.replace(" ", "_") + "/" + self.timeStamp + "/" + "res_" + str(self.res) + "_it_" + str(self.nb_iterations) + "_part_" + str(self.nb_particles)  + "_bur_" + str(self.nb_burn_in_iter) + "/"
			
		self.iter = 0
		self.__initialiseRandAccessVector()
		self.__initialiseRandomVector()
		self.__initialiseParticleAccessIndex()

		#*
		#* Post Calibration Filtering
		#*
		self.nb_burn_in_iter = self.burn_in_iter_children
		self.saveProgress = False

		self.postCalibrationPhase = True
		self.simulationName += "calibrated_"
		self.__SelectParameterMedian()
		self.__SelectInitialMapFromMean()
		
		self.__setupFigures()

		self.resampleParamBool = False
		self.plotPairPlot = False

		self.calibratedLikelihood_particle = np.zeros(self.nb_particles)

		self.lastPassage = True
		self.simulate = True
		self.simulationName += "simulated_"
		self.performSimulation()
		
		self.__closeAllFigs()
	
	# !############# Filtering ##################
	def __saveNewInitial(self):
		self.initialPresence = np.copy(self.W)

	#@jit
	def __filterFromWeights(self, t):
		indexes = self.__getSelectedIndexes(t)
		if self.plotParticleMovement and self.iter%15 == 0:
			self.__plotParticleMovement(indexes, t)
		self.W = self.W[:,:,indexes]
		self.initialPresence = self.initialPresence[:,:,indexes]
		# self.params = itemgetter(*indexes)(self.params)
		param_temp = copy.copy(self.params)
		for k in range(self.nb_particles):
			self.params[k] = copy.copy(param_temp[indexes[k]])
			self.params[k].eo = np.copy(param_temp[indexes[k]].eo)
		
		self.likelihood = self.likelihood[indexes]

	#@jit
	def __getSelectedIndexes(self, t):
		weights = self.likelihood / np.sum(self.likelihood)
		c_w = np.cumsum(weights)
		R = np.random.uniform(low=0.0, high=1.0 / self.nb_particles) + np.arange(self.nb_particles) / self.nb_particles
		indexes = np.argmax([c_w > r for r in R], axis = 1)

		if self.plotProgress:
			self.__plotEffSampleSize(self.iter, t, weights)
			self.__plotLikelihood(self.iter, t)
			self.__plotParams(self.iter, t)
		
		if self.saveProgress:
			self.__saveEffSampleSize(self.iter, t, weights)
			self.__saveLikelihood(self.iter, t)
			self.__saveParams(self.iter, t)
			# plt.pause(.1)
		if self.plotPairPlot and self.iter%15 == 0:
			self.__saveDataForPairPlot(t)
		return indexes

	#@jit
	def computeLikelihood_particle(self, t, particle):
		t_insitu = np.where(self.insituTimes == t)[0][0]
		
		M = self.W[self.samplingSites[t_insitu],particle].flatten()
		P_TP = self.binom[t_insitu] * (self.params[particle].p_TP ** self.k[t_insitu]) * ((1 - self.params[particle].p_TP) ** (self.n[t_insitu] - self.k[t_insitu]))
		P_FP = self.binom[t_insitu] * (self.params[particle].p_FP ** self.k[t_insitu]) * ((1 - self.params[particle].p_FP) ** (self.n[t_insitu] - self.k[t_insitu]))

		computedLikelihood = np.nanprod(M * P_TP + (1-M) * P_FP)
		return computedLikelihood
	
	def __SelectParameterMedian(self):
		c = np.ones(len(self.params))
		e = np.ones(len(self.params))
		D = np.ones(len(self.params))
		p_TP = np.ones(len(self.params))
		p_FP = np.ones(len(self.params))
		eo = np.ones([len(self.params), len(self.params[0].eo)])

		for p_i in range(len(self.params)):
			c[p_i] = self.params[p_i].c
			e[p_i] = self.params[p_i].e
			D[p_i] = self.params[p_i].D
			p_TP[p_i] = self.params[p_i].p_TP
			p_FP[p_i] = self.params[p_i].p_FP
			eo[p_i, :] = self.params[p_i].eo
		
		
		c = np.median(c)
		e = np.median(e)
		D = np.median(D)
		p_TP = np.median(p_TP)
		p_FP = np.median(p_FP)
		eo = np.median(eo, axis=0)

		for param in self.params:
			param.c = np.copy(c)
			param.e = np.copy(e)
			param.D = np.copy(D)
			param.p_TP = np.copy(p_TP)
			param.p_FP = np.copy(p_FP)
			param.eo = np.copy(eo)

	def returnAveragePresenceInMap(self):
		"""
		Returns the average presence over all repetitions
		"""
		return np.mean(self.W, 2)
	
	def doLastStepStuff(self, t):
		self.__plotPresenceLastStep(t)
		self.__saveFinalSetupEachYear(t)
	
	def __plotPresenceLastStep(self, year):
		presence = np.mean(self.W, axis=2)
		presence[presence == 0] = np.nan

		fig, axs = plt.subplots(1,1, figsize=(15, 6), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.001)
		divider = make_axes_locatable(axs)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		im = axs.imshow(presence, cmap='YlOrRd', vmin=0, vmax=1)
		axs.set_aspect('equal', adjustable='box')
		fig.colorbar(im, cax=cax, orientation='vertical')
		axs.get_xaxis().set_ticks([])
		axs.get_yaxis().set_ticks([])
		axs.set_title((year + 2006))
		
		# plt.pause(.1)

		folderpath = '../../figures/modeling/POMP/' + self.simulationName
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)
		fig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_final_step_presenceMap' + '_' + str(year + 2006) + '.png',bbox_inches='tight')

		plt.close(fig)
	

	def __saveFinalSetupEachYear(self, year):
		folderPath = '../../data/parameters/' + self.simulationName
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		
		filehandler = open(folderPath + str(self.lh_num) + "_finalPresence_" + str(year) + ".obj",'wb')
		pickle.dump(self.W, filehandler)
		filehandler.close()

	# !################################## Progression Monitoring ###############

	def __setupFigures(self):
		if self.plotProgress:
			self.__initialiseFilteringProgressPlots()
		
		if self.plotParticleMovement:
			self.__initialiseParticleMovementPlot()
		
		if self.plotBurnin:
			self.__initialiseBurninPlots()
			self.__initialiseBurninTraj()
		
		if self.plotPresenceMaps:
			self.__initialisePresenceMapsPlots()
		
		if self.plotInitialMaps:
			self.__initialiseInitialMapsPlots()

	def setPlotBurninTrajectory(self, plotBurnin = True):
		self.plotBurnin = plotBurnin
	
	def __storeBurnin(self, k, particle):
		self.burninTrajectories[k, particle] = np.sum(self.W[:,:,particle].flatten())

	def __initialiseBurninPlots(self):
		self.burninFig, self.burninAxs = plt.subplots(1, 1, figsize=(10, 10), facecolor='w', edgecolor='k', dpi = 200)
	
	def __initialiseBurninTraj(self):
		self.burninTrajectories = np.zeros([self.nb_burn_in_iter, self.nb_particles])
	
	def __plotBurnin(self):
		self.burninAxs.clear()
		self.burninAxs.plot(self.burninTrajectories)
		self.burninAxs.set_title("D: " + str(self.params0.D) + " c: " + str(self.params0.c) + " e: " + str(self.params0.e))
		
		self.__saveBurninPlot(self.iter)
	
	def setPlotParticleMovement(self, plotParticleMovement = True):
		self.plotParticleMovement = plotParticleMovement
	
	def __initialiseParticleMovementPlot(self):
		self.partMovFig, self.partMovAxs = plt.subplots(1, 1, figsize=(10, 10), facecolor='w', edgecolor='k', dpi = 200)
	
	def __plotParticleMovement(self, indexes, t):
		it = self.iter
		t_0 = np.where(self.insituTimes == t)[0][0]
		if t == self.insituTimes[-1]:
			it += 1
			t_1 = 0
		else:
			t_1 = t_0 + 1
		
		t_0 /= len(self.insituTimes - 1)
		t_1 /= len(self.insituTimes - 1)
		
		lines = np.array([np.arange(self.nb_particles), indexes])


		self.partMovAxs.plot(np.array([self.iter + t_0 + 0.05, it + t_1]), lines, 'b', linewidth=.1)

		if t_0 == 0:
			self.partMovAxs.plot(np.array([it + t_1]), np.array([indexes]), 'g.')
		else:
			self.partMovAxs.plot(np.array([it + t_1]), np.array([indexes]), 'r.')

		self.partMovAxs.plot(np.array([it + t_1, it + t_1 + 0.05]), lines[[-1,0],:], 'b', linewidth=.1)

	def setPlotPairPlot(self, plotPairPlot = True):
		self.plotPairPlot = plotPairPlot
	
	def __initialisePairPlot(self):
		self.pairplotData = pd.DataFrame(columns = ['step', 'likelihood', 'c', 'e', 'D', 'p_FP', 'p_TP', 'eo_0', 'eo_1', 'eo_2', 'eo_3', 'eo_4', 'eo_5', 'eo_6', 'eo_7', 'eo_8'])#, 'particleNumber'

		self.dataRecoder_c = np.ones(self.nb_particles)
		self.dataRecoder_e = np.ones(self.nb_particles)
		self.dataRecoder_D = np.ones(self.nb_particles)
		self.dataRecoder_p_FP = np.ones(self.nb_particles)
		self.dataRecoder_p_TP = np.ones(self.nb_particles)
		self.dataRecoder_eo_0 = np.ones(self.nb_particles)
		self.dataRecoder_eo_1 = np.ones(self.nb_particles)
		self.dataRecoder_eo_2 = np.ones(self.nb_particles)
		self.dataRecoder_eo_3 = np.ones(self.nb_particles)
		self.dataRecoder_eo_4 = np.ones(self.nb_particles)
		self.dataRecoder_eo_5 = np.ones(self.nb_particles)
		self.dataRecoder_eo_6 = np.ones(self.nb_particles)
		self.dataRecoder_eo_7 = np.ones(self.nb_particles)
		self.dataRecoder_eo_8 = np.ones(self.nb_particles)

	def __saveDataForPairPlot(self, t):

		for p in range(self.nb_particles):
			self.dataRecoder_c[p] = self.params[p].c
			self.dataRecoder_e[p] = self.params[p].e
			self.dataRecoder_D[p] = self.params[p].D
			self.dataRecoder_p_FP[p] = self.params[p].p_FP
			self.dataRecoder_p_TP[p] = self.params[p].p_TP
			self.dataRecoder_eo_0[p] = self.params[p].eo[0]
			self.dataRecoder_eo_1[p] = self.params[p].eo[1]
			self.dataRecoder_eo_2[p] = self.params[p].eo[2]
			self.dataRecoder_eo_3[p] = self.params[p].eo[3]
			self.dataRecoder_eo_4[p] = self.params[p].eo[4]
			self.dataRecoder_eo_5[p] = self.params[p].eo[5]
			self.dataRecoder_eo_6[p] = self.params[p].eo[6]
			self.dataRecoder_eo_7[p] = self.params[p].eo[7]
			self.dataRecoder_eo_8[p] = self.params[p].eo[8]

		newDataframe = pd.DataFrame(
			data = {
				'step': np.ones(self.nb_particles) * (self.iter + t / 10),
				# 'particleNumber': np.arange(self.nb_particles),
				'likelihood': -np.log(self.likelihood),
				'c': self.dataRecoder_c,
				'e': self.dataRecoder_e,
				'D': self.dataRecoder_D,
				'p_FP': self.dataRecoder_p_FP,
				'p_TP': self.dataRecoder_p_TP,
				'eo_0': self.dataRecoder_eo_0,
				'eo_1': self.dataRecoder_eo_1,
				'eo_2': self.dataRecoder_eo_2,
				'eo_3': self.dataRecoder_eo_3,
				'eo_4': self.dataRecoder_eo_4,
				'eo_5': self.dataRecoder_eo_5,
				'eo_6': self.dataRecoder_eo_6,
				'eo_7': self.dataRecoder_eo_7,
				'eo_8': self.dataRecoder_eo_8
			}, index = [np.arange(self.nb_particles)]
		)

		self.pairplotData = self.pairplotData.append(newDataframe, ignore_index=True)
	
	def __showPairPlot(self):
		self.pairplot = sns.pairplot(self.pairplotData, hue="step")
		

	def setPlotPresenceMaps(self, plotPresenceMaps = True):
		self.plotPresenceMaps = plotPresenceMaps
	
	def __initialisePresenceMapsPlots(self):
		self.mapsFig, self.mapsAxs = plt.subplots(3, 2, figsize=(10, 10), facecolor='w', edgecolor='k', dpi = 200)
		self.mapsAxs = self.mapsAxs.flatten()
		self.mapsFig.delaxes(self.mapsAxs[-1])
		self.__fillInitialPresenceMapsPlots()

	def __fillInitialPresenceMapsPlots(self):
		
		self.mapsAxs[0].set_title("initial")
		self.mapsAxs[0].invert_yaxis()
		self.mapsAxs[0].set_aspect('equal', adjustable='box')
		self.mapsAxs[0].get_xaxis().set_ticks([])
		self.mapsAxs[0].get_yaxis().set_ticks([])
		
		count = 1

		for k in self.insituTimes:
			self.mapsAxs[count].set_title(str(k + 2006))
			self.mapsAxs[count].invert_yaxis()
			self.mapsAxs[count].set_aspect('equal', adjustable='box')
			self.mapsAxs[count].get_xaxis().set_ticks([])
			self.mapsAxs[count].get_yaxis().set_ticks([])
			count += 1
	
	def __plotPresenceMap(self, index):
		presence = np.mean(self.W, axis=2)
		presence[presence == 0] = np.nan

		self.mapsAxs[index + 1].imshow(presence, vmin=0, vmax=1)
		# plt.pause(.1)
	
	def __clearPresenceMapPlot(self):
		for ax in self.mapsAxs:
			ax.clear()
		self.__fillInitialPresenceMapsPlots()

	def setPlotInitialMaps(self, plotInitialMaps = True):
		self.plotInitialMaps = plotInitialMaps
	
	def __initialiseInitialMapsPlots(self):
		self.initialMapsFig, self.initialMapsAxs = plt.subplots(3, 2, figsize=(10, 10), facecolor='w', edgecolor='k', dpi = 200)
		self.initialMapsAxs = self.initialMapsAxs.flatten()
		self.__fillInitialInitialMapsPlots()
	
	def __fillInitialInitialMapsPlots(self):
		
		self.initialMapsAxs[0].set_title("initial")
		
		self.initialMapsAxs[1].set_title("after burnin")
		

		for k in range(6):
			self.initialMapsAxs[k].invert_yaxis()
			self.initialMapsAxs[k].set_aspect('equal', adjustable='box')
			self.initialMapsAxs[k].get_xaxis().set_ticks([])
			self.initialMapsAxs[k].get_yaxis().set_ticks([])
		
		count = 2
		
		for k in self.insituTimes:
			self.initialMapsAxs[count].set_title(str(k + 2006))
			count += 1
	
	def __plotInitialMap(self, index):
		presence = np.mean(self.initialPresence, axis=2)
		presence[presence == 0] = np.nan

		self.initialMapsAxs[index + 2].imshow(presence, vmin=0, vmax=1)
		# plt.pause(.1)
	
	def __clearInitialMapPlot(self):
		for ax in self.initialMapsAxs:
			ax.clear()
		
		self.__fillInitialInitialMapsPlots()


	def setPlotProgressOption(self, plotProgress = True):
		self.plotProgress = plotProgress


	def setSaveProgressOption(self, saveProgress = True):
		self.saveProgress = saveProgress



	def setDisplayPlots(self, displayPlots = False):
		self.displayPlots = displayPlots
	
	def __initialiseFilteringProgressPlots(self):
		# likelihood, Neff, c, e, D, p_FP, p_TP, #EO params, 
		nbSuplots = 1 + 1 + 3 + 2 + len(self.params[0].eo)
		self.fig, self.axs = plt.subplots(int(np.ceil(nbSuplots/2)), 2, figsize=(10, 10), facecolor='w', edgecolor='k', dpi = 200)
		self.axs = self.axs.flatten()
		# self.fig.subplots_adjust(hspace = .5, wspace=.001)
		self.axs[0].set_title("Likelihood")
		self.axs[1].set_title("Eff Sample Size")
		self.axs[2].set_title("c")
		self.axs[3].set_title("e")
		self.axs[4].set_title("D")
		self.axs[5].set_title(r'$p_{FP}$')
		self.axs[6].set_title(r'$p_{TP}$')
		self.axs[7].set_title(r'$\alpha$')
		for k in range(8,len(self.axs)):
			self.axs[k].set_title(r'$\beta_' + str(k - 8) + '$')
		
		self.fig.tight_layout()

	def __plotLikelihood(self, iteration, t):
		self.axs[0].plot(np.ones(self.nb_particles) * (iteration + t / 10), -np.log(self.likelihood), 'b.', alpha=0.1,markeredgewidth=0.0)
		print(np.mean(-np.log(self.likelihood).flatten()))

	def __plotEffSampleSize(self, iteration, t, weights):
		self.axs[1].plot(iteration + t / 10, 1 / np.sum(weights ** 2), 'b.', alpha=0.8,markeredgewidth=0.0)
	
	def __saveEffSampleSize(self, iteration, t, weights):
		entry = iteration * 4 + (t if t < 2 else t - 4)
		self.saveProgress_eff_samp[entry] = 1 / np.sum(weights ** 2)

	def __saveLikelihood(self, iteration, t):
		entry = iteration * 4 + (t if t < 2 else t - 4)
		self.saveProgress_likelihood[entry,0] = np.nanmedian(-np.log(self.likelihood))
		self.saveProgress_likelihood[entry,1] = np.nanquantile(-np.log(self.likelihood), 0.025)
		self.saveProgress_likelihood[entry,2] = np.nanquantile(-np.log(self.likelihood), 0.975)
	
	def __saveParams(self, iteration, t):
		entry = iteration * 4 + (t if t < 2 else t - 4)

		c = np.ones(len(self.params))
		e = np.ones(len(self.params))
		D = np.ones(len(self.params))
		p_TP = np.ones(len(self.params))
		p_FP = np.ones(len(self.params))
		eo = np.ones([len(self.params), len(self.params[0].eo)])

		for p_i in range(len(self.params)):
			c[p_i] = self.params[p_i].c
			e[p_i] = self.params[p_i].e
			D[p_i] = self.params[p_i].D
			p_TP[p_i] = self.params[p_i].p_TP
			p_FP[p_i] = self.params[p_i].p_FP
			eo[p_i, :] = self.params[p_i].eo
		
		
		self.saveProgress_c[entry,0] = np.nanmedian(c)
		self.saveProgress_c[entry,1] = np.nanquantile(c, 0.025)
		self.saveProgress_c[entry,2] = np.nanquantile(c, 0.975)
		self.saveProgress_e[entry,0] = np.nanmedian(e)
		self.saveProgress_e[entry,1] = np.nanquantile(e, 0.025)
		self.saveProgress_e[entry,2] = np.nanquantile(e, 0.975)
		self.saveProgress_D[entry,0] = np.nanmedian(D)
		self.saveProgress_D[entry,1] = np.nanquantile(D, 0.025)
		self.saveProgress_D[entry,2] = np.nanquantile(D, 0.975)
		self.saveProgress_p_FP[entry,0] = np.nanmedian(p_FP)
		self.saveProgress_p_FP[entry,1] = np.nanquantile(p_FP, 0.025)
		self.saveProgress_p_FP[entry,2] = np.nanquantile(p_FP, 0.975)
		self.saveProgress_p_TP[entry,0] = np.nanmedian(p_TP)
		self.saveProgress_p_TP[entry,1] = np.nanquantile(p_TP, 0.025)
		self.saveProgress_p_TP[entry,2] = np.nanquantile(p_TP, 0.975)
		self.saveProgress_eo[entry,:,0] = np.nanmedian(eo, axis = 0)
		self.saveProgress_eo[entry,:,1] = np.nanquantile(eo, 0.025, axis = 0)
		self.saveProgress_eo[entry,:,2] = np.nanquantile(eo, 0.975, axis = 0)
	
	def __saveProgressToDisk(self):
		if self.saveProgress:
			folderPath = '../../data/parameters/' + self.simulationName
			if not os.path.exists(folderPath):
				os.makedirs(folderPath)
			
			# progress = type('obj', (object,), {'eff_samp' : self.saveProgress_eff_samp,'likelihood' : self.saveProgress_likelihood, 'c' : self.saveProgress_c, 'e' : self.saveProgress_e, 'D' : self.saveProgress_D, 'p_FP' : self.saveProgress_p_FP, 'p_TP' : self.saveProgress_p_TP, 'eo' : self.saveProgress_eo})
			
			filehandler = open(folderPath + str(self.lh_num) + "_progress_eff_samp.obj",'wb')
			pickle.dump(self.saveProgress_eff_samp, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_likelihood.obj",'wb')
			pickle.dump(self.saveProgress_likelihood, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_c.obj",'wb')
			pickle.dump(self.saveProgress_c, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_e.obj",'wb')
			pickle.dump(self.saveProgress_e, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_D.obj",'wb')
			pickle.dump(self.saveProgress_D, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_p_FP.obj",'wb')
			pickle.dump(self.saveProgress_p_FP, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_p_TP.obj",'wb')
			pickle.dump(self.saveProgress_p_TP, filehandler)
			filehandler.close()

			filehandler = open(folderPath + str(self.lh_num) + "_progress_eo.obj",'wb')
			pickle.dump(self.saveProgress_eo, filehandler)
			filehandler.close()

	def __plotParams(self, iteration, t):
		x = iteration + t / 10
		for k in range(self.nb_particles):
			self.axs[2].plot(x, self.params[k].c, 'b.', alpha=0.1,markeredgewidth=0.0)
			self.axs[3].plot(x, self.params[k].e, 'b.', alpha=0.1,markeredgewidth=0.0)
			self.axs[4].plot(x, self.params[k].D, 'b.', alpha=0.1,markeredgewidth=0.0)
			self.axs[5].plot(x, self.params[k].p_FP, 'b.', alpha=0.1,markeredgewidth=0.0)
			self.axs[6].plot(x, self.params[k].p_TP, 'b.', alpha=0.1,markeredgewidth=0.0)
			for eo_i in range(len(self.params[k].eo)):
				self.axs[7 + eo_i].plot(x, self.params[k].eo[eo_i], 'b.', alpha=0.1,markeredgewidth=0.0)
	
	# !########################## save data
	def __saveCalibratedParams(self):
		# index = np.argmax(self.likelihood)
		folderPath = '../../data/parameters/' + self.simulationName
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		
		filehandler = open(folderPath + str(self.lh_num) + "_params.obj",'wb')
		savedParam = self.params[0]
		pickle.dump(savedParam, filehandler)
		filehandler.close()
		filehandler = open(folderPath + str(self.lh_num) + "_initial.obj",'wb')
		savedInitialPresence = np.squeeze(self.initialPresence[:,:,0])
		pickle.dump(savedInitialPresence, filehandler)
		filehandler.close()

	def __saveCalibratedLikelihood(self):
		folderPath = '../../data/parameters/' + self.simulationName
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		
		filehandler = open(folderPath + str(self.lh_num) + "_likelihood_mean.obj",'wb')
		pickle.dump(np.mean(self.calibratedLikelihoodVector), filehandler)
		filehandler.close()
		
		filehandler = open(folderPath + str(self.lh_num) + "_likelihood_std.obj",'wb')
		pickle.dump(np.std(self.calibratedLikelihoodVector), filehandler)
		filehandler.close()
	
	def __saveFinalSetup(self):
		folderPath = '../../data/parameters/' + self.simulationName
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		
		filehandler = open(folderPath + str(self.lh_num) + "_finalInitialCondition.obj",'wb')
		pickle.dump(self.initialPresence, filehandler)
		filehandler.close()

	def __saveMapStep(self, t):
		# index = np.argmax(self.likelihood)
		folderPath = '../../data/parameters/' + self.simulationName
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)
		
		filehandler = open(folderPath + str(self.lh_num) + "_t_" + str(t) +  "_map.obj",'wb')
		pickle.dump(self.W, filehandler)
		filehandler.close()
	
	def __saveProgressPlot(self):
		if self.plotProgress:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.fig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_progress.pdf',bbox_inches='tight')
			self.fig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_progress.png',bbox_inches='tight')
			# self.fig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_progress.svg',bbox_inches='tight')
	
	def __saveBurninPlot(self, iter):
		if self.plotBurnin:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.burninFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_burnin'  + '_' + str(iter) + '.pdf',bbox_inches='tight')
			self.burninFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_burnin'  + '_' + str(iter) + '.png',bbox_inches='tight')
	
	def __savePresenceMapPlot(self, iter):
		if self.plotPresenceMaps:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.mapsFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_presenceMap' + '_' + str(iter) + '.pdf',bbox_inches='tight')
			self.mapsFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_presenceMap' + '_' + str(iter) + '.png',bbox_inches='tight')
	
	def __saveInitialPresenceMapPlot(self, iter):
		if self.plotInitialMaps:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.initialMapsFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_InitialPresenceMap' + '_' + str(iter) + '.pdf',bbox_inches='tight')
			self.initialMapsFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_InitialPresenceMap' + '_' + str(iter) + '.png',bbox_inches='tight')
	
	def __savePairPlot(self, iter):
		if self.plotPairPlot:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.pairplot.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_PairPlot' + '_' + str(iter) + '.pdf',bbox_inches='tight')
			self.pairplot.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_PairPlot' + '_' + str(iter) + '.png',bbox_inches='tight')
			plt.close()
	
	def __saveParticleMovementPlot(self, iter):
		if self.plotParticleMovement:
			folderpath = '../../figures/modeling/POMP/' + self.simulationName
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			# self.partMovFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_ParticleMovement' + '_' + str(iter) + '.pdf',bbox_inches='tight')
			self.partMovFig.savefig(folderpath + str(self.species) + '_lh_' + str(self.lh_num) + '_ParticleMovement' + '_' + str(iter) + '.png',bbox_inches='tight')
	
	def __closeAllFigs(self):
		if self.plotProgress:
			plt.close(self.fig)
		if self.plotBurnin:
			plt.close(self.burninFig)
		if self.plotPresenceMaps:
			plt.close(self.mapsFig)
		if self.plotInitialMaps:
			plt.close(self.initialMapsFig)
		if self.plotPairPlot:
			plt.close(self.pairplot)
		if self.plotParticleMovement:
			plt.close(self.partMovFig)