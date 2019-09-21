import numpy as np
from EOData import EOData_Metapop as EOdata_mp
from inSituDataMethods import speciesPresenceMetapop as spData_mp
from SDM import SDM_Metapop as sdm_metapop
import os
import pickle


class GPNP_Setup():
	def __init__(self, species):
		self.species = species
		self.res = 180
		self.ratio = 1

		self.__loadData(species, self.res)


		parameters = Parameters()
		self.parameters = parameters
	
	def __loadData(self, species, res):
		folderPath = '../../data/precomputed/' + species.replace(" ", "_") + "/"
		if not os.path.exists(folderPath):
			os.mkdir(folderPath)
		# ! Earth Observation data
		try:
			filehandler = open(folderPath + "metapopEOData.obj",'rb')
			self.metapopEOData = pickle.load(filehandler)
			filehandler.close()
		except:
			self.metapopEOData = EOdata_mp.getMetapopVariables()

			filehandler = open(folderPath + "metapopEOData.obj",'wb')
			pickle.dump(self.metapopEOData, filehandler)
			filehandler.close()
		
		# ! Earth Species Data
		try:
			filehandler = open(folderPath + "metapopSpeciesData.obj",'rb')
			self.metapopSpeciesData = pickle.load(filehandler)	
			filehandler.close()
		except:
			self.metapopSpeciesData = spData_mp.GetMetapopSpeciesData(species)

			filehandler = open(folderPath + "metapopSpeciesData.obj",'wb')
			pickle.dump(self.metapopSpeciesData, filehandler)
			filehandler.close()
		
		# ! initial presence form sdm
		try:
			filehandler = open(folderPath + "initialPresence.obj",'rb')
			self.initialPresence = pickle.load(filehandler)	
			filehandler.close()
		except:
			self.initialPresence = sdm_metapop.GetInitialPresenceProbabilitySDM(species)

			filehandler = open(folderPath + "initialPresence.obj",'wb')
			pickle.dump(self.initialPresence, filehandler)
			filehandler.close()


class GPNP_Setup_Pomp(GPNP_Setup):
	def __init__(self,species):
		GPNP_Setup.__init__(self,species)

		parameter = Parameters_pomp()

		parameter.from_vector(
			np.concatenate((
			self.parameters.to_vector(),
			np.array([
				parameter.p_FP,
				parameter.p_TP])))
		)

		self.parameters = parameter


class Parameters():
	def __init__(self):
		self.c = 1
		self.e = 1
		self.D = 100
		self.eo = np.ones(8)

	def to_vector(self):
		vector = np.concatenate((np.array([
			self.c,
			self.e,
			self.D]),
			self.eo))
		return np.array(vector)

	def from_vector(self, vector):
		self.c = vector[0]
		self.e = vector[1]
		self.D = vector[2]
		self.eo = vector[3:len(vector)]

	def direct(self, c, e, D, eo):
		self.c = c
		self.e = e
		self.D = D
		self.eo = eo

class Parameters_pomp(Parameters):
	def __init__(self):
		Parameters.__init__(self)
		self.p_FP = .01
		self.p_TP = .9

	def to_vector(self):
		vector = Parameters.to_vector(self)
		vector = np.concatenate((
			vector,
			np.array([
				self.p_FP,
				self.p_TP])))
		return vector

	def from_vector(self, vector):
		Parameters.from_vector(self, vector)
		self.eo = vector[3:(len(vector)-2)]
		self.p_FP = vector[-2]
		self.p_TP = vector[-1]
