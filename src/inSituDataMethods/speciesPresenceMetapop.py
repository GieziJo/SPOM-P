import numpy as np
from rpy2.robjects import r,pandas2ri

class MetapopSpeciesData:
	def __init__(self, speciesName, time, observed, sampled):
		self.speciesName = speciesName
		self.time = time
		self.observed = observed
		self.sampled = sampled


def GetMetapopSpeciesData(speciesName):
	years = np.array([2006, 2007, 2012, 2013])
	r('source("inSituDataMethods/createSpeciesRaster.R")')
	r('sampled <- getAllSampledSpeciesMatrix(\'' + speciesName + '\')')
	r('observed <- getAllObservedSpeciesMatrix(\'' + speciesName + '\')')
	Observed = np.array(r.get("observed")) 
	Sampled = np.array(r.get("sampled"))
	return MetapopSpeciesData(speciesName,years, Observed, Sampled)