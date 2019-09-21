from rpy2.robjects import r, pandas2ri
import numpy as np

def GetInitialPresenceProbabilitySDM(speciesName):
	r('source("SDM/SDM_Metapop.R")')
	r('P <- getInitialPresenceGuessSDM(\'' + speciesName + '\')')
	P = np.array(r.get("P"))
	return P