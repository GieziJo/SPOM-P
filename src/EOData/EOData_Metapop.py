from rpy2.robjects import r,pandas2ri
import numpy as np

class MetapopVariables:
    def __init__(self, EOMatrix):
        #? 0:#Rows, 1:#Cols, 2:#TimeSteps, 3:#params.eo - 1
        self.EOMatrix = EOMatrix

def getMetapopVariables(res):
    pandas2ri.activate()
    r('source("EOData/EOData_Metapop.R")')
    r('EOMatrix <- getEOMatrix_Metapop()')
    EOMatrix = np.array(r.get("EOMatrix"))
    return MetapopVariables(EOMatrix)