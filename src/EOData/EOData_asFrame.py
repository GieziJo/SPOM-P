from rpy2.robjects import r,pandas2ri
import numpy as np


def getEOdataframeAtTraps():
    pandas2ri.activate()
    r('source("EOData/EOData_Traps.R")')
    r('EOMatrix <- getEOMatrix_Traps()')
    EO_df = pandas2ri.ri2py_dataframe(r.get("EOMatrix"))
    return EO_df

def getEOdataframeAtPlots():
    pandas2ri.activate()
    r('source("EOData/EOData_asFrame.R")')
    r('EOMatrix <- getEOMatrix_Plots()')
    EO_df = pandas2ri.ri2py_dataframe(r.get("EOMatrix"))
    return EO_df