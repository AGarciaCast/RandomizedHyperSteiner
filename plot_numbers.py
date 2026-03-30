import numpy as np

from src.utils.delaunay2d import *
from collections import defaultdict 

import json
from pathlib import Path
from src.formatResults import *
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from src.smithMethods import *

from synthethicExperimentsUtils import *
    
warnings.filterwarnings("ignore", category=RuntimeWarning) 

FIG_PATH =  Path("./Figures/Results/failure")
RESULT_PATH =  Path("./Results/failure")

samplingParam= None


# nIters = 8
# convDiff = 1e-4


# numPoints = 5



# space = "Klein"
# sampleType = "PolyWrappedGauss"
# samplingParam= {"cov": np.eye(2)*0.01, "numPoly": numPoints, "t": 0.9}

# # precise = False
# # fstSize = 4

# # nIters = 3
# # # convDiff = 1e-1
# # # dist2Points = 1e-1
# # convDiff = 1e-1
# # dist2Points = 1e-5
# # numSamples = 1
# # slack=0
# # selection=None #Standard selection
# # method="EXH_GLOB"
# # # method="SLL"
# # nMaxExpansions = int(1*np.sqrt(numPoints))
# # resultExp = syntheticExperiments(method=method,
# #                                                 numSamples=numSamples, 
# #                                                 numPoints=numPoints,
# #                                                 nIters=nIters,
# #                                                 fstSize=fstSize,
# #                                                 convDiff=convDiff,
# #                                                 dist2Points=dist2Points,
# #                                                 precise=precise, 
# #                                                 sampleType=sampleType,
# #                                                 samplingParam=samplingParam,
# #                                                 space=space,
# #                                                 nMaxExpansions=nMaxExpansions,
# #                                                 selection=selection,
# #                                                 plot_graph="DT",
# #                                                 slack=slack,
# #                                                 seed=10,
# #                                                 fancy_plot=True)


# nIters = 3
# convDiff = 1e-1
# dist2Points = 1e-5
# numSamples = 1

# selection = ("Prob", (0.3, 0.6))
# temp = 0.7
# alpha = 0.7
# slack = 2


# title = "TestingExh"

# # "Euclidean", "Klein_3+Precise", "Klein_3+Simple", "Klein_4+Simple"


# fstSize = 4

# precise=False  

# nMaxExpansions = int(1*np.sqrt(numPoints))

# method="EXH_GLOB"
# #method="SLL"
# # method="EXH_LOC"
# resultExp = syntheticExperiments(method=method,
#                 numSamples=numSamples, 
#                 numPoints=numPoints,
#                 nIters=nIters,
#                 fstSize=fstSize,
#                 convDiff=convDiff,
#                 dist2Points=dist2Points,
#                 precise=precise, 
#                 sampleType=sampleType,
#                 samplingParam=samplingParam,
#                 space=space,
#                 seed=10,
#                 selection=selection,
#                 nMaxExpansions=nMaxExpansions,
#                 plot_graph="DT",
#                 num_epochs= 10000,
#                 lr=1e-2,
#                 fancy_plot=True,
#                 )



space = "Klein"
sampleType = "PolyWrappedGauss"
samplingParam= {"cov": np.eye(2)*0.02, "numPoly": 5, "t": 0.9}
# nIters = 4
# convDiff = 5e-3
nIters = 3
convDiff = 1e-1
dist2Points = 1e-5
numSamples = 1

selection = ("Prob", (0.3, 0.6))
temp = 0.7
alpha = 0.7
slack = 2


title = "TestingExh"

# "Euclidean", "Klein_3+Precise", "Klein_3+Simple", "Klein_4+Simple"


fstSize = 4

precise=False  

numPoints = 5
nMaxExpansions = int(1*np.sqrt(numPoints))

# method="EXH_GLOB"
method="SLL"
# method="EXH_LOC"
resultExp = syntheticExperiments(method=method,
                numSamples=numSamples, 
                numPoints=numPoints,
                nIters=nIters,
                fstSize=fstSize,
                convDiff=convDiff,
                dist2Points=dist2Points,
                precise=precise, 
                sampleType=sampleType,
                samplingParam=samplingParam,
                space=space,
                seed=30,
                selection=selection,
                nMaxExpansions=nMaxExpansions,
                plot_graph="DT",
                num_epochs= 10000,
                lr=1e-2,
                fancy_plot=True,
                )
