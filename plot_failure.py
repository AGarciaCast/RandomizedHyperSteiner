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

space = "Klein"
sampleType = "failure-case"
numeric_results = defaultdict(lambda: list())
space = "Klein"
# nIters = 8
# convDiff = 1e-4

precise = False
fstSize = 4

nIters = 3
convDiff = 1e-1
dist2Points = 1e-1
numSamples = 1
numPoints = 9
slack=0
selection=None #Standard selection
# method="EXH_GLOB"
method="SLL"
nMaxExpansions = int(3*np.sqrt(numPoints))
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
                                                nMaxExpansions=nMaxExpansions,
                                                selection=selection,
                                                plot_graph="DT",
                                                slack=slack,
                                                seed=10)