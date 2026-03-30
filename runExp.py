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


import argparse


warnings.filterwarnings("ignore", category=RuntimeWarning) 

FIG_PATH =  Path("./Figures/Results")
RESULT_PATH =  Path("./Results")


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--Method", default="SLL")
parser.add_argument("-n", "--numSamples", default=100, type=int)
parser.add_argument("-u", "--Uniform", action='store_true')
parser.add_argument("-c", "--Centered", action='store_true')
parser.add_argument("-i", "--Ideal", action='store_true')
parser.add_argument("-pc", "--PolyConv", action='store_true')


nIters = 3
convDiff = 1e-1
dist2Points = 1e-5
slack=1
selection=None #Standard selection

# Read arguments from command line
args = parser.parse_args()

method=args.Method
numSamples = args.numSamples

listNumPoints = list(range(50, 101, 10)) 



def pipelinePerf(listNumPoints, numeric_results, method="SLL", sampleType="EucUniform",
                numSamples=100, fstSize=4, nIters=100, convDiff=1e-2, dist2Points=1e-1,
                precise=True, space="Klein", selection=None, slack=1,
                samplingParam=None, plot_graph=None, verbose=False, seed=10):
    
    for numPoints in listNumPoints:
            
            nMaxExpansions=int(1*np.sqrt(numPoints))
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
                                                slack=slack,
                                                plot_graph=plot_graph,
                                                verbose=verbose,
                                                seed=seed)
            


            correctList, errVorList, avgTimeList, ratioList, perImprovList, avgFST3List, avgFST4List = resultExp

            correct = np.mean(correctList)*100
            avgTime = np.mean(avgTimeList)
            ratio = np.mean(ratioList)
            perImprov = np.mean(perImprovList)*100
            avgFST3 = np.mean(avgFST3List)
            avgFST4 = np.mean(avgFST4List)

            print(f"\r{mode}, {numPoints}: {correct}, {avgTime}, {ratio}, {perImprov}, {avgFST3}, {avgFST4}")

            numeric_results["mode"] += [mode]*numSamples
            numeric_results["numPoints"] += [numPoints]*numSamples
            numeric_results["correct"] += correctList.tolist()
            numeric_results["avgTime"] += avgTimeList.tolist()
            numeric_results["ratio"] += ratioList.tolist()
            numeric_results["perImprov"] += perImprovList.tolist()
            numeric_results["avgFST3"] += avgFST3List.tolist()
            numeric_results["avgFST4"] += avgFST4List.tolist()

      




if args.Uniform:
    print("-"*10 + "Uniform" + "-"*10)
    numeric_results = defaultdict(lambda: list())
    samplingParam= None
    space = "Klein"
    title = "Uniform"
    sampleType= "EucUniform"

    for mode in ["3+Precise", "3+Simple", "4+Precise", "4+Simple"]:
        fstSize = int(mode[0])
        precise = mode[2:] == "Precise"
        
        
        pipelinePerf(listNumPoints,
                    numeric_results,
                    method=method,
                    sampleType=sampleType,
                    numSamples=numSamples,
                    fstSize=fstSize, 
                    nIters=nIters,
                    convDiff=convDiff,
                    dist2Points=dist2Points,
                    precise=precise, 
                    space=space,
                    selection=selection,
                    slack=slack,
                    seed=10,
                    samplingParam=samplingParam,
                    plot_graph=None,
                    verbose=False,
                    )

            

        print("-"*30)

        df = pd.DataFrame(numeric_results)

        df.to_csv(
            RESULT_PATH / f'results_{title}_{method}.tsv',
            sep="\t",
            index=False
            )


if args.Centered:
    print("-"*10 + "Centered" + "-"*10)
    numeric_results = defaultdict(lambda: list())
    samplingParam= {"cov": np.eye(2)*0.5}
    title = "CenteredGauss"
    
    for mode in ["Euclidean",  "3+Precise", "3+Simple", "4+Precise", "4+Simple"]:
        if mode == "Euclidean":
            sampleType = "EucGauss"
            fstSize = 4
            space = "Euclidean"
            precise=True
        else:
            sampleType = "WrappedGauss"
            space = "Klein"
            fstSize = int(mode[0])
            precise = mode[2:] == "Precise"
                
        pipelinePerf(listNumPoints,
                    numeric_results,
                    method=method,
                    sampleType=sampleType,
                    numSamples=numSamples,
                    fstSize=fstSize, 
                    nIters=nIters,
                    convDiff=convDiff,
                    dist2Points=dist2Points,
                    precise=precise, 
                    space=space,
                    selection=selection,
                    slack=slack,
                    seed=10,
                    samplingParam=samplingParam,
                    plot_graph=None,
                    verbose=False,
                    )

           
            

        print("-"*30)

    
        df = pd.DataFrame(numeric_results)


        df.to_csv(
            RESULT_PATH / f'results_{title}_{method}.tsv',
            sep="\t",
            index=False
            )


if args.Ideal:
    print("-"*10 + "Ideal" + "-"*10)
    numeric_results = defaultdict(lambda: list())
    space = "Klein"
    sampleType = "PolyWrappedGauss"
    samplingParam= {"cov": np.eye(2)*0.5, "numPoly": 15, "t": 0.9}

    for mode in ["3+Precise", "3+Simple", "4+Precise", "4+Simple"]:
        fstSize = int(mode[0])
        precise = mode[2:] == "Precise"
       
        pipelinePerf(listNumPoints,
                    numeric_results,
                    method=method,
                    sampleType=sampleType,
                    numSamples=numSamples,
                    fstSize=fstSize, 
                    nIters=nIters,
                    convDiff=convDiff,
                    dist2Points=dist2Points,
                    precise=precise, 
                    space=space,
                    selection=selection,
                    slack=slack,
                    seed=10,
                    samplingParam=samplingParam,
                    plot_graph=None,
                    verbose=False,
                    )


        print("-"*30)


        df = pd.DataFrame(numeric_results)


        df.to_csv(
            RESULT_PATH / f'results_{sampleType}_{method}.tsv',
            sep="\t",
            index=False
            )

    

if args.PolyConv:
    print("-"*10 + "Poly convergence" + "-"*10)
    numeric_results = defaultdict(lambda: list())
    space = "Klein"
    sampleType = "PolyWrappedGauss"
    samplingParam= dict()
    
    if method == "EXH":
        precise = False
        fstSize = 3
    else:
        precise = True
        fstSize = 4
   

    title = "Conv2Ideal"
    
    for mode in ["3+1", "3+n", "4+1", "4+n", "5+1", "5+n", "6+1", "6+n"]:
        samplingParam["numPoly"] = int(mode[0])
        
        if mode[2]=="1":
            samplingParam["cov"]= np.eye(2)*0.01
            numPoints = samplingParam["numPoly"]
        else:
            samplingParam["cov"]= np.eye(2)*0.15
            numPoints = 20*samplingParam["numPoly"]

        nMaxExpansions=int(1*np.sqrt(numPoints))
        
        for t in [0.4, 0.6, 0.8, 0.9, 0.95, 0.98]:
            
            samplingParam["t"]= t

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
                                                annealing=None,
                                                nMaxExpansions=nMaxExpansions,
                                                selection=selection,
                                                slack=slack,
                                                seed=10)


            correctList, errVorList, avgTimeList, ratioList, perImprovList, avgFST3List, avgFST4List = resultExp
            
            correct = np.mean(correctList)*100
            avgTime = np.mean(avgTimeList)
            ratio = np.mean(ratioList)
            perImprov = np.mean(perImprovList)*100
            avgFST3 = np.mean(avgFST3List)
            avgFST4 = np.mean(avgFST4List)
            
            print(f"\r{mode}, {t}: {correct}, {avgTime}, {ratio}, {perImprov}, {avgFST3}, {avgFST4}")

            numeric_results["mode"] += [mode]*numSamples
            numeric_results["paramCurves"] += [t]*numSamples
            numeric_results["correct"] += correctList.tolist()
            numeric_results["avgTime"] += avgTimeList.tolist()
            numeric_results["ratio"] += ratioList.tolist()
            numeric_results["perImprov"] += perImprovList.tolist()
            numeric_results["avgFST3"] += avgFST3List.tolist()
            numeric_results["avgFST4"] += avgFST4List.tolist()
            
            

        print("-"*30)
        
        df = pd.DataFrame(numeric_results)

        df.to_csv(
            RESULT_PATH / f'results_{title}_{method}.tsv',
            sep="\t",
            index=False
            )
