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

FIG_PATH =  Path("./Figures/Results")
RESULT_PATH =  Path("./Results")

from src.exhaustiveMethods_local import vanillaMST
from src.embed.distances import *
from src.embed.tree_embedders import *

def seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    

################################################### CONFIGURATION ###################################################


# NUMPOLY = 3
# NUMPOLY = 4
# NUMPOLY = 5
# NUMPOLY = 6
# NUMPOLY = 7
# NUMPOLY = 8
# NUMPOLY = 9
NUMPOLY = 10



# METHOD = "EXH"
METHOD = "SLL"
# METHOD = "NJ"


##################################################### EXPERIMENTS ###################################################
    
# RHS exp3
if METHOD == "EXH" or METHOD == "SLL":
    numeric_results = defaultdict(lambda: list())
    sampleType = "PolyWrappedGauss" 
    space = "Klein"
    nIters = 3
    convDiff = 1e-1
    dist2Points = 1e-5
    numSamples = 3
    
    
    
    title = "Convergence"
    method=METHOD
    
    # "Euclidean", "Klein_3+Precise", "Klein_3+Simple", "Klein_4+Simple"
    for mode in ["Klein_4+Simple"]:
        
        fstSize = int(mode[6])
        precise = mode[8:] == "Precise"
                
        for t in [0.6, 0.8, 0.95, 0.99, 0.9999, 0.9999999999]:
            print(t)
            
            numPoints = 20*NUMPOLY
            samplingParam= {"cov": np.eye(2)*0.1, "numPoly": NUMPOLY, "t": t}

            #samplingParam= {"cov": np.eye(2)*0.5, "numPoly": numPoints, "t": 1-1e-10}

            
            nMaxExpansions=int(1*np.sqrt(numPoints))

            # Retry logic with different seeds
            max_retries = 100  # Maximum number of different seeds to try
            base_seed = 0
            resultExp = None

            for retry in range(max_retries):
                print(f"Attempt {retry+1} with seed {base_seed + retry}")
                current_seed = base_seed + retry
                try:
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
                                                        seed=10,
                                                        slack=2,
                                                        selection=None,
                                                        threshold_improvement=0.0,
                                                        num_epochs=10000,
                                                        lr=1e-2,
                                                        early_stopping=True,
                                                        patience=100,
                                                        min_delta=1e-6)
                    print(f"Success with seed {current_seed} for numPoints={numPoints}")
                    break
                except Exception as e:
                    #print(f"Error with seed {current_seed} for numPoints={numPoints}: {str(e)}")
                    if retry == max_retries - 1:
                        raise e  # Re-raise if all attempts failed
                    continue




            correctList, errVorList, avgTimeList, ratioList, perImprovList, avgFST3List, avgFST4List = resultExp

            # correct = np.mean(correctList)*100
            # avgTime = np.mean(avgTimeList)
            # avgTime_std = np.std(avgTimeList)
            #print(perImprovList)
            # ratio = np.mean(ratioList)
            # ratio_std = np.std(ratioList)
            # perImprov = np.mean(perImprovList)*100
            # perImprov_std = np.std(perImprovList)*100


            best_ratio = np.max(ratioList)
            best_perImprov = np.max(perImprovList)*100


            print(f"\r{t}, {np.max(avgTimeList)}, {best_perImprov}, {best_ratio}")
            numeric_results["t"] += [t]*numSamples
            numeric_results["mode"] += [mode]*numSamples
            numeric_results["numPoints"] += [numPoints]*numSamples
            numeric_results["correct"] += correctList.tolist()
            numeric_results["avgTime"] += avgTimeList.tolist()
            numeric_results["ratio"] += ratioList.tolist()
            numeric_results["perImprov"] += perImprovList.tolist()
            numeric_results["avgFST3"] += avgFST3List.tolist()
            numeric_results["avgFST4"] += avgFST4List.tolist()

    
            df = pd.DataFrame(numeric_results)


            df.to_csv(
                RESULT_PATH / f'results_{title}_{method}_{NUMPOLY}.tsv',
                sep="\t",
                index=False
                )

        print("-"*30)
        



# NJ exp3
if METHOD == "NJ":
    numeric_results = defaultdict(lambda: list())
    sampleType = "PolyWrappedGauss" 
    space = "Klein"
    nIters = 3
    convDiff = 1e-1
    dist2Points = 1e-5
    numSamples = 3
    title = "Convergence"
    method="NJ"
    
    for mode in ["NJ"]:
        
        for t in [0.6, 0.8, 0.95, 0.99, 0.9999, 0.9999999999]:
            
            numPoints = 20*NUMPOLY
            samplingParam= {"cov": np.eye(2)*0.1, "numPoly": NUMPOLY, "t": t}
            
            nMaxExpansions=int(1*np.sqrt(numPoints))

            # Retry logic with different seeds
            max_retries = 1000  # Maximum number of different seeds to try
            current_seed = 0
            resultExp = None

            avgTimeList = []
            ratioList = []
            perImprovList = []

            MSTvalList = []

            i = 0
            while i < numSamples:

                for retry in range(max_retries):
                    current_seed = current_seed + retry
                    try:        

                        seed_all(current_seed)

                        points_klein = sampleExperiments(sampleType, numPoints, samplingParam=samplingParam)
                        vertices_dict_terminal = {f"T{j}": points_klein[j] for j in range(numPoints)}
                        mstEdgeList, mstVert2Adj, MSTval = vanillaMST(vertices_dict_terminal, space=space)


                        # Neighbor-Joining

                        start_time_nj = time.process_time()
                        distance_matrix_points_klein = distance_matrix(points_klein)
                        njGraph, nj_networkx = neighbor_joining(points_klein, distance_matrix_points_klein) 
                        del distance_matrix_points_klein ###
                        nj_adjacency_matrix, nj_nodes_ordered = adjacency_matrix(nj_networkx) 

                        nj_adjacency_matrix = torch.tensor(nj_adjacency_matrix)
                        poincare = klein_to_poincare(points_klein)
                        poincare = torch.tensor(poincare)

                        nj_steiners = train_steiner_embeddings(
                            nj_adjacency_matrix,
                            terminals_poincare=poincare,
                            num_epochs=10000,
                            lr=1)

                        combined = torch.cat([poincare, nj_steiners], dim=0)
                        combined_klein = poincare_to_klein(combined)
                        dist_matrix = distance_matrix(combined_klein, klein_distance)        
                        nj_length = (0.5*(dist_matrix*nj_adjacency_matrix)).sum()
                        nj_time = time.process_time() - start_time_nj

                        # Convert tensor to numpy value and clean up memory
                        nj_length_value = nj_length.detach().cpu().numpy().item()
                        
                        # Clean up tensors to free memory
                        del nj_adjacency_matrix, poincare, nj_steiners, combined, combined_klein, dist_matrix, nj_length
                        
                        #   # Force garbage collection to free GPU/CPU memory
                        #   if torch.cuda.is_available():
                        #       torch.cuda.empty_cache()
                        #   gc.collect()
                                
                        print(f"Success with seed {current_seed} for numPoints={numPoints}")
                        break

                    except Exception as e:
                        print(f"Error with seed {current_seed} for numPoints={numPoints}: {str(e)}")
                        if retry == max_retries - 1:
                            raise e  # Re-raise if all attempts failed
                        continue

                avgTimeList.append(nj_time)
                ratioList.append(nj_length_value/MSTval)
                perImprovList.append((MSTval-nj_length_value)/MSTval)
                MSTvalList.append(MSTval)

                current_seed += 1

                i += 1


            avgTime = np.mean(avgTimeList)
            avgTime_std = np.std(avgTimeList)
            print(perImprovList)
            ratio = np.mean(ratioList)
            ratio_std = np.std(ratioList)
            perImprov = np.mean(perImprovList)*100
            perImprov_std = np.std(perImprovList)*100

            best_ratio = np.max(ratioList)
            best_perImprov = np.max(perImprovList)*100


            print(f"\r{numPoints}, {avgTime}+-{avgTime_std}, {best_perImprov}, {perImprov}+-{perImprov_std}, {ratio}+-{ratio_std}")
            numeric_results["t"] += [t]*numSamples
            numeric_results["mode"] += [mode]*numSamples
            numeric_results["numPoints"] += [numPoints]*numSamples
            numeric_results["avgTime"] += avgTimeList
            numeric_results["ratio"] += ratioList
            numeric_results["perImprov"] += perImprovList

    
            df = pd.DataFrame(numeric_results)


            df.to_csv(
                RESULT_PATH / f'results_{title}_{method}_{NUMPOLY}.tsv',
                sep="\t",
                index=False
                )

        print("-"*30)