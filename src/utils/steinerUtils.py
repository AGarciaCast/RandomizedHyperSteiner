
from src.utils.fullSteinerSolverHyperbolic import steinerPoint3Hyp, steinerPoints4Hyp, DISTANCE_HYP, hyperbolicInnerAngleTriangle
from src.utils.fullSteinerSolverEuclidean import steinerPoint3Euc, steinerPoints4Euc, l2Distance, euclideanInnerAngleTriangle
from math import inf #used for Dijkstra functions
from collections import defaultdict 
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import numpy as np

from src.embed.distances import *
from src.embed.tree_embedders import *
from src.utils.datastructures import *


DISTANCE_F = {**DISTANCE_HYP, 
             "Euclidean": l2Distance}


def innerAngleTriangle(u, v, w, space="Klein"):
    if space not in ["Klein", "Euclidean"]:
        raise ValueError("space should be either 'Klein' or 'Euclidean'")
    
    if space=="Klein":
        result = hyperbolicInnerAngleTriangle(u, v, w, model="Klein")
    else:
        result = euclideanInnerAngleTriangle(u, v, w)
    
    return result


def steinerPoint3(vert, model="Klein", dist2Points=1e-1, precise=True):

    if model in ["Klein", "Half"]:
        result = steinerPoint3Hyp(vert, model, precise=precise, dist2Points=dist2Points)
    elif model == "Euclidean":
        result = steinerPoint3Euc(vert, dist2Points=dist2Points)
    else:
        raise ValueError("model should be either 'Klein', 'Half', or 'Euclidean'")
    
    return result


def steinerPoints4(vert, topo, model="Klein", nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=True):

    if model in ["Klein", "Half"]:
        result = steinerPoints4Hyp(vert, topo, model, nIters=nIters, convDiff=convDiff,
                                   precise=precise, dist2Points=dist2Points)
    elif model == "Euclidean":
        result = steinerPoints4Euc(vert, topo, nIters=nIters, convDiff=convDiff, dist2Points=dist2Points)
    else:
        raise ValueError("model should be either 'Klein', 'Half', or 'Euclidean'")
    
    return result



def steinerRatio(vert, mst, model="Klein", idxTerminals=None, idxSteinerPoint=0,
                 nIters=100, convDiff=1e-2, dist2Points = 1e-1, precise=True):
    
    if idxSteinerPoint is None:
        idxTerminals = [i for i in range(len(vert))]
        
    numVert = len(vert)
    if numVert==3:
        return steinerRatio3(vert=vert,
                             mst=mst,
                             idxTerminals=idxTerminals,
                             model=model,
                             idxSteinerPoint=idxSteinerPoint,
                             precise=precise,
                             dist2Points=dist2Points
                             )
    elif numVert==4:
        return steinerRatio4(vert=vert,
                             mst=mst,
                             idxTerminals=idxTerminals,
                             model=model,
                             idxSteinerPoint=idxSteinerPoint,
                             nIters=nIters, 
                             convDiff=convDiff,
                             precise=precise,
                             dist2Points=dist2Points
                             )

def steinerRatio3(vert, mst, idxTerminals, model="Klein", idxSteinerPoint=0, dist2Points=1e-1, precise=True):
    
    steinerPoint = steinerPoint3(vert, model, precise=precise, dist2Points=dist2Points)
    smt = inf
    if steinerPoint is None:
        ratio = 1.0
        topology = None
        
    else:

        smt = 0
        for i in range(3):
            smt += DISTANCE_F[model](steinerPoint, vert[i])

        ratio = smt / mst
        
        if ratio >=1:
            ratio = 1.0
            steinerPoint = None
            topology = None
        else:
            topology = [[f"T{idxTerminals[i]}", f"S{idxSteinerPoint}"] for i in range(3)]

    return ratio, steinerPoint, topology, smt



def steinerRatio4(vert, mst, idxTerminals, model="Klein", idxSteinerPoint=0, nIters=100,
                  convDiff=1e-2, dist2Points=1e-1, precise=True):
    topologiesIdx = [[[0,1], [2, 3]], [[0,2], [1, 3]]]
    bestSteinerPoints = None
    bestSMT = inf
    bestTopo = None
    
    for topo in topologiesIdx:
        steinerPoints = steinerPoints4(vert, topo, model=model,
                                        nIters=nIters,
                                        convDiff=convDiff,
                                        dist2Points=dist2Points,
                                        precise=precise)
        
        
        if steinerPoints is None:
            ratio = 1.0
            fstTopology = None
        
        else:

            smt = DISTANCE_F[model](steinerPoints[0], steinerPoints[1])
            for i in range(2):
                for j in range(2):
                    smt += DISTANCE_F[model](steinerPoints[i], vert[topo[i][j]])
                        
            if smt < bestSMT:
                bestSMT = smt 
                bestSteinerPoints = steinerPoints
                bestTopo = topo
            
            
        if bestSteinerPoints is not None:  

            ratio = bestSMT / mst
            
            if ratio >=1:
                ratio = 1.0
                bestSteinerPoints = None
                fstTopology = None
            else:
                
                fstTopology = [[f"T{idxTerminals[bestTopo[i][j]]}", f"S{idxSteinerPoint + i}"] for i in range(2) for j in range(2)]
                fstTopology.append([f"S{idxSteinerPoint}", f"S{idxSteinerPoint + 1}"])
    
    
    return ratio, bestSteinerPoints, fstTopology, bestSMT



def bestSteinerFST4(vert, model="Klein", nIters=100,
                  convDiff=1e-2, dist2Points=1e-1, precise=True):
    topologiesIdx = [[[0,1], [2, 3]], [[0,2], [1, 3]]]
    bestSteinerPoints = None
    bestSMT = inf
    bestTopo = None
    
    for topo in topologiesIdx:
        steinerPoints = steinerPoints4(vert, topo, model=model,
                                        nIters=nIters,
                                        convDiff=convDiff,
                                        dist2Points=dist2Points,
                                        precise=precise)
        
        
        if steinerPoints is not None:
           
            smt = DISTANCE_F[model](steinerPoints[0], steinerPoints[1])
            for i in range(2):
                for j in range(2):
                    smt += DISTANCE_F[model](steinerPoints[i], vert[topo[i][j]])
                        
            if smt < bestSMT:
                bestSMT = smt 
                bestSteinerPoints = steinerPoints
                bestTopo = topo
      
    
    return  bestSteinerPoints, bestTopo


def checkAngles(steinerGraph, verticesDict, space="Klein"):
    vertices = verticesDict.keys()
    vert2Adj = defaultdict(list)
    for el in vertices:
        for e in steinerGraph:
            if el in e:
                vert2Adj[el]+=list(set(e) - {el})


    numCorrectRun = 0
    auxCount = 0
    for k, adjList in vert2Adj.items():
        numNeigh = len(adjList)
        if numNeigh > 1:
            angles = []
            auxNeigh = adjList[0]
            for j in range(1, numNeigh):
                angles.append(innerAngleTriangle(verticesDict[auxNeigh],
                                                verticesDict[k],
                                                verticesDict[adjList[j]],
                                                space=space))

            adjList = [auxNeigh] + [x for _, x in sorted(zip(angles, adjList[1:]))]


            auxIdx = 0 if numNeigh==2 else numNeigh
            angles = []

            for j in range(auxIdx):
                if k[0] == "S" or (adjList[j][0]=="T" and adjList[(j+1)%numNeigh][0]=="T"):
                    angles.append(innerAngleTriangle(verticesDict[adjList[j]],
                                                    verticesDict[k],
                                                    verticesDict[adjList[(j+1)%numNeigh]],
                                                    space=space))

            angles = np.array(angles)
            if k[0] == "S":
                bolAux = np.all((angles*180/np.pi>119) & (angles*180/np.pi<121))
            else:
                bolAux = np.all(angles*180/np.pi > 119)


            numCorrectRun += bolAux 
            auxCount += 1

    return numCorrectRun/auxCount


def plotSteinerTree(steinerGraph, verticesDict, mstGraph=None, edgesDT=None,
                    space="Klein", additional="MST", fig_path=None):
    
    fig = plt.figure(figsize=(6,6), dpi=200)
    ax = fig.add_subplot(111, aspect='equal') 
    
    if space=="Klein":
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
        ax.add_patch(circ)
        
    if mstGraph is not None or edgesDT is not None:
        if additional == "MST":
            H = nx.Graph(mstGraph)
        elif additional == "DT":
            H = nx.Graph([[f"T{p}" for p in edge] for edge in edgesDT])
        else:
            raise ValueError("additional should either 'MST' or 'DT'")

        nx.draw_networkx_edges(H, pos=verticesDict, style='--', alpha=0.4,)
    
    G = nx.Graph(steinerGraph)
    G.add_node("T0")
    color_map = []

    for node in G:
        if node[0]=="S":
            color_map.append('tab:blue')
        else: 
            
            color_map.append('tab:red') 
    if space=="Klein":
        node_size = [-30*DISTANCE_F["Euclidean"](np.array([0.0, 0.0]), verticesDict[p]) + 45 for p in  G.nodes()]
    else:
        node_size=50
        

    nx.draw(G, node_color=color_map, pos=verticesDict,node_size=node_size,)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Terminal',markerfacecolor='tab:red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Steiner',markerfacecolor='tab:blue', markersize=12),        
    ]

    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.10, 1))
    plt.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path / f"{space}_dtSMT.pdf", format='pdf', bbox_inches='tight')
    plt.show()




def plotSteinerTree_failure(steinerGraph, verticesDict, mstGraph=None, edgesDT=None,
                    space="Klein", additional="MST", fig_path=None):
    
    
        
    if mstGraph is not None and edgesDT is not None:
        fig = plt.figure(figsize=(6,6), dpi=200)
        ax = fig.add_subplot(111, aspect='equal') 
        
        if space=="Klein":
            circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
            ax.add_patch(circ)
            
        
      
        H = nx.Graph(mstGraph)
        DT = nx.Graph([[f"T{p}" for p in edge] for edge in edgesDT])
       

        nx.draw_networkx_edges(DT, pos=verticesDict, style='--', alpha=0.4, width=1.5)
        

        H.add_node("T0")
        color_map = []

        for node in H:
            if node[0]=="S":
                color_map.append('tab:blue')
            else: 
                
                color_map.append('tab:red') 
        if space=="Klein":
            node_size = [-30*DISTANCE_F["Euclidean"](np.array([0.0, 0.0]), verticesDict[p]) + 200 for p in  H.nodes()]
        else:
            node_size=50
            
        # node_size=60

        nx.draw(H, node_color=color_map, pos=verticesDict,node_size=node_size, width=3,)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Terminal',markerfacecolor='tab:red', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Steiner',markerfacecolor='tab:blue', markersize=12),        
        ]

        
        # plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.10, 1))
        plt.tight_layout()
        if fig_path is not None:
            fig.savefig(fig_path / f"failure/{space}_MST.pdf", format='pdf', bbox_inches='tight')
        plt.show()
        
        plt.close()
    
    fig = plt.figure(figsize=(6,6), dpi=200)
    ax = fig.add_subplot(111, aspect='equal') 
    
    if space=="Klein":
        circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
        ax.add_patch(circ)
        
        
    if mstGraph is not None or edgesDT is not None:        
        if additional == "MST":
            H = nx.Graph(mstGraph)
        elif additional == "DT":
            H = nx.Graph([[f"T{p}" for p in edge] for edge in edgesDT])
        else:
            raise ValueError("additional should either 'MST' or 'DT'")

        nx.draw_networkx_edges(H, pos=verticesDict, style='--', alpha=0.4, width=1.5)
    
    
    G = nx.Graph(steinerGraph)
    G.add_node("T0")
    color_map = []

    for node in G:
        if node[0]=="S":
            color_map.append('tab:blue')
        else: 
            
            color_map.append('tab:red') 
    if space=="Klein":
        node_size = [-30*DISTANCE_F["Euclidean"](np.array([0.0, 0.0]), verticesDict[p]) + 200 for p in  G.nodes()]
    else:
        node_size=50
        

    nx.draw(G, node_color=color_map, pos=verticesDict,node_size=node_size, width=3,)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Terminal',markerfacecolor='tab:red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Steiner',markerfacecolor='tab:blue', markersize=12),        
    ]

    
    # plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.10, 1))
    plt.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path / f"failure/{space}_dtSMT.pdf", format='pdf', bbox_inches='tight')
    plt.show()




def global_optimization(terminals_klein, vertices, graph, num_epochs=100, lr=0.01, verbose=False, plot=False,  early_stopping=True, patience=1000, min_delta=1e-6, ):
    terminals_poincare = klein_to_poincare(terminals_klein)

    steiners_klein, steiner_keys = extract_steiner_coordinates(vertices, extra_keys=True)
    
    # If there are no Steiner points to optimize, return vertices unchanged
    if steiner_keys is None or len(steiner_keys) == 0:
        return vertices
    
    steiners_poincare = klein_to_poincare(steiners_klein)

    hs_networks = edges_to_networkx(graph)
    hs_adjacency_matrix, hs_nodes_ordered = adjacency_matrix(hs_networks) 
    hs_adjacency_matrix = torch.tensor(hs_adjacency_matrix, dtype=torch.float64)
    

    terminals_poincare = torch.tensor(terminals_poincare, dtype=torch.float64)
    steiners_poincare = torch.tensor(steiners_poincare, dtype=torch.float64)
    
    optimized_steiners_poincare = train_steiner_embeddings(
            hs_adjacency_matrix,
            terminals_poincare= terminals_poincare,
            num_epochs=num_epochs,
            lr=lr,
            steiners_poincare = steiners_poincare,
            verbose=verbose,
            plot=plot,
            early_stopping=early_stopping, patience=patience, min_delta=min_delta, 
                           restore_best_weights=False
            )
    
    # Handle case where no Steiner points were optimized
    if optimized_steiners_poincare.numel() == 0:
        return vertices
    
    optimized_steiners_klein = poincare_to_klein(optimized_steiners_poincare)
    
    
    for i, key in enumerate(steiner_keys):
        vertices[key] = optimized_steiners_klein[i].detach().numpy()
    
    return vertices 




def lorentz_factor(point):
    """
    Compute the Lorentz factor γ = 1/√(1-|u|²) for a point in Klein disk.
    
    Args:
        point: numpy array of shape (..., 2) representing point(s) in Klein disk
        
    Returns:
        Lorentz factor(s)
    """
    norm_squared = np.sum(point**2, axis=-1)
    
    # Ensure we're inside the unit disk (add small epsilon for numerical stability)
    eps = 1e-10
    norm_squared = np.clip(norm_squared, 0, 1 - eps)
    
    return 1.0 / np.sqrt(1 - norm_squared)


def hyperbolic_barycenter(points, weights=None):
    """
    Compute hyperbolic barycenter using Lorentz factors.
    
    Args:
        points: numpy array of shape (n, 2) where each row is a point in Klein disk
        weights: optional weights for each point. If None, uniform weights are used.
        
    Returns:
        Barycenter point in Klein disk
    """
    points = np.array(points)
    n_points = len(points)
    
    if weights is None:
        weights = np.ones(n_points)
    else:
        weights = np.array(weights)
    
    # Compute Lorentz factors
    gamma = lorentz_factor(points)
    
    # Weighted Lorentz factors
    weighted_gamma = weights * gamma
    
    # Compute barycenter
    numerator = np.sum(weighted_gamma[:, np.newaxis] * points, axis=0)
    denominator = np.sum(weighted_gamma)
    
    return numerator / denominator