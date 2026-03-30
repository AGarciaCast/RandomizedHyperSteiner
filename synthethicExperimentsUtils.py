import numpy as np
import random

from src.utils.delaunay2d import *
from src.utils.hyperbolicWrappedGaussian import disc_pt_to_hyperboloid, hyperbolic_sampling, proj
from scipy.spatial.qhull import QhullError

from pathlib import Path
from src.formatResults import *
import warnings

from src.heuristicSteinerDT import heuristicSteinerDT
from src.utils.steinerUtils import plotSteinerTree, checkAngles, DISTANCE_F, plotSteinerTree_failure


try:
    from phcpy.phcpy2c3 import py2c_set_seed
    ALLOW_PRECISE = True

except ModuleNotFoundError:
    ALLOW_PRECISE = False


warnings.filterwarnings("ignore", category=RuntimeWarning) 

FIG_PATH =  Path("./Figures/Results")





def generate_klein_disk_triangles(central_radius=0.3, small_triangle_size=0.1):
    """
    Generate Klein disk triangle arrangement with controllable sizes.
    
    Parameters:
    -----------
    central_radius : float
        Size of the central equilateral triangle (default: 0.3)
    small_triangle_size : float
        Size of the small outward-pointing triangles (default: 0.1)
    
    Returns:
    --------
    np.array
        Array of shape (N, 2) containing Klein disk coordinates
        First 3 points are the central triangle vertices
        Remaining points are the outward triangle vertices
    """
    
    # Central equilateral triangle vertices
    central_triangle = np.array([
        [central_radius, -central_radius * 0.5],      # Bottom right
        [-central_radius, -central_radius * 0.5],     # Bottom left
        [0.0, central_radius * 0.833]                 # Top
    ])
    
    # Calculate center of central triangle
    triangle_center = np.mean(central_triangle, axis=0)
    
    # Start with central triangle vertices
    all_vertices = [central_triangle]
    
    # Create outward triangles for each vertex
    for i in range(3):
        vertex = central_triangle[i]
        
        # Direction from center to this vertex (outward direction)
        outward_direction = vertex - triangle_center
        outward_angle = np.arctan2(outward_direction[1], outward_direction[0])
        
        # Create two vertices that form an equilateral triangle with the original vertex
        # These extend outward from the central triangle
        angle1 = outward_angle + np.pi/3   # 60 degrees from outward direction
        angle2 = outward_angle - np.pi/3   # -60 degrees from outward direction
        
        new_vertex1 = vertex + small_triangle_size * np.array([np.cos(angle1), np.sin(angle1)])
        new_vertex2 = vertex + small_triangle_size * np.array([np.cos(angle2), np.sin(angle2)])
        
        # Only add vertices that are within the Klein disk (with margin)
        if np.linalg.norm(new_vertex1) < 0.9:
            all_vertices.append(new_vertex1.reshape(1, 2))
        if np.linalg.norm(new_vertex2) < 0.9:
            all_vertices.append(new_vertex2.reshape(1, 2))
    
    # Concatenate all vertices
    points = np.vstack(all_vertices)
    
    return points


def push_towards_boundary(points, push_factor=0.3, max_radius=0.95):
    """
    Push points in Klein disk towards the ideal points (boundary) in a controlled manner.
    
    Parameters:
    -----------
    points : np.array
        Array of shape (n, 2) containing points in Klein disk
    push_factor : float (0 to 1)
        Controls how much to push towards boundary
        0 = no change, 1 = push to boundary
    max_radius : float (0 to 1)
        Maximum allowed radius to prevent points from reaching boundary
    
    Returns:
    --------
    pushed_points : np.array
        New positions after pushing towards boundary
    """
    pushed_points = points.copy()
    
    for i in range(len(points)):
        x, y = points[i]
        
        # Calculate current distance from origin
        current_radius = np.sqrt(x**2 + y**2)
        
        # Skip if point is at origin
        if current_radius == 0:
            continue
            
        # Calculate direction (unit vector)
        direction_x = x / current_radius
        direction_y = y / current_radius
        
        # Calculate new radius using linear interpolation towards boundary
        # new_radius = current_radius + push_factor * (max_radius - current_radius)
        new_radius = current_radius + push_factor * (max_radius - current_radius)
        
        # Ensure we don't exceed max_radius
        new_radius = min(new_radius, max_radius)
        
        # Calculate new position
        pushed_points[i] = [new_radius * direction_x, new_radius * direction_y]
    
    return pushed_points






def seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    if ALLOW_PRECISE:
        py2c_set_seed(seed)


# SAMPLING FUNCTIONS

def meansPolygon(n, t):
    n = int(max(n, 1))
    t = np.clip(t, 0.0, 1.0)
    return np.array([[t*np.cos((2*np.pi*k)/n), t*np.sin((2*np.pi*k)/n)] for k in range(n)])

def uniformSamplingRest(numPoints):
    points = np.random.random(size=(2*numPoints, 2))*2 -1
    points = points[[DISTANCE_F["Euclidean"](p, np.zeros(2))<1 for p in points]]
    points = points[:numPoints]
    points = np.array([proj(p) for p in points])
    return points

def wrappedGaussian(numPoints, mean, cov, model="Klein"):
    hyperboloid_mean = disc_pt_to_hyperboloid(mean, metric='minkowski', model=model)
    gaussian_samples = hyperbolic_sampling(numPoints, hyperboloid_mean, cov**2, model=model)
    return gaussian_samples

def polyWrappedGaussSampling(numPoints, numPoly=10, t=0.88, cov=None, model="Klein"):
    """
    Sample points according to PolyWrappedGauss distribution.
    
    This function creates multiple clusters of points, where each cluster follows a 
    wrapped Gaussian distribution centered at points arranged in a regular polygon.
    
    Args:
        numPoints (int): Total number of points to sample
        numPoly (int): Number of polygon vertices (clusters)
        t (float): Radius of the polygon (between 0.0 and 1.0)
        cov (ndarray): Covariance matrix for the wrapped Gaussian (default: 0.25*eye(2))
        model (str): Hyperbolic model ("Klein" or "Euclidean")
    
    Returns:
        ndarray: Array of shape (numPoints, 2) containing the sampled points
    """
    if cov is None:
        cov = np.eye(2) * 0.25
    
    # Generate polygon vertices as cluster centers
    means = meansPolygon(n=numPoly, t=t)
    
    # Calculate points per cluster (ensure we get exactly numPoints total)
    points_per_cluster = numPoints // numPoly
    remaining_points = numPoints % numPoly
    
    points = []
    for i, mean in enumerate(means):
        # Add extra point to first few clusters if there are remaining points
        cluster_size = points_per_cluster + (1 if i < remaining_points else 0)
        cluster_points = wrappedGaussian(cluster_size, mean, cov, model=model)
        points.append(cluster_points)
    
    return np.concatenate(points)

def plotSampledPoints(points, sampleType="PolyWrappedGauss", samplingParam=None, 
                     showClusterCenters=True, figsize=(10, 8), title=None):
    """
    Plot points sampled from different distributions.
    
    Args:
        points (ndarray): Array of shape (n_points, 2) containing the sampled points
        sampleType (str): Type of sampling used ("PolyWrappedGauss", "WrappedGauss", "EucUniform", "EucGauss")
        samplingParam (dict): Parameters used for sampling (for showing cluster centers)
        showClusterCenters (bool): Whether to show cluster centers for PolyWrappedGauss
        figsize (tuple): Figure size (width, height)
        title (str): Custom title for the plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the sampled points
    ax.scatter(points[:, 0], points[:, 1], alpha=0.6, s=30, c='blue', label='Sampled Points')
    
    # Show cluster centers for PolyWrappedGauss
    if sampleType == "PolyWrappedGauss" and showClusterCenters and samplingParam is not None:
        numPoly = samplingParam.get("numPoly", 10)
        t = samplingParam.get("t", 0.88)
        cluster_centers = meansPolygon(n=numPoly, t=t)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                  c='red', s=100, marker='x', linewidths=3, label='Cluster Centers')
    
    # Add unit circle for hyperbolic space
    if sampleType in ["WrappedGauss", "PolyWrappedGauss"]:
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        ax.text(0.7, 0.7, 'Unit Disk', fontsize=12, alpha=0.7)
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set title
    if title is None:
        title = f"Points sampled using {sampleType}"
        if sampleType == "PolyWrappedGauss" and samplingParam:
            numPoly = samplingParam.get("numPoly", 10)
            t = samplingParam.get("t", 0.88)
            title += f" (n_clusters={numPoly}, radius={t})"
    
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# EXPERIMENT FUNCTIONS

def sampleExperiments(sampleType, numPoints, samplingParam=None):
    #TODO: What to do with the sampling when you try the Euclidean
    if sampleType=="EucUniform":
        points = uniformSamplingRest(numPoints)
    
    elif sampleType=="EucGauss":
        mean = np.zeros(2)
        cov = np.eye(2)

        if samplingParam is not None and isinstance(samplingParam, dict):
            if "mean" in samplingParam: 
                mean = samplingParam["mean"] 

            if "cov" in samplingParam:
                cov = samplingParam["cov"]

        points = np.random.multivariate_normal(mean, cov, numPoints) 

    elif sampleType=="WrappedGauss":
        mean = np.zeros(2)
        cov = np.eye(2)*0.25

        if samplingParam is not None and isinstance(samplingParam, dict):
            if "mean" in samplingParam: 
                mean = samplingParam["mean"] 

            if "cov" in samplingParam:
                cov = samplingParam["cov"]

        points = wrappedGaussian(numPoints, mean, cov, model="Klein")

    elif sampleType=="PolyWrappedGauss":

        cov = np.eye(2)*0.25
        numPoly = 10
        t = 0.88
        onlyMeans = False

        if samplingParam is not None and isinstance(samplingParam, dict):
            if "numPoly" in samplingParam:
                numPoly = samplingParam["numPoly"] 

            if "t" in samplingParam: 
                t = samplingParam["t"] 

            if "cov" in samplingParam:
                cov = samplingParam["cov"]
            

        means = meansPolygon(n=numPoly, t=t)

        points = []
        for mean in means:
            points.append(wrappedGaussian(numPoints//numPoly, mean, cov, model="Klein"))

        points = np.concatenate(points)
        
        
        
    elif sampleType=="failure-case":
        # assert numPoints == 9, "For the failure case, numPoints must be 9"
#         points=   np.array([
#     [0, 0],
#     [10, 0], 
#     [5, 8.66],
#     [0.58, -1],
#     [10.58, -1],
#     [9.42, -1],
#     [-0.58, -1],
#     [4.42, 9.66],
#     [5.58, 9.66],
# ])
    #     points=  np.array([
    # [-0.448, -0.388],  # Point 0
    # [0.448, -0.388],   # Point 1
    # [0.000, 0.388],    # Point 2
    # [-0.396, -0.478],  # Point 3
    # [0.500, -0.478],   # Point 4
    # [0.396, -0.478],   # Point 5
    # [-0.500, -0.478],  # Point 6
    # [-0.052, 0.478],   # Point 7
    # [0.052, 0.478],    # Point 8
    # ])    
    
    #     points =  push_towards_boundary(points, push_factor=0.5, max_radius=0.99)
        

        points=  np.array([
        
     [-0.59817631, -0.51806341],
 [ 0.59817631, -0.51806341],
 [ 0.      ,    0.689     ],
 [-0.51379182, -0.62018306],
 [ 0.65092133, -0.6222808 ],
 [ 0.51379182, -0.62018306],
[-0.65092133 ,-0.6222808 ],
 [-0.09054694 , 0.83233538],
[ 0.09054694 , 0.83233538]]
) 
        
        
        
        points = np.array([
        [-0.59817631, -0.51806341],
        [ 0.59817631, -0.51806341],
        [ 0.      ,    0.689     ],
        [-0.51379182, -0.62018306],
        [ 0.65092133, -0.6222808 ],
        [ 0.51379182, -0.62018306],
        [-0.65092133 ,-0.6222808 ],
        [-0.09054694 , 0.83233538],
        [ 0.09054694 , 0.83233538]
        ])

        # Define triangles as groups of indices
        triangles = [
        [0, 3, 6],
        [1, 4, 5],
        [2, 7, 8]
        ]

        # Scaling factor (>1 makes triangles bigger, <1 smaller)
        scale = 3

        new_points = points.copy()

        for tri in triangles:
            tri_points = points[tri]
            centroid = tri_points.mean(axis=0)
            new_points[tri] = centroid + scale * (tri_points - centroid)

        print(new_points)
        points =push_towards_boundary(new_points, push_factor=1/30, max_radius=0.92)
        
    else:
        raise ValueError("sampleType should either 'EucUniform', 'WrappedGauss', or 'PolyWrappedGauss'")
        
    return points

# MAIN FUNCTION

def syntheticExperiments(method="SSL", sampleType="EucUniform", numSamples=100, numPoints=3, fstSize=4, 
                         nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=True, space="Klein",
                         selection=None, nMaxExpansions=6,
                        slack=1, seed=None, samplingParam=None, plot_graph=None, verbose=False, 
                        threshold_improvement: float = 0,early_stopping=True, patience=100, min_delta=1e-6, 
                         num_epochs: int = 200,
<<<<<<< HEAD
                        lr: float = 0.01, expansion_mode="sqrt"):
=======
                        lr: float = 0.01, fancy_plot=False):
>>>>>>> main
                        
    if not ALLOW_PRECISE and precise and space=="Klein":
        raise ValueError("'precise' can't be True if PHCpy is not installed")
    
    if seed is not None:
        seed_all(seed)
    
    i = 0
   
    errorVoronoi = []
    perCorrect = []
    perTime = []
    accRatio = []
    totFST3 = []
    totFST4 = []
    steinerVals = []
    mstVals = []
    
    while i < numSamples:
        
       
        print(f"\rIter: {i}", end="")
            
        points = sampleExperiments(sampleType, numPoints, samplingParam=samplingParam)

        try:
         

            results = heuristicSteinerDT(points, 
                                method=method, 
                                space=space,
                                maxgroup=fstSize,
                                nIters=nIters, 
                                convDiff=convDiff,
                                dist2Points=dist2Points,
                                precise=precise, 
                                extendedResults=True,
                                selection=selection,
                                nMaxExpansions=nMaxExpansions,
                                slack=slack,
                                 threshold_improvement=threshold_improvement,
                                num_epochs=num_epochs,
                                lr=lr,
                                early_stopping=early_stopping, patience=patience, 
                                min_delta=min_delta,expansion_mode=expansion_mode
                              )

            
            resultGraph = results["resultGraph"]
            verticesDict = results["verticesDict"]
            
            if numPoints>2:
                runPer = checkAngles(resultGraph, verticesDict, space=space)
            else:
                runPer = 1
            
            accRatio += [results["ratio"]]
            perTime += [results["methodTime"]]
            totFST3 += [results["numFST3"]]
            totFST4 += [results["numFST4"]]
            perCorrect += [runPer]
            steinerVals += [results["steinerVal"]]
            mstVals += [results["mstVal"]]
            i += 1
            
            if verbose:
                print(i, runPer*100, results["methodTime"])

            if numSamples == 1 and plot_graph is not None:
                if sampleType == "failure-case" or fancy_plot:
                    plotSteinerTree_failure(resultGraph,
                                verticesDict, 
                                results["mstGraph"], 
                                results["edgesDT"], 
                                space, 
                                plot_graph,
                                FIG_PATH)

                else:
                    
                    plotSteinerTree(resultGraph,
                                    verticesDict, 
                                    results["mstGraph"], 
                                    results["edgesDT"], 
                                    space, 
                                    plot_graph,
                                    FIG_PATH)


        except QhullError:
            errorVoronoi+=[1]

    
    errorVoronoi = np.array(errorVoronoi)
    perCorrect = np.array(perCorrect)
    perTime = np.array(perTime)
    accRatio = np.array(accRatio)
    improvement = (1.0 - accRatio)
    totFST3 = np.array(totFST3)
    totFST4 = np.array(totFST4)
    steinerVals = np.array(steinerVals)
    mstVals = np.array(mstVals)



    return perCorrect, errorVoronoi, perTime, accRatio, improvement, totFST3, totFST4   
    #return resultGraph, verticesDict, perCorrect, errorVoronoi, perTime, accRatio, improvement, totFST3, totFST4, steinerVals, mstVals, points

