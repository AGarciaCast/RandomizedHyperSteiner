from src.utils.steinerUtils import *
from src.utils.delaunay2d import compute_Voronoi_Delaunay
from src.utils.graphsUtils import DisjSet
from collections import defaultdict 
from queue import PriorityQueue


try:
    from src.utils.fullSteinerSolverHyperbolic import *
    ALLOW_PRECISE = True

except ModuleNotFoundError:
    ALLOW_PRECISE = False


def expansion(nExpansions, points_og_list, verticesDictSteiner= None,
              space="Klein", triangMeth="DT", selection=("PROB", (0.3, 0.6)),
              dist2Points=1e-1, precise=True, idxSteinerPoint=0, baricenter=False):
    
    if verticesDictSteiner is None:
        verticesDictSteiner = dict()
    
    points_SMT = np.vstack(points_og_list)
    
    
    #print(nExpansions)
        
    if selection[0] != "ALL":
        p = np.random.uniform(*selection[1])
        
    for exp in range(nExpansions):
        
  
        #TODO: make a better optimization that is not that naive, i.e., only compute steiner point if not exists before
        steinerPoints = []
        if triangMeth == "DT":
            #{'original_points': S, 'Voronoi_points': V, 'Delaunay_triangulation': tri_list, 'Delaunay_edges': edges,
            # 'map_DelaunayEdge2Triangle': edge_map, 'map_DataPoint2VoronoiCell': voronoi_cell_map} 
            computedVoronoiDelaunay = compute_Voronoi_Delaunay(points_SMT, space = space)
            triangulationData = {"mapEdge2Triangle": computedVoronoiDelaunay["map_DelaunayEdge2Triangle"],
                                "triangulationListTriang": computedVoronoiDelaunay["Delaunay_triangulation"],
                                "triangulationListEdges": computedVoronoiDelaunay["Delaunay_edges"]
            }
            
        else:
            #TODO: obtain gabriel graph
            raise ValueError("triangMeth should be DT")
        
        triangulationListTriang = triangulationData["triangulationListTriang"]
        
        for idxPointsTriangle in triangulationListTriang:
            if selection[0] != "ALL" and not np.random.binomial(1, p):
                continue
            
            vert = points_SMT[idxPointsTriangle]
            
            if not baricenter:
                steinerPoint = steinerPoint3(vert, model=space, dist2Points=dist2Points,
                                            precise=precise)
                            
                if steinerPoint is not None:
                    steinerPoints.append(steinerPoint)
            else:
                if space == "Klein":
                    steinerPoint = hyperbolic_barycenter(vert)
                else:
                    steinerPoint = np.mean(vert, axis=0)
                
                steinerPoints.append(steinerPoint)
                
        
        points_SMT = np.vstack(points_og_list+steinerPoints)
    
    for p in steinerPoints:
        verticesDictSteiner[f"S{idxSteinerPoint}"] = p
        idxSteinerPoint += 1
    
    return verticesDictSteiner, idxSteinerPoint, triangulationData["triangulationListEdges"]





def delaunayMST(verticesDictTerminal, verticesDictSteiner=None, space="Klein"):
    
    pointsSMT = list(verticesDictTerminal.values())
    idices_points = list(verticesDictTerminal.keys()) 
    
    if verticesDictSteiner is not None:
        idices_points += list(verticesDictSteiner.keys())
        pointsSMT += list(verticesDictSteiner.values())
        
    numPoints = len(idices_points)
    # Building the edges-distance map:    
    edgesWeight = PriorityQueue()
    
    computedVoronoiDelaunay = compute_Voronoi_Delaunay(pointsSMT, space = space)
    triangulationListEdges =  computedVoronoiDelaunay["Delaunay_edges"]
    


    # Building the edges-distance map:
    edgesWeight = PriorityQueue()
    for (edge_idx0, edge_idx1) in triangulationListEdges:
            p0 = pointsSMT[edge_idx0]
            p1 = pointsSMT[edge_idx1]
            dist = DISTANCE_F[space](p1, p0)
            edgesWeight.put((dist, [idices_points[edge_idx0], idices_points[edge_idx1], dist]))
    
    
        
    auxGraphDisjSet = DisjSet(idxVert=idices_points)
    # This will store the resultant MST 
    mstEdgeList = [] 
    mstVert2Adj = defaultdict(set)



    # An index variable, used for result[] 
    e = 0
    
    fullMSTval = 0
    
    # Number of edges to be taken is less than to V-1 
    
    while e < numPoints - 1 and not edgesWeight.empty():         
        # Pick the smallest edge and increment 
        # the index for next iteration 
        u, v, w = edgesWeight.get()[1]
        x = auxGraphDisjSet.find(u) 
        y = auxGraphDisjSet.find(v) 

        # If including this edge doesn't 
        # cause cycle, then include it in result 
        # and increment the index of result 
        # for next edge 
        if x != y: 
            e = e + 1
            fullMSTval += w
            mstEdgeList.append([u, v]) 
            mstVert2Adj[u].add(v)
            mstVert2Adj[v].add(u)
            auxGraphDisjSet.union(x, y) 
    
    
    return mstEdgeList, mstVert2Adj, fullMSTval





def vanillaMST(verticesDictTerminal, verticesDictSteiner=None, space="Klein"):
    
    
    idices_points = list(verticesDictTerminal.keys()) 
    if verticesDictSteiner is not None:
        idices_points += list(verticesDictSteiner.keys())
        
    numPoints = len(idices_points)
    # Building the edges-distance map:    
    edgesWeight = PriorityQueue()
    
    
    
    for i, tagP0 in enumerate(idices_points):
        p0 = verticesDictTerminal[tagP0] if tagP0[0]=="T" else verticesDictSteiner[tagP0]
        
        for  j in range(i+1, numPoints):
            tagP1 = idices_points[j]
            p1 = verticesDictTerminal[tagP1] if tagP1[0]=="T" else verticesDictSteiner[tagP1]
            dist = DISTANCE_F[space](p1, p0)
            edgesWeight.put((dist, [tagP0, tagP1, dist]))

    
   
    
    
        
    auxGraphDisjSet = DisjSet(idxVert=idices_points)
    # This will store the resultant MST 
    mstEdgeList = [] 
    mstVert2Adj = defaultdict(set)



    # An index variable, used for result[] 
    e = 0
    
    fullMSTval = 0
    
    # Number of edges to be taken is less than to V-1 
    
    while e < numPoints - 1 and not edgesWeight.empty():         
        # Pick the smallest edge and increment 
        # the index for next iteration 
        u, v, w = edgesWeight.get()[1]
        x = auxGraphDisjSet.find(u) 
        y = auxGraphDisjSet.find(v) 

        # If including this edge doesn't 
        # cause cycle, then include it in result 
        # and increment the index of result 
        # for next edge 
        if x != y: 
            e = e + 1
            fullMSTval += w
            mstEdgeList.append([u, v]) 
            mstVert2Adj[u].add(v)
            mstVert2Adj[v].add(u)
            auxGraphDisjSet.union(x, y) 
    
    
    return mstEdgeList, mstVert2Adj, fullMSTval




def isOptimal(coordCandidate, coordAdj, space="Klein", slack=1):
    angles = []
    
    for j in range(3):
        angles.append(innerAngleTriangle(coordAdj[j],
                                            coordCandidate,
                                            coordAdj[(j+1)%3],
                                            space=space))

    angles = np.array(angles)
    
    return np.all((angles*180/np.pi>120-slack) & (angles*180/np.pi<120+slack))



def reduction(verticesDictTerminal, verticesDictSteiner, space="Klein", idxSteinerPoint=0, 
              dist2Points=1e-1, precise=True, nIters=100, convDiff=1e-2, slack=1, maxgroup=4):
    
    if verticesDictSteiner is None:
        verticesDictSteiner = dict()
    

    redoMST = True
    
    mstEdgeList, mstVert2Adj, fullMSTval = delaunayMST(verticesDictTerminal, 
                                                            verticesDictSteiner, 
                                                            space=space)
    
    i = 0
    
    # Steps A and B
    while True:
        redoMST = False
        
        frozenSteiner = list(verticesDictSteiner.keys())
        for idx in frozenSteiner:
            adj = mstVert2Adj[idx]
            
            if len(adj) == 1:
                mstVert2Adj[adj.pop()].remove(idx)
                mstVert2Adj.pop(idx)
                verticesDictSteiner.pop(idx)
                
                redoMST = True
                
            elif len(adj) == 2:
                edgeidx0 = adj.pop()
                edgeidx1 = adj.pop()
                mstVert2Adj[edgeidx0].remove(idx)
                mstVert2Adj[edgeidx0].add(edgeidx1)
                mstVert2Adj[edgeidx1].remove(idx)
                mstVert2Adj[edgeidx1].add(edgeidx0)
                mstVert2Adj.pop(idx)
                verticesDictSteiner.pop(idx)
                
                redoMST = True
            
            elif len(adj) > 4:
                for j in range(len(adj)):
                    mstVert2Adj[adj.pop()].remove(idx)
                    
                mstVert2Adj.pop(idx)
                verticesDictSteiner.pop(idx)
                redoMST = True
        
        if not redoMST and i>0:
            break
        
        # Steps C, D, E
        frozenSteiner = list(verticesDictSteiner.keys())
        for idx in frozenSteiner:
            adj = mstVert2Adj[idx]
            coordAdj = [verticesDictTerminal[tag] if tag[0]=="T" else verticesDictSteiner[tag] for tag in adj]
            coordCandidate = verticesDictSteiner[idx]
            
            if len(adj) == 3 and not isOptimal(coordCandidate, coordAdj, space=space, slack=slack):
                    
                steinerP = steinerPoint3(coordAdj,
                                            model=space,
                                            dist2Points=dist2Points,
                                            precise=precise)
                
                if steinerP is None:
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj.pop(idx)
                    verticesDictSteiner.pop(idx)
                else:
                    verticesDictSteiner[idx] = steinerP
        
        mstEdgeList, mstVert2Adj, fullMSTval = delaunayMST(verticesDictTerminal, 
                                                            verticesDictSteiner, 
                                                            space=space)
        
        i+=1
                    
            
        
    frozenSteiner = list(verticesDictSteiner.keys())
    random.shuffle(frozenSteiner)
    for idx in frozenSteiner:
        adj = mstVert2Adj[idx]
        adj_list = list(adj)
        coordAdj = [verticesDictTerminal[tag] if tag[0]=="T" else verticesDictSteiner[tag] for tag in adj_list]
        coordCandidate = verticesDictSteiner[idx]

        if len(adj) == 4 and maxgroup==4:
            
            bestSteinerPoints, bestTopo = bestSteinerFST4(coordAdj,
                                                model=space, nIters=nIters,
                                                convDiff=convDiff, dist2Points=dist2Points,
                                                precise=precise)
        
            if bestSteinerPoints is not None:
                mstVert2Adj[adj.pop()].remove(idx)
                mstVert2Adj[adj.pop()].remove(idx)
                mstVert2Adj[adj.pop()].remove(idx)
                mstVert2Adj[adj.pop()].remove(idx)
                mstVert2Adj.pop(idx)
                verticesDictSteiner.pop(idx)
                
                # remove from mstEdgeList
                mstEdgeList = [edge for edge in mstEdgeList if idx not in edge]
                
                
            
                verticesDictSteiner[f"S{idxSteinerPoint}"] = bestSteinerPoints[0]
                verticesDictSteiner[f"S{idxSteinerPoint + 1}"] = bestSteinerPoints[1]
                
                mstVert2Adj[f"S{idxSteinerPoint}"].add(f"S{idxSteinerPoint + 1}")
                mstVert2Adj[f"S{idxSteinerPoint + 1}"].add(f"S{idxSteinerPoint}")
                
                mstEdgeList.append([f"S{idxSteinerPoint}", f"S{idxSteinerPoint + 1}"])
                
                for i in range(2):
                    for j in range(2):
                        mstVert2Adj[adj_list[bestTopo[i][j]]].add(f"S{idxSteinerPoint + i}")
                        mstVert2Adj[f"S{idxSteinerPoint + i}"].add(adj_list[bestTopo[i][j]])
                        mstEdgeList.append([adj_list[bestTopo[i][j]], f"S{idxSteinerPoint + i}"])


                idxSteinerPoint += 2
        
        
    # Verification: Check that all Steiner points are present in MST edge list
    steiner_points = set(verticesDictSteiner.keys())
    edges_vertices = set()
    for edge in mstEdgeList:
        edges_vertices.add(edge[0])
        edges_vertices.add(edge[1])
    
    missing_steiner_points = steiner_points - edges_vertices
    
    if missing_steiner_points:
        raise ValueError(f"Steiner points not found in MST edge list: {missing_steiner_points}")
    
    # Optional: Also verify consistency with adjacency list
    adj_vertices = set(mstVert2Adj.keys())
    missing_from_adj = steiner_points - adj_vertices
    
    if missing_from_adj:
        raise ValueError(f"Steiner points not found in adjacency list: {missing_from_adj}")
        
    
    
    
    return verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj



# def reduction(verticesDictTerminal, verticesDictSteiner, space="Klein", idxSteinerPoint=0, 
#               dist2Points=1e-1, precise=True, nIters=100, convDiff=1e-2, slack=1, maxgroup=4):
    
#     if verticesDictSteiner is None:
#         verticesDictSteiner = dict()
    
#     def remove_vertex_completely(vertex_id, adj_dict, steiner_dict):
#         """
#         Completely remove a vertex from all data structures
#         """
#         if vertex_id in adj_dict:
#             # Remove this vertex from all its neighbors' adjacency lists
#             for neighbor in adj_dict[vertex_id].copy():
#                 if neighbor in adj_dict:
#                     adj_dict[neighbor].discard(vertex_id)
#             # Remove the vertex's own adjacency entry
#             del adj_dict[vertex_id]
        
#         # Remove from Steiner dictionary if it's a Steiner vertex
#         if vertex_id in steiner_dict:
#             del steiner_dict[vertex_id]
    
#     def adjacency_to_edges(adj_dict):
#         """
#         Convert adjacency dictionary to edge list, ensuring no duplicate edges
#         """
#         edge_set = set()
#         for vertex, neighbors in adj_dict.items():
#             for neighbor in neighbors:
#                 # Create canonical edge representation (sorted order)
#                 edge = tuple(sorted([vertex, neighbor]))
#                 edge_set.add(edge)
        
#         # Convert back to list format
#         return [[edge[0], edge[1]] for edge in edge_set]
    
#     def calculate_mst_value(edge_list, terminal_dict, steiner_dict, space):
#         """
#         Calculate total MST value from edge list
#         """
#         if not edge_list:
#             return 0.0
            
#         total_val = 0.0
#         distance_func = DISTANCE_F[space]
#         coord_lookup = terminal_dict.copy()
#         coord_lookup.update(steiner_dict)
        
#         for edge in edge_list:
#             u, v = edge
#             if u in coord_lookup and v in coord_lookup:
#                 total_val += distance_func(coord_lookup[u], coord_lookup[v])
        
#         return total_val

#     # Initial MST construction
#     mstEdgeList, mstVert2Adj, fullMSTval = vanillaMST(verticesDictTerminal, 
#                                                       verticesDictSteiner, 
#                                                       space=space)
    
#     iteration = 0
    
#     # Main reduction loop
#     while True:
#         changes_made = False
        
#         # Step A & B: Remove degree 1, 2, and high-degree vertices
#         vertices_to_remove = []
#         frozenSteiner = list(verticesDictSteiner.keys())
        
#         for idx in frozenSteiner:
#             if idx not in mstVert2Adj:
#                 continue
                
#             adj = mstVert2Adj[idx]
#             degree = len(adj)
            
#             if degree == 1:
#                 # Degree 1: remove completely
#                 vertices_to_remove.append(idx)
#                 changes_made = True
                
#             elif degree == 2:
#                 # Degree 2: remove and connect neighbors
#                 neighbors = list(adj)
#                 neighbor1, neighbor2 = neighbors[0], neighbors[1]
                
#                 # Remove vertex completely
#                 remove_vertex_completely(idx, mstVert2Adj, verticesDictSteiner)
                
#                 # Connect the two neighbors directly
#                 if neighbor1 in mstVert2Adj and neighbor2 in mstVert2Adj:
#                     mstVert2Adj[neighbor1].add(neighbor2)
#                     mstVert2Adj[neighbor2].add(neighbor1)
                
#                 changes_made = True
                
#             elif degree > 4:
#                 # High degree: remove completely
#                 vertices_to_remove.append(idx)
#                 changes_made = True
        
#         # Remove all flagged vertices
#         for vertex_id in vertices_to_remove:
#             remove_vertex_completely(vertex_id, mstVert2Adj, verticesDictSteiner)
        
#         if not changes_made and iteration > 0:
#             break
        
#         # Step C, D, E: Handle degree 3 vertices (Steiner point optimization)
#         frozenSteiner = list(verticesDictSteiner.keys())
#         for idx in frozenSteiner:
#             if idx not in mstVert2Adj:
#                 continue
                
#             adj = mstVert2Adj[idx]
            
#             if len(adj) == 3:
#                 adj_list = list(adj)
                
#                 # Get coordinates for all adjacent vertices
#                 coordAdj = []
#                 valid_coords = True
                
#                 for tag in adj_list:
#                     if tag[0] == "T":
#                         if tag in verticesDictTerminal:
#                             coordAdj.append(verticesDictTerminal[tag])
#                         else:
#                             valid_coords = False
#                             break
#                     else:
#                         if tag in verticesDictSteiner:
#                             coordAdj.append(verticesDictSteiner[tag])
#                         else:
#                             valid_coords = False
#                             break
                
#                 if valid_coords:
#                     coordCandidate = verticesDictSteiner[idx]
                    
#                     if not isOptimal(coordCandidate, coordAdj, space=space, slack=slack):
#                         steinerP = steinerPoint3(coordAdj,
#                                                model=space,
#                                                dist2Points=dist2Points,
#                                                precise=precise)
                        
#                         if steinerP is None:
#                             # Remove if no valid Steiner point
#                             remove_vertex_completely(idx, mstVert2Adj, verticesDictSteiner)
#                             changes_made = True
#                         else:
#                             # Update with new Steiner point
#                             verticesDictSteiner[idx] = steinerP
        
#         # Convert adjacency list to clean edge list
#         mstEdgeList = adjacency_to_edges(mstVert2Adj)
#         fullMSTval = calculate_mst_value(mstEdgeList, verticesDictTerminal, verticesDictSteiner, space)
        
#         iteration += 1
    
#     # Handle degree 4 vertices (FST4 optimization)
#     frozenSteiner = list(verticesDictSteiner.keys())
#     random.shuffle(frozenSteiner)
    
#     for idx in frozenSteiner:
#         if idx not in mstVert2Adj:
#             continue
            
#         adj = mstVert2Adj[idx]
        
#         if len(adj) == 4 and maxgroup == 4:
#             adj_list = list(adj)
            
#             # Get coordinates for all adjacent vertices
#             coordAdj = []
#             valid_coords = True
            
#             for tag in adj_list:
#                 if tag[0] == "T":
#                     if tag in verticesDictTerminal:
#                         coordAdj.append(verticesDictTerminal[tag])
#                     else:
#                         valid_coords = False
#                         break
#                 else:
#                     if tag in verticesDictSteiner:
#                         coordAdj.append(verticesDictSteiner[tag])
#                     else:
#                         valid_coords = False
#                         break
            
#             if valid_coords:
#                 bestSteinerPoints, bestTopo = bestSteinerFST4(coordAdj,
#                                                    model=space, nIters=nIters,
#                                                    convDiff=convDiff, dist2Points=dist2Points,
#                                                    precise=precise)
                
#                 if bestSteinerPoints is not None:
#                     # Remove the old degree-4 vertex completely
#                     remove_vertex_completely(idx, mstVert2Adj, verticesDictSteiner)
                    
#                     # Add two new Steiner vertices
#                     new_s1 = f"S{idxSteinerPoint}"
#                     new_s2 = f"S{idxSteinerPoint + 1}"
                    
#                     verticesDictSteiner[new_s1] = bestSteinerPoints[0]
#                     verticesDictSteiner[new_s2] = bestSteinerPoints[1]
                    
#                     # Initialize adjacency for new vertices
#                     mstVert2Adj[new_s1] = set()
#                     mstVert2Adj[new_s2] = set()
                    
#                     # Connect the two new Steiner vertices to each other
#                     mstVert2Adj[new_s1].add(new_s2)
#                     mstVert2Adj[new_s2].add(new_s1)
                    
#                     # Connect to terminals according to optimal topology
#                     for i in range(2):
#                         for j in range(2):
#                             terminal = adj_list[bestTopo[i][j]]
#                             steiner = new_s1 if i == 0 else new_s2
                            
#                             # Ensure terminal has adjacency entry
#                             if terminal not in mstVert2Adj:
#                                 mstVert2Adj[terminal] = set()
                            
#                             mstVert2Adj[terminal].add(steiner)
#                             mstVert2Adj[steiner].add(terminal)
                    
#                     idxSteinerPoint += 2
    
#     # Final cleanup: convert to edge list and ensure consistency
#     mstEdgeList = adjacency_to_edges(mstVert2Adj)
#     fullMSTval = calculate_mst_value(mstEdgeList, verticesDictTerminal, verticesDictSteiner, space)
    
#     # Verification: ensure all edges reference existing vertices
#     all_vertices = set(verticesDictTerminal.keys()) | set(verticesDictSteiner.keys())
#     clean_edges = []
#     for edge in mstEdgeList:
#         u, v = edge
#         if u in all_vertices and v in all_vertices:
#             clean_edges.append(edge)
    
#     if len(clean_edges) != len(mstEdgeList):
#         print(f"Warning: Filtered {len(mstEdgeList) - len(clean_edges)} invalid edges in reduction")
#         mstEdgeList = clean_edges
#         fullMSTval = calculate_mst_value(mstEdgeList, verticesDictTerminal, verticesDictSteiner, space)
    
#     return verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj, fullMSTval

def reduction_(verticesDictTerminal, verticesDictSteiner, space="Klein", idxSteinerPoint=0, 
              dist2Points=1e-1, precise=True, nIters=100, convDiff=1e-2, slack=1, maxgroup=4):
    
    keepReducing = True
    if verticesDictSteiner is None:
        verticesDictSteiner = dict()
    
    while keepReducing:
        redoMST = True
        keepReducing = False
        
        # Steps A and B
        while redoMST:
            
            mstEdgeList, mstVert2Adj, fullMSTval = delaunayMST(verticesDictTerminal, 
                                                              verticesDictSteiner, 
                                                              space=space)

            redoMST = False
            
            frozenSteiner = list(verticesDictSteiner.keys())
            for idx in frozenSteiner:
                adj = mstVert2Adj[idx]
                
                if len(adj) == 1:
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj.pop(idx)
                    verticesDictSteiner.pop(idx)
                    
                    redoMST = True
                    
                elif len(adj) == 2:
                    edgeidx0 = adj.pop()
                    edgeidx1 = adj.pop()
                    mstVert2Adj[edgeidx0].remove(idx)
                    mstVert2Adj[edgeidx0].add(edgeidx1)
                    mstVert2Adj[edgeidx1].remove(idx)
                    mstVert2Adj[edgeidx1].add(edgeidx0)
                    mstVert2Adj.pop(idx)
                    verticesDictSteiner.pop(idx)
                    
                    redoMST = True
        
        # Steps C, D, E
        frozenSteiner = list(verticesDictSteiner.keys())
        for idx in frozenSteiner:
            adj = mstVert2Adj[idx]
            coordAdj = [verticesDictTerminal[tag] if tag[0]=="T" else verticesDictSteiner[tag] for tag in adj]
            coordCandidate = verticesDictSteiner[idx]
            
            if len(adj) == 3 and not isOptimal(coordCandidate, coordAdj, space=space, slack=slack):
                    
                steinerP = steinerPoint3(coordAdj,
                                            model=space,
                                            dist2Points=dist2Points,
                                            precise=precise)
                
                if steinerP is None:
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj[adj.pop()].remove(idx)
                    mstVert2Adj.pop(idx)
                    verticesDictSteiner.pop(idx)
                else:
                    verticesDictSteiner[idx] = steinerP
                    
            
        
                keepReducing = True
            
            
            elif len(adj) == 4 and maxgroup==4:
                
                bestSteinerPoints, bestTopo = bestSteinerFST4(coordAdj,
                                                    model=space, nIters=nIters,
                                                    convDiff=convDiff, dist2Points=dist2Points,
                                                    precise=precise)
            
                if bestSteinerPoints is not None:
                    verticesDictSteiner[idx] = bestSteinerPoints[0]
                    verticesDictSteiner[f"S{idxSteinerPoint}"] = bestSteinerPoints[1]
                    idxSteinerPoint += 1
                
                    keepReducing = True
                
            elif len(adj) > 4:
                for i in range(len(adj)):
                    mstVert2Adj[adj.pop()].remove(idx)
                    
                mstVert2Adj.pop(idx)
                verticesDictSteiner.pop(idx)
                keepReducing = True
    
    
    
    return verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj, fullMSTval





def compute_tree_length(mst_edge_list, vertices_dict_terminal, vertices_dict_steiner, space="Klein"):
    """
    Compute the total length of a tree given its edge list and vertex coordinates.
    
    Args:
        mst_edge_list: List of edges in the tree
        vertices_dict_terminal: Dictionary of terminal vertex coordinates
        vertices_dict_steiner: Dictionary of Steiner vertex coordinates  
        space: Geometry space for distance calculation
    
    Returns:
        float: Total length of the tree
    """
        
    if not mst_edge_list:
        return 0.0
    
    distance_func = DISTANCE_F[space]
    total_length = 0.0
    
    # Create combined coordinate lookup
    coord_lookup = vertices_dict_terminal.copy()
    if vertices_dict_steiner:
        coord_lookup.update(vertices_dict_steiner)
    
    # Sum lengths of all edges
    for edge in mst_edge_list:
        u, v = edge
        coord_u = coord_lookup[u]
        coord_v = coord_lookup[v]
        edge_length = distance_func(coord_u, coord_v)
        total_length += edge_length
    
    return total_length


def reexpansion(verticesDictTerminal, verticesDictSteiner, mstVert2Adj, space="Klein", 
                idxSteinerPoint=0, dist2Points=1e-1, precise=True, slack=1):
    

    vertexAngleOK = True
    
    
    frozenIdx = list(mstVert2Adj.keys())
    for idx in frozenIdx:
        adj = mstVert2Adj[idx]
        coordCandidate = verticesDictTerminal[idx] if idx[0]=="T" else verticesDictSteiner[idx] 
        numAdj = len(adj)
        if numAdj>1:
            coordAdj = [(verticesDictTerminal[tag], tag) if tag[0]=="T" else (verticesDictSteiner[tag], tag) for tag in adj]
            for j in range(len(adj)):
                angle = innerAngleTriangle(coordAdj[j][0],
                                    coordCandidate,
                                    coordAdj[(j+1)%numAdj][0],
                                    space=space)
                
                if angle*180/np.pi<120-slack:
                    steinerP = steinerPoint3([coordAdj[j][0], coordAdj[(j+1)%numAdj][0], coordCandidate],
                                        model=space,
                                        dist2Points=dist2Points,
                                        precise=precise)
                    
                    if steinerP is not None:
                        verticesDictSteiner[f"S{idxSteinerPoint}"] = steinerP
                        idxSteinerPoint += 1
                        vertexAngleOK = False
        
    return verticesDictSteiner, idxSteinerPoint, vertexAngleOK




def exhaustiveMethod_local(points, space="Klein", triangMeth="DT", maxgroup=3,
                         nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=False,
                         selection=("PROB", (0.3, 0.6)), nMaxExpansions=6,
                         slack=1, extendedResults=False):
    """_summary_

    Args:
        points (_type_): _description_
        space (str, optional): "Klein" or "Euclidean". Defaults to "Klein".
        triangMeth (str, optional): _description_. Defaults to "DT".
        maxgroup (int, optional): _description_. Defaults to 4.
    """
    
    
    
    N = len(points)
    if N<3:
        resultGraph=list()
        if N==2:
            resultGraph.append(["T0", "T1"])
        
        verticesDict = dict()
        for i in range(N):
            verticesDict[f"T{i}"] = points[i]
            
        
        return resultGraph, verticesDict, 1, None, None
    
    if space not in ["Klein", "Euclidean"]:
        raise ValueError("space should be either 'Klein' or 'Euclidean'")
    
    if triangMeth not in ["DT", "GG"]:
        raise ValueError("space should be either 'DT' or 'GG'")
    
    if maxgroup not in [3, 4]:
        raise ValueError("maxgroup should be 3 or 4")

    if not ALLOW_PRECISE and precise and space=="Klein":
        raise ValueError("'precise' can't be True if PHCpy is not installed")
    

    nExpansions = 1

    verticesDictTerminal = dict()
    points_og_list = [] if type(points) is not list else points
    for i, p in enumerate(points):
        verticesDictTerminal[f"T{i}"] = p
        if type(points) is not list:
            points_og_list.append(p)
   
    
    verticesDictSteiner = dict()
    
    start_time_mst = time.process_time()
    ogMSTEdgeList, mstVert2Adj, fullMSTval  = delaunayMST(verticesDictTerminal, space=space)
    mstTime = time.process_time() - start_time_mst
    ogEdgesDT = None 
    idxSteinerPoint=0
    bestGraph = ogMSTEdgeList
    bestSteinerVal  = fullMSTval
    bestverticesDictSteiner = dict()
    
    start_time = time.process_time()
    
    auxSec = ("ALL", (1.0, 1.0))
    #auxSec = selection
    
    while nExpansions <= nMaxExpansions :
        #print(nExpansions, int(np.log2(nExpansions+1)), int(2*np.sqrt(nExpansions)-1),)
        pointsSMT = points_og_list+list(verticesDictSteiner.values())
        
        # EXPANSION
        # TODO: REMEMBER THAT THE FIRST VALUE IS HARDCODED BC WE ARE USING THE RANDOM VERSION
        verticesDictSteiner, idxSteinerPoint, edgesDT = expansion(int(2*np.sqrt(nExpansions)-1), pointsSMT, verticesDictSteiner, 
                                             space=space, triangMeth=triangMeth,
                                             selection=auxSec,
                                             dist2Points=dist2Points, precise=precise,
                                             idxSteinerPoint=idxSteinerPoint)
        
        
        
        if ogEdgesDT is None:
            ogEdgesDT = edgesDT 
            auxSec = selection

            
        
       
        auxVal = float("Inf")
        #print("hola")
        verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj = reduction(verticesDictTerminal,
                                                                                                verticesDictSteiner,
                                                                                                space=space, idxSteinerPoint=idxSteinerPoint, 
                                                                                                dist2Points=dist2Points, precise=precise, 
                                                                                                nIters=nIters, convDiff=convDiff, slack=slack,
                                                                                                maxgroup=maxgroup)
        
    
            

        #print("que tal")
        
        verticesDictSteiner, idxSteinerPoint, vertexAngleOK = reexpansion(verticesDictTerminal, verticesDictSteiner, 
                                                                    mstVert2Adj, space=space, 
                                                                    idxSteinerPoint=idxSteinerPoint,
                                                                    dist2Points=dist2Points,
                                                                    precise=precise, slack=slack)
        

  
        
        if not vertexAngleOK:
            
            
            mstEdgeList, mstVert2Adj, steinerVal = delaunayMST(verticesDictTerminal, 
                                                              verticesDictSteiner, 
                                                              space=space)
    
         
        
  
            if steinerVal < bestSteinerVal:
               
    
                verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj = reduction(verticesDictTerminal,
                                                                                                verticesDictSteiner,
                                                                                                space=space, idxSteinerPoint=idxSteinerPoint, 
                                                                                                dist2Points=dist2Points, precise=precise, 
                                                                                                nIters=nIters, convDiff=convDiff, slack=slack,
                                                                                                maxgroup=maxgroup)
            
        #print(steinerVal, bestSteinerVal)
        
        
        
        steinerVal = compute_tree_length(mstEdgeList, verticesDictTerminal, verticesDictSteiner, space)
        
        if steinerVal < bestSteinerVal:
            bestSteinerVal = steinerVal
            bestGraph = mstEdgeList
            bestverticesDictSteiner = verticesDictSteiner.copy()
            nExpansions = 1
        else:
            nExpansions += 1
            
        

    methodTime = time.process_time() - start_time
   
    # Rename steiner points
    verticesDict = verticesDictTerminal.copy()
    renaming = dict()
    for i, k, in enumerate(bestverticesDictSteiner.keys()):
        renaming[k] = f"S{i}"
        verticesDict[f"S{i}"] = bestverticesDictSteiner[k]


    
    resultGraph = []
    for v0, v1 in bestGraph:
        v0 = v0 if v0[0]=="T" else renaming[v0]
        v1 = v1 if v1[0]=="T" else renaming[v1]
        resultGraph.append([v0, v1])
    
    steinerVal = bestSteinerVal
    
    
    results = {
        "resultGraph":resultGraph,
        "verticesDict": verticesDict,
        "methodTime": methodTime,
        "steinerVal": steinerVal
    }
    
    
    if  extendedResults:
       
        results["mstVal"] = fullMSTval
        results["ratio"] = steinerVal/fullMSTval
        results["mstGraph"] = ogMSTEdgeList
        results["mstTime"] = mstTime
        #TODO: add count
        results["numFST3"] = -1
        results["numFST4"] = -1
        results["edgesDT"] = ogEdgesDT

       
    return  results



def exhaustiveMethod_CLASIC(points, space="Klein", triangMeth="DT", maxgroup=4,
                         nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=True,
                         selection=("All", (1.0, 1.0)), nMaxExpansions=6, annealing=(0.1, 0.7),
                         slack=1, extendedResults=False):
    """_summary_

    Args:
        points (_type_): _description_
        space (str, optional): "Klein" or "Euclidean". Defaults to "Klein".
        triangMeth (str, optional): _description_. Defaults to "DT".
        maxgroup (int, optional): _description_. Defaults to 4.
    """
    start_time = time.process_time()
    
    print(points.shape)
    
    N = len(points)
    if N<3:
        resultGraph=list()
        if N==2:
            resultGraph.append(["T0", "T1"])
        
        verticesDict = dict()
        for i in range(N):
            verticesDict[f"T{i}"] = points[i]
            
        
        return resultGraph, verticesDict, 1, None, None
    
    if space not in ["Klein", "Euclidean"]:
        raise ValueError("space should be either 'Klein' or 'Euclidean'")
    
    if triangMeth not in ["DT", "GG"]:
        raise ValueError("space should be either 'DT' or 'GG'")
    
    if maxgroup not in [3, 4]:
        raise ValueError("maxgroup should be 3 or 4")

    if not ALLOW_PRECISE and precise and space=="Klein":
        raise ValueError("'precise' can't be True if PHCpy is not installed")
    

    nExpansions = 1

    verticesDictTerminal = dict()
    points_og_list = [] if type(points) is not list else points
    for i, p in enumerate(points):
        verticesDictTerminal[f"T{i}"] = p
        if type(points) is not list:
            points_og_list.append(p)
   
    
    idxSteinerPoint=0
    start_time_MST = time.process_time()
    ogMSTEdgeList, mstVert2Adj, fullMSTval  = delaunayMST(verticesDictTerminal, space=space)
    mstTime = time.process_time() - start_time_MST

    bestGraph = ogMSTEdgeList
    bestSteinerVal = lastSteinerVal = firstSteinerVal = fullMSTval
    lastverticesDictSteiner = dict()
    bestverticesDictSteiner = dict()
    
    verticesDictSteiner = dict()
    ogEdgesDT = None
    
    if annealing is not None:
        temp = annealing[0]
        alpha = annealing[1]
    
    while nExpansions <= nMaxExpansions :
        
        pointsSMT = points_og_list+list(verticesDictSteiner.values())
        
        # EXPANSION
        verticesDictSteiner, idxSteinerPoint, edgesDT = expansion(nExpansions, pointsSMT, verticesDictSteiner, 
                                             space=space, triangMeth=triangMeth,
                                             selection=selection,
                                             dist2Points=dist2Points, precise=precise,
                                             idxSteinerPoint=idxSteinerPoint)
        
        
        if ogEdgesDT is None:
            ogEdgesDT = edgesDT 
        
        vertexAngleOK = False
        while not vertexAngleOK:
            verticesDictSteiner, idxSteinerPoint, mstEdgeList, mstVert2Adj, steinerVal = reduction(verticesDictTerminal,
                                                                                                   verticesDictSteiner,
                                                                                                    space=space, idxSteinerPoint=idxSteinerPoint, 
                                                                                                    dist2Points=dist2Points, precise=precise, 
                                                                                                    nIters=nIters, convDiff=convDiff, slack=slack,
                                                                                                    maxgroup=maxgroup)
            

                                                
            verticesDictSteiner, idxSteinerPoint, vertexAngleOK = reexpansion(verticesDictTerminal, verticesDictSteiner, 
                                                                       mstVert2Adj, space=space, 
                                                                       idxSteinerPoint=idxSteinerPoint,
                                                                       dist2Points=dist2Points,
                                                                       precise=precise, slack=slack)
            
            
            
        if steinerVal < bestSteinerVal:
            bestSteinerVal = steinerVal
            bestGraph = mstEdgeList
            bestverticesDictSteiner = verticesDictSteiner.copy()
        
        if annealing is not None:
            temp = alpha*temp 
        

        if steinerVal < lastSteinerVal:
            nExpansions = 1
            lastSteinerVal = steinerVal
            lastverticesDictSteiner = verticesDictSteiner.copy()
            
        elif steinerVal == lastSteinerVal:
            nExpansions += 1
        else:
            if annealing is not None:
                delta = (steinerVal - lastSteinerVal)/firstSteinerVal
                r = np.random.uniform(0, 1)
                if r < np.exp(-delta/(temp + 1e-16)):
                    nExpansions = 1
                    lastSteinerVal = steinerVal
                    lastverticesDictSteiner = verticesDictSteiner.copy()
                else:
                    nExpansions += 1
                    steinerVal = lastSteinerVal
                    verticesDictSteiner = lastverticesDictSteiner
            else:
                nExpansions += 1
                steinerVal = lastSteinerVal
                verticesDictSteiner = lastverticesDictSteiner


    methodTime = time.process_time() - start_time
    
    # Rename steiner points
    verticesDict = verticesDictTerminal.copy()
    renaming = dict()
    for i, k, in enumerate(bestverticesDictSteiner.keys()):
        renaming[k] = f"S{i}"
        verticesDict[f"S{i}"] = bestverticesDictSteiner[k]
    
    print(bestGraph)
    
    resultGraph = []
    for v0, v1 in bestGraph:
        v0 = v0 if v0[0]=="T" else renaming[v0]
        v1 = v1 if v1[0]=="T" else renaming[v1]
        resultGraph.append([v0, v1])
    
    steinerVal = bestSteinerVal
    
    results = {
        "resultGraph":resultGraph,
        "verticesDict": verticesDict,
        "methodTime": methodTime
    }
    
    
    print(verticesDict)
    print(resultGraph)
    
    if not extendedResults:
        results["steinerVal"] = steinerVal
    else:
        

        
        results["ratio"] = steinerVal/fullMSTval
        results["mstGraph"] = ogMSTEdgeList
        results["mstTime"] = mstTime
        #TODO: add count
        results["numFST3"] = -1
        results["numFST4"] = -1
    
        results["edgesDT"] = ogEdgesDT

       
    return  results
