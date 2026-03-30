from src.utils.steinerUtils import *
from src.exhaustiveMethods_local import isOptimal, vanillaMST, expansion, reduction, reexpansion, delaunayMST
# from functools import partial
from src.utils.delaunay2d import compute_Voronoi_Delaunay


try:
    from src.utils.fullSteinerSolverHyperbolic import *
    ALLOW_PRECISE = True

except ModuleNotFoundError:
    ALLOW_PRECISE = False


import heapq
import time
import numpy as np
from collections import defaultdict
from functools import partial



# def vanillaMST(verticesDictTerminal, verticesDictSteiner=None, space="Klein"):
#     """Optimized MST using heapq instead of PriorityQueue"""
    
#     # Collect all vertex IDs
#     terminal_ids = list(verticesDictTerminal.keys())
#     steiner_ids = list(verticesDictSteiner.keys()) if verticesDictSteiner else []
#     all_vertex_ids = terminal_ids + steiner_ids
#     num_vertices = len(all_vertex_ids)
    
#     if num_vertices < 2:
#         return [], defaultdict(set), 0.0
    
#     # Create coordinate lookup for faster access
#     coord_lookup = verticesDictTerminal.copy()
#     if verticesDictSteiner:
#         coord_lookup.update(verticesDictSteiner)
    
#     # Build edge heap more efficiently
#     edge_heap = []
#     distance_func = DISTANCE_F[space]
    
#     for i, vertex_i in enumerate(all_vertex_ids):
#         coord_i = coord_lookup[vertex_i]
        
#         for j in range(i + 1, num_vertices):
#             vertex_j = all_vertex_ids[j]
#             coord_j = coord_lookup[vertex_j]
            
#             dist = distance_func(coord_i, coord_j)
#             heapq.heappush(edge_heap, (dist, vertex_i, vertex_j))
    
#     # Initialize disjoint set
#     parent = {v: v for v in all_vertex_ids}
#     rank = {v: 0 for v in all_vertex_ids}
    
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])  # Path compression
#         return parent[x]
    
#     def union(x, y):
#         px, py = find(x), find(y)
#         if px == py:
#             return False
        
#         # Union by rank
#         if rank[px] < rank[py]:
#             px, py = py, px
#         parent[py] = px
#         if rank[px] == rank[py]:
#             rank[px] += 1
#         return True
    
#     # Build MST
#     mst_edges = []
#     mst_adjacency = defaultdict(set)
#     total_weight = 0.0
#     edges_added = 0
    
#     while edge_heap and edges_added < num_vertices - 1:
#         weight, u, v = heapq.heappop(edge_heap)
        
#         if union(u, v):
#             mst_edges.append([u, v])
#             mst_adjacency[u].add(v)
#             mst_adjacency[v].add(u)
#             total_weight += weight
#             edges_added += 1
    
#     return mst_edges, mst_adjacency, total_weight



# def clean(verticesDictTerminal, verticesDictSteiner, optimizer, space="Klein", 
#           dist2Points=1e-1, precise=True, slack=1):
#     """Optimized cleaning function with reduced MST recomputations"""
    
#     if verticesDictSteiner is None:
#         verticesDictSteiner = dict()

#     # Initial MST computation
#     mstEdgeList, mstVert2Adj, fullMSTval = vanillaMST(verticesDictTerminal, 
#                                                      verticesDictSteiner, space=space)
    
#     iteration = 0
#     while True:
#         structure_modified = False
        
#         # Process degree-1 and degree-2 vertices in single pass
#         steiner_vertices = list(verticesDictSteiner.keys())
#         for vertex_id in steiner_vertices:
#             if vertex_id not in mstVert2Adj:
#                 continue
                
#             neighbors = mstVert2Adj[vertex_id]
#             degree = len(neighbors)
            
#             if degree == 1:
#                 # Remove degree-1 vertex
#                 neighbor = neighbors.pop()
#                 mstVert2Adj[neighbor].discard(vertex_id)
#                 mstVert2Adj.pop(vertex_id, None)
#                 verticesDictSteiner.pop(vertex_id, None)
#                 structure_modified = True
                
#             elif degree == 2:
#                 # Connect the two neighbors directly
#                 neighbor1, neighbor2 = list(neighbors)
                
#                 # Remove connections to current vertex
#                 mstVert2Adj[neighbor1].discard(vertex_id)
#                 mstVert2Adj[neighbor2].discard(vertex_id)
                
#                 # Connect neighbors to each other
#                 mstVert2Adj[neighbor1].add(neighbor2)
#                 mstVert2Adj[neighbor2].add(neighbor1)
                
#                 # Remove current vertex
#                 mstVert2Adj.pop(vertex_id, None)
#                 verticesDictSteiner.pop(vertex_id, None)
#                 structure_modified = True
        
#         if not structure_modified and iteration > 0:
#             break
#         iteration += 1
        
#         # Optimize degree-3 vertices
#         steiner_vertices = list(verticesDictSteiner.keys())
#         for vertex_id in steiner_vertices:
#             if vertex_id not in mstVert2Adj:
#                 continue
                
#             neighbors = mstVert2Adj[vertex_id]
#             if len(neighbors) != 3:
#                 continue
                
#             # Get coordinates efficiently
#             neighbor_coords = []
#             for neighbor_id in neighbors:
#                 if neighbor_id[0] == "T":
#                     neighbor_coords.append(verticesDictTerminal[neighbor_id])
#                 else:
#                     neighbor_coords.append(verticesDictSteiner[neighbor_id])
            
#             current_coord = verticesDictSteiner[vertex_id]
            
#             if not isOptimal(current_coord, neighbor_coords, space=space, slack=slack):
#                 steiner_point = steinerPoint3(neighbor_coords, model=space,
#                                             dist2Points=dist2Points, precise=precise)
                
#                 if steiner_point is None:
#                     # Remove vertex if no valid Steiner point
#                     for neighbor in neighbors:
#                         mstVert2Adj[neighbor].discard(vertex_id)
#                     mstVert2Adj.pop(vertex_id, None)
#                     verticesDictSteiner.pop(vertex_id, None)
#                 else:
#                     # Update with optimized position
#                     verticesDictSteiner[vertex_id] = steiner_point
        
#         # Recompute MST only once per iteration
#         mstEdgeList, mstVert2Adj, fullMSTval = vanillaMST(verticesDictTerminal, 
#                                                          verticesDictSteiner, space=space)
    
#     # Check if optimization is needed
#     steiner_neighbors_exist = any(
#         any(neighbor[0] == "S" for neighbor in mstVert2Adj.get(vertex_id, []))
#         for vertex_id in verticesDictSteiner.keys()
#     )
    
#     if steiner_neighbors_exist:
#         verticesDictSteiner = optimizer(vertices=verticesDictSteiner, graph=mstEdgeList)
    
#     return verticesDictSteiner, mstEdgeList, mstVert2Adj, fullMSTval


# def edge_insertion(verticesDictTerminal, verticesDictSteiner, mstVert2Adj, 
#                   mstEdgeList, optimizer, space="Klein", idxSteinerPoint=0, 
#                   dist2Points=1e-1, precise=True, slack=1):
#     """Optimized edge insertion with reduced redundant checks"""
    
#     # Convert edge list to set for O(1) operations
#     edge_set = {tuple(sorted(edge)) for edge in mstEdgeList}
    
#     vertex_angle_ok = True
#     all_vertices = list(mstVert2Adj.keys())
    
#     # Cache coordinate lookups
#     coord_cache = {}
#     for vertex_id in all_vertices:
#         if vertex_id[0] == "T":
#             coord_cache[vertex_id] = verticesDictTerminal[vertex_id]
#         else:
#             coord_cache[vertex_id] = verticesDictSteiner[vertex_id]
    
#     for xi in all_vertices:
#         if xi not in mstVert2Adj:
#             continue
            
#         xi_coord = coord_cache[xi]
#         xi_neighbors = list(mstVert2Adj[xi])
        
#         for xj in xi_neighbors:
#             if xj not in mstVert2Adj or xi not in mstVert2Adj[xj]:
#                 continue
                
#             xj_coord = coord_cache[xj]
            
#             # Find minimum angle edge efficiently
#             min_angle = float('inf')
#             selected_xk = None
#             selected_xk_coord = None
            
#             for xl in mstVert2Adj[xj]:
#                 if xl == xi:
#                     continue
                    
#                 xl_coord = coord_cache[xl]
#                 angle = innerAngleTriangle(xi_coord, xj_coord, xl_coord, space=space)
                
#                 if angle * 180 / np.pi < 120 - slack and angle < min_angle:
#                     min_angle = angle
#                     selected_xk = xl
#                     selected_xk_coord = xl_coord
            
#             if selected_xk is not None:
#                 steiner_point = steinerPoint3([xi_coord, xj_coord, selected_xk_coord],
#                                             model=space, dist2Points=dist2Points, 
#                                             precise=precise)
                
#                 if steiner_point is not None:
#                     steiner_id = f"S{idxSteinerPoint}"
#                     verticesDictSteiner[steiner_id] = steiner_point
#                     coord_cache[steiner_id] = steiner_point
                    
#                     # Update adjacency structure
#                     mstVert2Adj[xi].discard(xj)
#                     mstVert2Adj[xj].discard(xi)
#                     mstVert2Adj[xj].discard(selected_xk)
#                     mstVert2Adj[selected_xk].discard(xj)
                    
#                     # Update edge set
#                     edge_set.discard(tuple(sorted([xi, xj])))
#                     edge_set.discard(tuple(sorted([xj, selected_xk])))
                    
#                     # Add new connections
#                     mstVert2Adj[steiner_id] = {xi, xj, selected_xk}
#                     mstVert2Adj[xi].add(steiner_id)
#                     mstVert2Adj[xj].add(steiner_id)
#                     mstVert2Adj[selected_xk].add(steiner_id)
                    
#                     # Add new edges
#                     edge_set.add(tuple(sorted([xi, steiner_id])))
#                     edge_set.add(tuple(sorted([xj, steiner_id])))
#                     edge_set.add(tuple(sorted([selected_xk, steiner_id])))
                    
#                     idxSteinerPoint += 1
#                     vertex_angle_ok = False
    
#     # Convert back to list format
#     mstEdgeList.clear()
#     mstEdgeList.extend([list(edge) for edge in edge_set])
    
#     if not vertex_angle_ok:
#         verticesDictSteiner = optimizer(vertices=verticesDictSteiner, graph=mstEdgeList)
    
#     return verticesDictSteiner, mstEdgeList, idxSteinerPoint


# def reduce_high_degree_vertices(verticesDictTerminal, verticesDictSteiner, 
#                                mstVert2Adj, mstEdgeList, space="Klein", 
#                                idxSteinerPoint=0, dist2Points=1e-1, precise=True):
#     """Optimized high-degree vertex reduction"""
    
#     # Convert edge list to set for O(1) operations
#     edge_set = {tuple(sorted(edge)) for edge in mstEdgeList}
#     all_vertices = list(mstVert2Adj.keys())
    
#     for xi in all_vertices:
#         if xi not in mstVert2Adj:
#             continue
            
#         neighbors = list(mstVert2Adj[xi])
        
#         if len(neighbors) > 3:
#             # Cache coordinate
#             xi_coord = verticesDictTerminal[xi] if xi[0] == "T" else verticesDictSteiner[xi]
            
#             # Cache neighbor data
#             neighbor_data = []
#             for neighbor_id in neighbors:
#                 if neighbor_id[0] == "T":
#                     coord = verticesDictTerminal[neighbor_id]
#                 else:
#                     coord = verticesDictSteiner[neighbor_id]
#                 neighbor_data.append((neighbor_id, coord))
            
#             available_indices = set(range(len(neighbor_data)))
#             num_reductions = len(neighbors) - 3
            
#             for _ in range(num_reductions):
#                 if len(available_indices) < 2:
#                     break
                
#                 # Find minimum angle pair
#                 min_angle = float('inf')
#                 best_pair_indices = None
                
#                 available_list = list(available_indices)
#                 for i in range(len(available_list)):
#                     for j in range(i + 1, len(available_list)):
#                         idx1, idx2 = available_list[i], available_list[j]
                        
#                         coord_k = neighbor_data[idx1][1]
#                         coord_l = neighbor_data[idx2][1]
                        
#                         angle = innerAngleTriangle(coord_k, xi_coord, coord_l, space=space)
                        
#                         if angle < min_angle:
#                             min_angle = angle
#                             best_pair_indices = (idx1, idx2)
                
#                 if best_pair_indices is None:
#                     break
                
#                 idx1, idx2 = best_pair_indices
#                 xk = neighbor_data[idx1][0]
#                 xk_coord = neighbor_data[idx1][1]
#                 xl = neighbor_data[idx2][0]
#                 xl_coord = neighbor_data[idx2][1]
                
#                 # Create Steiner point
#                 steiner_point = steinerPoint3([xi_coord, xk_coord, xl_coord],
#                                             model=space, dist2Points=dist2Points,
#                                             precise=precise)
                
#                 if steiner_point is not None:
#                     steiner_id = f"S{idxSteinerPoint}"
#                     verticesDictSteiner[steiner_id] = steiner_point
                    
#                     # Update structures
#                     mstVert2Adj[xi].discard(xk)
#                     mstVert2Adj[xk].discard(xi)
#                     mstVert2Adj[xi].discard(xl)
#                     mstVert2Adj[xl].discard(xi)
                    
#                     edge_set.discard(tuple(sorted([xi, xk])))
#                     edge_set.discard(tuple(sorted([xi, xl])))
                    
#                     mstVert2Adj[steiner_id] = {xi, xk, xl}
#                     mstVert2Adj[xi].add(steiner_id)
#                     mstVert2Adj[xk].add(steiner_id)
#                     mstVert2Adj[xl].add(steiner_id)
                    
#                     edge_set.add(tuple(sorted([xi, steiner_id])))
#                     edge_set.add(tuple(sorted([xk, steiner_id])))
#                     edge_set.add(tuple(sorted([xl, steiner_id])))
                    
#                     available_indices.discard(idx1)
#                     available_indices.discard(idx2)
                    
#                     idxSteinerPoint += 1
    
#     # Convert back to list
#     mstEdgeList.clear()
#     mstEdgeList.extend([list(edge) for edge in edge_set])
    
#     return verticesDictSteiner, mstVert2Adj, mstEdgeList, idxSteinerPoint


# def remove_degenerate_steiner_points(verticesDictTerminal, verticesDictSteiner, 
#                                    space="Klein", coalesce_threshold=1e-3):
#     """
#     Remove Steiner points that are too close to terminals or other Steiner points.
    
#     Args:
#         verticesDictTerminal: Dictionary of terminal vertices
#         verticesDictSteiner: Dictionary of Steiner vertices
#         space: Space type ("Klein" or "Euclidean")
#         coalesce_threshold: Distance threshold for coalescing
    
#     Returns:
#         verticesDictSteiner: Updated Steiner vertices dictionary
#     """
    
#     if not verticesDictSteiner:
#         return verticesDictSteiner, []
    
#     distance_func = DISTANCE_F[space]
    
#     # Keep iterating until no more removals
#     while True:
#         found_removal = False
#         steiner_ids = list(verticesDictSteiner.keys())
        
#         for steiner_id in steiner_ids:
#             if steiner_id not in verticesDictSteiner:  # Already removed
#                 continue
                
#             steiner_coord = verticesDictSteiner[steiner_id]
            
#             # Check distance to all terminals
#             for terminal_id, terminal_coord in verticesDictTerminal.items():
#                 dist = distance_func(steiner_coord, terminal_coord)
                
#                 if dist < coalesce_threshold:
#                     # Remove Steiner point - it's too close to a terminal
#                     del verticesDictSteiner[steiner_id]
#                     found_removal = True
#                     break
            
#             if found_removal:
#                 break
                
#             # Check distance to other Steiner points
#             other_steiner_ids = [sid for sid in steiner_ids if sid != steiner_id and sid in verticesDictSteiner]
            
#             for other_steiner_id in other_steiner_ids:
#                 other_steiner_coord = verticesDictSteiner[other_steiner_id]
#                 dist = distance_func(steiner_coord, other_steiner_coord)
                
#                 if dist < coalesce_threshold:
#                     # Keep the one with lexicographically smaller ID, remove the other
#                     if steiner_id < other_steiner_id:
#                         del verticesDictSteiner[other_steiner_id]
#                     else:
#                         del verticesDictSteiner[steiner_id]
#                     found_removal = True
#                     break
            
#             if found_removal:
#                 break
        
#         if not found_removal:
#             break
    
#     return verticesDictSteiner



# def exhaustiveMethod_global(points, space="Klein", triangMeth="DT", 
#                           dist2Points=1e-1, precise=False,
#                           selection=("PROB", (0.3, 0.6)), nMaxExpansions=6,
#                           slack=1, improvement_threshold=0.0, num_epochs=200, lr=1e-2, 
#                           coalesce_threshold_factor=0.1, 
#                           extendedResults=False):
#     """Optimized global exhaustive method for Steiner tree construction
    
#     Args:
#         points: List or array of input points
#         space: Geometry space ("Klein" or "Euclidean") 
#         triangMeth: Triangulation method ("DT" or "GG")
#         dist2Points: Distance parameter for Steiner point calculation
#         precise: Use precise calculation methods
#         selection: Selection strategy for triangle processing
#         nMaxExpansions: Maximum number of expansion iterations
#         slack: Tolerance for angle checking
#         improvement_threshold: Improvement threshold for recovering the Steiner topology of new solutions
#         num_epochs: Number of optimization epochs
#         lr: Learning rate for optimization
#         coalesce_threshold_factor: Factor of dist2Points for removing close Steiner points (default: 0.1)
#         cluster_threshold_factor: Factor of dist2Points for clustering Steiner points (default: 0.2)
#         extendedResults: Whether to return extended result information
    
#     Returns:
#         Dictionary with results or tuple depending on extendedResults flag
#     """
    
#     N = len(points)
    
#     # Handle trivial cases
#     if N < 3:
#         result_graph = [["T0", "T1"]] if N == 2 else []
#         vertices_dict = {f"T{i}": points[i] for i in range(N)}
        
#         if not extendedResults:
#             return result_graph, vertices_dict, 1, None, None
#         else:
#             return {
#                 "resultGraph": result_graph,
#                 "verticesDict": vertices_dict,
#                 "methodTime": 0.0,
#                 "steinerVal": 1.0 if N == 2 else 0.0,
#                 "mstVal": 1.0 if N == 2 else 0.0,
#                 "ratio": 1.0,
#                 "mstGraph": result_graph,
#                 "mstTime": 0.0,
#                 "numFST3": -1,
#                 "numFST4": -1,
#                 "edgesDT": []
#             }
    
#     # Validation
#     if space not in ["Klein", "Euclidean"]:
#         raise ValueError("space should be either 'Klein' or 'Euclidean'")
#     if triangMeth not in ["DT", "GG"]:
#         raise ValueError("triangMeth should be either 'DT' or 'GG'")
#     if not ALLOW_PRECISE and precise and space == "Klein":
#         raise ValueError("'precise' can't be True if PHCpy is not installed")
    
#     # Initialize data structures
#     vertices_dict_terminal = {f"T{i}": points[i] for i in range(N)}
#     vertices_dict_steiner = {}
    
#     # Convert points to numpy array for efficiency
#     points_array = np.array(points)
    
#     # Compute initial MST
#     start_time_mst = time.process_time()
#     og_mst_edges, mst_vert_adj, full_mst_val = vanillaMST(vertices_dict_terminal, space=space)
#     mst_time = time.process_time() - start_time_mst
    
#     # Initialize tracking variables
#     og_edges_dt = None
#     idx_steiner_point = 0
#     best_graph = og_mst_edges.copy()
#     best_steiner_val = full_mst_val
#     best_vertices_dict_steiner = {}
    
#     # Create optimizer function
#     optimizer = partial(global_optimization, terminals_klein=points_array, 
#                        num_epochs=num_epochs, lr=lr, verbose=False, plot=False)
    
#     n_expansions = 1
#     start_time = time.process_time()
    
#     # Main optimization loop
#     while n_expansions <= nMaxExpansions:
#         # Build current point set efficiently
#         if vertices_dict_steiner:
#             points_smt = points_array.tolist() + list(vertices_dict_steiner.values())
#         else:
#             points_smt = points_array.tolist()
        
#         # EXPANSION phase
#         vertices_dict_steiner, idx_steiner_point, edges_dt = expansion(
#             nExpansions=1, 
#             points_og_list=points_smt, 
#             verticesDictSteiner=vertices_dict_steiner,
#             space=space, 
#             triangMeth=triangMeth,
#             selection=selection,
#             precise=precise,
#             dist2Points=dist2Points,
#             idxSteinerPoint=idx_steiner_point
#         )
        
#         # Store original DT edges if first iteration
#         if og_edges_dt is None:
#             og_edges_dt = edges_dt
        
#         # CLEANING phase
#         vertices_dict_steiner, mst_edge_list, mst_vert_adj, steiner_val = clean(
#             vertices_dict_terminal, vertices_dict_steiner, optimizer,
#             space=space, dist2Points=dist2Points, precise=precise, slack=slack
#         )
        
#         # EDGE INSERTION phase
#         vertices_dict_steiner, mst_edge_list, idx_steiner_point = edge_insertion(
#             vertices_dict_terminal, vertices_dict_steiner, mst_vert_adj, mst_edge_list,
#             optimizer, space=space, idxSteinerPoint=idx_steiner_point,
#             dist2Points=dist2Points, precise=precise, slack=slack
#         )
        
#         # Recompute MST after edge insertion
#         mst_edge_list, mst_vert_adj, steiner_val = vanillaMST(
#             vertices_dict_terminal, vertices_dict_steiner, space=space
#         )
        
#         # HIGH DEGREE REDUCTION (only if improvement found)
#         if steiner_val - best_steiner_val < improvement_threshold*best_steiner_val:
#             # Remove degenerate Steiner points (too close to terminals or each other)
#             vertices_dict_steiner = remove_degenerate_steiner_points(
#                 vertices_dict_terminal, vertices_dict_steiner, 
#                 space=space, coalesce_threshold=dist2Points * coalesce_threshold_factor
#             )
            
#             # Recompute MST with cleaned Steiner points to get new topology
#             mst_edge_list, mst_vert_adj, steiner_val = vanillaMST(
#                 vertices_dict_terminal, vertices_dict_steiner, space=space
#             )
            
#             # CLEANING phase again after removals
#             vertices_dict_steiner, mst_edge_list, mst_vert_adj, steiner_val = clean(
#             vertices_dict_terminal, vertices_dict_steiner, optimizer,
#             space=space, dist2Points=dist2Points, precise=precise, slack=slack
#             )
        
            
#             #  Run degree reduction on the clean topology
#             vertices_dict_steiner, mst_vert_adj, mst_edge_list, idx_steiner_point = reduce_high_degree_vertices(
#                 vertices_dict_terminal, vertices_dict_steiner, mst_vert_adj, mst_edge_list,
#                 space=space, idxSteinerPoint=idx_steiner_point,
#                 dist2Points=dist2Points, precise=precise
#             )
            

        
#         # Update best solution if improvement found
#         if steiner_val < best_steiner_val:  # Add threshold for meaningful improvement
#             best_steiner_val = steiner_val
#             best_graph = mst_edge_list.copy()
#             best_vertices_dict_steiner = vertices_dict_steiner.copy()
#             n_expansions = 1  # Reset expansion counter
#         else:
#             n_expansions += 1
    
#     method_time = time.process_time() - start_time
    
#     # Prepare final results with consistent naming
#     vertices_dict = vertices_dict_terminal.copy()
#     renaming = {}
    
#     for i, vertex_id in enumerate(best_vertices_dict_steiner.keys()):
#         new_id = f"S{i}"
#         renaming[vertex_id] = new_id
#         vertices_dict[new_id] = best_vertices_dict_steiner[vertex_id]
    
#     # Rename vertices in result graph
#     result_graph = []
#     for v0, v1 in best_graph:
#         v0_renamed = v0 if v0[0] == "T" else renaming[v0]
#         v1_renamed = v1 if v1[0] == "T" else renaming[v1]
#         result_graph.append([v0_renamed, v1_renamed])
    
#     # Prepare return values
#     if not extendedResults:
#         return result_graph, vertices_dict, method_time, best_steiner_val, None
    
#     return {
#         "resultGraph": result_graph,
#         "verticesDict": vertices_dict,
#         "methodTime": method_time,
#         "steinerVal": best_steiner_val,
#         "mstVal": full_mst_val,
#         "ratio": best_steiner_val / full_mst_val if full_mst_val > 0 else 1.0,
#         "mstGraph": og_mst_edges,
#         "mstTime": mst_time,
#         "numFST3": -1,  # TODO: implement proper counting
#         "numFST4": -1,  # TODO: implement proper counting
#         "edgesDT": og_edges_dt
#     }
    
    
import heapq
import time
import numpy as np
from collections import defaultdict
from functools import partial


def ensure_numpy_coords(coords):
    """Ensure coordinates are numpy arrays for consistent processing"""
    if isinstance(coords, (list, tuple)):
        if len(coords) > 0 and isinstance(coords[0], (list, tuple, np.ndarray)):
            # List of coordinate arrays
            return [np.array(coord, dtype=np.float64) for coord in coords]
        else:
            # Single coordinate
            return np.array(coords, dtype=np.float64)
    elif isinstance(coords, np.ndarray):
        return coords.astype(np.float64)
    else:
        return np.array(coords, dtype=np.float64)


# def expansion(nExpansions, points_og_list, verticesDictSteiner=None,
#               space="Klein", triangMeth="DT", selection=("PROB", (0.3, 0.6)),
#               dist2Points=1e-1, precise=True, idxSteinerPoint=0, baricenter=False):
    
#     if verticesDictSteiner is None:
#         verticesDictSteiner = dict()
    
#     # Convert to numpy array once
#     points_og_array = np.array(points_og_list) if not isinstance(points_og_list, np.ndarray) else points_og_list
    
#     # Pre-compute random probability if needed
#     use_probability = selection[0] != "ALL"
#     if use_probability:
#         prob_range = selection[1]
    
#     # Collect all new Steiner points before adding to dictionary
#     new_steiner_points = []
    
#     for exp in range(nExpansions):
#         # Build current point set efficiently
#         if verticesDictSteiner:
#             steiner_array = np.array(list(verticesDictSteiner.values()))
#             points_SMT = np.vstack([points_og_array, steiner_array])
#         else:
#             points_SMT = points_og_array.copy()
        
#         if triangMeth == "DT":
#             try:
#                 computedVoronoiDelaunay = compute_Voronoi_Delaunay(points_SMT, space=space)
#                 triangulation_list = computedVoronoiDelaunay["Delaunay_triangulation"]
#                 triangulation_edges = computedVoronoiDelaunay["Delaunay_edges"]
#             except Exception as e:
#                 print(f"Warning: Delaunay triangulation failed: {e}")
#                 # Fallback: return without adding Steiner points for this expansion
#                 return verticesDictSteiner, idxSteinerPoint, []
#         else:
#             raise ValueError("triangMeth should be DT")
        
#         # Generate single random probability per expansion if needed
#         if use_probability:
#             p = np.random.uniform(*prob_range)
        
#         # Vectorized probability selection if applicable
#         if use_probability and len(triangulation_list) > 10:  # Worth vectorizing for larger sets
#             keep_triangles = np.random.binomial(1, p, len(triangulation_list)).astype(bool)
#             selected_triangles = [tri for tri, keep in zip(triangulation_list, keep_triangles) if keep]
#         else:
#             selected_triangles = triangulation_list
        
#         # Process triangles individually (fix array indexing issue)
#         for triangle_indices in selected_triangles:
#             if use_probability and len(triangulation_list) <= 10:  # Individual check for small sets
#                 if not np.random.binomial(1, p):
#                     continue
            
#             # Extract triangle vertices using proper indexing
#             triangle_vertices = points_SMT[triangle_indices]  # This gives us a 3x2 array
            
#             # Ensure triangle vertices are numpy arrays
#             triangle_vertices_clean = ensure_numpy_coords(triangle_vertices)        
            
#             if not baricenter:
#                 steinerPoint = steinerPoint3(triangle_vertices_clean, model=space, dist2Points=dist2Points,
#                                             precise=precise)
                            
#                 if steinerPoint is not None:
#                     new_steiner_points.append(steinerPoint)
#             else:
#                 if space == "Klein":
#                     steinerPoint = hyperbolic_barycenter(triangle_vertices_clean)
#                 else:
#                     steinerPoint = np.mean(triangle_vertices_clean, axis=0)
                
#                 new_steiner_points.append(steinerPoint)
    
#     # Add all new Steiner points to dictionary in batch
#     for point in new_steiner_points:
#         verticesDictSteiner[f"S{idxSteinerPoint}"] = point
#         idxSteinerPoint += 1
    
#     return verticesDictSteiner, idxSteinerPoint, triangulation_edges


# def vanillaMST(verticesDictTerminal, verticesDictSteiner=None, space="Klein"):
#     """Optimized MST using heapq instead of PriorityQueue"""
    
#     # Collect all vertex IDs
#     terminal_ids = list(verticesDictTerminal.keys())
#     steiner_ids = list(verticesDictSteiner.keys()) if verticesDictSteiner else []
#     all_vertex_ids = terminal_ids + steiner_ids
#     num_vertices = len(all_vertex_ids)
    
#     if num_vertices < 2:
#         return [], defaultdict(set), 0.0
    
#     # Create coordinate lookup for faster access
#     coord_lookup = verticesDictTerminal.copy()
#     if verticesDictSteiner:
#         coord_lookup.update(verticesDictSteiner)
    
#     # Build edge heap more efficiently
#     edge_heap = []
#     distance_func = DISTANCE_F[space]
    
#     for i, vertex_i in enumerate(all_vertex_ids):
#         coord_i = coord_lookup[vertex_i]
        
#         for j in range(i + 1, num_vertices):
#             vertex_j = all_vertex_ids[j]
#             coord_j = coord_lookup[vertex_j]
            
#             dist = distance_func(coord_i, coord_j)
#             heapq.heappush(edge_heap, (dist, vertex_i, vertex_j))
    
#     # Initialize disjoint set
#     parent = {v: v for v in all_vertex_ids}
#     rank = {v: 0 for v in all_vertex_ids}
    
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])  # Path compression
#         return parent[x]
    
#     def union(x, y):
#         px, py = find(x), find(y)
#         if px == py:
#             return False
        
#         # Union by rank
#         if rank[px] < rank[py]:
#             px, py = py, px
#         parent[py] = px
#         if rank[px] == rank[py]:
#             rank[px] += 1
#         return True
    
#     # Build MST
#     mst_edges = []
#     mst_adjacency = defaultdict(set)
#     total_weight = 0.0
#     edges_added = 0
    
#     while edge_heap and edges_added < num_vertices - 1:
#         weight, u, v = heapq.heappop(edge_heap)
        
#         if union(u, v):
#             mst_edges.append([u, v])
#             mst_adjacency[u].add(v)
#             mst_adjacency[v].add(u)
#             total_weight += weight
#             edges_added += 1
    
#     return mst_edges, mst_adjacency, total_weight



def clean(verticesDictTerminal, verticesDictSteiner, optimizer, space="Klein", 
          dist2Points=1e-1, precise=True, slack=1, maxgroup=4, nIters=50, convDiff=1e-3, idxSteinerPoint=0):
    """Optimized cleaning function with reduced MST recomputations"""
    
    if verticesDictSteiner is None:
        verticesDictSteiner = dict()

    # Initial MST computation
    mstEdgeList, mstVert2Adj, fullMSTval = delaunayMST(verticesDictTerminal, 
                                                     verticesDictSteiner, space=space)
    
    iteration = 0
    while True:
        structure_modified = False
        
        # Process degree-1 and degree-2 vertices in single pass
        steiner_vertices = list(verticesDictSteiner.keys())
        for vertex_id in steiner_vertices:
            if vertex_id not in mstVert2Adj:
                continue
                
            neighbors = mstVert2Adj[vertex_id]
            degree = len(neighbors)
            
            if degree == 1:
                # Remove degree-1 vertex
                neighbor = neighbors.pop()
                mstVert2Adj[neighbor].discard(vertex_id)
                mstVert2Adj.pop(vertex_id, None)
                verticesDictSteiner.pop(vertex_id, None)
                structure_modified = True
                
            elif degree == 2:
                # Connect the two neighbors directly
                neighbor1, neighbor2 = list(neighbors)
                
                # Remove connections to current vertex
                mstVert2Adj[neighbor1].discard(vertex_id)
                mstVert2Adj[neighbor2].discard(vertex_id)
                
                # Connect neighbors to each other
                mstVert2Adj[neighbor1].add(neighbor2)
                mstVert2Adj[neighbor2].add(neighbor1)
                
                # Remove current vertex
                mstVert2Adj.pop(vertex_id, None)
                verticesDictSteiner.pop(vertex_id, None)
                structure_modified = True
        
        if not structure_modified and iteration > 0:
            break
        iteration += 1
        
        # Optimize degree-3 vertices
        steiner_vertices = list(verticesDictSteiner.keys())
        for vertex_id in steiner_vertices:
            if vertex_id not in mstVert2Adj:
                continue
                
            neighbors = mstVert2Adj[vertex_id]
            if len(neighbors) != 3:
                continue
                
            # Get coordinates efficiently
            neighbor_coords = []
            for neighbor_id in neighbors:
                if neighbor_id[0] == "T":
                    neighbor_coords.append(verticesDictTerminal[neighbor_id])
                else:
                    neighbor_coords.append(verticesDictSteiner[neighbor_id])
            
            current_coord = verticesDictSteiner[vertex_id]
            
            if not isOptimal(current_coord, neighbor_coords, space=space, slack=slack):
                # Convert to numpy arrays for steinerPoint3
                neighbor_coords_clean = ensure_numpy_coords(neighbor_coords)
                
                steiner_point = steinerPoint3(neighbor_coords_clean, model=space,
                                            dist2Points=dist2Points, precise=precise)
                
                if steiner_point is None:
                    # Remove vertex if no valid Steiner point
                    for neighbor in neighbors:
                        mstVert2Adj[neighbor].discard(vertex_id)
                    mstVert2Adj.pop(vertex_id, None)
                    verticesDictSteiner.pop(vertex_id, None)
                else:
                    # Update with optimized position
                    verticesDictSteiner[vertex_id] = steiner_point
        
        # Recompute MST only once per iteration
        mstEdgeList, mstVert2Adj, fullMSTval = delaunayMST(verticesDictTerminal, 
                                                         verticesDictSteiner, space=space)
    
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
                
            
                verticesDictSteiner[f"S{idxSteinerPoint}"] = bestSteinerPoints[0]
                verticesDictSteiner[f"S{idxSteinerPoint + 1}"] = bestSteinerPoints[1]
                
                mstVert2Adj[f"S{idxSteinerPoint}"].add(f"S{idxSteinerPoint + 1}")
                mstVert2Adj[f"S{idxSteinerPoint + 1}"].add(f"S{idxSteinerPoint}")
                
                for i in range(2):
                    for j in range(2):
                        mstVert2Adj[adj_list[bestTopo[i][j]]].add(f"S{idxSteinerPoint + i}")
                        mstVert2Adj[f"S{idxSteinerPoint + i}"].add(adj_list[bestTopo[i][j]])


                idxSteinerPoint += 2
    
    
    # Check if optimization is needed
    steiner_neighbors_exist = any(
        any(neighbor[0] == "S" for neighbor in mstVert2Adj.get(vertex_id, []))
        for vertex_id in verticesDictSteiner.keys()
    )
    
    if optimizer is not None and steiner_neighbors_exist:
        verticesDictSteiner = optimizer(vertices=verticesDictSteiner, graph=mstEdgeList)
    
    return verticesDictSteiner, mstEdgeList, mstVert2Adj, idxSteinerPoint


def edge_insertion(verticesDictTerminal, verticesDictSteiner, mstVert2Adj, 
                  mstEdgeList, optimizer, space="Klein", idxSteinerPoint=0, 
                  dist2Points=1e-1, precise=True, slack=1):
    """Optimized edge insertion with reduced redundant checks"""
    
    # Convert edge list to set for O(1) operations
    edge_set = {tuple(sorted(edge)) for edge in mstEdgeList}
    
    vertex_angle_ok = True
    all_vertices = list(mstVert2Adj.keys())
    
    # Cache coordinate lookups
    coord_cache = {}
    for vertex_id in all_vertices:
        if vertex_id[0] == "T":
            coord_cache[vertex_id] = verticesDictTerminal[vertex_id]
        else:
            coord_cache[vertex_id] = verticesDictSteiner[vertex_id]
    
    for xi in all_vertices:
        if xi not in mstVert2Adj:
            continue
            
        xi_coord = coord_cache[xi]
        xi_neighbors = list(mstVert2Adj[xi])
        
        for xj in xi_neighbors:
            if xj not in mstVert2Adj or xi not in mstVert2Adj[xj]:
                continue
                
            xj_coord = coord_cache[xj]
            
            # Find minimum angle edge efficiently
            min_angle = float('inf')
            selected_xk = None
            selected_xk_coord = None
            
            for xl in mstVert2Adj[xj]:
                if xl == xi:
                    continue
                    
                xl_coord = coord_cache[xl]
                angle = innerAngleTriangle(xi_coord, xj_coord, xl_coord, space=space)
                
                if angle * 180 / np.pi < 120 - slack and angle < min_angle:
                    min_angle = angle
                    selected_xk = xl
                    selected_xk_coord = xl_coord
            
            if selected_xk is not None:
                # Convert coordinates to numpy arrays to avoid type issues
                coords_for_steiner = ensure_numpy_coords([xi_coord, xj_coord, selected_xk_coord])
                
                steiner_point = steinerPoint3(coords_for_steiner,
                                            model=space, dist2Points=dist2Points, 
                                            precise=precise)
                
                if steiner_point is not None:
                    steiner_id = f"S{idxSteinerPoint}"
                    verticesDictSteiner[steiner_id] = steiner_point
                    coord_cache[steiner_id] = steiner_point
                    
                    # Update adjacency structure
                    mstVert2Adj[xi].discard(xj)
                    mstVert2Adj[xj].discard(xi)
                    mstVert2Adj[xj].discard(selected_xk)
                    mstVert2Adj[selected_xk].discard(xj)
                    
                    # Update edge set
                    edge_set.discard(tuple(sorted([xi, xj])))
                    edge_set.discard(tuple(sorted([xj, selected_xk])))
                    
                    # Add new connections
                    mstVert2Adj[steiner_id] = {xi, xj, selected_xk}
                    mstVert2Adj[xi].add(steiner_id)
                    mstVert2Adj[xj].add(steiner_id)
                    mstVert2Adj[selected_xk].add(steiner_id)
                    
                    # Add new edges
                    edge_set.add(tuple(sorted([xi, steiner_id])))
                    edge_set.add(tuple(sorted([xj, steiner_id])))
                    edge_set.add(tuple(sorted([selected_xk, steiner_id])))
                    
                    idxSteinerPoint += 1
                    vertex_angle_ok = False
    
    # Convert back to list format
    mstEdgeList.clear()
    mstEdgeList.extend([list(edge) for edge in edge_set])
    
    if optimizer is not None and not vertex_angle_ok:
        verticesDictSteiner = optimizer(vertices=verticesDictSteiner, graph=mstEdgeList)
    
    return verticesDictSteiner, mstEdgeList, idxSteinerPoint


def reduce_high_degree_vertices(verticesDictTerminal, verticesDictSteiner, 
                               mstVert2Adj, mstEdgeList, optimizer, space="Klein", 
                               idxSteinerPoint=0, dist2Points=1e-1, precise=True):
    """Optimized high-degree vertex reduction"""
    
    # Convert edge list to set for O(1) operations
    edge_set = {tuple(sorted(edge)) for edge in mstEdgeList}
    all_vertices = list(mstVert2Adj.keys())
    
    for xi in all_vertices:
        if xi not in mstVert2Adj:
            continue
            
        neighbors = list(mstVert2Adj[xi])
        
        if len(neighbors) > 3:
            # Cache coordinate
            xi_coord = verticesDictTerminal[xi] if xi[0] == "T" else verticesDictSteiner[xi]
            
            # Cache neighbor data
            neighbor_data = []
            for neighbor_id in neighbors:
                if neighbor_id[0] == "T":
                    coord = verticesDictTerminal[neighbor_id]
                else:
                    coord = verticesDictSteiner[neighbor_id]
                neighbor_data.append((neighbor_id, coord))
            
            available_indices = set(range(len(neighbor_data)))
            num_reductions = len(neighbors) - 3
            
            for _ in range(num_reductions):
                if len(available_indices) < 2:
                    break
                
                # Find minimum angle pair
                min_angle = float('inf')
                best_pair_indices = None
                
                available_list = list(available_indices)
                for i in range(len(available_list)):
                    for j in range(i + 1, len(available_list)):
                        idx1, idx2 = available_list[i], available_list[j]
                        
                        coord_k = neighbor_data[idx1][1]
                        coord_l = neighbor_data[idx2][1]
                        
                        angle = innerAngleTriangle(coord_k, xi_coord, coord_l, space=space)
                        
                        if angle < min_angle:
                            min_angle = angle
                            best_pair_indices = (idx1, idx2)
                
                if best_pair_indices is None:
                    break
                
                idx1, idx2 = best_pair_indices
                xk = neighbor_data[idx1][0]
                xk_coord = neighbor_data[idx1][1]
                xl = neighbor_data[idx2][0]
                xl_coord = neighbor_data[idx2][1]
                
                # Create Steiner point
                coords_for_steiner = ensure_numpy_coords([xi_coord, xk_coord, xl_coord])
                
                steiner_point = steinerPoint3(coords_for_steiner,
                                            model=space, dist2Points=dist2Points,
                                            precise=precise)
                
                if steiner_point is not None:
                    steiner_id = f"S{idxSteinerPoint}"
                    verticesDictSteiner[steiner_id] = steiner_point
                    
                    # Update structures
                    mstVert2Adj[xi].discard(xk)
                    mstVert2Adj[xk].discard(xi)
                    mstVert2Adj[xi].discard(xl)
                    mstVert2Adj[xl].discard(xi)
                    
                    edge_set.discard(tuple(sorted([xi, xk])))
                    edge_set.discard(tuple(sorted([xi, xl])))
                    
                    mstVert2Adj[steiner_id] = {xi, xk, xl}
                    mstVert2Adj[xi].add(steiner_id)
                    mstVert2Adj[xk].add(steiner_id)
                    mstVert2Adj[xl].add(steiner_id)
                    
                    edge_set.add(tuple(sorted([xi, steiner_id])))
                    edge_set.add(tuple(sorted([xk, steiner_id])))
                    edge_set.add(tuple(sorted([xl, steiner_id])))
                    
                    available_indices.discard(idx1)
                    available_indices.discard(idx2)
                    
                    idxSteinerPoint += 1
    
    # Convert back to list
    mstEdgeList.clear()
    mstEdgeList.extend([list(edge) for edge in edge_set])
    
    if optimizer is not None:
        verticesDictSteiner = optimizer(vertices=verticesDictSteiner, graph=mstEdgeList)
    
    return verticesDictSteiner, mstVert2Adj, mstEdgeList, idxSteinerPoint


def remove_degenerate_steiner_points(verticesDictTerminal, verticesDictSteiner, 
                                   space="Klein", coalesce_threshold=1e-3):
    """
    Remove Steiner points that are too close to terminals or other Steiner points.
    
    Args:
        verticesDictTerminal: Dictionary of terminal vertices
        verticesDictSteiner: Dictionary of Steiner vertices
        space: Space type ("Klein" or "Euclidean")
        coalesce_threshold: Distance threshold for coalescing
    
    Returns:
        verticesDictSteiner: Updated Steiner vertices dictionary
        removed_points: List of removed Steiner point IDs
    """
    
    if not verticesDictSteiner:
        return verticesDictSteiner, []
    
    distance_func = DISTANCE_F[space]
    removed_points = []
    
    # Keep iterating until no more removals
    while True:
        found_removal = False
        steiner_ids = list(verticesDictSteiner.keys())
        
        for steiner_id in steiner_ids:
            if steiner_id not in verticesDictSteiner:  # Already removed
                continue
                
            steiner_coord = verticesDictSteiner[steiner_id]
            
            # Check distance to all terminals
            for terminal_id, terminal_coord in verticesDictTerminal.items():
                dist = distance_func(steiner_coord, terminal_coord)
                
                if dist < coalesce_threshold:
                    # Remove Steiner point - it's too close to a terminal
                    del verticesDictSteiner[steiner_id]
                    removed_points.append(steiner_id)
                    found_removal = True
                    break
            
            if found_removal:
                break
                
            # Check distance to other Steiner points
            other_steiner_ids = [sid for sid in steiner_ids if sid != steiner_id and sid in verticesDictSteiner]
            
            for other_steiner_id in other_steiner_ids:
                other_steiner_coord = verticesDictSteiner[other_steiner_id]
                dist = distance_func(steiner_coord, other_steiner_coord)
                
                if dist < coalesce_threshold:
                    # Keep the one with lexicographically smaller ID, remove the other
                    if steiner_id < other_steiner_id:
                        del verticesDictSteiner[other_steiner_id]
                        removed_points.append(other_steiner_id)
                    else:
                        del verticesDictSteiner[steiner_id]
                        removed_points.append(steiner_id)
                    found_removal = True
                    break
            
            if found_removal:
                break
        
        if not found_removal:
            break
    
    return verticesDictSteiner, removed_points




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



def exhaustiveMethod_global(points, space="Klein", triangMeth="DT", 
                          dist2Points=1e-1, precise=False,
                          selection=("PROB", (0.3, 0.6)), nMaxExpansions=6,
                          slack=1, threshold_improvement=0.0, num_epochs=200, lr=1e-2,  maxgroup=3,
                         nIters=100, convDiff=1e-2,  early_stopping=True, patience=100, min_delta=1e-6, 
                          extendedResults=False, expansion_mode="sqrt"):
    """Optimized global exhaustive method for Steiner tree construction
    
    Args:
        points: List or array of input points
        space: Geometry space ("Klein" or "Euclidean") 
        triangMeth: Triangulation method ("DT" or "GG")
        dist2Points: Distance parameter for Steiner point calculation
        precise: Use precise calculation methods
        selection: Selection strategy for triangle processing
        nMaxExpansions: Maximum number of expansion iterations
        slack: Tolerance for angle checking
        threshold_improvement: Improvement threshold for accepting new solutions
        num_epochs: Number of optimization epochs
        lr: Learning rate for optimization
        extendedResults: Whether to return extended result information
    
    Returns:
        Dictionary with results or tuple depending on extendedResults flag
    """
    
    N = len(points)
    
    # Handle trivial cases
    if N < 3:
        result_graph = [["T0", "T1"]] if N == 2 else []
        vertices_dict = {f"T{i}": points[i] for i in range(N)}
        
        if not extendedResults:
            return result_graph, vertices_dict, 1, None, None
        else:
            return {
                "resultGraph": result_graph,
                "verticesDict": vertices_dict,
                "methodTime": 0.0,
                "steinerVal": 1.0 if N == 2 else 0.0,
                "mstVal": 1.0 if N == 2 else 0.0,
                "ratio": 1.0,
                "mstGraph": result_graph,
                "mstTime": 0.0,
                "numFST3": -1,
                "numFST4": -1,
                "edgesDT": []
            }
    
    # Validation
    if space not in ["Klein", "Euclidean"]:
        raise ValueError("space should be either 'Klein' or 'Euclidean'")
    if triangMeth not in ["DT", "GG"]:
        raise ValueError("triangMeth should be either 'DT' or 'GG'")
    if not ALLOW_PRECISE and precise and space == "Klein":
        raise ValueError("'precise' can't be True if PHCpy is not installed")
    
    # Initialize data structures
    vertices_dict_terminal = {f"T{i}": points[i] for i in range(N)}
    vertices_dict_steiner = {}
    
    # Convert points to numpy array for efficiency
    points_array = np.array(points)
    
    # Compute initial MST
    start_time_mst = time.process_time()
    og_mst_edges, mst_vert_adj, full_mst_val = delaunayMST(vertices_dict_terminal, space=space)
    mst_time = time.process_time() - start_time_mst
    
    # Initialize tracking variables
    og_edges_dt = None
    idx_steiner_point = 0
    best_graph = og_mst_edges.copy()
    best_steiner_val = full_mst_val
    best_vertices_dict_steiner = {}
    
    # Create optimizer function
    optimizer = None
    if space=="Klein":
        optimizer = partial(global_optimization, terminals_klein=points_array, 
                        num_epochs=num_epochs, lr=lr, verbose=False, plot=False,  early_stopping=early_stopping, patience=patience, min_delta=min_delta, )
        
    n_expansions = 1
    start_time = time.process_time()
    
    # Main optimization loop
    while n_expansions <= nMaxExpansions:
        # Make a copy of current best Steiner vertices at the beginning of iteration
        # This ensures we can restore if no improvement is found
        vertices_dict_steiner = best_vertices_dict_steiner.copy()
        
        # Build current point set efficiently
        if vertices_dict_steiner:
            points_smt = points_array.tolist() + list(vertices_dict_steiner.values())
        else:
            points_smt = points_array.tolist()
        
        
        if expansion_mode =="constant":
            nExpansions=1
        elif expansion_mode =="linear":
            nExpansions=n_expansions
        else :  # sqrt mode
            nExpansions=int(2*np.sqrt(n_expansions)-1)
        
        # EXPANSION phase
        vertices_dict_steiner, idx_steiner_point, edges_dt = expansion(
            nExpansions=nExpansions, 
            points_og_list=points_smt, 
            verticesDictSteiner=vertices_dict_steiner,
            space=space, 
            triangMeth=triangMeth,
            selection=selection,
            precise=precise,
            dist2Points=dist2Points,
            idxSteinerPoint=idx_steiner_point,
            baricenter=True
        )
        
        # Store original DT edges if first iteration
        if og_edges_dt is None:
            og_edges_dt = edges_dt
            
        
        vertices_dict_steiner, idx_steiner_point, mst_edge_list, mst_vert_adj = reduction(vertices_dict_terminal,
                                                                                                vertices_dict_steiner,
                                                                                                space=space, idxSteinerPoint=idx_steiner_point, 
                                                                                                dist2Points=dist2Points, precise=precise, 
                                                                                               slack=slack,nIters=nIters, convDiff=convDiff, 
                                                                                                maxgroup=maxgroup)
        vertices_dict_steiner = optimizer(vertices=vertices_dict_steiner, graph=mst_edge_list)


    
        # EDGE INSERTION phase
        vertices_dict_steiner, mst_edge_list, idx_steiner_point = edge_insertion(
            vertices_dict_terminal, vertices_dict_steiner, mst_vert_adj, mst_edge_list,
            optimizer, space=space, idxSteinerPoint=idx_steiner_point,
            dist2Points=dist2Points, precise=precise, slack=slack
        )
        
        
        
        steiner_val = compute_tree_length(mst_edge_list, vertices_dict_terminal, vertices_dict_steiner, space)
        
        if steiner_val-best_steiner_val < threshold_improvement*best_steiner_val:

            vertices_dict_steiner, idx_steiner_point, mst_edge_list, mst_vert_adj = reduction(vertices_dict_terminal,
                                                                                                vertices_dict_steiner,
                                                                                                space=space, idxSteinerPoint=idx_steiner_point, 
                                                                                                dist2Points=dist2Points, precise=precise, 
                                                                                               slack=slack,nIters=nIters, convDiff=convDiff, 
                                                                                                maxgroup=maxgroup)
            vertices_dict_steiner = optimizer(vertices=vertices_dict_steiner, graph=mst_edge_list)
            steiner_val = compute_tree_length(mst_edge_list, vertices_dict_terminal, vertices_dict_steiner, space)

      
        
        # Update best solution if improvement found
        if steiner_val < best_steiner_val:  # Add threshold for meaningful improvement
            best_steiner_val = steiner_val
            best_graph = mst_edge_list.copy()
            best_vertices_dict_steiner = vertices_dict_steiner.copy()
            n_expansions = 1  # Reset expansion counter

        else:
            n_expansions += 1

    
    method_time = time.process_time() - start_time
    
    # Prepare final results with consistent naming
    vertices_dict = vertices_dict_terminal.copy()
    renaming = {}
    
    for i, vertex_id in enumerate(best_vertices_dict_steiner.keys()):
        new_id = f"S{i}"
        renaming[vertex_id] = new_id
        vertices_dict[new_id] = best_vertices_dict_steiner[vertex_id]
    
    # Rename vertices in result graph
    result_graph = []
    for v0, v1 in best_graph:
        v0_renamed = v0 if v0[0] == "T" else renaming[v0]
        v1_renamed = v1 if v1[0] == "T" else renaming[v1]
        result_graph.append([v0_renamed, v1_renamed])
    
    # Prepare return values
    if not extendedResults:
        return result_graph, vertices_dict, method_time, best_steiner_val, None
    
    return {
        "resultGraph": result_graph,
        "verticesDict": vertices_dict,
        "methodTime": method_time,
        "steinerVal": best_steiner_val,
        "mstVal": full_mst_val,
        "ratio": best_steiner_val / full_mst_val if full_mst_val > 0 else 1.0,
        "mstGraph": og_mst_edges,
        "mstTime": mst_time,
        "numFST3": -1,  # TODO: implement proper counting
        "numFST4": -1,  # TODO: implement proper counting
        "edgesDT": og_edges_dt
    }





def exhaustiveMethod_global_CLASIC(points, space="Klein", triangMeth="DT", 
                          dist2Points=1e-1, precise=False,
                          selection=("PROB", (0.3, 0.6)), nMaxExpansions=6,
                          slack=1, threshold_improvement=0.0, num_epochs=200, lr=1e-2, 
                          extendedResults=False):
    """Optimized global exhaustive method for Steiner tree construction
    
    Args:
        points: List or array of input points
        space: Geometry space ("Klein" or "Euclidean") 
        triangMeth: Triangulation method ("DT" or "GG")
        dist2Points: Distance parameter for Steiner point calculation
        precise: Use precise calculation methods
        selection: Selection strategy for triangle processing
        nMaxExpansions: Maximum number of expansion iterations
        slack: Tolerance for angle checking
        threshold_improvement: Improvement threshold for accepting new solutions
        num_epochs: Number of optimization epochs
        lr: Learning rate for optimization
        extendedResults: Whether to return extended result information
    
    Returns:
        Dictionary with results or tuple depending on extendedResults flag
    """
    
    N = len(points)
    
    # Handle trivial cases
    if N < 3:
        result_graph = [["T0", "T1"]] if N == 2 else []
        vertices_dict = {f"T{i}": points[i] for i in range(N)}
        
        if not extendedResults:
            return result_graph, vertices_dict, 1, None, None
        else:
            return {
                "resultGraph": result_graph,
                "verticesDict": vertices_dict,
                "methodTime": 0.0,
                "steinerVal": 1.0 if N == 2 else 0.0,
                "mstVal": 1.0 if N == 2 else 0.0,
                "ratio": 1.0,
                "mstGraph": result_graph,
                "mstTime": 0.0,
                "numFST3": -1,
                "numFST4": -1,
                "edgesDT": []
            }
    
    # Validation
    if space not in ["Klein", "Euclidean"]:
        raise ValueError("space should be either 'Klein' or 'Euclidean'")
    if triangMeth not in ["DT", "GG"]:
        raise ValueError("triangMeth should be either 'DT' or 'GG'")
    if not ALLOW_PRECISE and precise and space == "Klein":
        raise ValueError("'precise' can't be True if PHCpy is not installed")
    
    # Initialize data structures
    vertices_dict_terminal = {f"T{i}": points[i] for i in range(N)}
    vertices_dict_steiner = {}
    
    # Convert points to numpy array for efficiency
    points_array = np.array(points)
    
    # Compute initial MST
    start_time_mst = time.process_time()
    og_mst_edges, mst_vert_adj, full_mst_val = delaunayMST(vertices_dict_terminal, space=space)
    mst_time = time.process_time() - start_time_mst
    
    # Initialize tracking variables
    og_edges_dt = None
    idx_steiner_point = 0
    best_graph = og_mst_edges.copy()
    best_steiner_val = full_mst_val
    best_vertices_dict_steiner = {}
    
    # Create optimizer function
    optimizer = None
    if space=="Klein":
        optimizer = partial(global_optimization, terminals_klein=points_array, 
                        num_epochs=num_epochs, lr=lr, verbose=True, plot=True,  early_stopping=True, patience=1000, min_delta=1e-6, )
        
    n_expansions = 1
    start_time = time.process_time()
    auxSec = ("ALL", (1.0, 1.0))
    
    # Main optimization loop
    while n_expansions <= nMaxExpansions:
        # Make a copy of current best Steiner vertices at the beginning of iteration
        # This ensures we can restore if no improvement is found
        vertices_dict_steiner = best_vertices_dict_steiner.copy()
        
        # Build current point set efficiently
        if vertices_dict_steiner:
            points_smt = points_array.tolist() + list(vertices_dict_steiner.values())
        else:
            points_smt = points_array.tolist()
        
        
        
        
        # EXPANSION phase
        vertices_dict_steiner, idx_steiner_point, edges_dt = expansion(
            nExpansions=n_expansions, 
            points_og_list=points_smt, 
            verticesDictSteiner=vertices_dict_steiner,
            space=space, 
            triangMeth=triangMeth,
            selection=auxSec,
            precise=precise,
            dist2Points=dist2Points,
            idxSteinerPoint=idx_steiner_point
        )
        
        # Store original DT edges if first iteration
        if og_edges_dt is None:
            og_edges_dt = edges_dt
            auxSec = selection
            
        
        # CLEANING phase
        vertices_dict_steiner, mst_edge_list, mst_vert_adj, idx_steiner_point = clean(
            vertices_dict_terminal, vertices_dict_steiner, optimizer,
            space=space, dist2Points=dist2Points, precise=precise, slack=slack, idxSteinerPoint=idx_steiner_point
        )
        
        # EDGE INSERTION phase
        vertices_dict_steiner, mst_edge_list, idx_steiner_point = edge_insertion(
            vertices_dict_terminal, vertices_dict_steiner, mst_vert_adj, mst_edge_list,
            optimizer, space=space, idxSteinerPoint=idx_steiner_point,
            dist2Points=dist2Points, precise=precise, slack=slack
        )
        
        
        
        steiner_val = compute_tree_length(mst_edge_list, vertices_dict_terminal, vertices_dict_steiner, space)
        
        # HIGH DEGREE REDUCTION (only if improvement found)
        if steiner_val-best_steiner_val < threshold_improvement*best_steiner_val:
            # # First: Remove degenerate Steiner points (too close to terminals or each other)
            # vertices_dict_steiner, removed_points = remove_degenerate_steiner_points(
            #     vertices_dict_terminal, vertices_dict_steiner, 
            #     space=space, coalesce_threshold=dist2Points
            # )
            
            # # Second: Recompute MST with cleaned Steiner points to get new topology
            # mst_edge_list, mst_vert_adj, steiner_val = vanillaMST(
            #     vertices_dict_terminal, vertices_dict_steiner, space=space
            # )
            
            # CLEANING phase
            # vertices_dict_steiner, mst_edge_list, mst_vert_adj, idx_steiner_point = clean(
            # vertices_dict_terminal, vertices_dict_steiner, optimizer,
            # space=space, dist2Points=dist2Points, precise=precise, slack=slack, idxSteinerPoint=idx_steiner_point
            # )
                
            # Third: Run degree reduction on the clean topology
            vertices_dict_steiner, mst_vert_adj, mst_edge_list, idx_steiner_point = reduce_high_degree_vertices(
                vertices_dict_terminal, vertices_dict_steiner, mst_vert_adj, mst_edge_list,optimizer,
                space=space, idxSteinerPoint=idx_steiner_point,
                dist2Points=dist2Points, precise=precise
            )
            
            steiner_val = compute_tree_length(mst_edge_list, vertices_dict_terminal, vertices_dict_steiner, space)

        
        # Update best solution if improvement found
        if steiner_val < best_steiner_val:  # Add threshold for meaningful improvement
            best_steiner_val = steiner_val
            best_graph = mst_edge_list.copy()
            best_vertices_dict_steiner = vertices_dict_steiner.copy()
            n_expansions = 1  # Reset expansion counter
            # auxSec=("ALL", (1.0, 1.0))

        else:
            n_expansions += 1
            # auxSec=selection

    
    method_time = time.process_time() - start_time
    
    # Prepare final results with consistent naming
    vertices_dict = vertices_dict_terminal.copy()
    renaming = {}
    
    for i, vertex_id in enumerate(best_vertices_dict_steiner.keys()):
        new_id = f"S{i}"
        renaming[vertex_id] = new_id
        vertices_dict[new_id] = best_vertices_dict_steiner[vertex_id]
    
    # Rename vertices in result graph
    result_graph = []
    for v0, v1 in best_graph:
        v0_renamed = v0 if v0[0] == "T" else renaming[v0]
        v1_renamed = v1 if v1[0] == "T" else renaming[v1]
        result_graph.append([v0_renamed, v1_renamed])
    
    # Prepare return values
    if not extendedResults:
        return result_graph, vertices_dict, method_time, best_steiner_val, None
    
    return {
        "resultGraph": result_graph,
        "verticesDict": vertices_dict,
        "methodTime": method_time,
        "steinerVal": best_steiner_val,
        "mstVal": full_mst_val,
        "ratio": best_steiner_val / full_mst_val if full_mst_val > 0 else 1.0,
        "mstGraph": og_mst_edges,
        "mstTime": mst_time,
        "numFST3": -1,  # TODO: implement proper counting
        "numFST4": -1,  # TODO: implement proper counting
        "edgesDT": og_edges_dt
    }

