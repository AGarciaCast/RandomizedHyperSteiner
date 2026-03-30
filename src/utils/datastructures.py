import numpy as np
import networkx as nx

def extract_steiner_coordinates(vertices, extra_keys=False):
    """
    Extract Steiner node coordinates from vertices dictionary preserving their order.
    
    Parameters:
    -----------
    vertices : dict
        Dictionary containing vertex keys ('S0', 'S1', etc.) and their coordinate arrays as values.
        
    Returns:
    --------
    numpy.ndarray
        Array of Steiner node coordinates in order (S0, S1, S2, ...).
        Shape: (num_steiner_nodes, coordinate_dimension)
    """
    # Filter keys that start with 'S' (Steiner nodes)
    steiner_keys = [key for key in vertices.keys() if key.startswith('S')]
    
    # Sort by the numeric part to preserve order (S0, S1, S2, ...)
    steiner_keys_sorted = sorted(steiner_keys, key=lambda x: int(x[1:]))
    
    # Extract coordinates in order
    steiner_coordinates = [vertices[key] for key in steiner_keys_sorted]
    
    # Convert to numpy array if we have any Steiner nodes
    if steiner_coordinates:
        if extra_keys:
            return np.array(steiner_coordinates), steiner_keys_sorted
        else:
            return np.array(steiner_coordinates)
    else:
        # Return empty array with appropriate shape if no Steiner nodes
        # Klein coordinates are 2D, so empty array should have shape (0, 2)
        if extra_keys:
            return  np.array([]).reshape(0, 2), None
        else:
            return  np.array([]).reshape(0, 2)
        

def edges_to_networkx(graph):
    """
    Convert a list of edges to a NetworkX graph object.
    
    Parameters:
    -----------
    graph : list of list
        List of edges where each edge is represented as a list of two nodes.
        Example: [['T30', 'S17'], ['T25', 'S6'], ['T32', 'S15'], ...]
        
    Returns:
    --------
    networkx.Graph
        NetworkX undirected graph object containing the edges and nodes.
    """
    # Create an empty undirected graph
    G = nx.Graph()
    
    # Add edges to the graph (nodes are added automatically)
    for edge in graph:
        if len(edge) == 2:
            G.add_edge(edge[0], edge[1])
        else:
            raise ValueError(f"Each edge must contain exactly 2 nodes, got {len(edge)} in edge: {edge}")
    
    return G