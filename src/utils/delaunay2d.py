import itertools
import numpy
from scipy.spatial import ConvexHull

import numpy as np

from matplotlib.collections import LineCollection
from matplotlib import pyplot as plot

# --- Misc. geometry code -----------------------------------------------------

def norm2(X):
	return numpy.sqrt(numpy.sum(X ** 2))

def normalized(X):
	return X / norm2(X)

# --- Delaunay triangulation --------------------------------------------------

def get_triangle_normal(A, B, C):
	return normalized(numpy.cross(A, B) + numpy.cross(B, C) + numpy.cross(C, A))


def get_power_circumcenter(A, B, C):
	N = get_triangle_normal(A, B, C)
	return (-.5 / N[2]) * N[:2]


def is_ccw_triangle(A, B, C):
	M = numpy.concatenate([numpy.stack([A, B, C]), numpy.ones((3, 1))], axis = 1)
	return numpy.linalg.det(M) > 0


def get_power_triangulation(S, R):
	# Compute the lifted weighted points
	S_norm = numpy.sum(S ** 2, axis = 1) - R #instead of ...-R**2, because error in the paper where the formula gives R**2 and not R 
	S_lifted = numpy.concatenate([S, S_norm[:,None]], axis = 1)

	# Special case for 3 points
	if S.shape[0] == 3:
		if is_ccw_triangle(S[0], S[1], S[2]):
			return [[0, 1, 2]], numpy.array([get_power_circumcenter(*S_lifted)])
		else:
			return [[0, 2, 1]], numpy.array([get_power_circumcenter(*S_lifted)])

	# Compute the convex hull of the lifted weighted points
	hull = ConvexHull(S_lifted)
	
	# Extract the Delaunay triangulation from the lower hull
	tri_list = tuple([a, b, c] if is_ccw_triangle(S[a], S[b], S[c]) else [a, c, b]  for (a, b, c), eq in zip(hull.simplices, hull.equations) if eq[2] <= 0)
	
	# Compute the Voronoi points
	V = numpy.array([get_power_circumcenter(*S_lifted[tri]) for tri in tri_list])

	# Job done
	return tri_list, V


# --- Compute Voronoi cells ---------------------------------------------------

'''
Compute the segments and half-lines that delimits each Voronoi cell
  * The segments are oriented so that they are in CCW order
  * Each cell is a list of (i, j), (A, U, tmin, tmax) where
     * i, j are the indices of two ends of the segment. Segments end points are
       the circumcenters. If i or j is set to None, then it's an infinite end
     * A is the origin of the segment
     * U is the direction of the segment, as a unit vector
     * tmin is the parameter for the left end of the segment. Can be -1, for minus infinity
     * tmax is the parameter for the right end of the segment. Can be -1, for infinity
     * Therefore, the endpoints are [A + tmin * U, A + tmax * U]
'''

def get_voronoi_cells(S, V, tri_list):
    # Keep track of which circles are included in the triangulation
    vertices_set = frozenset(itertools.chain(*tri_list))
    #print('vertices_set:', vertices_set)

    # Keep track of which edge separate which triangles
    edge_map = { }
    for i, tri in enumerate(tri_list):
        for edge in itertools.combinations(tri, 2):
            edge = tuple(sorted(edge))
            if edge in edge_map:
                edge_map[edge].append(i)
            else:
                edge_map[edge] = [i]
    #print('edge_map:', edge_map)

    # For each triangle
    voronoi_cell_map = { i : [] for i in vertices_set }
    #print('voronoi_cell_map before big loop:', voronoi_cell_map)

    for i, (a, b, c) in enumerate(tri_list):
        # For each edge of the triangle
        #print('i:', i)
        #print('(a, b, c):', (a, b, c))
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
        # Finite Voronoi edge
            #print('u:', u) 
            #print('v:', v) 
            #print('w:', w)
            edge = tuple(sorted((u, v)))
            #print('edge:', edge)
            if len(edge_map[edge]) == 2:
                j, k = edge_map[edge]
                if k == i:
                    j, k = k, j
                
                # Compute the segment parameters
                U = V[k] - V[j]
                U_norm = norm2(U)

                # Add the segment
                voronoi_cell_map[u].append(((j, k), (V[j], U / U_norm, 0, U_norm)))
                #print('voronoi_cell_map[u]:', voronoi_cell_map[u])
            else: 
            # Infinite Voronoi edge
                # Compute the segment parameters
                A, B, C, D = S[u], S[v], S[w], V[i]
                U = normalized(B - A)
                I = A + numpy.dot(D - A, U) * U
                W = normalized(I - D)
                if numpy.dot(W, I - C) < 0:
                    W = -W
            
                # Add the segment
                voronoi_cell_map[u].append(((edge_map[edge][0], -1), (D,  W, 0, None)))
                voronoi_cell_map[v].append(((-1, edge_map[edge][0]), (D, -W, None, 0)))

    #print('voronoi_cell_map after big loop:', voronoi_cell_map)
    
    # Order the segments
    def order_segment_list(segment_list):
        # Pick the first element
        first = min((seg[0][0], i) for i, seg in enumerate(segment_list))[1]

        # In-place ordering
        segment_list[0], segment_list[first] = segment_list[first], segment_list[0]
        for i in range(len(segment_list) - 1):
            for j in range(i + 1, len(segment_list)):
                if segment_list[i][0][1] == segment_list[j][0][0]:
                    segment_list[i+1], segment_list[j] = segment_list[j], segment_list[i+1]
                    break

        # Job done
        return segment_list

    # Job done
    return { i : order_segment_list(segment_list) for i, segment_list in voronoi_cell_map.items() }

def convert_tri_list_to_edges(tri_list):
    edges = set()
    for l in tri_list:
        edges.add( frozenset({l[0], l[1]}) )
        edges.add( frozenset({l[1], l[2]}) )
        edges.add( frozenset({l[2], l[0]}) )
    return edges


def get_edge_map(tri_list):    
    edge_map = { }
    for i, tri in enumerate(tri_list):
        for edge in itertools.combinations(tri, 2):
            #print(edge)
            edge = tuple(sorted(edge))
            if edge in edge_map:
                edge_map[edge].append(i)
            else:
                edge_map[edge] = [i]  
    return edge_map


def convert_tri_list_to_edges_real(tri_list, V, edge_map, voronoi_cell_map):
    tri_list_real = []
    V_real = []
    edges_real = set()
    number_of_delaunay_edges_corresponding_to_halflines = 0
    for i,triangle in enumerate(tri_list):
        if np.linalg.norm(V[i]) < 1:
            edges_real.add( frozenset({triangle[0], triangle[1]}) )
            edges_real.add( frozenset({triangle[1], triangle[2]}) )
            edges_real.add( frozenset({triangle[2], triangle[0]}) )            
        else:
            triangle_edges = [tuple(sorted((triangle[0], triangle[1]))), tuple(sorted((triangle[1], triangle[2]))), tuple(sorted((triangle[2], triangle[0]))) ]
            for triangle_edge in triangle_edges:
                #print('triangle_edge:', triangle_edge)
                associated_triangles = edge_map[tuple(triangle_edge)] # all the triangles from which triangle_edge is an edge, should be 2 triangles in general, but 1 triangle for some cases   
                #print('associated_triangles:', associated_triangles)
                if len(associated_triangles)==2:
                    if min(np.linalg.norm(V[associated_triangles[0]]), np.linalg.norm(V[associated_triangles[1]]))<1:    
                        #print('One Voronoi point is outside and the other inside the unit disk.')
                        edges_real.add( frozenset({triangle_edge[0], triangle_edge[1]}) )
                    else: #that is: the two associated voronoi points (end points of the voronoi segment which is the dual of the delaunay edge) are outside the unit disk, then if the line intersect the unit disk and the intersections belong to the voronoi segment then the corresponding delaunay edge should be kept
                        #print('The two Voronoi points are outside the unit disk.')
                        m = ( V[associated_triangles[1]][1]-V[associated_triangles[0]][1] ) / ( V[associated_triangles[1]][0]-V[associated_triangles[0]][0] ) #coeff directeur de la droite passant par les voronoi points                    
                        p = V[associated_triangles[0]][1] - m*V[associated_triangles[0]][0] #ordonnée à l'origine
                        delta = 4*(m**2-p**2+1)
                        if delta>0: #that is if the line interects the cirlce at two places
                            #print('The line passing through the two Voronoi points is intersecting the disk.')
                            a = (1+m**2)
                            b = 2*m*p
                            c = p**2-1 
                            sol1_x = (-b-np.sqrt(delta))/(2*a)
                            sol2_x = (-b+np.sqrt(delta))/(2*a)
                            sol1_y = m*sol1_x+p
                            sol2_y = m*sol2_x+p
                            min_voronoi_x = min(V[associated_triangles[0]][0], V[associated_triangles[1]][0])
                            max_voronoi_x = max(V[associated_triangles[0]][0], V[associated_triangles[1]][0])       
                            min_voronoi_y = min(V[associated_triangles[0]][1], V[associated_triangles[1]][1]) 
                            max_voronoi_y = max(V[associated_triangles[0]][1], V[associated_triangles[1]][1])
                            if ( min_voronoi_x<=min(sol1_x,sol2_x) ) and ( max_voronoi_x>=max(sol1_x,sol2_x) ) and ( min_voronoi_y<=min(sol1_y,sol2_y) ) and ( max_voronoi_y>=max(sol1_y,sol2_y) ):
                                #print('The two intersections points belong to the Voronoi segment.')
                                edges_real.add( frozenset({triangle_edge[0], triangle_edge[1]}) )
                            #edges_real.add( frozenset({triangle_edge[0], triangle_edge[1]}) )
                        #else:
                            #print('The line passing through the two Voronoi points is not intersecting the disk, or is tangent to it.')
                elif len(associated_triangles)==1: #i.e. delaunay edge is the edge of only one triangle, i.e. the corresponding voronoi edge is a half-line starting from one voronoi point
                    #print('This is a half-line')
                    #if half-line intersects the unit circle then we keep it
#                     print('triangle_edge:', triangle_edge)
#                     print('i:', i)
#                     print('V[i]:', V[i])
#                     print('voronoi_cell_map[triangle_edge[0]]:', voronoi_cell_map[triangle_edge[0]])
#                     print('voronoi_cell_map[triangle_edge[1]]:', voronoi_cell_map[triangle_edge[1]])
                    for triangle_edge_index in [0,1]:
                        for border in voronoi_cell_map[triangle_edge[triangle_edge_index]]:
                            #print('border', border)
                            if border[0]==(i, -1): #half-line border from the i-th Voronoi point to infinite
#                                 print('The border (i, -1) is:', border)
#                                 print('V[i]:', V[i])
                                U = border[1][1] #unit vector direction of the half-line from Voronoi point to infinite
                                radial = -border[1][0] #border[1][0] should be equal to array(V[i])  
                                radial = radial/np.linalg.norm(radial) #unit vector direction from Voronoi point to origin of the Klein-Beltrami disk (center of the unit disk)
                                theta = np.arccos(np.dot(U,radial))
                                alpha = np.arcsin(1/np.linalg.norm(V[i])) #alpha is the limit angle for when the half-line is tangent to the unit circle
#                                 print('U', U)
#                                 print('radial', radial)
#                                 print('theta', theta)
#                                 print('alpha', alpha)
                                if theta<alpha or theta>(2*np.pi - alpha): #i.e. the half-line intersects the unit circle
                                    edges_real.add( frozenset({triangle_edge[0], triangle_edge[1]}) )
                                    number_of_delaunay_edges_corresponding_to_halflines +=1
#                                     print('The half-line intersects the unit circle.')
#                                 else:
#                                     print('The half-line does not intersect the unit circle.')                 
#     print('Number of Delaunay edges corresponding to half-lines:', number_of_delaunay_edges_corresponding_to_halflines)                                    
    return edges_real                     


def compute_Voronoi_Delaunay(z_beltrami, space = 'Euclidean'):

    S = np.array(z_beltrami)

    centers = np.zeros(S.shape)
    radii = np.zeros(S.shape[0])
    
    if space == 'Euclidean':
        for i in range(S.shape[0]):
            centers[i] = S[i]
            radii[i] = 0.0   

    elif (space == 'Klein' or space == 'Beltrami' or space == 'Cayley'):
        for i in range(S.shape[0]):
            centers[i] = S[i]/(2*np.sqrt(1-(np.linalg.norm(S[i]))**2))
            radii[i] = (np.linalg.norm(S[i]))**2/(4*(1-(np.linalg.norm(S[i]))**2))-1/np.sqrt(1-(np.linalg.norm(S[i]))**2)   

    # Compute the power triangulation of the circles
    tri_list, V = get_power_triangulation(centers, radii)
    voronoi_cell_map = get_voronoi_cells(centers, V, tri_list)
    edge_map = get_edge_map(tri_list)
    
    if space == 'Euclidean':
        edges = convert_tri_list_to_edges(tri_list)  

    elif (space == 'Klein' or space == 'Beltrami' or space == 'Cayley'):
        edges = convert_tri_list_to_edges_real(tri_list, V, edge_map, voronoi_cell_map)

    return {'original_points': S, 'Voronoi_points': V, 'Delaunay_triangulation': tri_list, 'Delaunay_edges': edges, 'map_DelaunayEdge2Triangle': edge_map, 'map_DataPoint2VoronoiCell': voronoi_cell_map}


def display(computed_Voronoi_Delaunay, margen=0, points=None, title='Voronoi diagram and Delaunay complex', space = 'Euclidean')  :
    
    S = computed_Voronoi_Delaunay['original_points'] 
    tri_list = computed_Voronoi_Delaunay['Delaunay_triangulation'] 
    voronoi_cell_map = computed_Voronoi_Delaunay['map_DataPoint2VoronoiCell']
    delaunay_edges = computed_Voronoi_Delaunay['Delaunay_edges']
    
    fig, ax = plot.subplots(figsize = (50, 50))
    plot.axis('equal')
    plot.axis('off')
    #plot.scatter(S[:, 0], S[:, 1], s=20, c='black')
    #plot.scatter(points[:, 0], points[:, 1], s=100, c=L, cmap='Spectral')

    # Set min/max display size, as Matplotlib does it wrong
    min_corner_x = numpy.amin(S, axis = 0)-margen
    max_corner_x = numpy.amax(S, axis = 0)+margen
    min_corner_y = numpy.amin(S, axis = 1)-margen
    max_corner_y = numpy.amax(S, axis = 1)+margen
    plot.xlim((min_corner_x[0], max_corner_x[0]))
    plot.ylim((min_corner_y[1], max_corner_y[1]))

    # Plot the Voronoi cells
    edge_map = { }
    for segment_list in voronoi_cell_map.values():
        for edge, (A, U, tmin, tmax) in segment_list:
            edge = tuple(sorted(edge))
            if edge not in edge_map:
                if tmax is None:
                    tmax = 10
                if tmin is None:
                    tmin = -10

                edge_map[edge] = (A + tmin * U, A + tmax * U)

    line_list = LineCollection(edge_map.values(), lw = 1., colors = 'red')
    line_list.set_zorder(0)
    ax.add_collection(line_list)
    
    line_list = LineCollection([(S[i], S[j]) for i, j in delaunay_edges], lw = 1., colors = 'green')
    line_list.set_zorder(0)
    ax.add_collection(line_list)
    
    if space != 'Euclidean':
        ax.add_artist(plot.Circle((0.0, 0.0), 1.0, fill = False, lw = 1., color = 'black'))

    plot.title(title)
    plot.show()


    