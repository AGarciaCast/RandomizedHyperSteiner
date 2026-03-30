import numpy as np
EPS=np.finfo(float).eps


def l2Distance(p, q):
    return ((p[0]-q[0])**2+(p[1]-q[1])**2)**0.5

def vert2sides(A, B, C):
    return (l2Distance(B,C), l2Distance(C,A), l2Distance(A,B))

def sides2angle(a, b, c):
    # Given the sides, returns the angle
    return np.arccos((a*a+c*c-b*b)/(2*a*c))

def sides2secant(a, b, c):
    # Given the sides, returns secant of that angle
    return 1/np.cos(sides2angle(b,a,c)-np.pi/6)


def in_hull4(sharedPoints, distPoints):
    result = True
    i = 0
    while result and i < 2:
        anchor = sharedPoints[i]
        angle = 0
        sharedVect = sharedPoints[(i+1)%2] - anchor
        for p in distPoints:
            side = p - anchor
            angle += np.arccos((sharedVect[0]*side[0] + sharedVect[1]*side[1])/(np.sqrt((sharedVect[0]**2 + sharedVect[1]**2)*( side[0]**2 + side[1]**2)))+EPS) 
        
        result = angle < np.pi
        i += 1
    
    return result


# +
def trilinear2cartesian(A, B, C, x, y, z, sides=None):
    """
    Given the sides and the Trilinear co-ordinates, returns the Cartesian co-ordinates
    
    A, B, C: Cartesian coordinates of triangle
    P has trilinear coordinates x : y : z, 
    sides: (a, b, c) where a is length(B, C)
    """
    if sides is None or type(sides) is not tuple or len(sides)!=3:
        sides = vert2sides(A, B, C)
        
    (a, b, c) = sides
    #check if int or float
    return (a*x*A + b*y*B + c*z*C)/(a*x + b*y + c*z)  

def euclideanInnerAngleTriangle(u, v, w):    
    sidesTriang = vert2sides(u, v, w)
    result = sides2angle(sidesTriang[0], sidesTriang[1], sidesTriang[2])
    return result


# -

def steinerPoint3Euc(vert, dist2Points=1e-1):
    # Based on the characterization of the Fermat point as the first isogonic center (in Euclidean plane)
    
    p1 = vert[0]
    p2 = vert[1]
    p3 = vert[2]
        
    b = lambda A,B,C,p,q,r: [(p*A[i]+q*B[i]+r*C[i])/(p+q+r) for i in [0,1]] 

    sidesTriang = vert2sides(p1, p2, p3)
    if np.all([sides2angle(sidesTriang[i], sidesTriang[(i+1)%3], sidesTriang[(i+2)%3])*180/np.pi < 119 for i in range(3)]):
        
        # trilinear coord of first isogonic center 
        x13 = [sides2secant(sidesTriang[i], sidesTriang[(i+1)%3], sidesTriang[(i+2)%3]) for i in range(3)]
        
        steinerP = trilinear2cartesian(p1, p2, p3, x13[0], x13[1], x13[2], sidesTriang)
        if (l2Distance(steinerP, p1)<dist2Points or 
            l2Distance(steinerP, p2)<dist2Points or 
            l2Distance(steinerP, p3)<dist2Points):
            steinerP = None
            
        return steinerP
            
    
    else:
        return None


def is_point_in_same_half_plane(point1, point2, point3, point4):
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

    cross_product_result_point3 = cross_product(point1, point2, point3)
    cross_product_result_point4 = cross_product(point1, point2, point4)

    # Check if both points are in the same half-plane
    return (cross_product_result_point3 >= 0 and cross_product_result_point4 >= 0) or \
           (cross_product_result_point3 < 0 and cross_product_result_point4 < 0)

def equilateral_triangle(point1, point2, point3=None):

    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])

    angle = 60 * np.pi / 180
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    vertex3_1 = point1 + np.dot(rotation_matrix, vector)
    
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    vertex3_2 = point1 + np.dot(rotation_matrix, vector)

    if point3 is None:
        return vertex3_1, vertex3_2
    elif is_point_in_same_half_plane(point1, point2, point3, vertex3_1):
        return vertex3_2
    else:
        return vertex3_1


def steinerPoints4Euc(vert, topo, nIters=100, convDiff=1e-2, dist2Points=1e-1):
    # Based on Smith’s Numerical Method
    
    p1 = vert[topo[0][0]]
    p2 = vert[topo[0][1]]
    p3 = vert[topo[1][0]]
    p4 = vert[topo[1][1]]
    
    
    d1, d2 = equilateral_triangle(p1, p2)
    
    beta = steinerPoint3Euc([d1, p3, p4], dist2Points=dist2Points)
    
    if beta is None:
        beta = steinerPoint3Euc([d2, p3, p4], dist2Points=dist2Points)
    
    if beta is None:
        return None
    
    alpha = steinerPoint3Euc([p1, p2, beta], dist2Points=dist2Points)
    
    if alpha is None:
        return None
    
    return alpha, beta
    
    """
    points = [[vert[topo[i][j]] for j in range(2)] for i in range(2)]
    
    
    steinerP = np.vstack([(vert[0] + vert[1] + vert[2]) / 3,
               (vert[1] + vert[2] + vert[3]) / 3])

    #steinerP = np.array([[3, 2], [4, 2]])

    for i in range(nIters):

        steinerP_old = np.copy(steinerP)
        mat = np.eye(4)
        val = np.ones(4)

        for k in range(2):

            auxDist = [1.0/(l2Distance(steinerP[0], steinerP[1])+ EPS)]
            auxDist = [1.0/(l2Distance(points[k][j], steinerP[(k+1)%2])+EPS) for j in range(2)] + auxDist

            deter = np.sum(auxDist) + EPS
            auxVal = -auxDist[-1]/deter

            for j in range(2):
                pos = k*2 + j
                val[pos] = (points[k][0][j]*auxDist[0] + points[k][1][j]*auxDist[1])/deter
                mat[pos][(pos + 2)%4] = auxVal

        steinerP = np.linalg.solve(mat, val).reshape(2,2)

        if (convDiff is not None and l2Distance(steinerP_old[0], steinerP[0])<convDiff and
            l2Distance(steinerP_old[1], steinerP[1])<convDiff):

            break          


    if (l2Distance(steinerP[0], vert[0])<dist2Points or 
            l2Distance(steinerP[0], vert[1])<dist2Points or 
            l2Distance(steinerP[0], vert[2])<dist2Points or 
            l2Distance(steinerP[0], vert[3])<dist2Points or 
            l2Distance(steinerP[1], vert[0])<dist2Points or 
            l2Distance(steinerP[1], vert[1])<dist2Points or 
            l2Distance(steinerP[1], vert[2])<dist2Points or 
            l2Distance(steinerP[1], vert[3])<dist2Points):
            return None

    
    return steinerP
    """

