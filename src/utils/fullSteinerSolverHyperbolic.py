
# +
import numpy as np
from scipy.optimize import fsolve
import random
import time
from sympy import symbols

try:
    from phcpy.solver import solve
    from phcpy.solver import standard_scale_system as scalesys
    from phcpy.solver import standard_scale_solutions as scalesols
    from phcpy.solutions import is_real, strsol2dict
    ALLOW_PRECISE = True
    
except ModuleNotFoundError:
    ALLOW_PRECISE = False
    
import matplotlib.pyplot as plt
EPS=np.finfo(float).eps



# +
def relu(x):
    return np.maximum(0, x)

def lorentzProduct(p, q):
    return (-1 + p[0]*q[0] + p[1]*q[1])

def lorentzBilinear(p, q):
    return lorentzProduct(p, q)/(np.sqrt(relu(lorentzProduct(p, p)*lorentzProduct(q, q))) + EPS)


# +

def kleinDistance(p, q):
    dist = np.arccosh(np.max([1.0, -lorentzBilinear(p, q)]))
    if np.isnan(dist):
        return EPS
    else:
        return dist



# -

def halfDistance(p, q):
    
    distCongPQ = np.sqrt(relu((q[0] - p[0])**2 + (q[1] + p[1])**2))
    distPQ = np.sqrt(relu((q[0] - p[0])**2 + (q[1] - p[1])**2))
    
    dist =  np.abs(np.log(distCongPQ + distPQ) - np.log(distCongPQ - distPQ))
    if np.isnan(dist):
        return EPS
    else:
        return dist

DISTANCE_HYP = {
    "Half": halfDistance,
    "Klein": kleinDistance
}


def coshHalfDist(p, q):
    return ((p[0] - q[0])**2 + p[1]**2 + q[1]**2)/(2*p[1]*q[1] + EPS)


def sinhHalfDist(p, q):
    distCongPQ = np.sqrt(relu((q[0] - p[0])**2 + (q[1] + p[1])**2))
    distPQ = np.sqrt(relu((q[0] - p[0])**2 + (q[1] - p[1])**2))
    
    return (distCongPQ*distPQ)/(2*p[1]*q[1] + EPS)


def hyperbolicInnerAngleTriangle(u, v, w, model="Klein"):
    # angle(uvw) using hyperbolic cosine rule
    if model == "Klein":   
        a = -lorentzBilinear(w, v)
        b = -lorentzBilinear(u, v)
        c = -lorentzBilinear(u, w) 
        
        aux = (a*b - c)/(np.sqrt(relu((a**2 - 1)*(b**2 -1))) + EPS)
    
    elif model == "Half":
        aux = (coshHalfDist(v, u)*coshHalfDist(v, w) - coshHalfDist(u, w))/(sinhHalfDist(v, u)*sinhHalfDist(v, w) + EPS)
        pass 
    
    else:
        raise ValueError("model should be either 'Klein' or 'Half'")
    
    
    return np.arccos(np.clip(aux, -1, 1))


def isopticCurve(p, q, s, model="Klein"):
    a, b = p
    c, d = q
    x, y = s
    
    if model == "Klein":
        
        result = -(a*y - x*b)*(c*y - x*d) + (b - y)*(d - y) + (x - a)*(x - c) +\
                        np.sqrt(relu((-(a*y - x*b)**2 + (b - y)**2 +(x - a)**2)*(-(c*y - x*d)**2 + (d - y)**2 + (x - c)**2)))/2     
        
    elif model == "Half":
        result = 2*((a -x)**2 + b**2 + y**2)*((c -x)**2 + d**2 + y**2) -4*y**2*((a - c)**2 + b**2 + d**2) +\
                        np.sqrt(relu((a -x)**2 + (b -y)**2))*np.sqrt(relu((a -x)**2 + (b + y)**2)) *\
                            np.sqrt(relu((c -x)**2 + (d -y)**2)) * np.sqrt(relu((c -x)**2 + (d + y)**2))
    
    else:
        raise ValueError("model should be either 'Klein' or 'Half'")
    
    return result



def systemIsopticCurves(p1, p2, p3, model="Klein"):
  
    def functs(params):
        return (isopticCurve(p1, p2, params, model),
                isopticCurve(p2, p3, params, model))
   
    return functs


# +

def plotIsoptics(p1, p2, p3, s, model="Klein"):
    X,Y = np.meshgrid(np.linspace(-1.15, 1.15, 500),
                          np.linspace(-1.15, 1.15, 500))
    circle = X**2 + Y**2 -1

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, aspect='equal') 
    ax.contour(X,Y, isopticCurve(p1, p2, [X, Y], model=model), 0, colors="tab:green")
    ax.contour(X,Y, isopticCurve(p2, p3, [X, Y], model=model), 0, colors="tab:red")
    ax.contour(X,Y, isopticCurve(p1, p3, [X, Y], model=model), 0, colors="tab:blue")
    ax.contour(X, Y, circle, 0, colors="grey", linestyles="dashed")
    ax.plot(s[0], s[1],  marker="o", c="k")

    plt.show()

def polIsoptic2Str(p, q, model="Klein", sym="x y", sym2 = None):
    a, b = p
    if sym2 is not None:
        c, d = symbols(sym2)
    else:
        c, d = q
    
    x, y = symbols(sym)
    if model=="Klein":
        eq = 3*a**2*c**2*y**4 - 6*a**2*c**2*y**2 + 3*a**2*c**2 - 6*a**2*c*d*x*y**3 +\
                6*a**2*c*d*x*y + 6*a**2*c*x*y**2 - 6*a**2*c*x + 3*a**2*d**2*x**2*y**2 +\
                a**2*d**2*x**2 + a**2*d**2*y**2 - a**2*d**2 - 8*a**2*d*x**2*y -\
                2*a**2*d*y**3 + 2*a**2*d*y + a**2*x**2*y**2 + 3*a**2*x**2 +\
                a**2*y**4 - a**2*y**2 - 6*a*b*c**2*x*y**3 + 6*a*b*c**2*x*y +\
                12*a*b*c*d*x**2*y**2 - 8*a*b*c*d*x**2 - 8*a*b*c*d*y**2 + 8*a*b*c*d -\
                4*a*b*c*x**2*y + 8*a*b*c*y**3 - 8*a*b*c*y - 6*a*b*d**2*x**3*y +\
                6*a*b*d**2*x*y + 8*a*b*d*x**3 - 4*a*b*d*x*y**2 - 8*a*b*d*x - 2*a*b*x**3*y -\
                2*a*b*x*y**3 + 8*a*b*x*y + 6*a*c**2*x*y**2 - 6*a*c**2*x - 4*a*c*d*x**2*y +\
                8*a*c*d*y**3 - 8*a*c*d*y - 8*a*c*x**2*y**2 + 12*a*c*x**2 - 8*a*c*y**4 +\
                8*a*c*y**2 - 2*a*d**2*x**3 - 8*a*d**2*x*y**2 + 2*a*d**2*x + 8*a*d*x**3*y +\
                8*a*d*x*y**3 + 4*a*d*x*y - 6*a*x**3 - 6*a*x*y**2 + 3*b**2*c**2*x**2*y**2 +\
                b**2*c**2*x**2 + b**2*c**2*y**2 - b**2*c**2 - 6*b**2*c*d*x**3*y +\
                6*b**2*c*d*x*y - 2*b**2*c*x**3 - 8*b**2*c*x*y**2 + 2*b**2*c*x +\
                3*b**2*d**2*x**4 - 6*b**2*d**2*x**2 + 3*b**2*d**2 + 6*b**2*d*x**2*y -\
                6*b**2*d*y + b**2*x**4 + b**2*x**2*y**2 - b**2*x**2 + 3*b**2*y**2 -\
                8*b*c**2*x**2*y - 2*b*c**2*y**3 + 2*b*c**2*y + 8*b*c*d*x**3 -\
                4*b*c*d*x*y**2 - 8*b*c*d*x + 8*b*c*x**3*y + 8*b*c*x*y**3 + 4*b*c*x*y +\
                6*b*d**2*x**2*y - 6*b*d**2*y - 8*b*d*x**4 - 8*b*d*x**2*y**2 +\
                8*b*d*x**2 + 12*b*d*y**2 - 6*b*x**2*y - 6*b*y**3 + c**2*x**2*y**2 +\
                3*c**2*x**2 + c**2*y**4 - c**2*y**2 - 2*c*d*x**3*y - 2*c*d*x*y**3 +\
                8*c*d*x*y - 6*c*x**3 - 6*c*x*y**2 + d**2*x**4 + d**2*x**2*y**2 -\
                d**2*x**2 + 3*d**2*y**2 - 6*d*x**2*y - 6*d*y**3 + 3*x**4 + 6*x**2*y**2 + 3*y**4
    
    elif model == "Half":
        eq = 3*a**4*c**4 - 12*a**4*c**3*x + 6*a**4*c**2*d**2 + 18*a**4*c**2*x**2 -\
                10*a**4*c**2*y**2 - 12*a**4*c*d**2*x - 12*a**4*c*x**3 + 20*a**4*c*x*y**2 +\
                3*a**4*d**4 + 6*a**4*d**2*x**2 - 6*a**4*d**2*y**2 + 3*a**4*x**4 - 10*a**4*x**2*y**2 +\
                3*a**4*y**4 - 12*a**3*c**4*x + 48*a**3*c**3*x**2 + 32*a**3*c**3*y**2 -\
                24*a**3*c**2*d**2*x - 72*a**3*c**2*x**3 - 56*a**3*c**2*x*y**2 + 48*a**3*c*d**2*x**2 +\
                32*a**3*c*d**2*y**2 + 48*a**3*c*x**4 + 16*a**3*c*x**2*y**2 - 32*a**3*c*y**4 -\
                12*a**3*d**4*x - 24*a**3*d**2*x**3 - 8*a**3*d**2*x*y**2 - 12*a**3*x**5 +\
                8*a**3*x**3*y**2 + 20*a**3*x*y**4 + 6*a**2*b**2*c**4 - 24*a**2*b**2*c**3*x +\
                12*a**2*b**2*c**2*d**2 + 36*a**2*b**2*c**2*x**2 - 20*a**2*b**2*c**2*y**2 -\
                24*a**2*b**2*c*d**2*x - 24*a**2*b**2*c*x**3 + 40*a**2*b**2*c*x*y**2 +\
                6*a**2*b**2*d**4 + 12*a**2*b**2*d**2*x**2 - 12*a**2*b**2*d**2*y**2 + 6*a**2*b**2*x**4 -\
                20*a**2*b**2*x**2*y**2 + 6*a**2*b**2*y**4 + 18*a**2*c**4*x**2 - 10*a**2*c**4*y**2 -\
                72*a**2*c**3*x**3 - 56*a**2*c**3*x*y**2 + 36*a**2*c**2*d**2*x**2 -\
                20*a**2*c**2*d**2*y**2 + 108*a**2*c**2*x**4 + 168*a**2*c**2*x**2*y**2 +\
                76*a**2*c**2*y**4 - 72*a**2*c*d**2*x**3 - 56*a**2*c*d**2*x*y**2 -\
                72*a**2*c*x**5 - 128*a**2*c*x**3*y**2 - 56*a**2*c*x*y**4 + 18*a**2*d**4*x**2 -\
                10*a**2*d**4*y**2 + 36*a**2*d**2*x**4 + 40*a**2*d**2*x**2*y**2 +\
                20*a**2*d**2*y**4 + 18*a**2*x**6 + 26*a**2*x**4*y**2 - 2*a**2*x**2*y**4 -\
                10*a**2*y**6 - 12*a*b**2*c**4*x + 48*a*b**2*c**3*x**2 + 32*a*b**2*c**3*y**2 -\
                24*a*b**2*c**2*d**2*x - 72*a*b**2*c**2*x**3 - 56*a*b**2*c**2*x*y**2 +\
                48*a*b**2*c*d**2*x**2 + 32*a*b**2*c*d**2*y**2 + 48*a*b**2*c*x**4 +\
                16*a*b**2*c*x**2*y**2 - 32*a*b**2*c*y**4 - 12*a*b**2*d**4*x - 24*a*b**2*d**2*x**3 -\
                8*a*b**2*d**2*x*y**2 - 12*a*b**2*x**5 + 8*a*b**2*x**3*y**2 + 20*a*b**2*x*y**4 -\
                12*a*c**4*x**3 + 20*a*c**4*x*y**2 + 48*a*c**3*x**4 + 16*a*c**3*x**2*y**2 -\
                32*a*c**3*y**4 - 24*a*c**2*d**2*x**3 + 40*a*c**2*d**2*x*y**2 - 72*a*c**2*x**5 -\
                128*a*c**2*x**3*y**2 - 56*a*c**2*x*y**4 + 48*a*c*d**2*x**4 + 16*a*c*d**2*x**2*y**2 -\
                32*a*c*d**2*y**4 + 48*a*c*x**6 + 128*a*c*x**4*y**2 + 112*a*c*x**2*y**4 + 32*a*c*y**6 -\
                12*a*d**4*x**3 + 20*a*d**4*x*y**2 - 24*a*d**2*x**5 - 32*a*d**2*x**3*y**2 - 8*a*d**2*x*y**4 -\
                12*a*x**7 - 36*a*x**5*y**2 - 36*a*x**3*y**4 - 12*a*x*y**6 + 3*b**4*c**4 -\
                12*b**4*c**3*x + 6*b**4*c**2*d**2 + 18*b**4*c**2*x**2 - 10*b**4*c**2*y**2 -\
                12*b**4*c*d**2*x - 12*b**4*c*x**3 + 20*b**4*c*x*y**2 + 3*b**4*d**4 + 6*b**4*d**2*x**2 -\
                6*b**4*d**2*y**2 + 3*b**4*x**4 - 10*b**4*x**2*y**2 + 3*b**4*y**4 + 6*b**2*c**4*x**2 -\
                6*b**2*c**4*y**2 - 24*b**2*c**3*x**3 - 8*b**2*c**3*x*y**2 + 12*b**2*c**2*d**2*x**2 -\
                12*b**2*c**2*d**2*y**2 + 36*b**2*c**2*x**4 + 40*b**2*c**2*x**2*y**2 + 20*b**2*c**2*y**4 -\
                24*b**2*c*d**2*x**3 - 8*b**2*c*d**2*x*y**2 - 24*b**2*c*x**5 - 32*b**2*c*x**3*y**2 -\
                8*b**2*c*x*y**4 + 6*b**2*d**4*x**2 - 6*b**2*d**4*y**2 + 12*b**2*d**2*x**4 +\
                8*b**2*d**2*x**2*y**2 + 12*b**2*d**2*y**4 + 6*b**2*x**6 + 6*b**2*x**4*y**2 -\
                6*b**2*x**2*y**4 - 6*b**2*y**6 + 3*c**4*x**4 - 10*c**4*x**2*y**2 + 3*c**4*y**4 -\
                12*c**3*x**5 + 8*c**3*x**3*y**2 + 20*c**3*x*y**4 + 6*c**2*d**2*x**4 -\
                20*c**2*d**2*x**2*y**2 + 6*c**2*d**2*y**4 + 18*c**2*x**6 + 26*c**2*x**4*y**2 -\
                2*c**2*x**2*y**4 - 10*c**2*y**6 - 12*c*d**2*x**5 + 8*c*d**2*x**3*y**2 + 20*c*d**2*x*y**4 -\
                12*c*x**7 - 36*c*x**5*y**2 - 36*c*x**3*y**4 - 12*c*x*y**6 + 3*d**4*x**4 -\
                10*d**4*x**2*y**2 + 3*d**4*y**4 + 6*d**2*x**6 + 6*d**2*x**4*y**2 - 6*d**2*x**2*y**4 -\
                6*d**2*y**6 + 3*x**8 + 12*x**6*y**2 + 18*x**4*y**4 + 12*x**2*y**6 + 3*y**8

    else:
        raise ValueError("model should be either 'Klein' or 'Half'")
    
        
        
    return f"{eq};"

def isSolIsoptic(p1, p2, p3, potentialSol, model="Klein", dist2Points=1e-1):
    
    return (DISTANCE_HYP[model](potentialSol, p1)>=dist2Points and
                DISTANCE_HYP[model](potentialSol, p2)>=dist2Points and
                DISTANCE_HYP[model](potentialSol, p3)>=dist2Points and
                np.all(np.isclose(np.zeros(3), [isopticCurve(p1, p2, potentialSol, model),
                                                isopticCurve(p1, p3, potentialSol, model),
                                                isopticCurve(p2, p3, potentialSol, model)])))
    


def solveSystemIsopticCurves(p1, p2, p3, model="Klein", precise=True, dist2Points=1e-1):
    
    result = None
    initialGuess = [p1, p3]
    if model == "Half":
        initialGuess+=[(p1+p2)/2, (p3+p2)/2, (p1+p3)/2]
        
    # First try fast generic non-poly solver
    for p in initialGuess:
    
        potentialSol = fsolve(systemIsopticCurves(p1, p2, p3, model), p)
        if isSolIsoptic(p1, p2, p3, potentialSol, model=model, dist2Points=dist2Points):
            result = potentialSol
            break
            
    # If not found solution then try robust polynomial homotopy-continuation solver
    if result is None and precise and ALLOW_PRECISE:
        # Create system pol equations
        systemEq = [polIsoptic2Str(p1, p2), polIsoptic2Str(p2, p3)]

        # Scale coefs for numerical stability
        systemEqNorm, scalingFactor = scalesys(systemEq)

        # Solve system of Polynomial eqs
        qsols = solve(systemEqNorm, verbose=False, tasks=1, precision="d")

        # Rescale solutions
        ssols = scalesols(len(systemEqNorm), qsols, scalingFactor)

        for sol in ssols:
            if is_real(sol, 1.0e-8):
                soldic = strsol2dict(sol)
                potentialSol = np.array([soldic["x"].real, soldic["y"].real])

                if isSolIsoptic(p1, p2, p3, potentialSol, model=model, dist2Points=dist2Points):
                    result = potentialSol
                    break

    return result


# -

def steinerPoint3Hyp(vert, model="Klein", precise=True, dist2Points=1e-1):
    existSol = True
    i=0
    while existSol and i<3:        
        angle = hyperbolicInnerAngleTriangle(vert[i],
                                             vert[(i+1)%3],
                                             vert[(i+2)%3],
                                             model=model)
        
        # 199 to remove some unnecesary edge cases
        existSol = angle*180/np.pi < 119
        i += 1
        
    if existSol:
        return solveSystemIsopticCurves(vert[0], vert[1], vert[2], model, precise=precise, dist2Points=dist2Points)
    else:
        return None


def samplePointFromIsoptic(p1, p2, p3=None, model="Klein"):
    a, b = p1
    c, d = p2
    
    def systemIsopticBisector(model="Klein"):
        def functs(params):
            x, y = params
            return (isopticCurve(p1, p2, params, model),
                    (2*x-a-c)*(a-c)+(2*y-b-d)*(b-d))
    
        return functs
    
    # coeficients of polynomial: bisector{p1,p2} intersection unit circle
    x2 = (a-c)**2 +(b-d)**2
    x1 = 2*(a*d -b*c)
    x0 = (a+c)**2/4 +(b+d)**2/4 -1
    
    isopticPoints = []
    for i in range(2):
        alpha =  (-x1 +((-1)**i)*np.sqrt(b**2 -4*x2*x0))/(2*x2 + EPS)
        x = (a+b)/2 +alpha*(d-b)
        y = (c+d)/2 + alpha*(a-c)
        isopticPoints.append(fsolve(systemIsopticBisector(model), 
                                    [x,y]))
    
    if p3 is None:
        result = isopticPoints[random.randint(0, 1)]
    else:
        result = isopticPoints[np.argmin([DISTANCE_HYP[model](p3, l) for l in isopticPoints])]
    
    
    return result

def steinerPoints4Hyp(vert, topo, model="Klein", nIters=100, convDiff=1e-2, dist2Points=1e-1, precise=True):
    p1 = vert[topo[0][0]]
    p2 = vert[topo[0][1]]
    p3 = vert[topo[1][0]]
    p4 = vert[topo[1][1]]
    
    """
    # Create system pol equations
    systemEq = [polIsoptic2Str(p1, p2, sym="x y"),
                polIsoptic2Str(p2, None, sym="x y", sym2="z w"),
                polIsoptic2Str(p3, p4, sym="z w"),
                polIsoptic2Str(p4, None, sym="z w", sym2="x y")
               ]

    # Scale coefs for numerical stability
    systemEqNorm, scalingFactor = scalesys(systemEq)

    # Solve system of Polynomial eqs
    qsols = solve(systemEqNorm, verbose=False, tasks=CORES, precision="d")

    # Rescale solutions
    ssols = scalesols(len(systemEqNorm), qsols, scalingFactor)
    result = None
    for sol in ssols:
        if is_real(sol, 1.0e-8):
            soldic = strsol2dict(sol)
            alpha = np.array([soldic["x"].real, soldic["y"].real])
            beta = np.array([soldic["z"].real, soldic["w"].real])

            if isSolIsoptic(p1, p2, beta, alpha, model) and isSolIsoptic(p3, p4, alpha, beta, model):
                result = [alpha, beta]
    
    
    return result
    """
    
    alpha = samplePointFromIsoptic(p1, p2, p3=p3, model=model)
    beta = steinerPoint3Hyp([alpha, p3, p4], model=model,
                            precise=precise, dist2Points=dist2Points)
    if beta is None:
        return None
    
    for i in range(nIters):

        alpha_old = alpha
        beta_old = beta

        alpha = steinerPoint3Hyp([beta, p1, p2], model=model,
                                  precise=precise, dist2Points=dist2Points)
        if alpha is None:
            return [alpha_old, beta]
        

        if (DISTANCE_HYP[model](alpha, p1)<dist2Points or 
            DISTANCE_HYP[model](alpha, p2)<dist2Points or 
            DISTANCE_HYP[model](alpha, p3)<dist2Points or 
            DISTANCE_HYP[model](alpha, p4)<dist2Points):

            return None
        
        
        beta = steinerPoint3Hyp([alpha, p3, p4], model=model, 
                                 precise=precise, dist2Points=dist2Points)
        if  beta is None:
            return [alpha, beta_old]
        
        
        if (DISTANCE_HYP[model](beta, p1)<dist2Points or 
            DISTANCE_HYP[model](beta, p2)<dist2Points or 
            DISTANCE_HYP[model](beta, p3)<dist2Points or 
            DISTANCE_HYP[model](beta, p4)<dist2Points):
            
            return None
                   
           
        """
        angles = np.array([hyperbolicInnerAngleTriangle(p1, alpha, beta, model=model),
                  hyperbolicInnerAngleTriangle(p2, alpha, beta, model=model),
                  hyperbolicInnerAngleTriangle(p3, beta, alpha, model=model),
                  hyperbolicInnerAngleTriangle(p4, beta, alpha, model=model)
        ])
        """
        
        if (convDiff is not None and DISTANCE_HYP[model](alpha_old, alpha)<convDiff and
            DISTANCE_HYP[model](beta_old, beta)<convDiff):
            
            break
    
    
    return [alpha, beta]

