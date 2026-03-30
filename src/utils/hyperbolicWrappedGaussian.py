# +
# import libraries
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Modified from: https://github.com/drewwilimitis/hyperbolic-learning/blob/master/ 


# convert array to klein/poincare disk
def hyperboloid_pts_to_disc(X, eps=1e-6, metric='lorentz', model="Klein"):
    model_pts = np.zeros((X.shape[0], X.shape[1]-1))
    poincare = float(model=="Poincare")
    if metric == 'minkowski':
        model_pts[:, 0] = X[:, 1] / ((X[:, 0] + poincare) + eps)
        model_pts[:, 1] = X[:, 2] / ((X[:, 0] + poincare) + eps)
    else:
        model_pts[:, 0] = X[:, 0] / ((X[:, 2] + poincare) + eps)
        model_pts[:, 1] = X[:, 1] / ((X[:, 2] + poincare) + eps)
    return model_pts

# convert single point to klein/poincare 
def hyperboloid_pt_to_disc(x, eps=1e-6, metric='lorentz', model="Klein"):
    model_pt = np.zeros((2, ))
    poincare = float(model=="Poincare")
    if metric == 'minkowski':
        model_pt[0] = x[1] / ((x[0] + poincare) + eps)
        model_pt[1] = x[2] / ((x[0] + poincare) + eps)
    else:
        model_pt[0] = x[0] / ((x[2] + poincare) + eps)
        model_pt[1] = x[1] / ((x[2] + poincare) + eps)
    return proj(model_pt)

# convert single point to hyperboloid
def poincare_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    mink_pt = np.zeros((3, ))
    r = norm(y)
    denominator = 1 - r**2 + eps
    if metric == 'minkowski':
        mink_pt[0] = 2/denominator * (1 + r**2)/2
        mink_pt[1] = 2/denominator * y[0]
        mink_pt[2] = 2/denominator * y[1]
    else:
        mink_pt[0] = 2/denominator * y[0]
        mink_pt[1] = 2/denominator * y[1]
        mink_pt[2] = 2/denominator * (1 + r**2)/2
    return mink_pt

# convert single point to hyperboloid
def klein_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    mink_pt = np.zeros((3, ))
    denominator = np.sqrt(np.max([0.0, 1 - np.dot(y, y)])) + eps
    if metric == 'minkowski':
        mink_pt[0] = 1/denominator 
        mink_pt[1] = y[0]/denominator  
        mink_pt[2] = y[1]/denominator 
    else:
        mink_pt[0] = y[0]/denominator
        mink_pt[1] = y[1]/denominator
        mink_pt[2] = 1/denominator 
    return mink_pt

def disc_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz', model="Klein"):
    result = None
    if model=="Klein":
        result = klein_pt_to_hyperboloid(y, eps=eps, metric=metric)
    elif model=="Poincare":
        result = poincare_pt_to_hyperboloid(y, eps=eps, metric=metric)
    else:
        raise ValueError("model should be either 'Klein' or 'Poincare'")
    
    return result

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 

# project within disk
def proj(theta,eps=0.1):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

#------------------------------------------------------------
#----- Wrapped Normal Distribution in Hyperboloid Model -----
#------------------------------------------------------------

# first get sample from standard multivariate gaussian 
def init_sample(dim=2, variance=None):
    """Sample v from normal distribution in R^n+1 with N(0, sigma)"""
    mean = np.zeros((dim))
    if variance is None:
        variance = np.eye(dim)
    v = np.random.multivariate_normal(mean, variance)
    tangent_0 = np.insert(v, 0, 0)
    return tangent_0

# define alternate minkowski/hyperboloid bilinear form
def lorentz_product(u, v):
    """Compute lorentz product with alternate minkowski/hyperboloid bilinear form"""
    return -minkowski_dot(u, v)

def lorentz_norm(u, eps=1e-5):
    """Compute norm in hyperboloid using lorentz product"""
    return np.sqrt(np.max([lorentz_product(u,u), eps]))

def parallel_transport(transport_vec, target_vec, base_vec, eps=1e-5):
    """Mapping between tangent spaces, transports vector along geodesic from v to u""" 
    alpha = -lorentz_product(base_vec, target_vec)
    frac = lorentz_product(target_vec - alpha*base_vec, transport_vec) / (alpha+1+eps)
    return transport_vec + frac*(base_vec + target_vec)

def exponential_map(u, mu):
    """Given v in tangent space of u, we project v onto the hyperboloid surface""" 
    first = np.cosh(lorentz_norm(u)) * mu 
    last = np.sinh(lorentz_norm(u)) * (u / lorentz_norm(u))
    return first + last

def logarithm_map(z, mu, eps=1e-5):
    """Given z in hyperboloid, we project z onto the tangent space at mu""" 
    alpha = -lorentz_product(mu, z)
    numer = np.arccosh(alpha) * (z - alpha*mu) 
    denom = np.sqrt(max(alpha**2 - 1, eps))
    return numer / denom

def hyperbolic_sampling(n_samples, mean, sigma, dim=2, model=None):
    """Generate n_samples from the wrapped normal distribution in hyperbolic space"""
    data = []
    mu_0 = np.insert(np.zeros((dim)), 0, 1) 
    for i in range(n_samples):
        init_v = init_sample(dim=dim, variance=sigma)
        tangent_u = parallel_transport(base_vec=mu_0, target_vec=mean, transport_vec=init_v)
        data.append(exponential_map(tangent_u, mean))
    data = np.array(data)
    if model is not None:
        return hyperboloid_pts_to_disc(data, metric='minkowski', model=model)
    else:
        return data

def log_pdf(z, mu, sigma):
    """Given sample z and parameters mu, sigma calculate log of p.d.f(z)""" 
    n = len(z) - 1
    mu_0 = np.insert(np.zeros((n)), 0, 1)
    u = logarithm_map(z, mu)
    v = parallel_transport(transport_vec=u, target_vec=mu_0, base_vec=mu)
    r = lorentz_norm(u)
    det_proj = (np.sinh(r) / r)**(n-1)
    pv = multivariate_normal.pdf(v[1:], mean=np.zeros((n)), cov=sigma)
    return np.log10(pv) - np.log10(det_proj)
