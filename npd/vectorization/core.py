import copy
import torch
import numpy as np
from itertools import chain

from halo import Halo

from torchph.nn.slayer import LinearRationalStretchedBirthLifeTimeCoordinateTransform
from torchph.nn import SLayerExponential

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def compute_SLayerExponential_parameters(
    dgms: dict[int, list[list]], 
    num_elements:int=20, 
    num_subsample:int=50000,
    return_points=False): 
    """Compute initializations for exponential structure element following 
    [Section 2, Royer19a]
    
    [Royer19a]
    Royer et al.
    ATOL: Measure Vectorization for Automatic Topologically-Oriented Learning
    see https://arxiv.org/abs/1909.13472    
    
    
    Parameters:
    -----------
        dgms: dict[int, list[list]]
            Dictionary of persistence diagrams for each dimension
            
            Each value is a list (of length M) of lists of persistence diagrams 
            (of length T), where M denotes the number of simulations and T denotes
            the number of persistence diagrams per simulation.

        num_elements: int
            Number of structure elements to use (controls dimensionality)
        num_subsample: int
            Number of samples to use for (internal) k-means clustering 
            (needs to be smaller than the total number of persistence diagrams 
            available per dimension, i.e., num_subsample <= M*T).
        return_points: bool
            If True, return ALL points used for clustering.
            
    Returns: 
    --------
        ci: torch.Tensor
            Centers for exponential structure elements, i.e., a tensor 
            of shape (num_elements, 2)
        si: torch.Tensor
            Scaling factors for exponential structure elements, i.e., a tensor
            of shape (num_elements, 1)
    """
    
    # collect all PDs and transform PDs to (birth,persistence) coordinates
    T = list(chain.from_iterable(dgms))
    for k,t in enumerate(T):
        if t.size == 0:
            T[k] = np.zeros((1,2), dtype=np.float32)
        else:
            t[:,1] = t[:,1]-t[:,0]

    # draw random sample from all PDs (of size num_subsample)
    selection = torch.randperm(len(T))[0:num_subsample]
    T = [torch.tensor(T[j])[:-1] for j in selection]
    
    # run k-means clustering
    km = KMeans(n_clusters=num_elements, init='k-means++')
    X = torch.cat(T)
    km.fit(X)
    
    ci = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(ci)
    dists,_ = neigh.kneighbors(ci, 2, return_distance=True)
    si = torch.tensor(dists[:,-1]/2.0, dtype=torch.float32).view(num_elements,1).repeat(1,2)   
    
    if return_points:
        return ci, si, X
    return ci, si


def compute_SLayerExponential_vectorization(
        dgms: dict[int, list[list]], 
        ci:torch.Tensor, 
        si:torch.Tensor, 
        nu:float=0.005):
    """
    Parameters:
    -----------
        dgms: dict[int, list[list]]
            Dictionary of persistence diagrams for each dimension
            
            Each value is a list (of length M) of lists of persistence diagrams 
            (of length T), where M denotes the number of simulations and T denotes
            the number of persistence diagrams per simulation.
        ci: torch.Tensor    
            Centers of the exponential structure elements, i.e., a tensor of shape
            (num_elements, 2)
        si: torch.Tensor
            Scaling factors for exponential structure elements, i.e., a tensor
            of shape (num_elements, 1)            
        nu: float
            Stretching parameter
    
    Returns:
    --------
        vecs: torch.Tensor
            Vectorized persistence diagrams, i.e., tensor of shape (M, T, num_elements)
            where M denotes the number of simulations and T denotes the number of 
            persistence diagrams per simulation.
    """
    
    num_elements = ci.shape[0]
    tf = LinearRationalStretchedBirthLifeTimeCoordinateTransform(nu)
    sl = SLayerExponential(
        num_elements, 
        2, 
        centers_init=ci, 
        sharpness_init=1./si)
    
    vecs = []
    for i in range(len(dgms)):
        inp = [tf(torch.tensor(a)) for a in dgms[i]]
        with torch.no_grad():
            out = sl(inp)
            vecs.append(out.detach().unsqueeze(0))
    vecs = torch.cat(vecs)

    return vecs   