"""Compute PH vectorizations using 

[Hofer19a]
Hofer et al.
Learning vectorizations of persistence barcodes
JMLR'19
see https://jmlr.csail.mit.edu/papers/v20/18-358.html
"""

import copy
import torch
import numpy as np
import argparse
from itertools import chain

# requires pip install halo
from halo import Halo

# requires torchph from https://github.com/c-hofer/torchph, see install instructions
from torchph.nn.slayer import LinearRationalStretchedBirthLifeTimeCoordinateTransform
from torchph.nn import SLayerExponential

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def setup_cmdline_parsing():
    generic_parser = argparse.ArgumentParser(
        description="""Compute PH vectorizations according to
        
        see 
        Hofer et al.
        "Learning Representations of Persistence Barcodes"
        JMLR'19
        """,formatter_class=argparse.RawTextHelpFormatter)
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument(
        "--dgms-inp-file", 
        metavar="<filename>",
        type=str, 
        default="dgms.pt", 
        help="Input file with persistence diagrams")
    group0.add_argument(
        "--vecs-out-base", 
        metavar="<prefix>",
        type=str, 
        default="vecs",
        help="Prefix name for output files")
    
    group1 = generic_parser.add_argument_group('Vectorization parameters')
    group1.add_argument(
        "--num-elements", 
        type=int, 
        choices=range(1, 100),
        metavar="[1-100]",
        default=20,
        help="Number of structure elements (controls dimensionality)")
    group1.add_argument(
        "--nu", 
        type=float, 
        default=0.005,
        metavar="<float>",
        help="start pulling towards 0 persistence at nu")
    
    group2 = generic_parser.add_argument_group('General parameters')
    group2.add_argument(
        "--subsample",
        metavar="<int>", 
        type=int, 
        default=50000) 
    
    return generic_parser



def compute_SLayerExponential_parameters(
    dgms: dict[int, list[list]], 
    num_elements:int=20, 
    num_subsample:int=50000,
    return_points=False): 
    """Compute initializations for exponential structure element following [Section 2, Royer19a]
    
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


def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)

    spinner = Halo(spinner='dots')

    spinner.start(f'Loading {args.dgms_inp_file}')
    raw_dgms = torch.load(args.dgms_inp_file)
    spinner.succeed()

    misc = {}
    for diag_dim,diag in raw_dgms.items():
        tmp = copy.deepcopy(diag)
        
        spinner.start(f'Compute SLayerExponential parameters for dim={diag_dim}')
        ci, si = compute_SLayerExponential_parameters(
            tmp, 
            args.num_elements, 
            args.subsample)
        spinner.succeed()
        
        spinner.start(f'Compute SLayerExponential vectorizations for dim={diag_dim}')
        vecs = compute_SLayerExponential_vectorization(
            diag, 
            ci, 
            si, 
            args.nu)
        spinner.succeed()

        misc[diag_dim] = {
            'ci': ci,
            'si': si,
            'vecs': vecs}

    vecs = torch.cat([misc[k]['vecs'] for k in misc.keys()], dim=2)
    
    misc_out_file = f'{args.vecs_out_base}_{args.num_elements}_{args.nu}_misc.pt'
    vecs_out_file = f'{args.vecs_out_base}_{args.num_elements}_{args.nu}.pt'

    spinner.start(f'Saving to {misc_out_file} and {vecs_out_file}')    
    torch.save(misc, f'{args.vecs_out_base}_{args.num_elements}_{args.nu}_misc.pt')
    torch.save(vecs, f'{args.vecs_out_base}_{args.num_elements}_{args.nu}.pt')
    spinner.succeed()
    
    
if __name__ == "__main__":
    main()





