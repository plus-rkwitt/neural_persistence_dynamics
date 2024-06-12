"""Code to pre-compute persistence diagrams from point clouds."""

import os
import glob
import h5py
import torch
from halo import Halo
import numpy as np
import argparse
import ripserplusplus as rpp_py
from collections import defaultdict


def setup_cmdline_parsing():
    generic_parser = argparse.ArgumentParser()
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument("--use-matlab-loader", action='store_true', default=False)  
    group0.add_argument("--simulation-dir", type=str, default=None)
    group0.add_argument("--simu-inp-file", type=str, default="simu_inp.pt")
    group0.add_argument("--prms-inp-file", type=str, default="prms_inp.pt")
    group0.add_argument("--simu-out-file", type=str, default="simu_out.pt")
    group0.add_argument("--dgms-out-file", type=str, default="dgms_out.pt")
    group0.add_argument("--prms-out-file", type=str, default="prms_out.pt")
    
    group1 = generic_parser.add_argument_group('PH computation arguments')
    group1.add_argument("--start", type=int, default=-1)
    group1.add_argument("--stop", type=int, default=-1)
    group1.add_argument("--compute-ph", action='store_true', default=False)  
    group1.add_argument("--max-dim", type=int, default=1)
    return generic_parser


def compute_rips_ph(args, PC_ten=None):
    dgms = defaultdict(list)
    
    N,T = 0,0
    if isinstance(PC_ten, torch.Tensor):
        N,T,_,_ = PC_ten.shape
    elif isinstance(PC_ten, list):
        N = len(PC_ten)
        T = len(PC_ten[0]) # assuming equal length!!!
    else:
        raise NotImplementedError()
    
    spinner = Halo(spinner='dots')
    for j in range(N): # nr. of simulations
        spinner.start('Computing Rips PH on simulation {}'.format(j))
        simu_dgms = defaultdict(list)
        for t in range(T): # nr. of time points
            data = None
            if isinstance(PC_ten, torch.Tensor):
                data = PC_ten[j,t].numpy()
            else:
                data = PC_ten[j][t].numpy()
            res = rpp_py.run("--format point-cloud --dim {}".format(args.max_dim), data)            
            for dim in range(args.max_dim+1):
                simu_dgms[dim].append(np.array([[u[0],u[1]] for u in res[dim]]))
    
        for dim in range(args.max_dim+1):
            dgms[dim].append(simu_dgms[dim])
        spinner.stop()
        
    return dgms    


def collect_from_matlab(args):
    CL_mat = [] # C,L parameters 
    PC_mat = [] # Point clouds from simulation
    
    CL_files = glob.glob(os.path.join(args.simulation_dir, 'CL*.mat'))
    
    for j, cl_file in enumerate(CL_files):
        base = os.path.basename(cl_file)
        ext = base.split('.')[-1]
        parts = base.split('.')[0].split('_')
        id = parts[-1]
        pc_file = os.path.join(
            args.simulation_dir, 
            'swarm3d_data_' + id + "." + ext)
        
        print('{} <-> {}'.format(
            cl_file,
            pc_file))
        
        cl_fid = h5py.File(cl_file, 'r') 
        CL_mat.append(np.array(cl_fid['CL']).T)

        pt_fid = h5py.File(pc_file)
        group = pt_fid['PP']
        for g in group:
            PC_mat.append(torch.tensor(np.array(pt_fid[g])))

    PC_ten = torch.stack(PC_mat)
    CL_ten = torch.vstack([torch.tensor(CL_mat[i]) for i in range(len(CL_mat))])
    return PC_ten, CL_ten
    

def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)
    
    if args.use_matlab_loader:
        # implements loading of point clouds from Matlab files (produced by [Giusti23a] Julia code)
        pts, aux = collect_from_matlab(args)    
        print('Saving simulation point clouds to {}'.format(args.simu_out_file))
        torch.save(pts, args.simu_out_file)
        print('Saving simulation parameters to {}'.format(args.prms_out_file))
        torch.save(aux, args.prms_out_file) 
    else:
        pts = torch.load(args.simu_inp_file)
        aux = torch.load(args.prms_inp_file)        
    
    
    if args.compute_ph:
        if args.start >= 0 and args.stop > 0:
            assert args.stop > args.start, 'Invalid index [start,stop]'
            dgms = compute_rips_ph(args, pts[args.start:args.stop])
        else:
            dgms = compute_rips_ph(args, pts)
    
        print('Saving diagrams to {}'.format(args.dgms_out_file))
        torch.save(dgms, args.dgms_out_file)


if __name__ == "__main__":
    main()






