"""Implementation of data loaders."""

import os
import itertools
import numpy as np

import torch
from torch.utils.data import Dataset, default_collate

from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph


class PHDataset(Dataset):
    """Dataset for PH vectorizations."""
    def __init__(self, vec_file: str, aux_file: str):   

        assert os.path.exists(vec_file)
        assert os.path.exists(aux_file)

        obs = torch.load(vec_file).float()   
        aux = torch.load(aux_file).float()
        N,T,D = obs.shape

        self.T = T
        self.N = N
        self.D = D

        self.aux_obs = aux
        self.evd_obs = torch.load(vec_file).float()   
        self.evd_msk = torch.ones_like(self.evd_obs).long()
        self.evd_tid = torch.arange(T).view(1,self.T).repeat(self.N,1)
        
        self.inp_tid = torch.clone(self.evd_tid)
        self.inp_obs = torch.clone(self.evd_obs)
        self.inp_msk = torch.clone(self.evd_msk)

        self.indices = self.evd_tid # default
        self.is_subsampled = False
    
    def __getitem__(self, idx):
        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx],
            'inp_msk' : self.inp_msk[idx],
            'inp_tid' : self.inp_tid[idx],
            'inp_tps' : self.inp_tid[idx]/self.T,
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx],
            'aux_obs' : self.aux_obs[idx],
            'raw_tid' : self.indices[idx]
        }
        return inp_and_evd

    @property
    def num_timepts(self):
        return self.T

    @property
    def num_aux_dim(self):
        return self.aux_obs.shape[1]
    
    @property
    def num_vec_dim(self):
        return self.D 
    
    @property
    def num_samples(self):
        return self.N
    
    def get_collate(self):
        def collate(batch):
            return default_collate(batch)
        return collate
        
    def subsample(self, indices):
        if self.is_subsampled:
            return 
        assert indices.shape[0] == self.N
        self.indices = indices
        
        for i in range(self.N):
            inp_msk = torch.zeros_like(self.inp_msk[i])
            inp_obs = torch.zeros_like(self.inp_obs[i])
            inp_tid = torch.zeros_like(self.inp_tid[i])
        
            idx_set = indices[i]
            inp_msk[0:len(idx_set)] = self.inp_msk[i][idx_set]
            inp_obs[0:len(idx_set)] = self.inp_obs[i][idx_set]
            inp_tid[0:len(idx_set)] = self.inp_tid[i][idx_set]

            bool_keep = torch.zeros(self.num_timepts, dtype=torch.long)
            bool_keep[idx_set] = 1

            self.evd_obs[i][bool_keep==0]=0
            self.evd_tid[i][bool_keep==0]=0
            self.evd_msk[i][bool_keep==0]=0
            
            self.inp_msk[i] = inp_msk
            self.inp_obs[i] = inp_obs
            self.inp_tid[i] = inp_tid
            
        self.is_subsampled = True

    def __len__(self):
        return self.N
    

class PointCloudDataset(Dataset):
    """Dataset for point clouds."""
    def __init__(self, pts_file, aux_file):   
        assert os.path.exists(pts_file), 'Point file {} does not exist!'.format(pts_file)
        assert os.path.exists(aux_file), 'Auxiliary file {} does not exist'.format(aux_file)

        obs = torch.load(pts_file)
        M, T, N, D = 0, 0, 0, 0
        if isinstance(obs, torch.Tensor):
            obs = obs.float()
            N,T,M,D = obs.shape
        elif isinstance(obs, list):
            # assumes all sequences have equal length and all are D-dim. point clouds
            N = len(obs)
            T = len(obs[0])
            D = obs[0][0].shape[1]
            
        aux = torch.load(aux_file).float()

        self.M = M # UNUSED
        self.T = T # sequence length
        self.N = N # nr. of points
        self.D = D # dim. of point clouds 

        tf = KNNGraph(k=6)
        self.aux_obs = aux
        self.pts_obs = {i: [tf(Data(pos=y)) for y in list(obs[i])] for i in range(self.N)}
        self.pts_msk = torch.ones(N,T,1, dtype=torch.long)
        self.indices = torch.arange(T).view(1,self.T).repeat(self.N,1)
        self.is_subsampled = False
        
    @property
    def num_aux_dim(self):
        return self.aux_obs.shape[1]

    @property
    def num_vec_dim(self):
        return 0

    @property
    def num_samples(self):
        return self.N

    @property
    def num_timepts(self):
        return self.T

    def subsample(self, indices):
        if self.is_subsampled:
            return
        assert indices.shape[0] == self.N
        self.indices = indices
        self.pts_msk.fill_(0)
        for i in range(self.N):    
            idx_set = indices[i]
            self.pts_obs[i] = [self.pts_obs[i][t] for t in idx_set]
            self.pts_msk[i][idx_set] = 1
        self.is_subsampled = True

    def get_collate(self):
        def collate(batch):
            pts_obs_batch = [b[0] for b in batch]
            pts_aux_batch = [b[1] for b in batch]
            pts_tid_batch = [b[2] for b in batch]
            pts_msk_batch = [b[3] for b in batch]
            return {
                'pts_obs_batch': list(itertools.chain(*pts_obs_batch)),
                'pts_aux_batch': default_collate(pts_aux_batch),
                'pts_tid_batch': default_collate(pts_tid_batch),
                'pts_msk_batch': default_collate(pts_msk_batch),
                'pts_cut_batch': torch.tensor([len(b) for b in pts_obs_batch][:-1]).cumsum(0),
            }
        return collate    
    
    def __getitem__(self, idx):
        return self.pts_obs[idx], self.aux_obs[idx], self.indices[idx], self.pts_msk[idx]
    
    def __len__(self):
        return self.N


class JointDataset(Dataset):  
    """Dataset for PH vectorizations and point clouds."""  
    def __init__(self, datasets): # first dataset needs to be the TDA one
        self.datasets = datasets
        assert len(np.unique([ds.num_samples for ds in self.datasets])) == 1
        assert len(np.unique([ds.num_timepts for ds in self.datasets])) == 1
        assert len(np.unique([ds.num_aux_dim for ds in self.datasets])) == 1
                            
        self.T = self.datasets[0].num_timepts
        self.N = self.datasets[0].num_samples
    
    @property
    def num_aux_dim(self):
        return self.datasets[0].num_aux_dim
    
    @property
    def num_timepts(self):
        return self.T
    
    @property
    def num_samples(self):
        return self.N
    
    @property 
    def num_vec_dim(self):
        assert hasattr(self.datasets[0], 'num_vec_dim')
        return self.datasets[0].num_vec_dim
    
    def __getitem__(self, idx):
        return [ds[idx] for ds in self.datasets]

    def subsample(self, indices):
        [ds.subsample(indices) for ds in self.datasets]

    def get_collate(self):
        def collate(batch):
            tda_obs_batch = [b[0] for b in batch]
            pts_obs_batch = [b[1][0] for b in batch]
            pts_aux_batch = [b[1][1] for b in batch]
            pts_tid_batch = [b[1][2] for b in batch]
            pts_msk_batch = [b[1][3] for b in batch]
            return {
                'tda_obs_batch': default_collate(tda_obs_batch),
                'pts_obs_batch': list(itertools.chain(*pts_obs_batch)),
                'pts_aux_batch': default_collate(pts_aux_batch),
                'pts_tid_batch': default_collate(pts_tid_batch),
                'pts_cut_batch': torch.tensor([len(b) for b in pts_obs_batch][:-1]).cumsum(0),
                'pts_msk_batch': default_collate(pts_msk_batch)
            }
        return collate
    
    def __len__(self):
        return len(self.datasets[0])


def create_sampling_indices(num_samples, num_timepts, N):
    return torch.stack([torch.randperm(num_timepts)[0:N].sort().values for _ in range(num_samples)])