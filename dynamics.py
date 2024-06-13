"""Neural Persistence Dynamics.

szeng, fgraf, rkwitt, muray, shuber, 2024
"""

import os
import sys
import random

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

import numpy as np
import matplotlib.pyplot as plt

from halo import Halo

from itertools import product
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from einops import reduce

# imports for latent ODE
from torchdiffeq import odeint

# imports for data loading and model components
from datasource import (
    PHDataset, 
    PointCloudDataset, 
    JointDataset, 
    create_sampling_indices)

from core import (
	SignatureHead,
	MTANHead,
	LatentStateHead,
	TDABackbone,
	PointNetBackbone,
	JointBackbone,
	VecReconNet, 
	LatentODEfunc, 
	PathToGaussianDecoder, 
	normal_kl)

from sklearn.metrics import r2_score, mean_squared_error


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# region cmdline parsing
def setup_cmdline_parsing():
    """Setup command line parsing."""
    
    description = """**Neural Persistence Dynamics**"""    
    generic_parser = argparse.ArgumentParser(description=Markdown(description, style="argparse.text"), formatter_class=RichHelpFormatter)
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument(
        "--log-out-file", 
		metavar="FILE",
        type=str, 
        default=None,
        help="Filename for output log file"
    )
    group0.add_argument(
        "--vec-inp-file", 
        metavar="FILE",
        type=str, 
        default=None,
        help="Filename for input PH vectorizations "
    )
    group0.add_argument(
        "--aux-inp-file", 
        metavar="FILE",
        type=str, 
        default=None,
        help="Filename for input regression targets (i.e., simulation parameters)"
    )
    group0.add_argument(
        "--pts-inp-file", 
        metavar="FILE",
        type=str, 
        default=None,
        help="Filename for input point cloud data"
    )
    group0.add_argument(
        "--net-out-file", 
		metavar="FILE",
        type=str, 
        default=None,
        help="Filename for output model file"
    )
    group0.add_argument(
        "--run-dir", 
		metavar="FOLDER",
        type=str, 
        default=None,
        help="Directory for tensorboard logs"
    )
    group0.add_argument(
        "--seed",
        metavar="INT",
        type=int, 
        default=42,
        help="Seed the model (default: %(default)s)"
    )
    group0.add_argument(
        "--experiment-id", 
        metavar="STR",
        type=str, 
        default="42",
        help="Experiment identifier (can be string)"
    )

    group1 = generic_parser.add_argument_group('Training arguments')
    group1.add_argument(
        "--batch-size", 
        metavar="INT",
        type=int, 
        default=64,
        help="Batch size (default: %(default)s)"
    )
    group1.add_argument(
        "--lr", 
        metavar="FLOAT",
        type=float, 
        default=1e-3,
        help="Learning rate (default: %(default)s)"
    )
    group1.add_argument(
        "--n-epochs", 
        metavar="INT",
        type=int, 
        default=210,
        help="Number of training epochs (default: %(default)s)"
	)
    group1.add_argument(
        "--restart", 
        metavar="INT",
        type=int, 
        default=30,
        help="1/2 cycle length of cyclic cosine LR annealing (default: %(default)s)"
	)
    group1.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to run the model on (default: %(default)s)"
	)
    group1.add_argument(
        "--kl-weight", 
        metavar="FLOAT",
        type=float, 
        default=1e-3,
        help="KL-div. weight (default: %(default)s)"
	)
    group1.add_argument(
        "--aux-weight", 
        metavar="FLOAT",
        type=float, 
        default=1000.0,
        help="Weighting of the regression loss term (default: %(default)s)"
	)
    group1.add_argument(
        "--weight-decay", 
        metavar="FLOAT",
        type=float, 
        default=1e-3,
        help="Weight decay (default: %(default)s)"
	)

    group2 = generic_parser.add_argument_group('Model configuration arguments')
    group2.add_argument(
        "--z-dim", 
		metavar="INT",
        type=int, 
        default=16, 
        help="Dimensionality of latent space Z (default: %(default)s)"
    )
    group2.add_argument(
        "--ode-h-dim", 
        metavar="INT",
        type=int, 
        default=30, 
        help="Hidden dimensionality of ODE func (default: %(default)s)"
    )
    group2.add_argument(
        "--mtan-h-dim",
        metavar="INT", 
        type=int, 
        default=128, 
        help="Hidden dimensionality of mTAN module (default: %(default)s)"
    )
    if "z_signature" in sys.argv:
        group2.add_argument(
			"--sig-depth",
            metavar="INT", 
			type=int, 
			default=3, 
			help="Path signature depth (default: %(default)s)"
		)
    if any([x in ['ptsdyn_only', 'joint'] for x in sys.argv]):
        group2.add_argument(
			"--pointnet-dim", 
			metavar="INT",
			type=int, 
			default=32, 
			help="PointNet++ hidden dimensionality (default: %(default)s)"
		)
    group2.add_argument(
        "--reconnet-h-dim",
        metavar="INT",
        type=int,
        default=32,
        help="Hidden dimensionality of reconstruction net (default: %(default)s)"
    )
    group2.add_argument(
        "--backbone",
        choices=[
            "topdyn_only", 
            "ptsdyn_only",
            "joint"],
        default="topdyn_only",
        help="Model variants  (default: %(default)s)"
    )
    group2.add_argument(
        "--processor", 
        choices=[
			'z_signature', 
			'z_laststate',
			'z_mtantwins',
			'z_meanstate'], 
        default="z_signature", 
        help="Method to summarize the latent trajectory (default: %(default)s)"
	)

    group3 = generic_parser.add_argument_group("Data preprocessing arguments")
    group3.add_argument(
        "--tps-frac", 
        type=float, 
        default=0.5, 
        help="Fraction of timepoints to use."
    )
    return generic_parser
# endregion


def run_epoch(
    args, 
    dl, 
    t: torch.Tensor, 
    modules, 
    optimizer, 
    aux_loss_fn=None, 
    tracker=None, 
    mode:str='train'):
	epoch_aux_loss = epoch_kld = epoch_loss = epoch_log_pxz = 0. 
	
	if mode=='train':
		modules.train()        
	else:
		modules.eval()    
	
	aux_p = [] # predictions 
	aux_t = [] # ground truth
	
	for batch in dl:
		# (1) run through recognition model
		out, evd_obs, evd_msk, aux_obs = modules['recog_net'](batch, args.device)
		
		# (2) draw z(t_0) from initial state distribution Normal(z|mu(x),sigma(x))
		qz0_mean, qz0_logvar = out[:, :args.z_dim], out[:, args.z_dim:]
		epsilon = torch.randn(qz0_mean.size()).to(args.device)
		z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
		
		# (3) integrate latent ODE forward to time points in t
		zs = odeint(modules['lnode_net'], z0, t, method="euler").permute(1, 0, 2)
		
		# (4) run through reconstruction network and parametrize p(x|z), then comp. log-likelihood
		rec = modules['recon_net'](zs)
		pxz = modules['ptogd_net'](rec.unsqueeze(0))     
		log_pxz = -pxz.log_prob(evd_obs.unsqueeze(0))
		log_pxz[evd_msk.unsqueeze(0)==0]=0
		log_pxz = log_pxz.mean(dim=0)
		log_pxz = reduce(log_pxz, "b ... -> b", "sum")
		numel = reduce(evd_msk.unsqueeze(0)[0], "b ... -> b", "sum")
		log_pxz /= numel

		# (5) compute KL divergence between approximate posterior q(z|x) and prior p(z)        
		pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(args.device)
		kld = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1).unsqueeze(0)
	
		# (6) summarize latent path and run summary through regression network
		aux_enc = modules['processor'](zs) 
		aux_out = modules['regressor'](aux_enc)
		aux_loss = aux_loss_fn(aux_out.flatten(), aux_obs.flatten())
		
		# (7) compute augmented ELBO
		loss = (log_pxz + args.kl_weight*kld).mean() + args.aux_weight*aux_loss
		
		if mode == 'train':
			optimizer.zero_grad()
			loss.backward()        
			optimizer.step()
		
		# track predictions and ground truths
		with torch.no_grad():
			aux_p.append(aux_out.detach().cpu())
			aux_t.append(aux_obs.detach().cpu())    
			
		epoch_aux_loss += aux_loss.item()
		epoch_kld += kld.mean().item()
		epoch_log_pxz += log_pxz.mean().item()
		epoch_loss += loss.item()

	if tracker is not None:
		tracker['epoch_aux_loss'].append(epoch_aux_loss/len(dl))
		tracker['epoch_kld'].append(epoch_kld/len(dl))
		tracker['epoch_log_pxz'].append(epoch_log_pxz/len(dl))
		tracker['epoch_loss'].append(epoch_loss/len(dl))
		tracker['epoch_aux_p'].append(torch.cat(aux_p))
		tracker['epoch_aux_t'].append(torch.cat(aux_t))


def create_recog_backbone(args):
	if args.backbone == 'topdyn_only': return TDABackbone(args)
	elif args.backbone == 'ptsdyn_only': return PointNetBackbone(args)
	elif args.backbone == 'joint': return JointBackbone(args)
	else: raise NotImplementedError    


def create_recon_backbone(args):
	if args.backbone == 'topdyn_only':
		return VecReconNet(
			args.z_dim, 
			args.reconnet_h_dim, 
			args.vec_inp_dim)
	elif args.backbone == 'ptsdyn_only':
		return VecReconNet(
			args.z_dim, 
			args.reconnet_h_dim, 
			args.pointnet_dim)
	elif args.backbone == 'joint':
		return VecReconNet(
			args.z_dim, 
			args.reconnet_h_dim, 
			args.pointnet_dim + args.vec_inp_dim)
	else:
		raise NotImplementedError


def load_data(args):
	if args.backbone == 'topdyn_only':
		return PHDataset(
			args.vec_inp_file, 
			args.aux_inp_file)
	elif args.backbone == 'ptsdyn_only':
		return PointCloudDataset(
			args.pts_inp_file, 
			args.aux_inp_file)
	elif args.backbone == 'joint':
		ds_topdyn = PHDataset(
			args.vec_inp_file, 
			args.aux_inp_file)    
		ds_ptsdyn = PointCloudDataset(
			args.pts_inp_file,
			args.aux_inp_file)
		return JointDataset([ds_topdyn, ds_ptsdyn])
	else:
		raise NotImplementedError()


def create_processor(args):
	if args.processor == 'z_mtantwins':        
		return MTANHead(
			mtan_input_dim = args.z_dim,
			mtan_hidden_dim = args.mtan_h_dim,
			num_timepts = args.num_timepts,
			use_atanh=False)
	elif args.processor == 'z_signature':
		return SignatureHead(
			in_channels = args.z_dim, 
			sig_depth = args.sig_depth)
	elif args.processor == 'z_laststate': 
		return LatentStateHead(
			in_channels=args.z_dim, 
			type="last")
	elif args.processor=='z_meanstate': 
		return LatentStateHead(
			in_channels=args.z_dim, 
			type="mean")
	else:
		raise NotImplementedError()


def main():
	term_rows, term_cols = os.popen('stty size', 'r').read().split() 
	os.environ["COLUMNS"] = str(400)

	trn_tracker = defaultdict(list) # track trn stats
	tst_tracker = defaultdict(list) # track tst stats
	
	spinner = Halo(spinner='dots')
	
	# setup command line parsing
	parser = setup_cmdline_parsing()
	args = parser.parse_args()
	print(args)
	
	# setup tensorboard if run_dir is given
	writer = None
	if args.run_dir is not None:
		writer = SummaryWriter(os.path.join(args.run_dir, args.experiment_id))
	
	spinner.start('Loading data')
	ds = load_data(args)
	spinner.succeed()

	spinner.start('Patching command line args')
	args_dict = vars(args)
	args_dict['vec_inp_dim'] = ds.num_vec_dim
	args_dict['num_aux_dim'] = ds.num_aux_dim
	args_dict['num_timepts'] = ds.num_timepts
	spinner.succeed()

	spinner.start('Subsampling time points')
	if args.tps_frac > 0:
		assert args.tps_frac < 1, 'Timepoint subsampling not in range (0,1)'
		indices = create_sampling_indices(len(ds), args.num_timepts, int(args.tps_frac*args.num_timepts)) 
		ds.subsample(indices) 
	spinner.succeed()
	
	# make sure splits are the same for a given seed
	split_generator = torch.Generator().manual_seed(args.seed) 
	trn_set, tst_set = torch.utils.data.random_split(ds, [0.8, 0.2], generator=split_generator)

	loader_generator = torch.Generator().manual_seed(args.seed)
	dl_trn = DataLoader(trn_set, 
						batch_size=args.batch_size, 
						shuffle=True, 
						num_workers=4,
						collate_fn=ds.get_collate(),
						worker_init_fn=seed_worker,
						generator=loader_generator)
	dl_tst = DataLoader(tst_set, 
						batch_size=args.batch_size, 
						shuffle=False, 
						collate_fn=ds.get_collate())

	recog_backbone = create_recog_backbone(args) 
	recon_backbone = create_recon_backbone(args) 
	processor = create_processor(args)

	modules = nn.ModuleDict(
	{
		"recog_net": recog_backbone,
		"recon_net": recon_backbone,
		"lnode_net": LatentODEfunc(args.z_dim, args.ode_h_dim),
		"ptogd_net": PathToGaussianDecoder(nn.Identity(), initial_sigma=1.0),
		"processor": processor,
		"regressor": nn.Sequential(
				nn.Linear(processor.get_outdim(), args.num_aux_dim),
				nn.Tanh())})
	modules = modules.to(args.device)
	
	num_params = 0
	for p in modules.parameters():
		num_params += p.numel()
	print(f'Number of parameters is {num_params}')

	# discretize T=[0,1]
	assert args.num_timepts is not None, "Nr. of timepoints not set!"
	t = torch.linspace(0, 1.0, args.num_timepts).to(args.device)
	
	optimizer = Adam([
				{'params': modules['recog_net'].parameters(), 'weight_decay': 1e-3},
				{'params': modules['recon_net'].parameters()},
				{'params': modules['lnode_net'].parameters()},
				{'params': modules['ptogd_net'].parameters()},
				{'params': modules['processor'].parameters()},
				{'params': modules['regressor'].parameters()}],
			lr=args.lr, weight_decay=args.weight_decay
	)
	
	optimizer = Adam(modules.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)
	aux_loss_fn = nn.MSELoss()

	for epoch_cnt in range(args.n_epochs):
		# training 
		run_epoch(args, 
			dl_trn, 
			t, 
			modules, 
			optimizer, 
			aux_loss_fn, 
			trn_tracker, 
			mode='train')
		# testing
		with torch.no_grad():
			run_epoch(args, 
				dl_tst, 
				t, 
				modules, 
				optimizer,
				aux_loss_fn, 
				tst_tracker, 
				mode='test')
		scheduler.step()
		
		# what scores to track throughout training/testing
		scorefns = {'r2s': r2_score, 
					'mse': mean_squared_error}
		trackers = {'trn': trn_tracker, 
					'tst': tst_tracker}
		
		# compute scores
		scores = defaultdict(list)
		for scorefn_key, trackers_key in product(scorefns, trackers):  
			key_str = scorefn_key + "_" + trackers_key   
			for aux_d in range(args.num_aux_dim):
				tmp = scorefns[scorefn_key](
					trackers[trackers_key]['epoch_aux_t'][-1][:,aux_d].numpy(),
					trackers[trackers_key]['epoch_aux_p'][-1][:,aux_d].numpy())
				scores[key_str].append(tmp)
				if writer:
					writer.add_scalar("{}_{}/{}".format(scorefn_key, aux_d, trackers_key), tmp, epoch_cnt)
				
		
		print('{:04d} | trn_loss={:.4f} | trn_aux={:0.4f} | trn_kld={:0.4f} | trn_log_pxz={:0.4f} | avg_trn_mse={:0.4f} | avg_tst_mse={:0.4f} | avg_tst_r2s={:0.4f} | lr={:0.6f}'.format(
			epoch_cnt,
			trn_tracker['epoch_loss'][-1],
			trn_tracker['epoch_aux_loss'][-1],
			trn_tracker['epoch_kld'][-1],
			trn_tracker['epoch_log_pxz'][-1],
			np.mean(scores['mse_trn']),
			np.mean(scores['mse_tst']),
			np.mean(scores['r2s_tst']),
			scheduler.get_last_lr()[-1]))
		
		if writer:
			writer.add_scalar("r2s_avg/trn", np.mean(scores['r2s_trn']), epoch_cnt)
			writer.add_scalar("r2s_avg/tst", np.mean(scores['r2s_tst']), epoch_cnt)
			
		for aux_d in range(args.num_aux_dim):
			if writer:
				plt.plot(
					tst_tracker['epoch_aux_t'][-1][:,aux_d], 
					tst_tracker['epoch_aux_p'][-1][:,aux_d], '.')
				writer.add_figure('r2s/tst_scatter_{}'.format(aux_d), plt.gcf(), epoch_cnt)
				plt.close()

	if writer:
		writer.close()
	if args.log_out_file:
		torch.save((trn_tracker, tst_tracker, args), args.log_out_file)
	if args.net_out_file:
		torch.save(modules.state_dict(), args.net_out_file)


if __name__ == "__main__":
	main()
