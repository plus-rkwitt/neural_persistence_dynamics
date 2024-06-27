"""Implementation of the mTAN baseline model (w/o dynamics)."""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict

from halo import Halo

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

from npd.nn.core import (
    TDABaselineBackbone,
    PointNetBaselineBackbone,
    JointBaselineBackbone
)

from npd.data.core import (
    PHDataset,
    PointCloudDataset,
    JointDataset,
    create_sampling_indices)


def setup_cmdline_parsing():
    description = """**Baseline**"""
    generic_parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"),
        formatter_class=RichHelpFormatter)
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument(
        "--log-out-file",
        metavar="FIlE",
        type=str,
        default=None,
        help="Filename for output log file (default: None)"
    )
    group0.add_argument(
        "--vec-inp-file",
        metavar="FILE",
        type=str,
        default=None,
        help="Filename for input PH vectorizations"
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
        "--weight-decay",
        metavar="FLOAT",
        type=float,
        default=1e-3,
        help="Weight decay (default: %(default)s)"
    )

    group2 = generic_parser.add_argument_group('Baseline model configuration arguments')
    group2.add_argument(
        "--mtan-h-dim",
        metavar="INT",
        type=int,
        default=128,
        help="Hidden dimensionality of mTAN module (default: %(default)s)"
    )
    group2.add_argument(
        "--mtan-embed-time",
        type=int,
        metavar="INT",
        default=128,
        help="Dimensionality of time embedding.")
    group2.add_argument(
        "--mtan-num-queries",
        type=int,
        metavar="INT",
        default=128,
        help="Number of queries (default: %(default)s)"
    )
    group2.add_argument(
        "--pointnet-dim",
        type=int,
        metavar="INT",
        default=32,
        help="PointNet++ hidden dimensionality (default: %(default)s)"
    )
    group2.add_argument(
        "--backbone", choices=[
            'topdyn_only',
            'ptsdyn_only',
            'joint'],
        default="topdyn_only")

    group3 = generic_parser.add_argument_group("Data preprocessing arguments")
    group3.add_argument(
        "--tps-frac",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="Fraction of timepoints to use."
    )
    return generic_parser


def run_epoch(args, dl, modules, optimizer, aux_loss_fn=None, tracker=None, mode='train'):
    epoch_loss = epoch_instances = 0.

    if mode == 'train':
        modules.train()
    else:
        modules.eval()

    aux_p = [] # predictions
    aux_t = [] # ground truth

    for batch in dl:
        aux_enc, _, _, aux_obs = modules['recog_net'](batch, args.device)
        aux_out = modules['regressor'](aux_enc)
        loss = aux_loss_fn(aux_out.flatten(), aux_obs.flatten())

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            aux_p.append(aux_out.detach().cpu())
            aux_t.append(aux_obs.detach().cpu())

        epoch_loss += loss.item()
        epoch_instances += aux_obs.shape[0]

    if tracker is not None:
        tracker['epoch_loss'].append(epoch_loss/len(dl))
        tracker['epoch_aux_p'].append(torch.cat(aux_p))
        tracker['epoch_aux_t'].append(torch.cat(aux_t))


def create_recog_backbone(args):
    if args.backbone == 'topdyn_only':
        return TDABaselineBackbone(args)
    elif args.backbone == 'ptsdyn_only':
        return PointNetBaselineBackbone(args)
    elif args.backbone == 'joint':
        return JointBaselineBackbone(args)

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

    raise NotImplementedError()


def main():
    trn_tracker = defaultdict(list)  # track training stats
    tst_tracker = defaultdict(list)  # track testing stats

    spinner = Halo(spinner='dots')

    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)

    writer = None
    if args.run_dir is not None:
        writer = SummaryWriter(os.path.join(args.run_dir, args.experiment_id))

    spinner.start('Loading data')
    ds = load_data(args)
    spinner.succeed('Loaded data!')

    spinner.start('Patching command line args')
    args_dict = vars(args)
    args_dict['vec_inp_dim'] = ds.num_vec_dim
    args_dict['num_aux_dim'] = ds.num_aux_dim
    args_dict['num_timepts'] = ds.num_timepts
    spinner.succeed('Patched command line args!')

    spinner.start('Subsampling')
    if args.tps_frac > 0:
        assert args.tps_frac < 1, 'Timepoint subsampling not in range (0,1)'
        indices = create_sampling_indices(len(ds), args.num_timepts, int(args.tps_frac*args.num_timepts))
        ds.subsample(indices)
    spinner.succeed('Subsampled!')

    generator = torch.Generator()
    trn_set, tst_set = torch.utils.data.random_split(ds, [0.8, 0.2], generator=generator)
    dl_trn = DataLoader(trn_set, batch_size=args.batch_size, shuffle=True, collate_fn=ds.get_collate())
    dl_tst = DataLoader(tst_set, batch_size=args.batch_size, shuffle=False, collate_fn=ds.get_collate())

    recog_backbone = create_recog_backbone(args)

    modules = nn.ModuleDict({
        "recog_net":recog_backbone
    })
    modules.add_module("regressor", nn.Sequential(
        nn.Linear(
            args.mtan_h_dim,
            args.num_aux_dim), nn.Tanh()))
    modules = modules.to(args.device)

    num_params = 0
    for p in modules.parameters():
        num_params += p.numel()
    print(f'Number of parameters is {num_params}')

    assert args.num_timepts is not None, "Nr. of timepoints not set!"
    t = torch.linspace(0, 1.0, args.num_timepts).to(args.device)

    optimizer = Adam(modules.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)
    aux_loss_fn = nn.MSELoss()

    for epoch_cnt in range(args.n_epochs):
        run_epoch(args, dl_trn, modules, optimizer,
                  aux_loss_fn, trn_tracker, mode='train')

        with torch.no_grad():
            run_epoch(args, dl_tst, modules, optimizer,
                      aux_loss_fn, tst_tracker, mode='test')
        scheduler.step()

        scorefns = {'r2s': r2_score,
                    'mse': mean_squared_error}
        trackers = {'trn': trn_tracker,
                    'tst': tst_tracker}

        scores = defaultdict(list)
        for scorefn_key, trackers_key in product(scorefns, trackers):
            key_str = scorefn_key + "_" + trackers_key
            for aux_d in range(args.num_aux_dim):
                tmp = scorefns[scorefn_key](
                    trackers[trackers_key]['epoch_aux_t'][-1][:,aux_d],
                    trackers[trackers_key]['epoch_aux_p'][-1][:,aux_d])
                scores[key_str].append(tmp)
                if writer:
                    writer.add_scalar("{}_{}/{}".format(scorefn_key, aux_d, trackers_key), tmp, epoch_cnt)

        print(f"{epoch_cnt:04d} | "
              f"trn_loss={trn_tracker['epoch_loss'][-1]:.4f} | "
              f"avg_trn_mse={np.mean(scores['mse_trn']):0.4f} | "  
              f"avg_tst_mse={np.mean(scores['mse_tst']):0.4f} | "
              f"avg_tst_r2s={np.mean(scores['r2s_tst']):0.4f} |" 
              f"lr={scheduler.get_last_lr()[-1]:0.6f}"
              )

        if writer:
            writer.add_scalar("r2s_avg/trn", np.mean(scores['r2s_trn']), epoch_cnt)
            writer.add_scalar("r2s_avg/tst", np.mean(scores['r2s_tst']), epoch_cnt)

        for aux_d in range(args.num_aux_dim):
            if writer:
                plt.plot(
                    tst_tracker['epoch_aux_t'][-1][:, aux_d],
                    tst_tracker['epoch_aux_p'][-1][:, aux_d], '.')
                writer.add_figure('r2s/tst_scatter_{}'.format(aux_d), plt.gcf(), epoch_cnt)
                plt.close()

    if writer:
        writer.close()
    if args.log_out_file:
        torch.save((trn_tracker, tst_tracker, args), args.log_out_file)


if __name__ == "__main__":
    main()
