"""Compute PH vectorizations using 

[Hofer19a]
Hofer et al.
Learning vectorizations of persistence barcodes
JMLR'19
see https://jmlr.csail.mit.edu/papers/v20/18-358.html
"""

import copy
import torch

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from halo import Halo

from npd.vectorization.core import (
    compute_SLayerExponential_parameters,
    compute_SLayerExponential_vectorization)


def setup_cmdline_parsing():
    description = """**Compute PD vectorizations**"""    
    generic_parser = argparse.ArgumentParser(description=Markdown(description, style="argparse.text"), formatter_class=RichHelpFormatter)
    
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument(
        "--dgms-inp-file", 
        metavar="FILE",
        type=str, 
        default="dgms.pt", 
        help="Input file with persistence diagrams")
    group0.add_argument(
        "--vecs-out-base", 
        metavar="PREFIX",
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
        help="Start pulling towards 0 persistence at nu")
    
    group2 = generic_parser.add_argument_group('General parameters')
    group2.add_argument(
        "--subsample",
        metavar="<int>", 
        type=int, 
        default=50000) 
    
    return generic_parser


def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)

    spinner = Halo(spinner='dots')

    spinner.start(f'Loading {args.dgms_inp_file}')
    raw_dgms = torch.load(args.dgms_inp_file, weights_only=False)
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
