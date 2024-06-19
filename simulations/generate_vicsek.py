"""Code to generate the Volume-Exclusion data."""


import os
import sys
import json
import torch
from random import SystemRandom

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from sisyphe.models import Vicsek
from simulation_utils import ProgressBar


def setup_argsparse():
    description = """**Vicsek simulation**"""    
    parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"), 
        formatter_class=RichHelpFormatter)
    parser.add_argument(
        "--simulations", 
        metavar="INT",
        type=int, 
        default=1, 
        help="Number of simulations to run (default: %(default)s)"
    )
    parser.add_argument(
        "--points", 
        metavar="INT",
        type=int, 
        default=50,
        help="Number of points (default: %(default)s)"
    )
    parser.add_argument(
        "--steps", 
        metavar="INT",
        type=int, 
        default=1000,
        help='Number of simulation steps at dt=0.01 (default: %(default)s)'
    )
    parser.add_argument(
        "--freq", 
        metavar="INT",
        type=int, 
        default=10,
        help="Frequency of keeping observations (default: %(default)s)"
    )
    parser.add_argument(
        "--dim", 
        metavar="INT",
        type=int, 
        default=3,
        help="Dimension of the simulation space (default: %(default)s)"
    )
    parser.add_argument(
        "--device", 
        metavar="STR",
        type=str, 
        default="cuda:0",
        help="Device to run the simulation on (default: %(default)s)"
    )
    parser.add_argument(
        "--root", 
        metavar="FOLDER",
        type=str, 
        default="../data/volume_exclusion/",
        help="Root folder to save the data (default: %(default)s)"
    )
    parser.add_argument(
        "--id", 
        metavar="STR",
        type=str, 
        default=-1,
        help="ID of the simulation (default: random)",
    )
    args = parser.parse_args()
    if args.id == -1:
        args.id = int(SystemRandom().random() * 100000)
    return args


class VicsekSimulator:
    def __init__(
        self,
        num_points: int,
        steps: int,
        freq: int,
        dim: int = 3,
        device: float = 'cuda:0',
        box_size: float = 20.,
        dt: float = 0.01,
        ) -> None:
        self.num_points = num_points
        self.steps = steps
        self.freq = freq
        self.dim = dim
        self.device = device
        self.box_size = box_size
        self.dt = dt
        
        self.fixed_params = {
            'box_size': box_size,
            'dt': dt
        }

    def simulate(self, rad, C, sigma, nu):
        pos = self.box_size*torch.rand(self.num_points, self.dim, device=self.device)
        vel = torch.randn(self.num_points, self.dim, device=self.device)
        vel = vel/torch.norm(vel, dim=1).reshape((self.num_points,1))
    
        params = {
            'pos': pos,
            'vel': vel,
            'interaction_radius': rad,
            'sigma': sigma,
            'nu' : nu,
            'v' : C,
            **self.fixed_params
        }
        
        simu = Vicsek(**params)

        positions = []
        for step, i in zip(simu, range(self.steps)):
            if i%10 == 0:
                positions.append(step['position'])
        return positions
    

def generate_param():
    rad = torch.Tensor(1).uniform_(0.5,5.).item()
    C = torch.Tensor(1).uniform_(0.5,5.).item()
    sigma = torch.Tensor(1).uniform_(0.,2.0).item()
    nu = torch.Tensor(1).uniform_(0.5,5.0).item()
    return rad, C, sigma, nu


def main():
    args = setup_argsparse()
    simulator = VicsekSimulator(
        num_points=args.points, 
        steps=args.steps, 
        freq=args.freq, 
        dim=args.dim, 
        device=args.device)

    dir = os.path.join(args.root, f"id_{args.id}")
    if not os.path.isdir(dir): 
        os.makedirs(dir)
    
    pbar = ProgressBar(args.simulations)
    positions, velocities, params = [], [], []
    n = 0
    m = 0
    while n < args.simulations:
        param = generate_param()
        pos = simulator.simulate(*param)
        if pos is not None:
            positions.append(pos)
            params.append(param)
            n += 1
            pbar.update()
    print("\n", m, "/", n)
        
    with open (f'{dir}setup.json', 'w') as f:
        json.dump(simulator.__dict__, f, ensure_ascii=True, indent=4)
    torch.save(params, os.path.join(dir, 'target.pt'))
    torch.save(positions, os.path.join(dir, 'positions.pt'))
    torch.save(velocities, os.path.join(dir, 'velocities.pt'))


if __name__ == "__main__":
    main()