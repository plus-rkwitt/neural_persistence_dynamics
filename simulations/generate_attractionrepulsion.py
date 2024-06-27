"""Code to generate the Attraction-Repulsion data."""

import os
import sys
import json
import torch

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from random import SystemRandom

from sisyphe.models import AttractionRepulsion
from simulation_utils import ProgressBar


def setup_argsparse():
    description = """**Attraction-Repulsion simulation**"""    
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
        default=200,
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
        default="../data/attraction_repulsion/",
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


class AttractionRepulsionSimulator:
    def __init__(
        self,
        num_points: int,
        steps: int,
        freq: int,
        dim: int = 3,
        device: float = 'cuda:0',
        friction: float = 1.,
        Ca: float = 1.,
        la: float = 1.,
        box_size: float = 1.,
        dt: float = 0.01,
        isaverage: bool = False,
        p: int = 1,
    ) -> None:
        self.num_points = num_points
        self.steps = steps
        self.freq = freq
        self.dim = dim
        self.device = device

        self.fixed_params = {
            'friction': friction,
            'Ca': Ca,
            'la': la,
            'box_size': box_size,
            'dt': dt,
            'isaverage': isaverage,
            'p': p
        }

    def simulate(self, rad, propulsion, Cr, lr):
        pos = (torch.rand((self.num_points, self.dim), device=self.device) - 0.5)
        vel = torch.randn(self.num_points, self.dim, device=self.device)
        vel = vel/torch.norm(vel, dim=1, keepdim=True)

        params = {
            'pos': pos,
            'interaction_radius': rad,
            'vel': vel,
            'propulsion': propulsion,        
            'Cr': Cr,
            'lr': lr,
            **self.fixed_params
        } 
        
        simu = AttractionRepulsion(**params)
        pos_vel = torch.stack([torch.stack((step['position'], step['velocity'])).cpu() for step, i in zip(simu, range(self.steps)) if i % self.freq == 0], dim=1)
        return pos_vel


def main():
    args = setup_argsparse()
    
    simulator = AttractionRepulsionSimulator(
        num_points=args.points, 
        steps=args.steps, 
        freq=args.freq, 
        dim=args.dim, 
        device=args.device)
    
    rad = torch.tensor(2.).pow(torch.Tensor(args.simulations).uniform_(-2/3, 2/3))
    prop = torch.tensor(2.).pow(torch.Tensor(args.simulations).uniform_(-2, 2))
    Cr = torch.tensor(2.).pow(torch.Tensor(args.simulations).uniform_(-1, 1))
    lr = torch.tensor(2.).pow(torch.Tensor(args.simulations).uniform_(-1.5, 0.5))
    params = torch.stack([rad, prop, Cr, lr], dim=1)

    directory_ = os.path.join(args.root, f"id_{args.id}")
    if not os.path.isdir(directory_):
        os.makedirs(directory_)

    pbar = ProgressBar(args.simulations)
    positions, velocities = [], []
    for r, p, c, l in params:
        pos, vel = simulator.simulate(float(r), float(p), float(c), float(l))
        positions.append(pos)
        velocities.append(vel)
        pbar.update()
        
    with open (f'{directory_}setup.json', 'w') as f:
        json.dump(simulator.__dict__, f, ensure_ascii=True, indent=4)

    torch.save(params, os.path.join(directory_, 'target.pt'))
    torch.save(positions, os.path.join(directory_, 'positions.pt'))
    torch.save(velocities, os.path.join(directory_, 'velocities.pt'))


if __name__ == "__main__":
    main()
