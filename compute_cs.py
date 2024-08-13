"""
Script to prepare the crocker stacks from the persistence diagrams.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from teaspoon.TDA.Persistence import maxPers, CROCKER_Stack
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def max_pers_dgms(dgms_exp: list):
    m_values = np.array(
        [maxPers(dgms) for dgms_t in dgms_exp for dgms in dgms_t]
    )

    return m_values.max()*0.9 // 2


def featurize(pds, max_eps_=1.0, num_stops_=10, alpha=None):
    if alpha is None:
        alpha = [0.]

    n_pds = len(pds)
    n_tps = len(pds[0])
    n_dim = num_stops_ * len(alpha) * n_tps
    vecs = torch.zeros(n_pds, n_dim)
    for n in tqdm(range(n_pds)):
        cs = CROCKER_Stack(pds[n], maxEps=max_eps_,
                           numStops=num_stops_, alpha=alpha,
                           plotting=False)
        vecs[n] = torch.tensor(cs).view(-1)
    return vecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgms_file", type=str,
                        help="path to the dgms file")
    args = parser.parse_args()

    if not os.path.isfile(args.dgms_file):
        logging.fatal("DGMS file does not exist.")
        sys.exit(1)

    root_dir = os.path.dirname(args.dgms_file)
    root_dir = os.path.abspath(root_dir)

    logging.debug("Start preparing the data.")

    if not os.path.isfile(os.path.join(root_dir, 'crocker_vecs.pt')):
        dgms = torch.load(args.dgms_file)

        logging.debug("find maxEps value over ALL DGMS")
        max_eps_h0 = np.array(dgms[0]).flatten().max() / 3
        max_eps_h1 = np.array([
            np.array(dd).max() for d in dgms[1] for dd in d
        ]).max() / 2
        max_eps = [np.ceil(max_eps_h0), np.ceil(max_eps_h1)]

        logging.debug("find max alpha value over ALL DGMS")
        max_alpha = max_pers_dgms(dgms[0])

        num_hom_levels = len(dgms)
        for idx in range(num_hom_levels):
            # vectorizing the persistence diagrams
            vecs_ = featurize(dgms[idx], max_eps_=max_eps[idx], num_stops_=10,
                              alpha=np.linspace(0.01, max_alpha, 8))

            # storing the vectorized persistence diagrams
            torch.save(vecs_, os.path.join(root_dir, f'crocker_vecs_h{idx}.pt'))

            # deleting, so that RAM is not too overloaded while processing the
            # others
            del vecs_

        # starting of fresh
        vecs_ = [
            torch.load(
                os.path.join(root_dir, f'crocker_vecs_h{idx}.pt')
            ) for idx in range(num_hom_levels)
        ]

        torch.save(torch.hstack(vecs_), os.path.join(root_dir, f'crocker_vecs.pt'))

    logging.debug("Preparing data done.")


if __name__ == '__main__':
    main()
