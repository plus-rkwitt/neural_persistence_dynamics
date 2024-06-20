<img src="assets/npd.jpg" height="180" />

This is the offical repository for [Neural Persistence Dynamics](https://arxiv.org/abs/2405.15732).

If you use the code please cite as:

```bibtex
@misc{Zeng24a,
      title={Neural Persistence Dynamics}, 
      author={Sebastian Zeng and Florian Graf and Martin Uray and Stefan Huber and Roland Kwitt},
      year={2024},
      eprint={2405.15732},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contents
- [Setup](#setup)
- [Replicating experiments (with precomputed simulations)](#replicating-experiments)
- [Running your own simulations](#running-your-own-simulations)

## Setup
In the following, we assume that the repository has been cloned into `/tmp/neural_persistence_dynamics`.

### Setup a new Anaconda environment
```bash
conda create -n "pytorch23" python=3.10
conda activate pytorch23
```

### Create folder structure
```bash
cd /tmp/neural_persistence_dynamics
mkdir -p data # stores all data
mkdir -p logs # stores all logs
mkdir -p 3rdparty # 3rdparty code
mkdir -p runs # stores all tensorboard related stuff
```

### Installing ```pytorch```
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python -c 'import torch' # check
```

### Installing `torchph`
```bash
cd /tmp/neural_persistence_dynamics
cd 3rdparty
git clone https://github.com/c-hofer/torchph.git 
pip install -e .
python -c 'import torchph' # check
```

### Installing `torchdiffeq`
```bash
pip3 install torchdiffeq
python -c 'import torchdiffeq' # check
```

### Installing `ripser-plusplus`
```bash
conda install cmake
pip3 install git+https://github.com/simonzhang00/ripser-plusplus.git
```

### Installing `signatory`
```bash
cd /tmp/neural_persistence_dynamics
cd 3rdparty
git clone https://github.com/patrick-kidger/signatory.git
python setup.py install 
python -c 'import signatory' # check
```

### Installing other required packages
```bash
pip install tensorboard permetrics halo einops h5py rich_argparse gdown matplotlib torch_geometric
```

## Replicating experiments

In the following, we replicate the experiments with the `dorsogna-1k` data from then paper. All other experiments follow the same procedure.

### Downloading precomputed simulation data
```bash
cd /tmp/neural_persistence_dynamics
mkdir -p data/Giusti23a/1k
python download.py --dataset dorsogna-1k --destination data/Giusti23a/1k
```
### Computing Vietoris-Rips persistence diagrams 
```bash
 python compute_pds.py \
    --simu-inp-file data/Giusti23a/1k/simu_1k.pt \
    --prms-inp-file data/Giusti23a/1k/prms_1k.pt \
    --dgms-out-file data/Giusti23a/1k/dgms_1k_vr_h0h1.pt \
    --compute-ph
```
This will compute all Vietoris-Rips persistence diagrams for $H_0$ and $H_1$ 
(by default; in case you also want $H_2$ add `--max-dim 2` on the command line).
Diagrams are saved to `data/Giusti23a/1k/dgms_1k_vr_h0h1.pt`.

### Computing vectorizations

Next, we can compute the vectorizations:
```python
python compute_vec.py \
    --dgms-inp-file data/Giusti23a/1k/dgms_1k_vr_h0h1.pt \
    --vecs-out-base data/Giusti23a/1k/vecs \ 
    --num-elements 20 \
    --nu 0.005 \
    --subsample 50000
```
The `dorsogna-1k` dataset contains 1,000 simulations with 100 time points. Hence, we 
have a total of 100,000 available persistence diagrams (for $H_0$ and $H_1$). 

In this example,
we randomly subsample 50,000 to parametrize the vectorization, meaning 50,000 diagrams are 
used to compute the centers of 20 exponential structure elements. Overall, this 
yields 20-dimensional vectorizations per diagram and dimension. The relevant output
file is named `vecs_20_0.005.pt` and, for the given setting, contains 40-dimensional
vectors per time point.

### Training / Evaluation
We are now ready to train and evaluate the continuous latent variable model using 
`dynamics.py`. 

```bash
python dynamics.py \
    --vec-inp-file data/Giusti23a/1k/vecs_20_0.005.pt \
    --aux-inp-file data/Giusti23a/1k/prms_1k_norm.pt \
    --batch-size 64 \
    --lr 0.001 \
    --n-epochs 210 \
    --kl-weight 0.001 \
    --aux-weight 1000 \
    --restart 30 \
    --device cuda:0 \
    --z-dim 16 \
    --tps-frac 0.5 \
    --weight-decay 1e-3 \
    --run-dir runs/ \
    --log-out-file logs/logfile.pt \
    --backbone topdyn_only \
    --processor z_mtantwins \
    --mtan-h-dim 64 \
    --experiment-id debug \
    --seed 9000
```
Here,  `--tps-frac 0.5` specifies that we only want to 
keep 50% of (all 100) time points for training, the `--aux-inp-file` holds a (M, 2) tensor with normalized (to 
[-1,1]) simulation parameters that we wish to predict, and 
`--kl-weight` as well as `--aux-weight` specify the 
weight given to the KL divergence in the ELBO and the 
regression objective, respectively.

**Note**: we normalize simulation parameter ranges to 
[-1,1] for training *only*, but evaluate on the actual
range for R2 and SMAPE computation later. 

**Note**: Overall, the run in the above mentioned configuration should yield an average
testing R2 score of approx. 0.845.

### Monitoring progress

You can monitor training progress by starting a tensorboard
as follows:

```bash
cd /tmp/neural_persistence_dynamics
tensorboard --logdir=runs --port <PORT>
```

### Evaluation

The console output currently only provide information 
about the overall (extended) ELBO, the training/testing 
MSE (averaged over all parametes) and the training/testing 
R2 score (averaged over all parameters). 


### Saving / Loading models

## Precomputed simulation data

We provide precomputed simulations and persistence 
diagrams. They can be downloaded as follows:

```bash
cd /tmp/neural_persistence_dynamics
mkdir -p data/<OUTFOLDER>
python download.py --dataset <DATASETNAME> --destination data/OUTFOLDER
```
`OUTFOLDER` is the desired output folder name under `data/`, 
`DATASETNAME` is the name of the dataset as referenced in the 
paper, i.e., `dorsogna-1k`, `dorsogna-10k`, `volex-10k` and
`vicsek-10k`.

## Running your own simulations

To run your own simulations, you also need to install the `sysiphe` package via

```bash
pip3 install sisyphe
python -c 'import torchdiffeq' # check
```

Simulation code for (almost) all simulations can be found under `simulations`. 
Currently, the variations in the simulation model parameters are fixed (according 
to the paper), but are easy to modify if required.


### `dorsogna-1k`

To simulate the `dorsogna-1k` data, we used the original simulation code provided
[here](https://github.com/ldarrick/paths-of-persistence-diagrams).

### `dorsogna-10k`

Use `generate_attractionrepulsion.py` for this dataset. In particular, we used 

```bash
cd /tmp/neural_persistence_dynamics
cd simulations
python generate_attractionrepulsion.py \
    --device cuda:0 \
    --freq 10 \
    --points 200 \
    --dim 3 \
    --root ../data/attraction_repulsion \
    --steps 1000 \
    --simulations 10000
```
to generate the `dorsogna-10k` data. 

### `volex-10k`

Use `generate_volumeexclusion.py` for this dataset. In particular, we used 

```bash
cd /tmp/neural_persistence_dynamics
cd simulations
python generate_volumeexclusion.py \
    --device cuda:0 \
    --freq 10 \
    --points 50 \
    --dim 3 \
    --root ../data/volume_exclusion \
    --steps 1000 \
    --simulations 10000
```
to generate the `volex-10k` data. 


### `vicsek-10k`

Use `generate_vicsek.py` for this dataset. In particular, we used 

```bash
python generate_vicsek.py \
    --device cuda:0 \
    --freq 10 \
    --points 200 \
    --dim 3 \
    --root ../data/vicsek \
    --steps 1000 \
    --simulations 10000
```

to generate the `vicsek-10k` data.

## Test system

Ubuntu 22.04, running kernel 5.15.0-112-generic, CUDA 11.8 (NVIDIA driver 555.42.02, 2x GeForce RTX 3090) and PyTorch 2.3. 

## Notebooks

Notebooks can be found under `notebooks`.
