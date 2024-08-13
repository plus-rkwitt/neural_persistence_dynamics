"""Implementation of PSK kernel approach from [Giusti23a]."""

import os
import sys
import time
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import explained_variance_score

import argparse
from rich.markdown import Markdown
from rich_argparse import RichHelpFormatter

from sklearn.model_selection import ShuffleSplit
from sktime.dists_kernels import SignatureKernel

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from permetrics import RegressionMetric


def setup_cmdline_parsing():
    description = """**PSK computation**"""    
    generic_parser = argparse.ArgumentParser(
        description=Markdown(description, style="argparse.text"), 
        formatter_class=RichHelpFormatter)
    
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument(
        "--vecs-inp-file",
        metavar="FILE", 
        type=str, 
        default="vecs.pt",
        help="Input file with vectorized PDs (default: %(default)s)"
    )
    group0.add_argument(
        "--prms-inp-file",
        metavar="FILE", 
        type=str,
        default="prms.pt",
        help="Input file with target simulation parameters (default: %(default)s)"
    )
    group0.add_argument(
        "--kern-out-base", 
        metavar="FILE",
        type=str, 
        default="/tmp/kern",
        help="Output file basename for precomputed kernel matrices (default: %(default)s)"
    ) 
    group0.add_argument(
        "--stat-out-file", 
        metavar="FILE",
        type=str, 
        default="/tmp/stat.pt",
        help="Output file with regression statistics (default: %(default)s)"
    )
    group1 = generic_parser.add_argument_group('PSK parameters')
    group1.add_argument(
        "--level",
        metavar="INT", 
        type=int, 
        default=2,
        help="Level of the signature kernel (default: %(default)s)"
    )
    return generic_parser


def lag_transform(X, lags=None):
    """Lag transform of the time series."""
    if lags is None:
        lags = [0, 1, 2]

    lagged_X = defaultdict()
    num_experiment, num_time_steps, dimension_time_series = X.shape
    for i in range(len(lags)):
        Y = torch.zeros(num_experiment, num_time_steps, dimension_time_series * (i + 1))
        for j, l in enumerate(lags[:i+1]):
            Y[:, l:, j * dimension_time_series:(j + 1) * dimension_time_series] = X[:, 0:num_time_steps - l, :]
        lagged_X[lags[i]] = Y
    return lagged_X


def time_subsample(vecs, sample_rate=0.2):
    num_experiments, num_diagrams, num_time_steps = vecs.shape
    num_keep = int(num_time_steps*sample_rate)
    vecs_ss = torch.zeros(num_experiments, num_diagrams, num_keep)
    for i in range(vecs.shape[0]):
        vecs_ss[i] = vecs[i, :, torch.randperm(num_time_steps)[0:num_keep]]
    return vecs_ss


def run_regression(K, lags, vecs, y, C_s, e_s, n_splits=10,
                   test_size=.20, id_=0):

    rmses, r2vals, smapes, expvars  = [], [], [], []
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)
    metric = RegressionMetric()

    # evaluate over CV folds
    for _, (trn_idx, tst_idx) in enumerate(cv.split(vecs)):

        best_score = 1e9
        best_c, best_e, best_l = 0, 0, 0
        svr = SVR(kernel='precomputed')
        
        # cross-validation for hyperparameters (over lags and SVC
        # hyperparameters)
        for lag_ in lags:
            # training portion of precomputed kernel matrix
            K_trn = K[lag_][trn_idx, :][:, trn_idx]
            y_trn = y[trn_idx, id_]

            # testing portion of precomputed kernel matrix
            K_tst = K[lag_][tst_idx, :][:, trn_idx]
            y_tst = y[tst_idx, id_]

            for C in C_s:
                for e in e_s:
                    svr.C = C
                    svr.epsilon = e
                    # only split once in inner loop (for speedup)
                    inner_cv = ShuffleSplit(n_splits=1, test_size=.20)
                    
                    trn_idx_cv, tst_idx_cv = next(inner_cv.split(range(K_trn.shape[0])))                
                    K_trn_cv = K_trn[trn_idx_cv, :][:, trn_idx_cv]
                    K_tst_cv = K_trn[tst_idx_cv, :][:, trn_idx_cv]
                    y_trn_cv = y_trn[trn_idx_cv]
                    y_tst_cv = y_trn[tst_idx_cv]
                    
                    svr.fit(K_trn_cv, y_trn_cv)
                    y_hat_cv = svr.predict(K_tst_cv)

                    # use parameter MSE for hyperparameter selection
                    score = mean_squared_error(y_tst_cv, y_hat_cv)
                    
                    if score < best_score:
                        best_c = C  # C-param of SVM
                        best_e = e  # epsilon-param of SVM
                        best_l = lag_  # lag
                        best_score = score
        
        svr.C = best_c
        svr.epsilon = best_e
        
        # re-train on full training portion of kernel matrix
        K_trn = K[best_l][trn_idx, :][:, trn_idx]
        K_tst = K[best_l][tst_idx, :][:, trn_idx]
        y_trn = y[trn_idx, id_]
        y_tst = y[tst_idx, id_]
        svr.fit(K_trn, y_trn)

        # predict on testing portion
        y_hat = svr.predict(K_tst)
        
        # compute scores
        rmses.append(metric.root_mean_squared_error(y_tst.numpy(), y_hat))
        smapes.append(metric.symmetric_mean_absolute_percentage_error(y_tst.numpy(), y_hat))
        r2vals.append(r2_score(y_tst.numpy(), y_hat))
        expvars.append(explained_variance_score(y_tst.numpy(), y_hat))

    return rmses, r2vals, smapes, expvars


def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)
    
    assert os.path.exists(args.prms_inp_file), f"File {args.prms_inp_file} not found!"
    assert os.path.exists(args.vecs_inp_file), f"File {args.vecs_inp_file} not found!"
    
    prms = torch.load(args.prms_inp_file)
    vecs = torch.load(args.vecs_inp_file)

    # determine nr. of parameters
    prms_ids = list(range(prms.shape[1]))
    
    vecs = vecs.permute(0, 2, 1)
    num_time_series, dimension_time_series, num_time_steps = vecs.shape
    print(f'{num_time_series} time series of dim {dimension_time_series} '
          f'with {num_time_steps} timepoints and '
          f'{len(prms_ids)} aux. variables!')
    
    data = vecs.permute(0, 2, 1).view(-1, num_time_steps)
    scaler = MinMaxScaler()
    scaler.fit(data)
    vecs = torch.tensor(scaler.transform(data)).view(num_time_series,
                                                     num_time_steps,
                                                     dimension_time_series)
    
    lags = [0, 1, 2]  # 3 lags used ion [Giusti23a]
    lagged_vecs = lag_transform(vecs)

    K_ss = defaultdict(list)
    for l in lags:
        kern_out_file = args.kern_out_base + "_level_{}_lag_{}.pt".format(args.level, l)
        if os.path.exists(kern_out_file):
            print('Loading {}'.format(kern_out_file))
            K_ss[l] = torch.load(kern_out_file)
        else:
            t0=time.time()
            sk = SignatureKernel(normalize=True, level=args.level)
            K_ss[l] = sk.transform(lagged_vecs[l].permute(0,2,1).numpy())
            print('Computed {} in {} sec'.format(kern_out_file, time.time()-t0))
            torch.save(K_ss[l], kern_out_file)

    C_s = np.logspace(-3, 1, 5)  # from [Giusti23a] paper
    e_s = np.logspace(-4, 1, 5)  # from [Giusti23a] paper

    stats = defaultdict(list)
    cv_runs = 10
    for aux_d in prms_ids:
        rmses_ss, r2vals_ss, smapes_ss, expvars_ss = run_regression(K_ss, lags, vecs, prms, C_s, e_s, cv_runs, 0.2, aux_d)
        print('[{}]: RMSE={:0.4f} +/- {:0.4f} | R2={:0.4f} +/- {:0.4f} | SMAPE={:0.4f} +/- {:0.4f} | ExpVar={:0.4f} +/- {:0.4f} '.format(
            aux_d,
            np.mean(rmses_ss),
            np.std(rmses_ss),
            np.mean(r2vals_ss),
            np.std(r2vals_ss),
            np.mean(smapes_ss),
            np.std(smapes_ss),
            np.mean(expvars_ss),
            np.std(expvars_ss)
        ))            
    
        [stats['r2s_param'+str(aux_d)].append(tmp) for tmp in r2vals_ss]   # r2 per CV run and parameter
        [stats['rmse_param'+str(aux_d)].append(tmp) for tmp in rmses_ss]   # RMSE per CV run and parameter
        [stats['smape_param'+str(aux_d)].append(tmp) for tmp in smapes_ss] # SMAPE per CV run and parameter
        [stats['expvar_param'+str(aux_d)].append(tmp) for tmp in expvars_ss] # SMAPE per CV run and parameter
        
    
    torch.save(stats, args.stat_out_file)
    

if __name__ == "__main__":
    main()
