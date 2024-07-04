import argparse
import os.path
import signal
import sys

import pickle

import torch
from collections import defaultdict

import numpy as np
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from skopt.space import Categorical

from skopt import BayesSearchCV

import logging

import time
import datetime


def sigint_handler(sig, frame):
    logging.fatal('SIGINT was sent (probably Ctrl+C).')
    time.sleep(1)
    sys.exit(-1)


def setup_cmdline_parsing():
    generic_parser = argparse.ArgumentParser()
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument("--prms_file", type=str,
                        help="path to the Parameter file")
    group0.add_argument("--vecs_file", type=str,
                        help="path to the Crocker Stack Vector file")
    group0.add_argument("--skip_y", type=int, nargs='+', default=[],
                        help="labels to skip for parameter search")

    group1 = generic_parser.add_argument_group('Training arguments')
    group1.add_argument("--debug", action="store_true", help="debug mode")
    group1.add_argument("--jobs", type=int, default=8,
                        help="number of jobs for parameter search")
    group1.add_argument("--n_iter", type=int, default=200,
                        help="number of iterations for bayes search")

    group2 = generic_parser.add_argument_group('Model configuration arguments')
    group2.add_argument("--svr_c", type=float, nargs='+', default=None,
                        help="C Parameter for sklearn SVR.")
    group2.add_argument("--svr_tol", type=float, nargs='+', default=None,
                        help="tol Parameter for sklearn SVR.")
    group2.add_argument("--data_scaling", type=bool, nargs='+', default=None,
                        help="tol Parameter for sklearn SVR.")

    return generic_parser


def get_bayes_search_opt(args):
    pipe = Pipeline([
        ("scaling", "passthrough"),
        ("model", LinearSVR(dual='auto', max_iter=1 if args.debug else 100000))
    ])

    return BayesSearchCV(
        pipe,
        {
            'scaling': Categorical([MinMaxScaler(), StandardScaler(), None]),
            'model__C': (1e-7, 1e-3, 'log-uniform'),
            'model__tol': (1e-7, 1e-2, 'log-uniform'),
        },
        n_iter=args.n_iter if not args.debug else 2,
        n_points=1,
        n_jobs=args.jobs, cv=4,
        scoring='neg_mean_squared_error',
        verbose=10, random_state=0)


def evaluate_cv(x_, y_, args, n_splits=10, scale=False, c_=1e-4, tol_=1e-6):
    results_ = defaultdict(list)

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
    for cv_run, (trn_idx, tst_idx) in enumerate(cv.split(x_)):

        trn_vecs = x_[trn_idx]
        tst_vecs = x_[tst_idx]
        trn_parameter = y_[trn_idx]
        tst_parameter = y_[tst_idx]

        if scale:
            scaler = MinMaxScaler()
            scaler.fit(trn_vecs)
            trn_vecs = scaler.transform(trn_vecs)
            tst_vecs = scaler.transform(tst_vecs)

        # needs quite some heavy regularization
        svr = LinearSVR(dual="auto", random_state=0, C=c_, tol=tol_,
                        max_iter=1 if args.debug else 100000)
        svr.fit(trn_vecs, trn_parameter.view(-1))
        yhat = np.abs(svr.predict(tst_vecs))

        results_['yhat'].append(yhat)
        results_['ytru'].append(tst_parameter.view(-1).numpy())
        results_['r2'].append(r2_score(tst_parameter, yhat))
        results_['mse'].append(mean_squared_error(tst_parameter, yhat))

        logging.info(f'{cv_run:04d} | R2={results_["r2"][-1]:0.4f} | '
                     f'MSE={results_["mse"][-1]:0.4f}')

    return results_


def evaluate(y_label_idx, args, c_=1e-4, tol_=1e-6, scale=False, random_seed=0):
    results_ = defaultdict(list)

    x_train, x_test, y_train, y_test = get_train_test_data(random_seed)

    if scale:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    # needs quite some heavy regularization
    svr = LinearSVR(dual="auto", random_state=0, C=c_, tol=tol_,
                    max_iter=1 if args.debug else 50000)

    svr.fit(x_train, y_train[y_label_idx].view(-1).numpy())
    yhat = np.abs(svr.predict(x_test))

    results_['yhat'].append(yhat)
    results_['r2'].append(r2_score(y_test[y_label_idx], yhat))
    results_['mse'].append(mean_squared_error(y_test[y_label_idx], yhat))

    logging.info(f'{random_seed} | R2={results_["r2"][-1]:0.4f} | '
                 f'MSE={results_["mse"][-1]:0.4f}')

    return results_


def get_train_test_data(vecs, prms, number_y, random_state=0):
    np.random.seed(random_state)

    msk_ = np.random.rand(vecs.shape[0]) < 0.75

    train_data_x_, test_data_x_ = vecs[msk_], vecs[~msk_]

    train_data_y_, test_data_y_ = {}, {}

    for idx_ in range(number_y):
        y = prms[:, idx_].ravel()
        train_data_y_[idx_] = y[msk_]
        test_data_y_[idx_] = y[~msk_]

    return train_data_x_, test_data_x_, train_data_y_, test_data_y_


def main():
    signal.signal(signal.SIGINT, sigint_handler)

    parser = setup_cmdline_parsing()
    args = parser.parse_args()

    log_handlers = [logging.StreamHandler()]
    if not args.debug:
        ts = datetime.datetime.fromtimestamp(time.time()).strftime(
            '%Y-%m-%d_%H:%M:%S')
        log_handlers += [logging.FileHandler(
            f"{os.getcwd()}/out/"
            f"{ts}_crocker_stack_svc_hyp-tun.log"
        )]

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers
    )


    logging.debug(f"Called using the following parameters: {args}")

    logging.info("Start loading data.")

    logging.info(f"Loading Parameter File: {args.prms_file}")
    if not os.path.isfile(args.prms_file):
        logging.fatal("PRMS file does not exist.")
        sys.exit(1)

    try:
        prms = torch.load(args.prms_file)
    except RuntimeError:
        with open(args.prms_file,  'rb') as fp:
            prms = pickle.load(fp)

    number_y = prms.shape[1]

    logging.info(f"Loading Crocker Stack Vector File: {args.vecs_file}")
    if not os.path.isfile(args.vecs_file):
        logging.fatal("Vecs file does not exist.")
        sys.exit(1)
    vecs = torch.load(args.vecs_file)

    logging.info("Data loaded")

    bayes_search = True
    if args.svr_c or args.svr_tol or args.data_scaling:
        logging.info("SVR Parameter supplied. No optimization performed.")
        bayes_search = False
        if not (args.svr_c and args.svr_tol and args.data_scaling):
            logging.fatal("One model parameter (C, tol, scaling) not supplied.")
            sys.exit(1)
        if len(args.svr_c) != len(args.svr_tol) != len(args.data_scaling):
            logging.fatal("Number of parameter (C, tol, scaling) do not match.")
            sys.exit(1)
        if len(args.svr_c) != number_y:
            logging.fatal("Number of parameter (C, tol, scaling) do not match to "
                          "the number of labels (y).")
            sys.exit(1)


    all_m, all_r = [], []

    for idx in range(number_y):
        if idx in args.skip_y:
            logging.warning(f"Skipping label {idx}, as excluded by cli-param.")
            continue

        if bayes_search:
            train_data_X, test_data_X, \
                train_data_y, test_data_y = get_train_test_data(vecs, prms, number_y)

            logging.info(f"Starting BayesSearchCV for {idx}")
            opt = get_bayes_search_opt(args)
            opt.fit(train_data_X, train_data_y[idx])

            logging.info(f"test score: {opt.score(test_data_X, test_data_y[idx])}")
            logging.info(f"best params: {opt.best_params_}")

            best_C = opt.best_params_['model__C']
            best_tol = opt.best_params_['model__tol']
            best_scaling = opt.best_params_['scaling']
        else:
            logging.info("Using supplied model parameter")
            best_C = args.svr_c[idx]
            best_tol = args.svr_tol[idx]
            best_scaling = args.data_scaling[idx]

        logging.debug(f"Evaluating Model for Parameter for y[{idx}]:"
                      f" C={best_C}, tol={best_tol}, scaling={best_scaling}")

        res = [
            evaluate(idx, args, scale=best_scaling is not None, c_=best_C,
                     tol_=best_tol,
                     random_seed=it_) for it_ in range(10)
        ]

        mse_ = np.array([r["mse"] for r in res])
        r2_ = np.array([r["r2"] for r in res])
        all_m.append(mse_)
        all_r.append(r2_)

        logging.info(f'MSE for best model (mean over 5 times) for '
                     f'Parameter {idx}: {np.mean(mse_)} +/- {np.std(mse_)}')
        logging.info(f'R2 for best model (mean over 5 times) for '
                     f'Parameter {idx}: {np.mean(r2_)} +/- {np.std(r2_)}')

    if all_m:
        logging.info(f"Final results over all labels: "
                     f"mse = {np.mean(np.array(all_m))} +/- {np.std(np.array(all_m))}, "
                     f"r^2 = {np.mean(np.array(all_r))} +/- {np.std(np.array(all_r))}")


if __name__ == '__main__':
    main()
