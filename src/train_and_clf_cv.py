#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 30 Nov 2017
# Last modified : 22 Aug 2019

"""
Gaussian linear classifier or multi-class logistic regression on i-vectors.
Gaussian linear classifier with uncertainty on i-vector posterior dists.
"""

import os
import sys
import argparse
from time import time
import h5py
import numpy as np
import scipy
import sklearn
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.linear_model import LogisticRegressionCV
from glcu import GLCU
from glc import GLC


def kfold_cv_train_set(train_feats, train_labels, args):
    """ Run k-fold cross validation on train set

    Args:
        train_feats (np.ndarray): training features (n_samples x dim)
        train_labels (np.ndarray): corresponding labels
        args: args

    Returns:
        np.float64: average classification accuracy over k-folds
        np.float64: average cross-entropy loss over k-folds
    """

    skf = SKFold(n_splits=args.nf, shuffle=True, random_state=0)

    # [acc, x_entropy]
    scores = np.zeros(shape=(args.nf, 2))
    i = 0
    for trn_ixs, dev_ixs in skf.split(train_feats, train_labels):

        (_, _, acc, xen), _, _ = run_clf(train_feats[trn_ixs],
                                         train_labels[trn_ixs],
                                         train_feats[dev_ixs],
                                         train_labels[dev_ixs], args)

        scores[i, :2] = acc, xen

        i += 1

    return np.mean(scores[:, 0]), np.mean(scores[:, 1])


def run_clf(train_feats, train_labels, test_feats, test_labels, args):
    """ Train and classify using Gaussian linear classifier or
    multi-class logistic regression """

    if args.clf == 'glc':
        glc = GLC(est_prior=True)
        glc.train(train_feats, train_labels)

        train_pred = glc.predict(train_feats)
        train_prob = glc.predict(train_feats, return_probs=True)
        test_pred = glc.predict(test_feats)
        test_prob = glc.predict(test_feats, return_probs=True)

    elif args.clf == "lr":
        mclr = LogisticRegressionCV(Cs=[0.01, 0.1, 0.2, 0.5, 1.0, 10.0],
                                    multi_class='multinomial', cv=5,
                                    random_state=0, n_jobs=1,
                                    class_weight='balanced',
                                    max_iter=3000)
        mclr.fit(train_feats, train_labels)

        train_pred = mclr.predict(train_feats)
        train_prob = mclr.predict_proba(train_feats)

        test_pred = mclr.predict(test_feats)
        test_prob = mclr.predict_proba(test_feats)

    else:

        glcu = GLCU(args.trn, cov_type='diag', est_prior=True)

        glcu.train_b(train_feats, train_labels, args.bs)

        train_prob = glcu.predict_b(train_feats, return_labels=False,
                                    bsize=args.bs)
        train_pred = np.argmax(train_prob, axis=1)

        test_prob = glcu.predict_b(test_feats, return_labels=False,
                                   bsize=args.bs)
        test_pred = np.argmax(test_prob, axis=1)

    train_acc = np.mean(train_labels == train_pred) * 100.
    train_xen = log_loss(train_labels, train_prob)
    test_acc = np.mean(test_labels == test_pred) * 100.
    test_xen = log_loss(test_labels, test_prob)

    return (train_acc, train_xen, test_acc, test_xen), test_pred, test_prob


def print_and_save_scores(scores, res_f, ovr):
    """ Print and save scores """

    # print('scores:', scores.shape)

    dev_acc_ix = np.argmax(scores[:, 0])
    dev_xen_ix = np.argmin(scores[:, 1])

    print("      dev_A dev_X   trn_A  trn_X  test_A  test_X  xtr_iter")
    print("acc {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} \
{:.4f} {:.0f}".format(*scores[dev_acc_ix]))
    print("xen {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} \
{:.4f} {:.0f}".format(*scores[dev_xen_ix]))

    header = "\ndev_acc,dev_xen,train_acc,train_xen,test_acc,test_xen,xtr_iter"
    if ovr:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(res_f, mode) as fpw:
        np.savetxt(fpw, scores, fmt='%.4f', header=header)
    print("Saved to", res_f)

    with open(res_f.replace('results_', 'best_score_'), 'w') as fpw:
        np.savetxt(fpw, scores[dev_acc_ix].reshape(1, -1),
                   fmt='%.4f', header=header[1:])


def run(train_h5, test_h5, max_iters, args):
    """ Train and classify for every iteration of extracted i-vector """

    # columns in each row:
    # dev_acc, dev_xen, train_acc, train_xen, test_acc, test_xen, xtr_iter
    scores = []

    train_labels = np.loadtxt(args.train_label_f, dtype=int)
    test_labels = np.loadtxt(args.train_label_f.replace("train", "test"),
                             dtype=int)

    if min(train_labels) == 1:
        train_labels -= 1
    if min(test_labels) == 1:
        test_labels -= 1

    if args.final:
        args.start = max_iters

    for i in range(args.start, max_iters+1):

        if str(i) not in train_h5:
            continue

        score = [0, 0, 0, 0, 0, 0, i]

        train_feats = train_h5.get(str(i))[()]
        test_feats = test_h5.get(str(i))[()]

        if train_feats.shape[0] != train_labels.shape[0]:
            train_feats = train_feats.T
        if test_feats.shape[0] != test_labels.shape[0]:
            test_feats = test_feats.T

        if args.clf in ("glc", "lr"):
            # Get only the mean parameter from the post. dist
            dim = train_feats.shape[1] // 2
            train_feats = train_feats[:, :dim]
            test_feats = test_feats[:, :dim]

        score[:2] = kfold_cv_train_set(train_feats, train_labels, args)

        score[2:6], test_pred, test_prob = run_clf(train_feats, train_labels,
                                                   test_feats, test_labels,
                                                   args)
        scores.append(score)

        if args.verbose:
            print("i-vec xtr no: {:4d}/{:4d}".format(i, max_iters), end=" ")
            print("{:.2f} {:.4f} {:.2f} {:.4f} {:.2f} \
{:.4f} {:.0f}".format(*score))

    return np.asarray(scores), np.asarray(test_pred, dtype=int), test_prob


def main():
    """ main method """

    stime = time()

    args = parse_arguments()

    train_ivecs_h5f = args.train_ivecs_h5

    if not os.path.exists(train_ivecs_h5f):
        print(train_ivecs_h5f, "not found.")
        sys.exit()

    test_ivecs_h5f = train_ivecs_h5f.replace("train_", "test_")
    if not os.path.exists(test_ivecs_h5f):
        print(test_ivecs_h5f, "not found.")
        sys.exit()

    max_iters = int(os.path.splitext(os.path.basename(
        train_ivecs_h5f))[0].split("_")[-1][1:])

    mbase = os.path.splitext(
        os.path.basename(train_ivecs_h5f))[0].split("_")[-2]
    print('max_iters :', max_iters, 'model_base:', mbase)

    try:

        train_h5f = h5py.File(train_ivecs_h5f, 'r')
        train_h5 = train_h5f.get('ivecs')

        test_h5f = h5py.File(test_ivecs_h5f, 'r')
        test_h5 = test_h5f.get('ivecs')

        # results file
        res_f = os.path.realpath(os.path.dirname(train_ivecs_h5f) + "/../")

        sfx = "_" + mbase + "_" + str(max_iters) + "_cv"

        if args.final:
            sfx += "_final"

        res_f += "/results/results_" + args.clf + sfx + ".txt"

        if os.path.exists(res_f) and args.ovr is False:
            print(res_f, 'already EXISTS.')
            sys.exit()

        scores, test_pred, test_prob = run(train_h5, test_h5, max_iters, args)

        print_and_save_scores(scores, res_f, args)

        np.savetxt(res_f.replace("results_", "test_pred_"),
                   test_pred.reshape(-1, 1), fmt="%d")
        np.savetxt(res_f.replace("results_", "test_prob_"),
                   test_prob, fmt="%f")

    except IOError as err:
        print(err)

    finally:
        train_h5f.close()
        test_h5f.close()

    print("== Done: {:.2f} sec ==".format(time() - stime))


def parse_arguments():
    """ parse command line args """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_ivecs_h5", help="path to train_ivecs.h5 file")
    parser.add_argument("train_label_f", help="path to train label file")
    parser.add_argument("clf", default="glc", type=str,
                        choices=["glcu", "glc", "lr"],
                        help="Choice of classifier: glcu or glc or lr")
    parser.add_argument("-trn", type=int, default=10,
                        help='GLCU training iters')
    parser.add_argument("-bs", type=int, default=1500,
                        help="batch size for training GLCU")
    parser.add_argument("-nf", type=int, default=5,
                        help="Number of folds for k-fold CV")
    parser.add_argument("-start", type=int, default=1, help="start iter num")
    parser.add_argument("-mkl", default="1", help="MKL threads")
    parser.add_argument("--final", action="store_true",
                        help="use only final iteration of emebddings")
    parser.add_argument("--ovr", action="store_true",
                        help="over-write results file")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--versions", action="store_true", help="verbose")
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = args.mkl
    os.environ['MKL_NUM_THREADS'] = args.mkl

    if args.versions:
        versions()

    return args

def versions():
    """ Print versions of packages """

    print("python :", sys.version)
    print("numpy  :", np.__version__)
    print("scipy  :", scipy.__version__)
    print("h5py   :", h5py.__version__)
    print("sklearn:", sklearn.__version__)

if __name__ == "__main__":

    main()
