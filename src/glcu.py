#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 15 Feb 2018
# Last modified : 15 Feb 2018

"""
Gaussian Linear Classifier with Uncertainty (GLCU)
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import scipy.sparse as sparse
from scipy.special import logsumexp as lse


class GLCU:
    """ Gaussian Linear Classifier with uncertainty """

    def __init__(self, trn_iters=5, cov_type='diag', est_prior=False):
        """ Initialize model configuration """

        if cov_type != "diag":
            print("cov_type should be diag. Others not implemented.")
            sys.exit()

        self.cov_type = cov_type
        self.trn_iters = trn_iters
        self.est_prior = est_prior

        self.cmus = None
        self.scov = None
        self.m2l = None
        self.c_sizes = None
        self.dim = None
        self.priors = None

        self.tmp_dir = tempfile.mkdtemp() + "/"

    def compute_cmus_shared_cov(self, means, labels):
        """ Compute class means and shared covariance matrix """

        _, self.c_sizes = np.unique(labels, return_counts=True)
        self.c_sizes = self.c_sizes.astype(np.float32)
        noc = len(self.c_sizes)

        if self.est_prior:
            self.priors = (self.c_sizes /
                           self.c_sizes.sum()).astype(np.float32)
        else:
            self.priors = np.ones(shape=(noc, 1), dtype=np.float32) / noc

        N = means.shape[0]  # number of samples
        # means 2 label mapping
        self.m2l = sparse.csc_matrix((np.ones(N), (range(N), labels)),
                                     dtype=np.float32)
        self.cmus = self.m2l.T.dot(means) / self.c_sizes.reshape(-1, 1)

        # -- sanity check --
        # global_mu = np.mean(ivecs, axis=0).reshape(1, -1)
        # global_cov = np.cov(ivecs.T, bias=True)
        # tmp = (class_mus - global_mu) * np.sqrt(class_sizes.reshape(-1, 1))
        # acc = tmp.T.dot(tmp) / N
        # shared_cov = global_cov - acc  # OR weighted with-in class cov

        tmp = means - self.m2l.dot(self.cmus)
        self.scov = (tmp.T.dot(tmp) / N).astype(np.float32)

    def train(self, data, labels, covs=None):
        """ Train using EM

        Args:
            data (np.ndarray): If cov_type is `diag` then,
                shape is n_docs x [twice dim]. Every row is twice the
                dimension of latent variable, where the first half is mean,
                and second half is log std.dev). Otherwise data represents
                only means of shape n_docs x dim.
            labels (np.ndarray): Labels for n_docs. Shape is n_docs
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
                then data represents only mus and covs are passed as parameters
        """

        if min(labels) == 1:
            labels -= 1

        if self.cov_type == 'diag':
            self.dim = int(data.shape[1] / 2)
            mus = data[:, :self.dim]
            covs = np.exp(2 * data[:, self.dim:])

        else:
            self.dim = data.shape[1]
            mus = data

        self.compute_cmus_shared_cov(mus, labels)

        gammas = 1. / covs

        eye = np.eye(mus.shape[1], dtype=np.float32)  # Identity matrix

        spre = np.linalg.inv(self.scov)  # shared precision

        # Parameters of the posterior distribution of latent variables
        #  p(Z|W,Mus,S)
        alphas = np.zeros_like(mus, dtype=np.float32)
        lambdas = np.zeros(shape=(mus.shape[0], self.dim, self.dim),
                           dtype=np.float32)

        for _ in range(self.trn_iters):

            # E - step: get the posterior of latent variables Z
            # i.e, q(z_n) = N(z_n | alpha_n, Lambda_n.inv())
            # alpha_n = [I + D.inv().dot(Gamma_n)].inv().dot(w_n - mu_n)
            # Lambda_n = D + Gamma_n

            tmp = mus - self.m2l.dot(self.cmus)
            if self.cov_type == 'diag':
                for n in range(mus.shape[0]):
                    # alphas[n, :] = np.linalg.inv((self.scov * gammas[n, :])
                    #                             + eye).dot(tmp[n, :])
                    alphas[n, :] = np.linalg.solve(
                        (self.scov * gammas[n, :]) + eye, tmp[n, :])
                    lambdas[n, :, :] = spre + np.diag(gammas[n, :])
            else:
                for n in range(mus.shape[0]):
                    # alphas[n, :] = np.linalg.inv(
                    # self.scov.dot(gammas[n, :, :])
                    #                             + eye).dot(tmp[n, :])
                    alphas[n, :] = np.linalg.solve(
                        self.scov.dot(gammas[n, :, :]) + eye, tmp[n, :])

                lambdas = gammas + spre

            # M - step: maximize w.r.t params (class mus,
            #                                  shared precision matrix)
            # M.a maximize w.r.t. class mus
            # mu_l = (1/N_l) \sum_{n in l} (w_n - alpha_n)

            self.cmus = (self.m2l.T.dot(mus - alphas) /
                         self.c_sizes.reshape(-1, 1))

            # M.b maximize w.r.t. shared cov (precision) matrix
            # S = 1/N [ (\sum_n \Lambda_n.inv()) +
            #           (alpha_n - (w_n - mu_l)) (alpha_n - (w_n - mu_l)).T ]

            tmp = alphas - (mus - self.m2l.dot(self.cmus))
            self.scov = (np.linalg.inv(lambdas).sum(axis=0) +
                         tmp.T.dot(tmp)) / mus.shape[0]
            spre = np.linalg.inv(self.scov)

    def e_step(self, mus, covs, i, bsize):
        """ E step - get the posterior distribution of latent variables.

        Args:
        -----
            mus (np.ndarray): means of data points or embeddings
            covs (np.ndarray): covs of data points or embeddings
            i (int): training iteration number
            bsize (int): batch size

        Returns:
        --------
            np.ndarray (alphas): Means of posterior dist. of latent vars.
        """

        eye = np.eye(mus.shape[1], dtype=np.float32)  # Identity matrix

        gammas = 1. / covs

        spre = np.linalg.inv(self.scov).astype(np.float32)  # shared precision

        # Parameters of the posterior distribution of latent variables
        # p(Z|W,Mus,S) = N(Z | alphas, Lambdas)

        alphas = np.zeros_like(mus, dtype=np.float32)

        # E - step: get the posterior of latent variables
        #           q(z_n) ~ N(alpha_n, Lambda_n.inv())
        # alpha_n = [I + D.inv().dot(Gamma_n)].inv().dot(w_n - mu_n)
        # Lambda_n = D + Gamma_n

        tmp = mus - self.m2l.dot(self.cmus)

        sdoc = 0
        edoc = bsize

        bno = 0  # batch number
        while sdoc < edoc:

            lambdas_batch = np.zeros(shape=(edoc-sdoc, self.dim, self.dim),
                                     dtype=np.float32)
            # print('\r    e-step batch num \
            # {:2d}/{:2d}'.format(bno, mus.shape[0]//bsize), end=" ")

            if self.cov_type == 'diag':

                for n in range(sdoc, edoc, 1):
                    alphas[n, :] = np.linalg.solve((
                        self.scov * gammas[n, :]) + eye, tmp[n, :])
                    lambdas_batch[n-sdoc, :, :] = spre + np.diag(gammas[n, :])

                np.save(self.tmp_dir + "lambdas_" + str(bno) + "_" + str(i),
                        lambdas_batch)

            else:

                for n in range(sdoc, edoc, 1):
                    alphas[n, :] = np.linalg.solve(
                        self.scov.dot(gammas[n, :, :]) + eye, tmp[n, :])
                    lambdas_batch[n-sdoc, :, :] = gammas[n, :, :] + spre

                np.save(self.tmp_dir + "lambdas_" + str(bno) + "_" + str(i),
                        lambdas_batch)

            bno += 1
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        return alphas

    def m_step(self, mus, alphas, i, bsize):
        """  M - step: maximize w.r.t params (class mus, shared precision matrix)

            # 1. maximize w.r.t. class mus
            # mu_l = (1/N_l) SUM{n in Class_l} (w_n - alpha_n)

            self.cmus = (self.m2l.T.dot(mus - alphas) /
                         self.c_sizes.reshape(-1, 1))

            # 2. maximize w.r.t. shared cov (precision) matrix
            # S = 1/N (SUM_n Lambda_n.inv()) +
            #               (alpha_n - (w_n - mu_l)) (alpha_n - (w_n - mu_l)).T

        """

        # 1 maximizing w.r.t. class means
        self.cmus = self.m2l.T.dot(mus - alphas) / self.c_sizes.reshape(-1, 1)

        # 2 maximizing w.r.t. shared covariance
        self.scov = np.zeros_like(self.scov, dtype=np.float32)

        sdoc = 0
        edoc = bsize
        bno = 0

        tmp = alphas - (mus - self.m2l.dot(self.cmus))
        while sdoc < edoc:

            # print('\r    m-step batch num {:2d}'.format(bno), end=" ")
            lambdas_batch = np.load(self.tmp_dir + "lambdas_" + str(bno) +
                                    "_" + str(i) + ".npy").astype(np.float32)

            self.scov += (np.linalg.inv(lambdas_batch).sum(axis=0) +
                          tmp[sdoc:edoc, :].T.dot(tmp[sdoc:edoc, :]))

            bno += 1
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        self.scov /= mus.shape[0]

    def train_b(self, data, labels, covs=None, bsize=4096):
        """ Train using EM

        Args:
            data (np.ndarray): If cov_type is `diag` then,
                shape is n_docs x [twice dim]. Every row is twice the
                dimension of latent variable, where the first half is mean,
                and second half is log std.dev). Otherwise data represents
                only means of shape n_docs x dim.
            labels (np.ndarray): Labels for n_docs. Shape is n_docs
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
                then data represents only means and covs are passed as
                parameters
            bsize (int): batch size
        """

        if min(labels) == 1:
            labels -= 1

        if self.cov_type == 'diag':
            self.dim = int(data.shape[1] / 2)
            mus = (data[:, :self.dim]).astype(np.float32)
            covs = np.exp(2 * data[:, self.dim:]).astype(np.float32)

        else:
            self.dim = data.shape[1]
            mus = data.astype(np.float32)

        self.compute_cmus_shared_cov(mus, labels)

        if bsize > mus.shape[0]:
            # if batch size is > no. of examples
            bsize = mus.shape[0]

        for i in range(self.trn_iters):

            # print('Iter :{:d}'.format(i))

            alphas = self.e_step(mus, covs, i, bsize)
            # print()
            self.m_step(mus, alphas, i, bsize)
            # print()

            os.system("rm -rf " + self.tmp_dir + "/*.npy")

    def predict(self, data, return_labels=False, covs=None):
        """ Predict posterior probabilities or labels given the
             test data (means and covs).

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape
                is `n_docs x twice_dim` (every row is twice the dimension
                of latent variable, where the first half is mean,
                and second half is log std.dev). Otherwise data represents
                only means of shape n_docs x dim.
            return_labels (boolean): Returns labels if True,
                else returns log-likelihoods
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is
                full then data represents only means and covs are passed
                as parameters

        Returns:
            labels (np.ndarray):  Posterior probabilities or predicted
                labels for `n_docs`
        """

        if self.cov_type == 'diag':
            mus = data[:, :self.dim]
            covs = np.exp(2 * data[:, self.dim:])

            tot_covs = np.zeros(shape=(mus.shape[0], self.dim, self.dim),
                                dtype=np.float32)
            for n in range(mus.shape[0]):
                tot_covs[n, :, :] = self.scov + np.diag(covs[n, :])

        else:
            mus = data
            tot_covs = self.scov + covs

        # inv_tot_covs = np.linalg.inv(tot_covs)
        sgn, log_dets = np.linalg.slogdet(tot_covs)

        const = -0.5 * self.dim * np.log(2 * np.pi)

        if (sgn < 0).any():
            print("WARNING: Det of tot_covs is Negative.")
            sys.exit()

        # class conditional log-likelihoods
        cc_llh = np.zeros(shape=(mus.shape[0], self.cmus.shape[0]),
                          dtype=np.float32)

        for n in range(mus.shape[0]):  # for each doc
            tmp = self.cmus - mus[n, :]
            z = (tmp.reshape(tmp.shape[0], tmp.shape[1], 1) @
                 tmp.reshape(tmp.shape[0], 1, tmp.shape[1]))
            cc_llh[n, :] = const - (
                0.5 * (log_dets[n] + (z * np.linalg.inv(
                    tot_covs[n, :, :])).sum(axis=1).sum(axis=1)))

            # cc_llh[n, :] = -log_dets[n]
            # for i in range(self.cmus.shape[0]):  # given a class, i
            #    cc_llh[n, i] -= (np.outer(tmp[i, :], tmp[i, :]) *
            #                     inv_tot_covs[n, :, :]).sum()

        if self.est_prior:
            if return_labels:
                ret_val = np.argmax(cc_llh + np.log(self.priors).T, axis=1)
            else:
                cc_llh += np.log(self.priors).T
                ret_val = np.exp(cc_llh.T - lse(cc_llh, axis=1)).T

        else:
            if return_labels:
                ret_val = np.argmax(cc_llh, axis=1)
            else:
                ret_val = np.exp(cc_llh.T - lse(cc_llh, axis=1)).T

        return ret_val

    def predict_b(self, data, return_labels=False, covs=None, bsize=1500):
        """ Predict posterior probabilities or labels given the test data
            (means and covs).

        Args:
            data (np.ndarray): If cov_type is `diag` then, shape is
            `n_docs x [twice dim]`
            (every row is twice the dimension of latent variable, where the
             first half is mean, and second half is log std.dev).
             Otherwise data represents only means of shape n_docs x dim.
            return_labels (boolean): Returns labels if True,
                else returns log-likelihoods
            covs (np.ndarray): Shape is n_docs x dim x dim. If cov_type is full
            then data represents only means and covs are passed as parameters
            bsize (int): batch size

        Returns:
            labels (np.ndarray): Posterior probabilities or predicted labels
                for n_docs.
        """

        if bsize > data.shape[0]:
            bsize = data.shape[0]

        mus = (data[:, :self.dim]).astype(np.float32)

        # class conditional log-likelihoods
        cc_llh = np.zeros(shape=(mus.shape[0], self.cmus.shape[0]),
                          dtype=np.float32)

        const = -0.5 * self.dim * np.log(2 * np.pi)

        sdoc = 0
        edoc = bsize

        while sdoc < edoc:

            tot_covs = np.zeros(shape=(edoc-sdoc, self.dim, self.dim),
                                dtype=np.float32)

            if self.cov_type == 'diag':
                #  covs = np.exp(2 * data[:, self.dim:])
                for n in range(sdoc, edoc, 1):
                    tot_covs[n-sdoc, :, :] = (self.scov +
                                              np.diag(np.exp(
                                                  2. * data[n, self.dim:])))

            else:
                tot_covs = self.scov + covs[sdoc:edoc, :, :]

            # inv_tot_covs = np.linalg.inv(tot_covs)
            sgn, log_dets = np.linalg.slogdet(tot_covs)

            if (sgn < 0).any():
                print("WARNING: Det of tot_covs is Negative.")
                sys.exit()

            for n in range(sdoc, edoc, 1):  # for each doc in the batch
                tmp = self.cmus - mus[n, :]
                z = (tmp.reshape(tmp.shape[0], tmp.shape[1], 1) @
                     tmp.reshape(tmp.shape[0], 1, tmp.shape[1]))
                cc_llh[n, :] = const - (0.5 *
                                        (log_dets[n-sdoc] +
                                         (z * np.linalg.inv(
                                             tot_covs[n-sdoc, :, :])
                                         ).sum(axis=1).sum(axis=1)))
            sdoc += bsize
            edoc += bsize
            if edoc > mus.shape[0]:
                edoc = mus.shape[0]

        if self.est_prior:
            if return_labels:
                ret_val = np.argmax(cc_llh + np.log(self.priors).T, axis=1)
            else:
                cc_llh += np.log(self.priors).T
                ret_val = np.exp(cc_llh.T - lse(cc_llh, axis=1)).T

        else:
            if return_labels:
                ret_val = np.argmax(cc_llh, axis=1)
            else:
                ret_val = np.exp(cc_llh.T - lse(cc_llh, axis=1)).T

        return ret_val


def main():
    """ main method """

    np.random.seed(0)
    N, K = 5, 2
    means = np.random.randn(N, K)
    sigmas = np.log(1 / np.sqrt(np.random.gamma(shape=1.5, scale=1.5,
                                                size=(N, K))))

    labels = np.random.randint(0, 2, size=(N))
    data = np.concatenate((means, sigmas), axis=1)

    glcu = GLCU(ARGS.trn, cov_type='diag')
    glcu.train(data, labels)

    glcu2 = GLCU(ARGS.trn, cov_type="diag")
    glcu2.train_b(data, labels, bsize=ARGS.bs)

    print('---------')
    print(glcu.scov)
    print(glcu2.scov)
    print(np.allclose(glcu.scov, glcu2.scov))

    print('----------')
    print(glcu.predict(data))
    print(glcu2.predict_b(data))
    print(np.allclose(glcu.predict(data), glcu2.predict_b(data,
                                                          bsize=ARGS.bs)))

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument('-trn', type=int, default=5)
    PARSER.add_argument('-bs', type=int, default=5)
    ARGS = PARSER.parse_args()
    main()
