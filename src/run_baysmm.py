#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# author : Santosh Kesiraju
# e-mail : kcraj2[AT]gmail[DOT]com
# Date created : 22 Oct 2018
# Last modified : 22 Oct 2018

"""
Train Bayesian SMM and extract parameters of
document i-vector (embedding) posterior distributions.
"""

import os
import re
import sys
import json
import logging
import argparse
import traceback
from time import time
import scipy.io as sio
import torch
import torch.optim
import numpy as np
from baysmm import BaySMM
import utils


def get_mtx_list_for_extraction(extract_list):
    """ Get mtx file list for which i-vectors needs to be extracted """

    ext = os.path.basename(extract_list).split(".")[-1]

    mtx_list = []

    if ext == "mtx":
        mtx_list = [extract_list]
    else:
        mtx_list = utils.read_simple_flist(extract_list)

    return mtx_list


def check_and_load_training_data(args):
    """ Check and load training data """

    error_flag = False

    mtx_file = args.mtx_file
    stats_mtx = sio.mmread(mtx_file).tocsc()

    vocab = utils.read_simple_flist(args.vocab_file)
    if len(vocab) == stats_mtx.shape[0]:
        stats_mtx = stats_mtx.T

    if len(vocab) != stats_mtx.shape[1]:
        print("Error: Vocabulary size ({:d}) does not match".format(len(vocab)),
              "dimension 1 of stats ({:d})".format(stats_mtx.shape[1]))
        error_flag = True

    labels = None
    if error_flag:
        sys.exit()

    return stats_mtx, vocab, labels


def create_model(train_mtx, config, args):
    """ Create a new SMM model or load an existing model if found. """

    if config['latest_trn_model']:

        params = utils.load_params(config['latest_trn_model'])

        model = BaySMM(params['m'], config, args.cuda)
        model.T.data = params['T']
        model.Q.data = params['Q']
        model.T.requires_grad_(True)
        model.m.requires_grad_(True)
        model.Q.requires_grad_(True)

        logging.info("Loaded parameters from: %s", config['latest_trn_model'])

    else:
        ubm = utils.estimate_ubm(train_mtx.T.tocsc())
        logging.info("UBM initialized.")
        model = BaySMM(ubm, config, cuda=config['cuda'])
        logging.info("Model created.")

    if config['cuda']:
        model.to_device(torch.device("cuda"))

    return model


def create_optimizers(model, config):
    """ Create optimizers for SMM model """

    if config['optim'] == 'adagrad':
        torch_optim = torch.optim.Adagrad
    else:
        torch_optim = torch.optim.Adam

    if config['update_ubm']:
        opt_t = torch_optim([model.T, model.m], lr=config['eta_t'])
    else:
        opt_t = torch_optim([model.T], lr=config['eta_t'])

    opt_q = torch_optim([model.Q], lr=config['eta_q'])
    optims = {'Q': opt_q, 'T': opt_t}

    return optims


def batch_wise_training(model, optims, dset, config, args):
    """ Batch-wise training by updating all parameters at once
    (no EM style alternating updates) """

    logging.info('Batch-wise training on %d %s', config['n_docs'], 'docs')

    loss_iters = torch.zeros(1, 2).to(dtype=model.dtype, device=model.device)

    loss, kld = model.compute_total_loss_batch_wise(dset, args.nb,
                                                    use_params='all')
    logging.info("Initial ELBO: %.1f %s: %.1f", -loss.detach().cpu().numpy(),
                 "KLD", kld.detach().cpu().numpy())
    torch.cuda.empty_cache()

    loss_iters[0, 0] = loss
    loss_iters[0, 1] = kld

    for i in range(config['trn_done'], config['trn_iters']):

        loss_sum = torch.Tensor([0., 0.]).to(device=model.device,
                                             dtype=torch.float).view(1, -1)

        stime = time()

        for _, data_dict in enumerate(dset.yield_batches(args.nb)):

            optims['Q'].zero_grad()
            optims['T'].zero_grad()

            loss, kld = model.compute_loss(data_dict, use_params='all')

            loss.backward()

            loss_sum += torch.Tensor(
                [loss.data.clone(), kld.data.clone()]).to(
                    device=model.device).view(1, -1)

            optims['Q'].step()

            if model.config['hyper']['reg_t'] == 'l1':
                model.orthant_projection(optims['T'])
            else:
                optims['T'].step()

            torch.cuda.empty_cache()

            model.sample()

        loss_iters = torch.cat((loss_iters, loss_sum))

        logging.info("Iter: %4d/%4d %s: %.1f %s: %.1f %s: %.2f", i+1,
                     config['trn_iters'], "ELBO",
                     -loss_sum[0, 0].detach().cpu().numpy(), "KLD",
                     loss_sum[0, 1].detach().cpu().numpy(), "Time per iter",
                     (time() - stime))

        model.config['trn_done'] = i+1
        if (i+1) % config['save'] == 0:
            utils.save_model(model)

    model.Q.requires_grad_(False)
    model.T.requires_grad_(False)
    model.m.requires_grad_(False)

    loss_iters = loss_iters.cpu().numpy()

    loss, kld = model.compute_total_loss_batch_wise(dset, args.nb,
                                                    use_params='all')
    torch.cuda.empty_cache()

    loss_iters = np.concatenate((loss_iters,
                                 np.asarray([loss.cpu().item(),
                                             kld.cpu().item()]).reshape(1, -1)))
    return model, loss_iters


def train_model(args):
    """ Train model """

    train_mtx, vocab, train_labels = check_and_load_training_data(args)

    # -- configuration --

    config = utils.create_baysmm_config(args)

    config['vocab_size'] = len(vocab)
    config['n_docs'] = train_mtx.shape[0]
    config['dtype'] = 'float'

    # -- end of configuration --

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        filename=config['exp_dir'] + 'training.log', filemode='a',
                        level=getattr(logging, args.log.upper(), None))
    print("Log file:", config['exp_dir'] + 'training.log')
    if args.v:
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('PyTorch version: %s', str(torch.__version__))

    model = create_model(train_mtx, config, args)

    # params = utils.load_params(eng_model_h5_file)
    # model.Q.data = params['Q'].to(device=model.device)

    # if config['cuda']:
    #    utils.estimate_approx_num_batches(model, train_mtx)

    if args.trn <= config['trn_done']:
        logging.info('Found model that is already trained.')
        return

    config['trn_iters'] = args.trn

    optims = create_optimizers(model, config)

    # optims['Q'] = None

    utils.save_config(config)

    dset = utils.SMMDataset(train_mtx, train_labels, len(vocab), 'unsup')

    dset.to_device(model.device)

    if args.batchwise:

        model, loss_iters = batch_wise_training(model, optims, dset,
                                                config, args)

    else:

        logging.info('Training on {:d} docs.'.format(config['n_docs']))
        loss_iters = model.train_me(dset, optims, args.nb)

    t_sparsity = utils.t_sparsity(model)

    utils.write_info(model.config, "Sparsity in T: {:.2f}".format(t_sparsity))

    logging.info("Initial ELBO: {:.1f}".format(-loss_iters[0][0]))
    logging.info("  Final ELBO: {:.1f}".format(-loss_iters[-1][0]))
    logging.info("Sparsity in T: {:.2f}".format(t_sparsity))
    utils.save_model(model)

    base = os.path.basename(args.mtx_file).split('.')[0]
    sfx = "_T{:d}".format(config['trn_done'])
    utils.save_loss(loss_iters, model.config, base, sfx)
    # utils.plot_loss(loss_iters, model.config, base)


def extract_ivector_posteriors(args):
    """ Extract posterior distribution of i-vectors using existing model """

    # -- configuration --

    cfg_f = os.path.dirname(os.path.realpath(args.model_f)) + "/config.json"
    config = json.load(open(cfg_f, 'r'))
    os.makedirs(config['tmp_dir'] + 'ivecs/', exist_ok=True)
    config['xtr_done'] = 0
    config['xtr_iters'] = args.xtr
    config['nth'] = args.nth

    # -- end of configuration

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        filename=config['exp_dir'] + 'extraction.log',
                        level=getattr(logging, args.log.upper(), None))
    print("Log file:", config['exp_dir'] + 'extraction.log')
    if args.v:
        logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('PyTorch version: %s', str(torch.__version__))

    mtx_list = get_mtx_list_for_extraction(args.extract_list)

    logging.info("Number of files for extraction: %d", len(mtx_list))

    params = utils.load_params(args.model_f)

    for mtx_file in mtx_list:

        data_mtx = sio.mmread(mtx_file).tocsc()
        if data_mtx.shape[0] == config['vocab_size']:
            data_mtx = data_mtx.T

        sbase = os.path.basename(mtx_file).split(".")[0]
        mbase = os.path.basename(args.model_f).split(".")[0]

        out_file = config['ivecs_dir'] + sbase + "_" + mbase + "_e"
        out_file += str(config['xtr_iters']) + ".h5"

        if os.path.exists(out_file):
            logging.info("i-vector posterior distributions were %s %s",
                         "already extracted and saved in", out_file)
            continue

        logging.info('Extracting i-vectors for %d %s', data_mtx.shape[0], 'docs')

        # Create model and copy existing parameters
        model = BaySMM(params['m'], config, args.cuda)
        model.T.data = params['T']
        model.T.requires_grad_(False)
        model.m.requires_grad_(False)

        # Create dataset object
        dset = utils.SMMDataset(data_mtx, None, config['vocab_size'], 'unsup')
        dset.to_device(model.device)

        # Reset i-vector posterior parameters
        model.init_var_params(data_mtx.shape[0])

        # move model to device (CUDA if available)
        model.to_device(model.device)

        # Create optimizer
        if config['optim'] == 'adam':
            opt_e = torch.optim.Adam([model.Q], lr=config['eta_q'])
        else:
            opt_e = torch.optim.Adagrad([model.Q], lr=config['eta_q'])

        # extract
        loss_iters = model.extract_ivector_posteriors(dset, opt_e, sbase,
                                                      args.nb)

        sfx = "_" + mbase + "_e{:d}".format(config['xtr_iters'])
        utils.save_loss(loss_iters, model.config, "xtr_" + sbase, sfx)

        # utils.merge_ivecs(config, sbase, mbase, config['xtr_iters'])
        utils.merge_ivecs_v2(config, sbase, mbase, config['xtr_iters'], args.nb)


def main():
    """ main method """

    args = parse_arguments()

    gpu_id = -1

    while True:

        try:

            stime = time()

            if args.cuda and gpu_id == -1:
                gpu_ids = utils.get_free_gpus()
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
                print("Using GPU ID:", gpu_ids[0])

            if args.phase == 'train':

                train_model(args)

            else:

                if os.path.exists(args.model_f):
                    extract_ivector_posteriors(args)
                else:
                    print(args.model_f, "NOT FOUND.")
                    sys.exit()

            print(".. done {:.2f} sec".format(time() - stime))
            break

        except RuntimeError as err:

            torch.cuda.empty_cache()

            if re.search(r"CUDA out of memory", str(err)):
                print("\n" + str(err),
                      "\nIncreasing the number of batches to", end=" ")
                args.nb = args.nb + 2
                print(args.nb)

            else:
                print('{0}'.format(str(err)))
                traceback.print_tb(err.__traceback__)
                sys.exit()

            continue


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    sub_parsers = parser.add_subparsers(help='phase (train or extract)',
                                        dest='phase')
    sub_parsers.required = True

    train_parser = sub_parsers.add_parser('train', help='train i-vector model\
                                                   on the stats (mtx_file)')

    train_parser.add_argument("mtx_file", help="path to Doc-by-Word stats \
                                               (scipy.sparse format)")

    xtr_parser = sub_parsers.add_parser('extract',
                                        help="extract i-vectors for the given \
                                        stats (mtx_file) using a specified \
                                        existing model")
    xtr_parser.add_argument("extract_list", help="path to Doc-by-Word stats \
                                                  (scipy.sparse format) or \
                                                  list of mtx files")

    train_parser.add_argument("vocab_file", help="path to vocabulary file")
    train_parser.add_argument("out_dir", help="path to output directory")

    train_parser.add_argument("-var_p", type=float, default=10.0,
                              help="initial precision of var. dist (default: %(default)s)")
    train_parser.add_argument("-R", default=1, type=int,
                              help="no. of samples for re-parametrization \
                              (default: %(default)s)")
    train_parser.add_argument("-K", default=50, type=int,
                              help="i-vector dim (default: %(default)s)")
    train_parser.add_argument("-lw", default=1e+00, type=float,
                              help="prior precision for i-vectors \
                              (default: %(default)s)")
    train_parser.add_argument("-rt", default="l1",
                              help="l1 or l2 regularization for bases T \
                              (default: %(default)s)")
    train_parser.add_argument("-lt", default=1e-4, type=float,
                              help='reg. constant for bases T \
                              (default: %(default)s)')
    train_parser.add_argument("-optim", default='adam',
                              choices=['adam', 'adagrad'],
                              help="choice of optimizer")
    train_parser.add_argument("-eta_q", type=float, default=0.001,
                              help="learning rate for i-vector variational \
                              dist. (default: %(default)s)")
    train_parser.add_argument("-eta_t", type=float, default=0.001,
                              help='learning rate for T \
                              (default: %(default)s)')
    train_parser.add_argument("-s", type=float, default=1.0,
                              help='scale bag-of-words statistics')
    train_parser.add_argument("-trn", type=int, default=2000,
                              help="number of training iterations \
                              (default: %(default)s)")
    train_parser.add_argument('-nb', default=1, type=int,
                              help='number of batches (default: %(default)s)')
    train_parser.add_argument('-save', default=1000, type=int,
                              help='save every nth intermediate model \
                              (default: %(default)s)')
    train_parser.add_argument('-mkl', default=1, type=int,
                              help='number of MKL threads \
                              (default: %(default)s)')
    train_parser.add_argument('-log', choices=['info', 'debug', 'warning'],
                              default='INFO', help='logging level')
    train_parser.add_argument('--batchwise', action='store_true',
                              help='Batch-wise training for large datasets \
                              (default: %(default)s)')
    train_parser.add_argument('--update_ubm', action='store_true',
                              help='Update UBM during training \
                              (default: %(default)s)')
    train_parser.add_argument('--ovr', action='store_true',
                              help='over-write the exp dir \
                              (default: %(default)s)')
    train_parser.add_argument("--nocuda", action='store_true',
                              help='Do not use GPU (default: %(default)s)')
    train_parser.add_argument('--v', action='store_true', help='verbose \
                              (default: %(default)s)')

    xtr_parser.add_argument("model_f", help="path to trained model file")
    xtr_parser.add_argument("-xtr", type=int, default=2000, help="number of \
                            extraction iterations (default: %(default)s)")
    xtr_parser.add_argument("-nth", type=int, default=20,
                            help="save every nth extracted i-vector \
                            (default: %(default)s)")
    xtr_parser.add_argument('-nb', default=1, type=int,
                            help='number of batches \
                            (default: %(default)s)')
    xtr_parser.add_argument('-mkl', default=1, type=int,
                            help='number of MKL threads \
                            (default: %(default)s)')
    xtr_parser.add_argument('-log', choices=['INFO', 'DEBUG', 'WARNING'],
                            default='INFO', help='logging level')
    xtr_parser.add_argument("--nocuda", action='store_true',
                            help='Do not use GPU \
                            (default: %(default)s)')
    xtr_parser.add_argument('--v', action='store_true',
                            help='verbose (default: %(default)s)')

    args = parser.parse_args()

    torch.set_num_threads(args.mkl)
    torch.manual_seed(0)
    args.cuda = not args.nocuda and torch.cuda.is_available()

    return args

if __name__ == "__main__":

    if int(torch.__version__.split(".")[1]) < 4 and int(torch.__version__.split(".")[0]) == 0:
        print("Requires pytorch version > 0.4.0",
              "Current version is:", torch.__version__)
        sys.exit()

    torch.manual_seed(0)

    main()
