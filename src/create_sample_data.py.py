#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# create_sample_data.py.py
# @Author : Santosh Kesiraju (kcraj2@gmail.com)
# @Date   : 8/22/2019, 2:49:26 PM

"""
Create sample data from 20 Newsgroups
"""

import os
import codecs
import argparse
import numpy as np
import scipy.io as sio
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """ main method """

    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)

    # just 3 categories
    cats = ['alt.atheism', 'sci.space', 'rec.autos']
    train_ng = fetch_20newsgroups(subset='train', categories=cats)
    test_ng = fetch_20newsgroups(subset='test', categories=cats)

    cvect = CountVectorizer(strip_accents='ascii', min_df=2)
    train_dbyw = cvect.fit_transform(train_ng.data)
    test_dbyw = cvect.transform(test_ng.data)

    print("Train stats:", train_dbyw.shape, "labels:", train_ng.target.shape)
    print("Test stats :", test_dbyw.shape, "labels:", test_ng.target.shape)

    sio.mmwrite(args.out_dir + "train.mtx", train_dbyw)
    sio.mmwrite(args.out_dir + "test.mtx", test_dbyw)

    np.savetxt(args.out_dir + "train.labels", train_ng.target, fmt="%d")
    np.savetxt(args.out_dir + "test.labels", test_ng.target, fmt="%d")

    with codecs.open(args.out_dir + "vocab", "w", "utf-8") as fpw:
        fpw.write("\n".join(cvect.vocabulary_))

    with open(args.out_dir + "mtx.flist", "w") as fpw:
        fpw.write(os.path.realpath(args.out_dir) + "/train.mtx\n")
        fpw.write(os.path.realpath(args.out_dir) + "/test.mtx")

    print("Stats, labels, vocab saved in", args.out_dir)


def parse_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="path to output dir")
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    main()
