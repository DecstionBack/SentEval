# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
from exutil import dotdict
import argparse
import logging
from os.path import join as pjoin

reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='DisSent SentEval Evaluation')
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID, we map all model's gpu to this id")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# Set PATHs
GLOVE_PATH = '/home/anie/glove/glove.840B.300d.txt'
PATH_SENTEVAL = '/home/anie/SentEval'
PATH_TO_DATA = '/home/anie/SentEval/data/senteval_data/'
MODEL_PATH = pjoin(params.outputdir, params.outputmodelname + ".pickle.encoder")

assert os.path.isfile(MODEL_PATH) and os.path.isfile(GLOVE_PATH), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',
                  'SICKEntailment', 'MRPC', 'STS14']

# define senteval params
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    # We map cuda to the current cuda device
    map_locations = {}
    for d in range(4):
        if d != params.gpu_id:
            map_locations['cuda:{}'.format(d)] = "cuda:{}".format(params.gpu_id)

    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH, map_location=map_locations)
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
