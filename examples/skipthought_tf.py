# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

"""
Example of file to compare skipthought vectors with our InferSent model
Modified: we change this file so it loads in Tensorflow bi-directional SkipThought model
"""
import logging
from exutil import dotdict
import sys
import tensorflow as tf

# Set PATHs
PATH_TO_SENTEVAL = '/home/anie/SentEval'
PATH_TO_DATA = '/home/anie/SentEval/data/senteval_data/'
PATH_TO_SKIPTHOUGHT = '/home/anie/models/research/skip_thoughts'

sys.path.insert(0, PATH_TO_SENTEVAL)
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

import senteval

cluster = "arthur"

if cluster == "deep":
    VOCAB_FILE = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/vocab.txt"
    EMBEDDING_MATRIX_FILE = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/embeddings.npy"
    CHECKPOINT_PATH = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
elif cluster == "cresta":
    VOCAB_FILE = "/home/anie/Documents/models/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
    EMBEDDING_MATRIX_FILE = "/home/anie/Documents/models/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
    CHECKPOINT_PATH = "/home/anie/Documents/models/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
else:
    VOCAB_FILE = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
    EMBEDDING_MATRIX_FILE = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
    CHECKPOINT_PATH = "/home/anie/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"

def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = encoder.encode([str(' '.join(sent))
                                        if sent!= [] else '.' for sent in batch],
                                     verbose=False, use_eos=True)
    return embeddings


# Set params for SentEval
params_senteval = {'usepytorch': True,
                   'task_path': PATH_TO_DATA,
                   'batch_size': 512}

params_senteval = dotdict(params_senteval)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config_gpu) as session:
        encoder = encoder_manager.EncoderManager()

        encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                           vocabulary_file=VOCAB_FILE,
                           embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                           checkpoint_path=CHECKPOINT_PATH)

        params_senteval.encoder = encoder
        se = senteval.SentEval(params_senteval, batcher, prepare)
        # se.eval(['DIS', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'SICKRelatedness',
        #          'SICKEntailment', 'MRPC', 'STS14'])
        se.eval(['DIS'])
        #results_transfer = se.eval(['PDTB_IMEX', 'PDTB_EX'])

        #print(results_transfer)
