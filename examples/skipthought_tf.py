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
sys.setdefaultencoding('utf8')


# Set PATHs
PATH_TO_SENTEVAL = '/afs/cs.stanford.edu/u/anie/SentEval'
PATH_TO_DATA = '/deep/u/anie/SentEval/data/senteval_data/'

sys.path.insert(0, PATH_TO_SENTEVAL)

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

import senteval

VOCAB_FILE = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/vocab.txt"
EMBEDDING_MATRIX_FILE = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/embeddings.npy"
CHECKPOINT_PATH = "/deep/u/anie/skip_thoughts/skip_thoughts_bi_2017_02_16/model.ckpt-500008"


def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = encoder.encode(session, [str(' '.join(sent))
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
    encoder = encoder_manager.EncoderManager()

    configuration.model_config(bidirectional_encoder=True)

    encoder.load_model(configuration.model_config(),
                       vocabulary_file=VOCAB_FILE,
                       embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                       checkpoint_path=CHECKPOINT_PATH)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        params_senteval.encoder = encoder
        se = senteval.SentEval(params_senteval, batcher, prepare)
        se.eval(['DIS', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',
                 'SICKEntailment', 'MRPC', 'STS14'])
