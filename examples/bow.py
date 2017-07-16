# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

import os
import sys
import numpy as np
import logging
import torch

from exutil import dotdict
import data

# Set PATHs
PATH_TO_SENTEVAL = '/afs/cs.stanford.edu/u/anie/SentEval'
PATH_TO_DATA = '/deep/u/anie/SentEval/data/senteval_data/'
PATH_TO_GLOVE = '/deep/u/anie/glove/glove.840B.300d.txt'
RUN_DIR = '/deep/u/anie/SentEval/run_dir/bow/'
                
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


"""
Note: 

The user has to implement two functions:
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params" 
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""


def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.word_vec = data.get_wordvec(PATH_TO_GLOVE, params.word2id)
    return


def batcher(params, batch):
    batch = [sent if sent!=[] else ['.'] for sent in batch]
    embeddings = []
    
    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            sentvec.append(params.word_vec['.'])
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold':5}
params_senteval = dotdict(params_senteval)

# set gpu device
torch.cuda.set_device(0)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    if not os.path.exists(RUN_DIR):
        os.makedirs(RUN_DIR)
    file_handler = logging.FileHandler("{0}/log.txt".format(RUN_DIR))
    logging.getLogger().addHandler(file_handler)

    se = senteval.SentEval(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STS14']
    results = se.eval(transfer_tasks)

    
    
    
    
    
    
