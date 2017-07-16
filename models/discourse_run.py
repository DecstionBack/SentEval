"""
Build the classifier and run it
"""

# Plan:
# move the model here (not training here)
# and have the model here load in train_dir from elsewhere
# eventually this whole thing can just run on the cloud (or locally)

# model after bow.py

import sys
import numpy as np
import logging
import torch
import tensorflow as tf

from examples.exutil import dotdict
import examples.data as data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("pytorch_cuda", 1, "specify the PyTorch GPU number")

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_GLOVE = 'glove/glove.840B.300d.txt'

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
    # this is the same. we no longer create trimmed files
    _, params.word2id = data.create_dictionary(samples)
    params.word_vec = data.get_wordvec(PATH_TO_GLOVE, params.word2id)

    # TODO: in order to store self.word_vec, must build this internally
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

    return

# TODO: 1. change the vocab building
# TODO: 2. add the padding function
def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)

    # embeddings = []
    #
    # for sent in batch:
    #     sentvec = []
    #     for word in sent:
    #         if word in params.word_vec:
    #             sentvec.append(params.word_vec[word])
    #     if not sentvec:
    #         sentvec.append(params.word_vec['.'])
    #     sentvec = np.mean(sentvec, 0)
    #     embeddings.append(sentvec)
    #
    # embeddings = np.vstack(embeddings)

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold':10}
params_senteval = dotdict(params_senteval)

# set gpu device
torch.cuda.set_device(FLAGS.pytorch_cuda)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    # build TF session here
    pass