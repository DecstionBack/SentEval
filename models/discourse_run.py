"""
Build the classifier and run it
"""

# Plan:
# move the model here (not training here)
# and have the model here load in train_dir from elsewhere
# eventually this whole thing can just run on the cloud (or locally)

# model after bow.py

import sys
import os
import json
import numpy as np
import logging
import torch
import tensorflow as tf

from examples.exutil import dotdict
import examples.data as data
from os.path import join as pjoin
from discourse_classifier import Encoder, SequenceClassifier

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("state_size", 256, "hidden dimension")
tf.app.flags.DEFINE_integer("layers", 1, "number of hidden layers")
tf.app.flags.DEFINE_integer("epochs", 8, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("max_seq_len", 35, "number of time steps to unroll for BPTT, also the max sequence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "dimension of GloVE vector to use")
tf.app.flags.DEFINE_integer("learning_rate_decay_epoch", 1, "Learning rate starts decaying after this epoch.")
tf.app.flags.DEFINE_float("dropout", 0., "probability of dropping units")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("seed", 123, "random seed to use")
tf.app.flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.8, "amount to decrease learning rate")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("print_every", 5, "How many iterations to do per print.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
tf.app.flags.DEFINE_string("embed_path", "None", "Path to the trimmed GLoVe embedding")
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
tf.app.flags.DEFINE_integer("best_epoch", 1, "enter the best epoch to use")
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
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)
    return


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)

    return embeddings


# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',
                  'SICKEntailment', 'MRPC', 'STS14']

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval = dotdict(params_senteval)

# set gpu device
torch.cuda.set_device(FLAGS.pytorch_cuda)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def main(_):
    # build the model here

    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)

    assert FLAGS.embed_path is not "None", "must pick a loading path"

    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
    logging.getLogger().addHandler(file_handler)
    embed_path = FLAGS.embed_path
    embed_size = FLAGS.embed_size

    with open(os.path.join(FLAGS.run_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(FLAGS.seed)

        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        with tf.variable_scope("model", initializer=initializer):
            encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            sc = SequenceClassifier(encoder, FLAGS, embed_size, embed_path)

        # restore the model here
        best_epoch = FLAGS.best_epoch
        model_saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

        assert FLAGS.restore_checkpoint is not None, "we must be able to reload the model"
        logging.info("restore model from best epoch %d" % best_epoch)
        checkpoint_path = pjoin(FLAGS.restore_checkpoint, "dis.ckpt")
        model_saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

        se = senteval.SentEval(params_senteval, batcher, prepare)
        results_transfer = se.eval(transfer_tasks)

        print results_transfer

if __name__ == "__main__":
    tf.app.run()
