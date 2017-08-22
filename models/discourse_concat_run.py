"""
Load in and run concatenated discourse SNLI model
"""

import sys
import os
import json
import numpy as np
import logging
import torch
import tensorflow as tf

from os.path import join as pjoin
from discourse_classifier import Encoder, SequenceClassifier

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("state_size", 256, "hidden dimension")
tf.app.flags.DEFINE_integer("layers", 1, "number of hidden layers")
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
# tf.app.flags.DEFINE_string("embed_path", "None", "Path to the trimmed GLoVe embedding")
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
tf.app.flags.DEFINE_integer("best_epoch", 1, "enter the best epoch to use")
tf.app.flags.DEFINE_integer("label_size", 14, "enter the number of labels")
tf.app.flags.DEFINE_string("cluster", "cres", "cres/deep the path")
tf.app.flags.DEFINE_boolean("concat", False, "if flag True, bidirectional does concatenation not average")
tf.app.flags.DEFINE_boolean("temp_max", False, "if flag true, will use Temporal Max Pooling")
tf.app.flags.DEFINE_boolean("temp_mean", False, "if flag true, will use Temporal Mean Pooling")
tf.app.flags.DEFINE_string("rnn", "lstm", "lstm/gru architecture choice")

# Set PATHs
if FLAGS.cluster == "deep":
    PATH_TO_SENTEVAL = '/afs/cs.stanford.edu/u/anie/SentEval'
    PATH_TO_DATA = '/deep/u/anie/SentEval/data/senteval_data/'
    PATH_TO_GLOVE = '/deep/u/anie/glove/glove.840B.300d.txt'
else:
    PATH_TO_SENTEVAL = '/home/anie/Documents/SentEval'
    PATH_TO_DATA = '/home/anie/Documents/SentEval/data/senteval_data/'
    PATH_TO_GLOVE = '/home/anie/Documents/discourse/data/glove.6B/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
    params.discourse.build_vocab([' '.join(s) for s in samples], tokenize=False)
    params.snli.build_vocab([' '.join(s) for s in samples], tokenize=False)
    return


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]

    # (corpus_length, embed for each sent)
    discourse_embed = params.discourse.encode(sentences, bsize=params.batch_size, tokenize=False)
    snli_embed = params.snli.encode(sentences, bsize=params.batch_size, tokenize=False)

    # concatenate on the final dimension
    assert len(discourse_embed.shape) == 2 and len(snli_embed.shape) == 2
    embeddings = np.hstack([discourse_embed, snli_embed])
    return embeddings


# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',
                  'SICKEntailment', 'STS14']
# MRPC

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval = dotdict(params_senteval)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def main(_):
    # build the model here

    if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)

    # assert FLAGS.embed_path is not "None", "must pick a loading path"

    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
    logging.getLogger().addHandler(file_handler)
    embed_path = PATH_TO_GLOVE  # FLAGS.embed_path
    embed_size = FLAGS.embed_size

    with open(os.path.join(FLAGS.run_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Graph().as_default(), tf.device('/gpu:1'), tf.Session(config=config) as session:
        tf.set_random_seed(FLAGS.seed)

        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

        # with tf.variable_scope("model", initializer=initializer):
        #     encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
        #     sc = SequenceClassifier(session, encoder, FLAGS, embed_size, FLAGS.label_size, embed_path)

        with tf.variable_scope("discourse", reuse=None, initializer=initializer):
            encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            discourse_sc = SequenceClassifier(session, encoder, FLAGS, embed_size, FLAGS.label_size, embed_path)

        with tf.variable_scope("snli", reuse=None, initializer=initializer):
            # preparation for SNLI
            snli_encoder = Encoder(size=FLAGS.state_size, num_layers=FLAGS.layers)
            snli_sc = SequenceClassifier(session, snli_encoder, FLAGS, embed_size, FLAGS.label_size, embed_path)

        params_senteval.discourse = discourse_sc
        params_senteval.snli = snli_sc

        # restore the model here
        # (two models are stored together)
        best_epoch = FLAGS.best_epoch
        model_saver = tf.train.Saver(max_to_keep=FLAGS.keep)

        assert FLAGS.restore_checkpoint is not None, "we must be able to reload the model"
        logging.info("restore model from best epoch %d" % best_epoch)
        checkpoint_path = pjoin(FLAGS.restore_checkpoint, "dis.ckpt")
        model_saver.restore(session, checkpoint_path + ("-%d" % best_epoch))

        se = senteval.SentEval(params_senteval, batcher, prepare)
        results_transfer = se.eval(transfer_tasks)

        print results_transfer


if __name__ == "__main__":
    tf.app.run()
