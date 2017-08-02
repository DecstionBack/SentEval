"""
Using discourse discrimination
to produce sentence embeddings
"""

# ====== Model Definition ======

import time
import os
import sys
import logging

import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs

FLAGS = tf.app.flags.FLAGS

# this is set as DEBUG
# logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, num_layers):
        self.size = size
        self.keep_prob = tf.placeholder(tf.float32)
        lstm_cell = rnn_cell.BasicLSTMCell(self.size)
        lstm_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob, seed=123)
        self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    def encode(self, inputs, masks, reuse=False, scope_name=""):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: (time_step, length, size), notice that input is "time-major"
                        instead of "batch-major".
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        with vs.variable_scope(scope_name + "Encoder", reuse=reuse):
            inp = inputs
            mask = masks
            encoder_outputs = None

            with vs.variable_scope("EncoderCell") as scope:
                srclen = tf.reduce_sum(mask, reduction_indices=1)
                (fw_out, bw_out), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell,
                                                                                         self.encoder_cell, inp, srclen,
                                                                                         scope=scope, dtype=tf.float32)
                out = fw_out + bw_out

            # this is extracting the last hidden states
            encoder_outputs = tf.add(output_state_fw[-1][1], output_state_bw[-1][1])

        return out, encoder_outputs

class SequenceClassifier(object):
    def __init__(self, session, encoder, flags, embed_size, label_size, embed_path):
        self.encoder = encoder
        self.embed_size = embed_size
        self.embed_path = embed_path
        self.flags = flags
        self.label_size = label_size
        self.session = session

        self.learning_rate = flags.learning_rate
        max_gradient_norm = flags.max_gradient_norm
        keep = flags.keep
        dropout = flags.dropout
        learning_rate_decay = flags.learning_rate_decay

        self.seqX = tf.placeholder(tf.float32, [None, None, self.embed_size])
        self.seqX_mask = tf.placeholder(tf.int32, [None, None])

        self.labels = tf.placeholder(tf.int32, [None])

        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(self.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay)
        self.global_step = tf.Variable(0, trainable=False)

        # main computation graph is here
        self.seqX_w_matrix, self.seqX_rep = self.encoder.encode(self.seqX, self.seqX_mask)

    # we need padding (no need to batch, but need to pad)
    def get_sent_embedding(self, session, sentX, sentX_mask):
        input_feed = {}
        input_feed[self.seqX] = sentX
        input_feed[self.seqX_mask] = sentX_mask

        input_feed[self.encoder.keep_prob] = 1.

        output_feed = [self.seqX_rep]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'embed_path'), 'warning : you need to pass in embed_path'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_glove(word_dict)
        logging.info('Vocab size : {0}'.format(len(self.word_vec)))

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        if tokenize: from nltk.tokenize import word_tokenize
        sentences = [s.split() if not tokenize else word_tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        return word_dict

    def get_glove(self, word_dict):
        assert hasattr(self, 'embed_path'), 'warning : you need to pass in embed_path'
        # create word_vec with glove vectors
        word_vec = {}
        with open(self.embed_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        logging.info('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
        return word_vec

    # good, this is fixed
    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        # load in the batch here
        embed = np.zeros((len(batch), len(batch[0]), self.embed_size))
        # padding is set to be 0!!
        mask = np.zeros((len(batch), len(batch[0])), dtype=np.float32)

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[i, j, :] = self.word_vec[batch[i][j]]
                mask[i, j] = 1.  # the pad would be 0

        return embed, mask

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        # this mimics the InferSent's encode() function
        # so far, this really isn't doing much...except sorting and indexing
        tic = time.time()
        if tokenize: from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else ['<s>'] + word_tokenize(s) + ['</s>'] for s in
                     sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors (note this is
        # a bit different for our discourse classifier)
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn(
                    'No words in "{0}" (idx={1}) have glove vectors. Replacing by "</s>"..'.format(sentences[i], i))
                s_f = ['</s>']
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : {0}/{1} ({2} %)'.format(n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        # this should be fine now? Hopefully.
        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch, batch_mask = self.get_batch(sentences[stidx:stidx + bsize])
            batch = self.get_sent_embedding(self.session, batch, batch_mask)
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print(
            'Speed : {0} sentences/s (bsize={1})'.format(round(len(embeddings) / (time.time() - tic), 2), bsize))
        return embeddings

    def extract_sent(self, positions, sent):
        list_sent = sent.tolist()
        extracted_sent = []
        for i in range(sent.shape[0]):
            if positions[i]:
                extracted_sent.append(list_sent[i])
        return extracted_sent
