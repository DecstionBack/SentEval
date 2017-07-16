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

tf.app.flags.DEFINE_integer("state_size", 256, "hidden dimension")
tf.app.flags.DEFINE_integer("layers", 1, "number of hidden layers")
tf.app.flags.DEFINE_integer("epochs", 8, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("max_seq_len", 35, "number of time steps to unroll for BPTT, also the max sequence length")
tf.app.flags.DEFINE_integer("embedding_size", 100, "dimension of GloVE vector to use")
tf.app.flags.DEFINE_integer("learning_rate_decay_epoch", 1, "Learning rate starts decaying after this epoch.")
tf.app.flags.DEFINE_float("dropout", 0.2, "probability of dropping units")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("seed", 123, "random seed to use")
tf.app.flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "initial learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.8, "amount to decrease learning rate")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_integer("print_every", 5, "How many iterations to do per print.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
tf.app.flags.DEFINE_string("dataset", "ptb", "ptb/wikitext-103 select the dataset to use")
tf.app.flags.DEFINE_string("task", "but", "choose the task: but/cause")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding")
tf.app.flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
tf.app.flags.DEFINE_boolean("dev", False, "if flag true, will run on dev dataset in a pure testing mode")
tf.app.flags.DEFINE_boolean("correct_example", False, "if flag false, will print error, true will print out success")
tf.app.flags.DEFINE_integer("best_epoch", 1, "enter the best epoch to use")
tf.app.flags.DEFINE_integer("num_examples", 30, "enter the best epoch to use")

logging.basicConfig(level=logging.INFO)

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
            encoder_outputs = tf.add(output_state_fw[0][1], output_state_bw[0][1])

        return out, encoder_outputs

class SequenceClassifier(object):
    def __init__(self, encoder, flags, vocab_size, vocab, rev_vocab, embed_size):
        # task: ["but", "cause"]

        batch_size = flags.batch_size
        max_seq_len = flags.max_seq_len
        self.encoder = encoder
        self.embed_size = embed_size
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.vocab_size = vocab_size
        self.flags = flags
        self.label_size = 2

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
        self.seqX_w_matrix, self.seqX_rep = self.encoder.encode(self.seqX_inputs, self.seqX_mask)


    # we need padding (no need to batch, but need to pad)
    def get_sent_embedding(self, session, sentX, sentX_mask):
        input_feed = {}
        input_feed[self.seqA] = sentX
        input_feed[self.seqA_mask] = sentX_mask

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

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def encode(self, session, sentences, bsize=64, tokenize=True, verbose=False):
        # this mimics the InferSent's encode() function
        # so far, this really isn't doing much...except sorting and indexing
        tic = time.time()
        if tokenize: from nltk.tokenize import word_tokenize
        sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else ['<s>'] + word_tokenize(s) + ['</s>'] for s in
                     sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without glove vectors (note this is
        # a bit different for our discourse classifier)
        # TODO: build self.word_vec
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

        embeddings = []
        # TODO: self.get_batch
        for stidx in range(0, len(sentences), bsize):
            # batch = Variable(self.get_batch(sentences[stidx:stidx + bsize]), volatile=True)
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print(
            'Speed : {0} sentences/s (bsize={1})'.format(round(len(embeddings) / (time.time() - tic), 2), bsize))
        return embeddings


    def test(self, session, because_tokens, because_mask, but_tokens, but_mask, labels):
        input_feed = {}
        input_feed[self.seqA] = because_tokens
        input_feed[self.seqA_mask] = because_mask
        input_feed[self.seqB] = but_tokens
        input_feed[self.seqB_mask] = but_mask
        input_feed[self.labels] = labels

        input_feed[self.encoder.keep_prob] = 1.

        output_feed = [self.loss, self.logits]

        outputs = session.run(output_feed, input_feed)

        return outputs[0], outputs[1]

    def extract_sent(self, positions, sent):
        list_sent = sent.tolist()
        extracted_sent = []
        for i in range(sent.shape[0]):
            if positions[i]:
                extracted_sent.append(list_sent[i])
        return extracted_sent

    def setup_cause_effect(self):
        # seqA: but, seqB: because, this will learn to differentiate them
        seqA_w_matrix, seqA_c_vec = self.encoder.encode(self.seqA_inputs, self.seqA_mask)
        seqB_w_matrix, seqB_c_vec = self.encoder.encode(self.seqB_inputs, self.seqB_mask, reuse=True)

        self.seqA_rep = seqA_c_vec
        self.seqB_rep = seqB_c_vec

        # for now we just use context vector
        # we create additional perspectives

        # seqA_c_vec: (batch_size, hidden_size)
        persA_B_mul = seqA_c_vec * seqB_c_vec
        persA_B_sub = seqA_c_vec - seqB_c_vec
        persA_B_avg = (seqA_c_vec + seqB_c_vec) / 2.0

        # logits is [batch_size, label_size]
        self.logits = rnn_cell._linear([seqA_c_vec, seqB_c_vec, persA_B_mul, persA_B_sub, persA_B_avg],
                                       self.label_size, bias=True)

    def detokenize_batch(self, sent):
        # sent: (N, sentence_padded_length)
        def detok_sent(sent):
            outsent = ''
            for t in sent:
                if t > 0:  # only take out pad, but not unk
                    outsent += self.rev_vocab[t] + " "
            return outsent
        return [detok_sent(s) for s in sent]
