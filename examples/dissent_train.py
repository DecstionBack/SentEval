"""
We do fine-tuning on PDTB

need to have a classifier that takes an encoder!
so we can compare with InferSent (max reuse the code)

50 minutes
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import csv
import os
import torch
from exutil import dotdict
import argparse
import logging
from os.path import join as pjoin
import codecs
import numpy as np
import copy
import torch.optim as optim
from torch.autograd import Variable

import logging

reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='DisSent SentEval Evaluation')
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID, we map all model's gpu to this id")
parser.add_argument("--search_start_epoch", type=int, default=-1, help="Search from [start, end] epochs ")
parser.add_argument("--search_end_epoch", type=int, default=-1, help="Search from [start, end] epochs")
parser.add_argument("--dis", action='store_true', help="run on DIS")
parser.add_argument("--pdtb", action='store_true', help="run on PDTB")
parser.add_argument("--mlp", action='store_true', help="use MLP")
parser.add_argument("--bilinear", action='store_true',
                    help="Vector dimension too large, do not use BiLinear interaction")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")


params, _ = parser.parse_known_args()

"""
Logging
"""
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

if not os.path.exists(params.outputdir):
    os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/senteval_log.txt".format(params.outputdir))
logging.getLogger().addHandler(file_handler)

# set gpu device
torch.cuda.set_device(params.gpu_id)

# Set PATHs
GLOVE_PATH = '/home/anie/glove/glove.840B.300d.txt'
PATH_SENTEVAL = '/home/anie/SentEval'
PATH_TO_DATA = '/home/anie/SentEval/data/senteval_data/'

assert os.path.isfile(GLOVE_PATH), 'Set GloVe PATH'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
from senteval.tools.classifier import LogReg, MLP, FCNet

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['PDTB']

if not params.mlp:
    params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                               'seed': 1111, 'kfold': 5, 'bilinear': params.bilinear,
                               'classifier': 'LogReg'})
else:
    params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                               'seed': 1111, 'kfold': 5, 'classifier': 'MLP', 'nhid': 512,
                               'bilinear': params.bilinear})

def prepare(params, samples):
    params.encoder.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.encoder.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings


"""
Data Loader
(similar to PDTB_Eval)
"""


class PDTB_Data(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : Penn Discourse Treebank 2.0 Classification*****\n\n')
        self.seed = seed
        train1 = self.loadFile(os.path.join(taskpath, 's1.train'))
        train2 = self.loadFile(os.path.join(taskpath, 's2.train'))
        trainlabels = open(os.path.join(taskpath, 'labels.train')).read().splitlines()

        valid1 = self.loadFile(os.path.join(taskpath, 's1.dev'))
        valid2 = self.loadFile(os.path.join(taskpath, 's2.dev'))
        validlabels = open(os.path.join(taskpath, 'labels.dev')).read().splitlines()

        test1 = self.loadFile(os.path.join(taskpath, 's1.test'))
        test2 = self.loadFile(os.path.join(taskpath, 's2.test'))
        testlabels = open(os.path.join(taskpath, 'labels.test')).read().splitlines()

        # sort data (by s2 first) to reduce padding
        sorted_train = sorted(zip(train2, train1, trainlabels), key=lambda z: (len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels), key=lambda z: (len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels), key=lambda z: (len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))

        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {
            'train': (train1, train2, trainlabels),
            'valid': (valid1, valid2, validlabels),
            'test': (test1, test2, testlabels)
        }

        dico_label = ['Instantiation',
                      'Synchrony',
                      'Pragmatic cause',
                      'List',
                      'Asynchronous',
                      'Restatement',
                      'Alternative',
                      'Conjunction',
                      'Cause',
                      'Concession',
                      'Contrast']
        self.dico_label = {k: v for v, k in enumerate(dico_label)}

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.encode('utf-8').split() for line in f.read().splitlines()]


"""
Classifier Building
"""


class FineTuneClassifier(object):
    def __init__(self, params):
        # decoder must be a PyTorchClassifier object that we can fit
        # encoder is the original encoder we can backprop to
        self.encoder = params.encoder
        self.pdtb = PDTB_Data(PATH_TO_DATA + '/PDTB')

        # build up vocab
        self.pdtb.do_prepare(params, prepare)

        config = {'nclasses': len(self.pdtb.dico_label), 'seed': self.pdtb.seed,
                  'usepytorch': params.usepytorch,
                  'cudaEfficient': True,
                  'classifier': params.classifier, 'nhid': params.nhid, 'maxepoch': 100,
                  'nepoches': 1,
                  'noreg': True}  # originally False, but now set as True

        self.nclasses = config['nclasses']
        self.featdim = self.encoder.enc_lstm_dim * 5 * 2
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.cudaEfficient = False if 'cudaEfficient' not in config else \
            config['cudaEfficient']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else \
            'pytorch-' + config['classifier']
        self.nepoches = None if 'nepoches' not in config else \
            config['nepoches']
        self.maxepoch = None if 'maxepoch' not in config else \
            config['maxepoch']
        self.noreg = False if 'noreg' not in config else config['noreg']

    def trainepoch(self, nepoches=1):
        # 1. batch iterate on X, y (like PDTB_Eval)
        # 2. encode X, decode h, get loss on y

        # a difference: classifier is permutating at each turn
        # meaning, true randomness. Here we only permute once...not as good

        input1, input2, train_labels = self.pdtb.data["train"]
        self.encoder.train()
        self.clf.model.train()

        n_labels = len(train_labels)
        for _ in range(self.nepoch, self.nepoch + nepoches):
            all_costs = []

            iter = 0
            for ii in range(0, n_labels, self.clf.batch_size):
                batch1 = input1[ii:ii + self.clf.batch_size]
                batch2 = input2[ii:ii + self.clf.batch_size]

                iter += len(batch1)

                mylabels = train_labels[ii:ii + self.clf.batch_size]
                y = [self.pdtb.dico_label[y] for y in mylabels]
                ybatch = Variable(torch.LongTensor(y), requires_grad=False).cuda()

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    sent1 = [' '.join(s) for s in batch1]
                    sent2 = [' '.join(s) for s in batch2]

                    u = self.encoder.encode_trainable(sent1)
                    v = self.encoder.encode_trainable(sent2)

                    Xbatch = torch.cat((u, v, u - v, u * v, (u + v) / 2.), 1)
                    output = self.clf.model(Xbatch)

                    # loss
                    loss = self.clf.loss_fn(output, ybatch)
                    all_costs.append(loss.data[0])

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Update parameters
                    self.optimizer.step()

                if len(all_costs) == params.log_interval:
                    logging.info('iter {}; loss {}'.format(
                        iter,
                        round(np.mean(all_costs), 2)))
                    all_costs = []

        self.nepoch += nepoches

    def score(self, test=False):
        # compute the score for dev set
        if not test:
            input1, input2, labels = self.pdtb.data["valid"]
        else:
            input1, input2, labels = self.pdtb.data["test"]

        self.clf.model.eval()
        self.encoder.eval()

        correct = 0
        for i in range(0, len(labels), self.clf.batch_size):
            batch1 = input1[i:i + self.clf.batch_size]
            batch2 = input2[i:i + self.clf.batch_size]

            mylabels = labels[i:i + self.clf.batch_size]
            y = [self.pdtb.dico_label[y] for y in mylabels]
            ybatch = torch.LongTensor(y)

            assert len(batch1) == len(batch2) and len(batch1) > 0

            sent1 = [' '.join(s) for s in batch1]
            sent2 = [' '.join(s) for s in batch2]

            u = self.encoder.encode_trainable(sent1, volatile=True)
            v = self.encoder.encode_trainable(sent2, volatile=True)

            Xbatch = torch.cat((u, v, u - v, u * v, (u + v) / 2.), 1)
            ybatch = Variable(ybatch, volatile=True)
            if self.cudaEfficient:
                ybatch = ybatch.cuda()

            output = self.clf.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()

        accuracy = 1.0 * correct / len(labels)
        return accuracy

    def fit(self, early_stop=True):
        # we can't call prepare split, X and y are not PyTorch tensor yet

        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Training
        while not stop_train and self.nepoch <= self.maxepoch:
            self.trainepoch(nepoches=self.nepoches)
            accuracy = self.score()
            logging.info("epoch {} finished with dev accuracy {}".format(self.nepoch, accuracy))
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                # this feels slow and unnecessary?
                bestmodel = copy.deepcopy(self.clf.model)
                bestencoder = copy.deepcopy(self.encoder)
            elif early_stop:
                if early_stop_count >= 5:
                    stop_train = True
                early_stop_count += 1
        self.clf.model = bestmodel
        self.encoder = bestencoder
        return bestaccuracy

    def run(self):
        # similar to split classifier, this method is the MAIN method
        # will be called by outside to get task dev/test accuracy
        logging.info('Training {0} with standard validation..'
                     .format(self.modelname))
        regs = [10 ** t for t in range(-5, -1)] if self.usepytorch else \
              [2 ** t for t in range(-2, 4, 1)]
        if self.noreg:
            regs = [0.]
        scores = []
        for reg in regs:
            logging.info("Searching reg {}".format(reg))
            if self.usepytorch:
                if self.classifier == 'LogReg':
                    self.clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses,
                                 l2reg=reg, seed=self.seed,
                                 cudaEfficient=self.cudaEfficient,
                                  batch_size=32)
                elif self.classifier == 'MLP':
                    self.clf = FCNet(inputdim=self.featdim, hiddendim=self.nhid,
                              nclasses=self.nclasses, l2reg=reg,
                              seed=self.seed, cudaEfficient=self.cudaEfficient,
                              batch_size=32)

                # this will actually encompass parameters from encoder and clf
                # an optimizer for each model

                # this is a possible point of failure, watch out!
                self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.clf.model.parameters()),
                                            weight_decay=self.clf.l2reg)
                self.fit()
            else:
                raise Exception("Must use PyTorch")
            cur_dev_acc = self.score()
            logging.info("Epoch {} dev accuracy {}".format(self.nepoch,cur_dev_acc))
            scores.append(round(100 * cur_dev_acc, 2))
        logging.info([('reg:' + str(regs[idx]), scores[idx])
                      for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score \
                    {1}'.format(optreg, devaccuracy))

        logging.info('Evaluating...')
        # retrain with best hyper-param
        if self.usepytorch:
            if self.classifier == 'LogReg':
                self.clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses,
                             l2reg=optreg, seed=self.seed,
                             cudaEfficient=self.cudaEfficient)
            elif self.classifier == 'MLP':
                self.clf = FCNet(inputdim=self.featdim, hiddendim=self.nhid,
                          nclasses=self.nclasses, l2reg=optreg, seed=self.seed,
                          cudaEfficient=self.cudaEfficient)
            # small hack : MultiNLI/SNLI specific
            if self.nepoches:
                self.clf.nepoches = self.nepoches
            if self.maxepoch:
                self.clf.maxepoch = self.maxepoch
            self.fit()

        testaccuracy = self.score(test=True)
        testaccuracy = round(100 * testaccuracy, 2)

        return devaccuracy, testaccuracy


def write_to_dis_csv(file_name, epoch, results_transfer, print_header=False):
    header = ['Epoch', 'Result']
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if print_header:
            writer.writerow(header)
        results = ['Epoch {}'.format(epoch)]
        if params.dis:
            acc = results_transfer['DIS']['acc']
        elif params.pdtb:
            acc = results_transfer['PDTB']['acc']
        else:
            raise Exception("must be one of two: dis or pdtb")

        results.append("{0:.2f}".format(acc))

        writer.writerow(results)


# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    # NOTE: we probably can only afford search a small number of epochs
    # let's first start with 1 epoch, see how it goes.

    # We map cuda to the current cuda device
    # this only works when we set params.gpu_id = 0
    map_locations = {}
    for d in range(4):
        if d != params.gpu_id:
            map_locations['cuda:{}'.format(d)] = "cuda:{}".format(params.gpu_id)

    # collect number of epochs trained in directory
    model_files = filter(lambda s: params.outputmodelname + '-' in s and 'encoder' not in s,
                         os.listdir(params.outputdir))
    epoch_numbers = map(lambda s: s.split(params.outputmodelname + '-')[1].replace('.pickle', ''), model_files)
    # ['8', '7', '9', '3', '11', '2', '1', '5', '4', '6']
    # this is discontinuous :)
    epoch_numbers = map(lambda i: int(i), epoch_numbers)
    epoch_numbers = sorted(epoch_numbers)  # now sorted

    suffix = "_finetune"  # to mark difference with other methods
    if params.mlp:
        suffix += "_mlp"

    csv_file_name = 'senteval_results.csv' if len(transfer_tasks) == 10 else "_".join(transfer_tasks) + suffix + ".csv"

    filtered_epoch_numbers = filter(lambda i: params.search_start_epoch <= i <= params.search_end_epoch,
                                    epoch_numbers)
    assert len(
        filtered_epoch_numbers) >= 1, "the epoch search criteria [{}, {}] returns null, available epochs are: {}".format(
        params.search_start_epoch, params.search_end_epoch, epoch_numbers)

    first = True
    for epoch in filtered_epoch_numbers:
        logging.info("******* Epoch {} Evaluation *******".format(epoch))
        model_name = params.outputmodelname + '-{}.pickle'.format(epoch)
        model_path = pjoin(params.outputdir, model_name)

        dissent = torch.load(model_path, map_location=map_locations)
        params_senteval.encoder = dissent.encoder  # this might be good enough
        params_senteval.encoder.set_glove_path(GLOVE_PATH)

        # se = senteval.SentEval(params_senteval, batcher, prepare)
        # results_transfer = se.eval(transfer_tasks)

        clf = FineTuneClassifier(params_senteval)
        devacc, testacc = clf.run()

        logging.debug('Dev acc : {0} Test acc : {1} for PDTB\n'.format(devacc, testacc))
        results_transfer = {'devacc': devacc, 'acc': testacc, 'ndev': len(clf.pdtb.data['valid'][0]),
                            'ntest': len(clf.pdtb.data['test'][0])}

        logging.info(results_transfer)

        # now we sift through the result dictionary and save results to csv
        write_to_dis_csv(pjoin(params.outputdir, csv_file_name), epoch, results_transfer, first)
        first = False