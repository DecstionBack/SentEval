'''
DIS - Discourse
'''
from __future__ import division

import codecs
import os
import logging
import numpy as np

from tools.validation import SplitClassifier

class DISEval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : Discourse Classification*****\n\n')
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
        sorted_train = sorted(zip(train2, train1, trainlabels), key=lambda z:(len(z[0]), len(z[1]), z[2]))
        train2, train1, trainlabels = map(list, zip(*sorted_train))

        sorted_valid = sorted(zip(valid2, valid1, validlabels), key=lambda z:(len(z[0]), len(z[1]), z[2]))
        valid2, valid1, validlabels = map(list, zip(*sorted_valid))

        sorted_test = sorted(zip(test2, test1, testlabels), key=lambda z:(len(z[0]), len(z[1]), z[2]))
        test2, test1, testlabels = map(list, zip(*sorted_test))


        self.samples = train1 + train2 + valid1 + valid2 + test1 + test2
        self.data = {
            'train': (train1, train2, trainlabels),
            'valid': (valid1, valid2, validlabels),
            'test': (test1, test2, testlabels)
        }

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    # TODO: change this function
    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.encode('utf-8').split() for line in f.read().splitlines()]

    def run(self, params, batcher):
        self.X, self.y = {}, {}
        # TODO: change this part
        # dico_label = {'but': 0, 'because': 1, 'when': 2, 'if': 3, 'for example': 4, 'so': 5, 'before': 6, 'still': 7}
        dico_label = {'but': 0, 'because': 1, 'when': 2, 'if': 3, 'so': 4}
        for key in self.data:
            if key not in self.X: self.X[key] = []
            if key not in self.y: self.y[key] = []

            input1, input2, mylabels = self.data[key]
            enc_input = []
            n_labels = len(mylabels)
            for ii in range(0, n_labels, params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)
                    enc_input.append(np.hstack(( enc1, enc2, enc1 * enc2, enc1 - enc2, (enc1 + enc2) / 2.)))
                if (ii*params.batch_size) % (20000*params.batch_size) == 0:
                    logging.info("PROGRESS (encoding): %.2f%%" % (100 * ii / n_labels))
            self.X[key] = np.vstack(enc_input)
            self.y[key] = [dico_label[y] for y in mylabels] # we are zero-indexed so -1

        config_classifier = {'nclasses':8, 'seed':self.seed, 'usepytorch':params.usepytorch, 'cudaEfficient': True,\
                            'classifier':params.classifier, 'nhid': params.nhid, 'maxepoch':15, 'nepoches':1, 'noreg':True}
        clf = SplitClassifier(self.X, self.y, config_classifier)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1} for SNLI\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev':len(self.data['valid'][0]), 'ntest':len(self.data['test'][0])}
