# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
SemEval 2016 Task 5 Aspect-Based Sentiment Analysis

We reduce this task to polarity prediction without predicting categories

http://alt.qcri.org/semeval2016/task5/
https://github.com/magizbox/underthesea/wiki/SemEval-2016-Task-5
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import logging
import numpy as np
from re import compile as _Re

from senteval.tools.validation import KFoldClassifier

_unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split


def split_unicode_chrs(text):
    return [chr for chr in _unicode_chr_splitter(text) if chr]


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


# TODO: need about 2 hours on this...crap...
class ABSA_CHEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : ABSA_CH *****\n\n')
        self.seed = seed

        self.train_phone = self.loadFile(os.path.join(task_path, 'ABSA_CH_PHNS_train.tsv'))
        self.test_phone = self.loadFile(os.path.join(task_path, 'ABSA_CH_PHNS_test.tsv'))

        self.train_cam = self.loadFile(os.path.join(task_path, 'ABSA_CH_CAME_train.tsv'))
        self.test_cam = self.loadFile(os.path.join(task_path, 'ABSA_CH_CAME_test.tsv'))

    def do_prepare(self, params, prepare):
        samples = self.train_phone['X'] + self.test_phone['X'] + self.train_cam['X'] + self.test_cam['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        absa_data = {'X': [], 'y': []}
        tgt2idx = {'negative': 0, 'positive': 1}
        # no longer latin-1
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                target, sample = line.strip().split('\t')[1:]
                sample = split_unicode_chrs(sample.decode('utf-8'))  # Chinese splitting, char-level!!
                assert target in tgt2idx, target
                absa_data['X'].append(sample)
                absa_data['y'].append(tgt2idx[target])
        return absa_data

    def run(self, params, batcher):
        dict_cam = self.single_run(params, batcher,
                                   self.train_cam['X'], self.train_cam['y'],
                                   self.test_cam['X'], self.test_cam['y'], field="Camera")
        dict_phn = self.single_run(params, batcher,
                                   self.train_phone['X'], self.train_phone['y'],
                                   self.test_phone['X'], self.test_phone['y'], field="Phone")
        return merge_two_dicts(dict_cam, dict_phn)

    def single_run(self, params, batcher, train_X, train_y, test_X, test_y, field):
        # batcher is the algorithm
        train_embeddings, test_embeddings = [], []

        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(train_X, train_y),
                                     key=lambda z: (len(z[0]), z[1]))
        train_samples = [x for (x, y) in sorted_corpus_train]
        train_labels = [y for (x, y) in sorted_corpus_train]

        sorted_corpus_test = sorted(zip(test_X, test_y),
                                    key=lambda z: (len(z[0]), z[1]))
        test_samples = [x for (x, y) in sorted_corpus_test]
        test_labels = [y for (x, y) in sorted_corpus_test]

        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')

        config_classifier = {'nclasses': 2, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'nhid': params.nhid, 'kfold': params.kfold}
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': np.array(test_labels)},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('\n' + field + ' Dev acc : {0} Test acc : {1} \
            for ABSA_CH\n'.format(devacc, testacc))
        return {'{} devacc'.format(field): devacc, '{} acc'.format(field): testacc,
                '{} ndev'.format(field): len(train_X), '{} ntest'.format(field): len(test_X)}
