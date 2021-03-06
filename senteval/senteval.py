# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.trec import TRECEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.mrpc import MRPCEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, \
    STSBenchmarkEval, STS_SPBenchmarkEval
from senteval.sst import SSTEval
from senteval.rank import ImageCaptionRetrievalEval
from senteval.dis import DISEval
from senteval.ABSA_CH import ABSA_CHEval
from senteval.ABSA_SP import ABSA_SPEval
from senteval.pdtb import PDTB_Eval
from senteval.pdtb_ex import PDTB_EX_Eval
from senteval.pdtb_imex import PDTB_IMEX_Eval
from senteval.dat import DAT_EVAL
from senteval import utils

class SentEval(object):
    def __init__(self, params, batcher, prepare=None):
        # setting default parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # set up bilinear projection, with learnable matrix W
        params.bilinear = False if 'bilinear' not in params else params.bilinear

        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        # sanity check
        # assert params.classifier in ['LogReg', 'MLP']
        # if params.classifier == 'MLP':
        #     assert params.nhid > 0, 'When using an MLP, \
        #         you need to set params.nhid>0'
        # if not params.usepytorch and params.classifier == 'MLP':
        #     assert False, 'No MLP implemented in scikit-learn'

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16', 'DIS', 'PDTB', 'DAT',
                           'PDTB_EX', 'PDTB_IMEX',
                           'ABSA_CH', 'ABSA_SP', 'STS_SP']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        if name == 'CR':
            self.evaluation = CREval(tpath + '/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/SST/fine', nclasses=5, seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/SICK',
                                                  seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/STS/STSBenchmark',
                                               seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/SICK',
                                                 seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/SNLI', seed=self.params.seed)
        elif name == 'DIS':
            self.evaluation = DISEval(tpath + '/DIS', seed=self.params.seed)
        elif name == 'PDTB':
            self.evaluation = PDTB_Eval(tpath + '/PDTB', seed=self.params.seed)
        elif name == "PDTB_EX":
            self.evaluation = PDTB_EX_Eval(tpath + '/PDTB_EX', seed=self.params.seed)
        elif name == "PDTB_IMEX":
            self.evaluation = PDTB_IMEX_Eval(tpath + '/PDTB_IMEX', seed=self.params.seed)
        elif name == 'DAT':
            self.evaluation = DAT_EVAL(tpath + '/DAT', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/STS/' + fpath,
                                                  seed=self.params.seed)
        elif name == 'ImageCaptionRetrieval':
            self.evaluation = ImageCaptionRetrievalEval(tpath + '/COCO',
                                                        seed=self.params.seed)
        elif name == 'ABSA_CH':
            self.evaluation = ABSA_CHEval(tpath + '/ABSA_CH', seed=self.params.seed)
        elif name == 'ABSA_SP':
            self.evaluation = ABSA_SPEval(tpath + '/ABSA_SP', seed=self.params.seed)
        elif name == 'STS_SP':
            self.evaluation = STS_SPBenchmarkEval(tpath + '/STS_SP/STSBenchmark', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
