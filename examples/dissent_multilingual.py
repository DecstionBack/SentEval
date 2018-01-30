"""
Similar to `dissent_eval.py` but just for Multilingual models
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

import logging

reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description='DisSent SentEval Evaluation')
parser.add_argument("--outputdir", type=str, default='sandbox/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='dis-model')
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID, we map all model's gpu to this id")
parser.add_argument("--search_start_epoch", type=int, default=-1, help="Search from [start, end] epochs ")
parser.add_argument("--search_end_epoch", type=int, default=-1, help="Search from [start, end] epochs")
parser.add_argument("--lang", type=str, default='CH', help="CH|SP, change language")

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
if params.lang == 'SP':
    GLOVE_PATH = '/home/anie/fasttext/wiki.es.vec'
elif params.lang == "CH":
    GLOVE_PATH = '/home/anie/fasttext/wiki.zh.vec'

PATH_SENTEVAL = '/home/anie/SentEval'
PATH_TO_DATA = '/home/anie/SentEval/data/senteval_data/'

assert os.path.isfile(GLOVE_PATH), 'Set GloVe PATH'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings


def write_to_csv(file_name, epoch, results_transfer, print_header=False):
    header = ['Epoch', 'ABSA_CH:Phone', 'ABSA_CH:Camera'] if params.lang == 'CH' else ['Epoch', 'ABSA_SP:Restaurant', 'STS_SP']
    acc_header = ['ABSA_CH:Phone', 'ABSA_CH:Camera'] if params.lang == 'CH' else ['ABSA_SP:Restaurant']
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if print_header:
            writer.writerow(header)

        # then process result_transfer to print to file
        # since each test has different dictionary entry, we process them separately...
        results = ['Epoch {}'.format(epoch)]
        acc_s = []
        for h in acc_header:
            h, field = h.split(":")
            acc = results_transfer[h]['{} acc'.format(field)]
            acc_s.append(acc)
            results.append("{0:.2f}".format(acc))  # take 2 digits, and manually round later

        if params.lang == "SP":
            sts14_pear_wmean = results_transfer['STS_SP']['all']['pearson']['wmean']
            sts14_pear_mean = results_transfer['STS_SP']['all']['pearson']['mean']

            results.append("{0:.4f}/{0:.4f}".format(sts14_pear_wmean, sts14_pear_mean))

        writer.writerow(results)


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['ABSA_CH'] if params.lang == 'CH' else ['ABSA_SP', 'STS_SP']

# define senteval params
# Can choose to use MLP instead
params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                           'seed': 1111, 'kfold': 5})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

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

    csv_file_name = 'senteval_results.csv'

    # original setting
    if params.search_start_epoch == -1 or params.search_end_epoch == -1:
        # Load model
        MODEL_PATH = pjoin(params.outputdir, params.outputmodelname + ".pickle.encoder")

        params_senteval.infersent = torch.load(MODEL_PATH, map_location=map_locations)
        params_senteval.infersent.set_glove_path(GLOVE_PATH)

        se = senteval.SentEval(params_senteval, batcher, prepare)
        results_transfer = se.eval(transfer_tasks)

        logging.info(results_transfer)
    else:
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
            params_senteval.infersent = dissent.encoder  # this might be good enough
            params_senteval.infersent.set_glove_path(GLOVE_PATH)

            se = senteval.SentEval(params_senteval, batcher, prepare)
            results_transfer = se.eval(transfer_tasks)

            logging.info(results_transfer)

            # now we sift through the result dictionary and save results to csv
            write_to_csv(pjoin(params.outputdir, "senteval_results.csv"), epoch, results_transfer, first)

            first = False