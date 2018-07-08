from __future__ import absolute_import, division, unicode_literals

import sys
import csv
import os
import torch
from torch.autograd import Variable
import numpy as np
from exutil import dotdict
import argparse
import logging
from os.path import join as pjoin
from data import TextEncoder, pad_batch

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
parser.add_argument("--pdtb", action='store_true', help="run on PDTB, PDTB_IMEX, PDTB_EX")
parser.add_argument("--dat", action='store_true', help="run on DAT")
parser.add_argument("--mlp", action='store_true', help="use MLP")
parser.add_argument("--bilinear", action='store_true',
                    help="Vector dimension too large, do not use BiLinear interaction")

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
# GLOVE_PATH = '/home/anie/glove/glove.840B.300d.txt'
PATH_SENTEVAL = '/home/anie/SentEval'
PATH_TO_DATA = '/home/anie/SentEval/data/senteval_data/'
# Word embedding now comes with pickle file

bpe_encoder_path = '/home/anie/DisExtract/transformer/params/encoder_bpe_40000.json'
bpe_vocab_path = '/home/anie/DisExtract/transformer/params/vocab_40000.bpe'
params_path = '/home/anie/DisExtract/transformer/params/'

"""
BPE encoder
"""
text_encoder = TextEncoder(bpe_encoder_path, bpe_vocab_path)
encoder = text_encoder.encoder

# add special token (embedding is already there)
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad_id):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad_id).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval

# maybe can trunkcate samples?? here we follow skipthought_tf
def prepare(params, samples):
    # params.infersent.build_vocab([' '.join(s) for s in samples],
    #                              tokenize=False)
    return

def batcher(params, batch):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    num_sents = []
    # numericalize into BPE format
    for sent in sentences:
        num_sent = text_encoder.encode([sent], verbose=False, lazy=True)[0]
        num_sents.append([encoder['_start_']] + num_sent + [encoder['_end_']])

    sent_batch = pad_batch(num_sents, encoder['_pad_'])
    sent_lengths = (sent_batch[:, :-1] != encoder['_pad_']).sum(axis=1) # numpy
    sent_batch = Variable(torch.from_numpy(sent_batch)).cuda(params.gpu_id)
    sent_mask = make_std_mask(sent_batch, encoder['_pad_'])

    embeddings = params.infersent.encode(sent_batch, sent_mask)
    embeddings = params.infersent.pick_h(embeddings, sent_lengths)

    return embeddings.data.cpu().numpy()


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
            # this does not print two tasks...only print the first one
            # fix it...
            if 'PDTB' in results_transfer:
                acc = results_transfer['PDTB']['acc']
            elif 'PDTB_EX' in results_transfer:
                acc = results_transfer['PDTB_EX']['acc']
            elif 'PDTB_IMEX' in results_transfer:
                acc = results_transfer['PDTB_IMEX']['acc']
            else:
                raise Exception("task not in PDTB range")
        elif params.dat:
            acc = results_transfer['DAT']['acc']
        else:
            raise Exception("must be one of two: dis or pdtb or dat")

        results.append("{0:.2f}".format(acc))

        writer.writerow(results)


def write_to_csv(file_name, epoch, results_transfer, print_header=False):
    header = ['Epoch', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14',
              "ACC_AVG"]
    acc_header = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC']
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if print_header:
            writer.writerow(header)
        # then process result_transfer to print to file
        # since each test has different dictionary entry, we process them separately...
        results = ['Epoch {}'.format(epoch)]
        acc_s = []
        for h in acc_header:
            acc = results_transfer[h]['acc']
            acc_s.append(acc)
            results.append("{0:.2f}".format(acc))  # take 2 digits, and manually round later
        pear = results_transfer['SICKRelatedness']['pearson']
        results.append("{0:.4f}".format(pear))
        acc = results_transfer['SICKEntailment']['acc']
        acc_s.append(acc)
        results.append("{0:.2f}".format(acc))

        mprc_acc = results_transfer['MRPC']['acc']
        mprc_f1 = results_transfer['MRPC']['f1']

        acc_s.append(mprc_acc)

        results.append("{0:.2f}/{0:.2f}".format(mprc_acc, mprc_f1))

        sts14_pear_wmean = results_transfer['STS14']['all']['pearson']['wmean']
        sts14_pear_mean = results_transfer['STS14']['all']['pearson']['mean']

        results.append("{0:.4f}/{0:.4f}".format(sts14_pear_wmean, sts14_pear_mean))

        mean_acc = sum(acc_s) / float(len(acc_s))
        results.append("{0:.4f}".format(mean_acc))

        writer.writerow(results)


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
if params.dis:
    transfer_tasks = ['DIS']
elif params.pdtb:
    transfer_tasks = ['PDTB_IMEX']  # 'PDTB_EX'
elif params.dat:
    transfer_tasks = ['DAT']
else:
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'SICKRelatedness',
                      'SICKEntailment', 'MRPC', 'STS14']

# define senteval params
if params.mlp:
    # keep nhid the same as DisSent model (otherwise we can try 1024)
    params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                               'seed': 1111, 'kfold': 5, 'classifier': 'MLP', 'nhid': 512,
                               'bilinear': params.bilinear})
else:
    params_senteval = dotdict({'usepytorch': True, 'task_path': PATH_TO_DATA,
                               'seed': 1111, 'kfold': 5, 'bilinear': params.bilinear})

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

    suffix = ""
    if params.mlp:
        suffix += "_mlp"
    if params.bilinear:
        suffix += "_bilinear"

    csv_file_name = 'senteval_results.csv' if len(transfer_tasks) == 10 else "_".join(transfer_tasks) + suffix + ".csv"

    # original setting
    if params.search_start_epoch == -1 or params.search_end_epoch == -1:
        # Load model
        MODEL_PATH = pjoin(params.outputdir, params.outputmodelname + ".pickle")

        params_senteval.infersent = torch.load(MODEL_PATH) # , map_location=map_locations
        params_senteval.infersent.eval()

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

            dissentT = torch.load(model_path)  # , map_location=map_locations
            dissentT.eval()

            params_senteval.infersent = dissentT  # this might be good enough

            se = senteval.SentEval(params_senteval, batcher, prepare)
            results_transfer = se.eval(transfer_tasks)

            logging.info(results_transfer)

            # now we sift through the result dictionary and save results to csv
            if not params.dis and not params.pdtb and not params.dat:
                write_to_csv(pjoin(params.outputdir, "senteval_results.csv"), epoch, results_transfer, first)
            else:
                write_to_dis_csv(pjoin(params.outputdir, csv_file_name), epoch, results_transfer, first)
            first = False
