import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import logging
import time
import numpy as np

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(sentences, seq_length, tokenizer, silent=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(sentences):
        tokens_a = tokenizer.tokenize(example)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5 and not silent:
            print("*** Example ***")
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def bert_encode(sentences, max_seq_length=128, is_cuda=False):
    features = convert_examples_to_features(
        sentences=sentences, seq_length=max_seq_length, tokenizer=tokenizer)

    if is_cuda:
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
    else:
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)

    final_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_masks,
                                    output_all_encoded_layers=False)

    # the [CLS] position
    if is_cuda:
        return final_encoder_layers[:, 0].data.cpu().numpy()
    else:
        return final_encoder_layers[:, 0].data.numpy()

import pickle
from collections import defaultdict

corpora = []
with open('/mnt/fs5/anie/lm1b/training-monolingual/news.2011.en.shuffled.toked.txt') as fin:
    for line in fin:
        corpora.append(line.strip())


word_to_sent = defaultdict(list)

for i, sent in enumerate(corpora):
    for word in sent.split():
        word_to_sent[word].append(i)

train_words, test_words = pickle.load(open("./train_test_words.pkl", 'rb'))

idx_to_sent_embedding = {}


def gen_train_test(train_words, bsize=32, print_every=100):
    # generate training set
    train_sent_vec = []
    train_word_vec = []

    start = time.time()

    processed_sent = 0

    for word_i, word in enumerate(train_words):
        all_sent_vec = []

        # all_sents = [corpora[i] for i in word_to_sent[word]] # if len(corpora[i].split()) < 50
        all_sents = []
        all_sent_indices = []
        for sent_i in word_to_sent[word]:

            # first check if it's already embedded
            if sent_i in idx_to_sent_embedding:
                all_sent_vec.append(idx_to_sent_embedding[sent_i])
            else:
                word_list = corpora[sent_i].split()
                if len(word_list) > 50:
                    content_word_idx = word_list.index(word)
                    # select the window
                    all_sents.append(' '.join(word_list[max(content_word_idx - 25, 0):content_word_idx + 25]))
                else:
                    all_sents.append(corpora[sent_i])
                all_sent_indices.append(sent_i)

        assert len(all_sents) == len(all_sent_indices)

        for idx in range(0, len(all_sents), bsize):
            sents = all_sents[idx:idx + bsize]
            embeddings = bert_encode(sents, max_seq_length=50, is_cuda=True)

            for j, sent_idx in enumerate(all_sent_indices[idx:idx + bsize]):
                idx_to_sent_embedding[sent_idx] = embeddings[j, :]

            all_sent_vec.append(embeddings)
            processed_sent += 1

            if processed_sent % print_every == 0:
                logger.info(time.time() - start)

        sent_vec = np.vstack(all_sent_vec)

        train_sent_vec.append(np.squeeze(np.mean(sent_vec, axis=0)))
        train_word_vec.append(word_vec[word])

    return train_sent_vec, train_word_vec

def build_bert_wordvec(train_words, test_words):
    # BERT uses WordPiece to tokenize
    # but frequent enough words remain as original words
    word_vectors = model.embeddings.word_embeddings.weight
    vocab_dict = tokenizer.vocab

    word_vec = {}

    # we build a dictionary from word to vectors
    for word in train_words + test_words:
        if word not in vocab_dict:
            raise Exception("{} word not found".format(word))
        else:
            word_vec[word] = word_vectors[vocab_dict[word]]

    return word_vec

if __name__ == '__main__':

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model = BertModel.from_pretrained('bert-base-cased')

    model.eval()

    word_vec = build_bert_wordvec(train_words, test_words)

    # load to GPU
    model.to(0)

    train_sent_vec, train_word_vec = gen_train_test(train_words[:1000], bsize=32, print_every=100)
    test_sent_vec, test_word_vec = gen_train_test(test_words[:100], bsize=32, print_every=100)

    # Currently we select the [CLS] token
    # Then we can select the actual word's position

    print("saving bert pickle file...")
    pickle.dump([train_sent_vec, train_word_vec, test_sent_vec, test_word_vec], open('./bert_cls_train_test.pkl', 'wb'))

    print("bert pickle file saved...")
