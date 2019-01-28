import time
import pickle
import torch
import argparse
import numpy as np
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

def gen_train_test(encoder, train_words, bsize=32, print_every=100):
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
                if len(word_list) > 30:
                    content_word_idx = word_list.index(word)
                    # select the window
                    all_sents.append(' '.join(word_list[min(content_word_idx - 15, 0):content_word_idx + 15]))
                else:
                    all_sents.append(corpora[sent_i])
                all_sent_indices.append(sent_i)

        assert len(all_sents) == len(all_sent_indices)

        for idx in range(0, len(all_sents), bsize):
            sents = all_sents[idx:idx + bsize]
            embeddings = encoder.encode(sents, tokenize=False, bsize=bsize)

            for j, sent_idx in enumerate(all_sent_indices[idx:idx + bsize]):
                idx_to_sent_embedding[sent_idx] = embeddings[j, :]

            all_sent_vec.append(embeddings)
            processed_sent += 1

            if processed_sent % print_every == 0:
                print(time.time() - start)

        sent_vec = np.vstack(all_sent_vec)

        train_sent_vec.append(np.squeeze(np.mean(sent_vec, axis=0)))
        train_word_vec.append(encoder.word_vec[word])

    return train_sent_vec, train_word_vec

if __name__ == '__main__':
    infersent = torch.load('infersent.allnli.pickle')
    infersent.set_glove_path("/home/anie/glove/glove.840B.300d.txt")
    infersent.build_vocab(corpora, tokenize=False)

    infersent.eval()

    train_sent_vec, train_word_vec = gen_train_test(infersent, train_words[:1000], bsize=32, print_every=1000)
    test_sent_vec, test_word_vec = gen_train_test(infersent, test_words[:100], bsize=32, print_every=1000)

    print("saving pickle file...")
    pickle.dump([train_sent_vec, train_word_vec, test_sent_vec, test_word_vec], open('./infersent_train_test.pkl', 'wb'))

    print("pickle file saved...")

    del infersent

    print("start dissent")

    map_locations = {}
    for d in range(4):
        map_locations['cuda:{}'.format(d)] = "cuda:{}".format(0)

    dissent = torch.load("/mnt/fs5/anie/DisExtract/exps/books_all/dis-model-10.pickle", map_location=map_locations)

    dissent = dissent.encoder
    dissent.set_glove_path("/home/anie/glove/glove.840B.300d.txt")
    dissent.build_vocab(corpora, tokenize=False)

    dissent.eval()

    train_sent_vec, train_word_vec = gen_train_test(dissent, train_words[:1000], bsize=32, print_every=1000)
    test_sent_vec, test_word_vec = gen_train_test(dissent, test_words[:100], bsize=32, print_every=1000)

    print("saving dissent pickle file...")
    pickle.dump([train_sent_vec, train_word_vec, test_sent_vec, test_word_vec], open('./infersent_train_test.pkl', 'wb'))

    print("dissent pickle file saved...")
