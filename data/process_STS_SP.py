"""
Make this into the same format as STSBenchmark
(with fewer columns)

similarity_score	S1	S2
"""

import os
import numpy as np

pjoin = os.path.join

output_directory = "senteval_data/STS_SP/STSBenchmark/"
if not os.path.exists(output_directory):
	os.makedirs(output_directory)
output_prefix = pjoin(output_directory, "sts-")
output_suffix = ".csv"

all_data = []

corpora = {
	14: ["li65", "news", "wikipedia"],
	15: ["newswire", "wikipedia"],
	17: ["track3.es-es"]
}

for yr in [14,15,17]:
	directory = "STS{}".format(yr)
	for corpus in corpora[yr]:
		gs_file = "senteval_data/STS_SP/STS{}/STS.gs.{}.txt".format(yr, corpus)
		input_file = "senteval_data/STS_SP/STS{}/STS.input.{}.txt".format(yr, corpus)
		similarity_scores = [line[:-1] for line in open(gs_file).readlines()]
		s1_list, s2_list = zip(*[line.strip().split("\t") for line in open(input_file).readlines()])
		all_data += ["\t".join(t) for t in zip(similarity_scores, s1_list, s2_list)]

all_data = list(set(all_data))
np.random.shuffle(all_data)

splits = ["train", "dev", "test"]

n_pairs = len(all_data)
train_proportion = 0.9
test_proportion = (1 - train_proportion) / 2
dev_proportion = test_proportion
test_number = int(np.ceil(test_proportion*n_pairs))
dev_number = int(np.ceil(dev_proportion*n_pairs))
train_number = n_pairs - (test_number + dev_number)

data = {
	"train": all_data[:train_number],
	"dev": all_data[train_number:(train_number+dev_number)],
	"test": all_data[(train_number+dev_number):]
}

assert(len(data["train"]) + len(data["test"]) + len(data["dev"]) == n_pairs)

for split in splits:
	output_file = "{}{}{}".format(output_prefix, split, output_suffix)

	w = open(output_file, "w")
	w.write("")
	w.close()

	w = open(output_file, "a")
	for d in data[split]:
		line = d + "\n"
		w.write(line)
	w.close()

